"""
nanoGPT - A minimal, single-file GPT implementation in PyTorch.

This implementation follows the GPT-2 architecture with:
- Learnable token and positional embeddings
- Pre-norm transformer blocks (LayerNorm before attention/FFN)
- Multi-head causal self-attention with scaled dot-product
- GELU-activated feed-forward networks
- Residual connections

Features:
- Flash Attention support (PyTorch 2.0+)
- KV-cache with eviction for long sequence generation
- Multi-GPU training via DDP (small models) or FSDP (>1B params)
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- Attention masks for padded sequences
- Learning rate scheduling with warmup

Based on the architecture from "Attention Is All You Need" (Vaswani et al., 2017)
and "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).
"""

import functools
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ParallelMode(Enum):
    """Training parallelism strategies."""
    NONE = "none"           # Single GPU
    DDP = "ddp"             # DistributedDataParallel (replicate model on each GPU)
    FSDP = "fsdp"           # Fully Sharded Data Parallel (shard model across GPUs)
    FSDP_HYBRID = "hybrid"  # FSDP with intra-node sharding, DDP across nodes


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""

    vocab_size: int = 50257  # GPT-2 vocabulary size
    block_size: int = 1024   # Maximum sequence length (context window)
    n_layer: int = 12        # Number of transformer blocks
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768        # Embedding dimension
    dropout: float = 0.1     # Dropout probability
    bias: bool = True        # Use bias in Linear layers and LayerNorms
    use_flash_attention: bool = True  # Use Flash Attention if available

    # KV-cache settings
    max_cache_length: Optional[int] = None  # Max KV-cache length (None = unlimited)
    cache_eviction_strategy: str = "sliding_window"  # "sliding_window" or "attention_sink"
    sink_tokens: int = 4  # Number of initial tokens to always keep (for attention_sink)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

    def estimate_params(self) -> int:
        """Estimate total parameters (useful for choosing parallel strategy)."""
        # Embeddings
        emb_params = self.vocab_size * self.n_embd + self.block_size * self.n_embd

        # Per-layer params
        # Attention: Q,K,V projection (3*n_embd*n_embd) + output projection (n_embd*n_embd)
        attn_params = 4 * self.n_embd * self.n_embd
        # MLP: up projection (n_embd*4*n_embd) + down projection (4*n_embd*n_embd)
        mlp_params = 8 * self.n_embd * self.n_embd
        # LayerNorms: 2 per layer, each has n_embd*2 params
        ln_params = 4 * self.n_embd

        layer_params = attn_params + mlp_params + ln_params
        total_layer_params = self.n_layer * layer_params

        # Final LayerNorm + LM head (tied with embeddings, so not counted twice)
        final_params = 2 * self.n_embd

        return emb_params + total_layer_params + final_params


# ============================================================================
# KV-Cache with Eviction
# ============================================================================

@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation with eviction support.

    Supports two eviction strategies for long sequences:
    1. sliding_window: Keep only the most recent max_length tokens
    2. attention_sink: Keep first sink_tokens + recent tokens (StreamingLLM approach)

    The attention_sink strategy is based on "Efficient Streaming Language Models
    with Attention Sinks" (Xiao et al., 2023) - initial tokens act as "sinks"
    that stabilize attention patterns.
    """

    key: torch.Tensor           # (batch, n_head, seq_len, head_dim)
    value: torch.Tensor         # (batch, n_head, seq_len, head_dim)
    max_length: Optional[int] = None
    eviction_strategy: str = "sliding_window"
    sink_tokens: int = 4
    _position_offset: int = 0   # Tracks actual position for RoPE compatibility

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value and apply eviction if necessary.

        Args:
            key: New key tensor (batch, n_head, new_len, head_dim)
            value: New value tensor (batch, n_head, new_len, head_dim)

        Returns:
            Updated (key, value) tensors for attention computation
        """
        # Concatenate new K,V
        self.key = torch.cat([self.key, key], dim=2)
        self.value = torch.cat([self.value, value], dim=2)

        # Apply eviction if over max_length
        if self.max_length is not None and self.key.size(2) > self.max_length:
            self._evict()

        return self.key, self.value

    def _evict(self):
        """Apply eviction strategy to reduce cache size."""
        current_len = self.key.size(2)
        tokens_to_remove = current_len - self.max_length

        if tokens_to_remove <= 0:
            return

        if self.eviction_strategy == "sliding_window":
            # Simple: remove oldest tokens
            self.key = self.key[:, :, tokens_to_remove:, :]
            self.value = self.value[:, :, tokens_to_remove:, :]
            self._position_offset += tokens_to_remove

        elif self.eviction_strategy == "attention_sink":
            # Keep sink tokens (first N) + most recent tokens
            # This preserves attention stability per StreamingLLM
            keep_recent = self.max_length - self.sink_tokens

            if self.sink_tokens > 0 and current_len > self.sink_tokens:
                # Keep first sink_tokens and last keep_recent tokens
                sink_k = self.key[:, :, :self.sink_tokens, :]
                sink_v = self.value[:, :, :self.sink_tokens, :]
                recent_k = self.key[:, :, -keep_recent:, :]
                recent_v = self.value[:, :, -keep_recent:, :]

                self.key = torch.cat([sink_k, recent_k], dim=2)
                self.value = torch.cat([sink_v, recent_v], dim=2)

                # Position offset tracks where "recent" starts in absolute terms
                self._position_offset = current_len - keep_recent
            else:
                # Fallback to sliding window if not enough tokens for sinks
                self.key = self.key[:, :, tokens_to_remove:, :]
                self.value = self.value[:, :, tokens_to_remove:, :]
                self._position_offset += tokens_to_remove

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self.key.size(2)

    @property
    def position_offset(self) -> int:
        """Position offset for correct positional encoding after eviction."""
        return self._position_offset

    @classmethod
    def empty(
        cls,
        batch_size: int,
        n_head: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        max_length: Optional[int] = None,
        eviction_strategy: str = "sliding_window",
        sink_tokens: int = 4,
    ) -> 'KVCache':
        """Create an empty KV cache."""
        key = torch.empty(batch_size, n_head, 0, head_dim, device=device, dtype=dtype)
        value = torch.empty(batch_size, n_head, 0, head_dim, device=device, dtype=dtype)
        return cls(
            key=key,
            value=value,
            max_length=max_length,
            eviction_strategy=eviction_strategy,
            sink_tokens=sink_tokens,
        )


# ============================================================================
# Model Components
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with KV-cache and eviction support.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = (
            config.use_flash_attention and
            hasattr(F, 'scaled_dot_product_attention')
        )

        if not self.use_flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Handle KV cache
        new_cache = None
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if use_cache:
            new_cache = KVCache(
                key=k.clone(),
                value=v.clone(),
                max_length=self.config.max_cache_length,
                eviction_strategy=self.config.cache_eviction_strategy,
                sink_tokens=self.config.sink_tokens,
            )
            if kv_cache is not None:
                new_cache._position_offset = kv_cache._position_offset

        q_len = q.size(2)
        kv_len = k.size(2)

        if self.use_flash:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(B, 1, q_len, kv_len).bool()

            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(kv_cache is None and attention_mask is None)
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            if kv_cache is None:
                att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                att = att.masked_fill(mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, q_len, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, new_cache


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        attn_out, new_cache = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    """
    GPT Language Model with support for large-scale training.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = self.get_num_params()
        logger.info(f"GPT model initialized with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[KVCache]]]:
        device = idx.device
        B, T = idx.size()

        # Handle position offset from cache eviction
        pos_offset = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
            pos_offset = kv_caches[0].seq_len

        # For evicted caches, we need absolute positions
        # but clamp to block_size for the embedding lookup
        pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=device)
        pos = pos.clamp(max=self.config.block_size - 1)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)

        new_kv_caches = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(
                x,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(new_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if attention_mask is not None:
                loss_mask = attention_mask.view(-1)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                targets_flat = torch.where(loss_mask == 1, targets_flat, torch.tensor(-1, device=device))
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        max_cache_length: Optional[int] = None,
        cache_eviction_strategy: str = "sliding_window",
    ) -> torch.Tensor:
        """
        Generate tokens with KV-cache and optional eviction for long sequences.

        Args:
            idx: Input token indices (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            use_cache: Use KV-cache
            max_cache_length: Max cache size (enables eviction if set)
            cache_eviction_strategy: "sliding_window" or "attention_sink"
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0 or None, got {top_k}")
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        # Override config cache settings if provided
        original_max_cache = self.config.max_cache_length
        original_strategy = self.config.cache_eviction_strategy

        if max_cache_length is not None:
            self.config.max_cache_length = max_cache_length
            self.config.cache_eviction_strategy = cache_eviction_strategy

        was_training = self.training
        self.eval()

        try:
            kv_caches = None

            for _ in range(max_new_tokens):
                if use_cache and kv_caches is not None:
                    idx_cond = idx[:, -1:]
                else:
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

                logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches, use_cache=use_cache)

                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

        finally:
            if was_training:
                self.train()
            # Restore original config
            self.config.max_cache_length = original_max_cache
            self.config.cache_eviction_strategy = original_strategy

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        use_fused = False
        if device_type == 'cuda':
            try:
                test_opt = torch.optim.AdamW([torch.zeros(1)], fused=True)
                del test_opt
                use_fused = True
            except (TypeError, RuntimeError):
                pass

        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        logger.info(f"Using fused AdamW: {use_fused}")
        return optimizer

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                        iter_num: int = 0, best_val_loss: float = float('inf'), **extra_state):
        checkpoint = {
            'model': self.state_dict(),
            'config': self.config,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint.update(extra_state)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu') -> Tuple['GPT', dict]:
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        logger.info(f"Loaded checkpoint from {path}")
        return model, checkpoint


# ============================================================================
# Training Utilities
# ============================================================================

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError(f"Data length ({len(data)}) must be > block_size ({block_size})")
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_batch_padded(
    sequences: List[List[int]],
    block_size: int,
    device: str,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    max_len = min(block_size, max(len(s) for s in sequences))

    x = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    y = torch.full((batch_size, max_len), -1, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        x[i, :seq_len] = torch.tensor(seq[:seq_len])
        if seq_len > 1:
            y[i, :seq_len - 1] = torch.tensor(seq[1:seq_len])
        attention_mask[i, :seq_len] = 1

    return x.to(device), y.to(device), attention_mask.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(iter_num: int, warmup_iters: int, lr_decay_iters: int,
           min_lr: float, max_lr: float) -> float:
    if iter_num < warmup_iters:
        return max_lr * (iter_num + 1) / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# Multi-GPU Training: DDP and FSDP
# ============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if not torch.distributed.is_initialized():
        if 'RANK' in os.environ:
            torch.distributed.init_process_group(backend='nccl')
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            rank, local_rank, world_size = 0, 0, 1
    else:
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = torch.distributed.get_world_size()

    return rank, local_rank, world_size, (rank == 0)


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wrap_model_ddp(model: GPT, local_rank: int) -> nn.Module:
    """Wrap with DistributedDataParallel."""
    return DDP(model, device_ids=[local_rank])


def get_fsdp_wrap_policy(min_num_params: int = 100_000_000) -> Callable:
    """
    Create FSDP wrap policy that wraps transformer blocks.

    For models >1B params, we wrap each Block separately to enable
    parameter sharding across GPUs.
    """
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    # Wrap Block modules (each transformer layer)
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )


def wrap_model_fsdp(
    model: GPT,
    local_rank: int,
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD",
) -> nn.Module:
    """
    Wrap model with Fully Sharded Data Parallel for large models (>1B params).

    FSDP shards model parameters, gradients, and optimizer states across GPUs,
    enabling training of models that don't fit in single GPU memory.

    Sharding strategies:
    - FULL_SHARD: Shard everything (most memory efficient)
    - SHARD_GRAD_OP: Shard gradients and optimizer states only
    - NO_SHARD: Like DDP, for comparison
    - HYBRID_SHARD: Full shard within node, replicate across nodes

    Args:
        model: GPT model to wrap
        local_rank: Local GPU rank
        mixed_precision: Use bfloat16/float16 for compute
        cpu_offload: Offload parameters to CPU (slower but saves GPU memory)
        sharding_strategy: One of FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
        BackwardPrefetch,
    )

    # Map string to enum
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mp_policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    # CPU offload (trades speed for memory)
    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None

    # Wrap policy - wraps each Block for fine-grained sharding
    wrap_policy = get_fsdp_wrap_policy()

    # Create FSDP model
    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_policy,
        auto_wrap_policy=wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        limit_all_gathers=True,  # Memory optimization
        use_orig_params=True,  # Better for optimizer state dicts
    )

    return fsdp_model


def get_fsdp_state_dict(model: nn.Module) -> dict:
    """Get full state dict from FSDP model for checkpointing."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    return state_dict


def load_fsdp_state_dict(model: nn.Module, state_dict: dict):
    """Load state dict into FSDP model."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )

    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(state_dict)


def choose_parallel_strategy(config: GPTConfig, world_size: int) -> ParallelMode:
    """
    Automatically choose the best parallelism strategy based on model size.

    Rules:
    - < 500M params: DDP is fine
    - 500M - 2B params: FSDP recommended
    - > 2B params: FSDP required, consider CPU offload

    Args:
        config: Model configuration
        world_size: Number of GPUs

    Returns:
        Recommended ParallelMode
    """
    estimated_params = config.estimate_params()

    if world_size == 1:
        return ParallelMode.NONE

    if estimated_params < 500_000_000:  # < 500M
        logger.info(f"Model has {estimated_params/1e6:.0f}M params - using DDP")
        return ParallelMode.DDP

    elif estimated_params < 2_000_000_000:  # 500M - 2B
        logger.info(f"Model has {estimated_params/1e6:.0f}M params - using FSDP")
        return ParallelMode.FSDP

    else:  # > 2B
        logger.info(f"Model has {estimated_params/1e9:.1f}B params - using FSDP (consider CPU offload)")
        return ParallelMode.FSDP


# ============================================================================
# Tokenizers
# ============================================================================

class CharTokenizer:
    UNK_TOKEN = '<UNK>'
    PAD_TOKEN = '<PAD>'

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.itos = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.stoi = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for i, ch in enumerate(chars, start=2):
            self.itos[i] = ch
            self.stoi[ch] = i
        self.vocab_size = len(self.itos)
        self.pad_token_id = 0
        self.unk_token_id = 1

    def encode(self, text: str) -> list:
        return [self.stoi.get(c, self.unk_token_id) for c in text]

    def decode(self, indices: list) -> str:
        return ''.join([self.itos.get(i, self.UNK_TOKEN) for i in indices
                       if i != self.pad_token_id])


class BPETokenizerWrapper:
    """Wrapper for cortical BPE tokenizer."""

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, bpe_tokenizer: Any):
        self.bpe = bpe_tokenizer
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        for i, token in enumerate(self.special_tokens):
            self.stoi[token] = i
            self.itos[i] = token

        offset = len(self.special_tokens)
        for i, word in enumerate(sorted(self.bpe.vocab)):
            idx = i + offset
            self.stoi[word] = idx
            self.itos[idx] = word

        self.vocab_size = len(self.stoi)
        self.pad_token_id = self.stoi[self.PAD_TOKEN]
        self.unk_token_id = self.stoi[self.UNK_TOKEN]
        self.bos_token_id = self.stoi[self.BOS_TOKEN]
        self.eos_token_id = self.stoi[self.EOS_TOKEN]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self.bpe.tokenize(text)
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for token in tokens:
            ids.append(self.stoi.get(token, self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        tokens = []
        special_ids = {self.pad_token_id, self.unk_token_id,
                       self.bos_token_id, self.eos_token_id}
        for idx in indices:
            if skip_special and idx in special_ids:
                continue
            tokens.append(self.itos.get(idx, self.UNK_TOKEN))
        return ' '.join(tokens)


# ============================================================================
# Demo
# ============================================================================

def demo(
    use_fsdp: bool = False,
    gradient_accumulation_steps: int = 1,
    cpu_offload: bool = False,
):
    """
    Demo with automatic DDP/FSDP selection.

    Usage:
        # Single GPU
        python nanogpt.py

        # Multi-GPU with DDP (auto-selected for small models)
        torchrun --nproc_per_node=4 nanogpt.py

        # Multi-GPU with FSDP (for large models, or force with flag)
        torchrun --nproc_per_node=4 nanogpt.py --fsdp

        # FSDP with CPU offload (for very large models)
        torchrun --nproc_per_node=4 nanogpt.py --fsdp --cpu_offload
    """
    import urllib.request

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    rank, local_rank, world_size, is_master = setup_distributed()

    if world_size > 1:
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    if is_master:
        print("=" * 60)
        print("nanoGPT Demo")
        print(f"GPUs: {world_size}, Grad accum: {gradient_accumulation_steps}")
        print(f"FSDP: {use_fsdp}, CPU offload: {cpu_offload}")
        print("=" * 60)

    # Hyperparameters
    micro_batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_iters = 100
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    grad_clip = 1.0

    # Get data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, 'input.txt')
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    if is_master and not os.path.exists(data_path):
        print("Downloading dataset...")
        try:
            urllib.request.urlretrieve(data_url, data_path)
        except Exception as e:
            sample = "To be, or not to be, that is the question.\n" * 1000
            with open(data_path, 'w') as f:
                f.write(sample)

    if world_size > 1:
        torch.distributed.barrier()

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Create model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        max_cache_length=512,  # Enable cache eviction
        cache_eviction_strategy="attention_sink",
        sink_tokens=4,
    )

    model = GPT(config)

    # Choose parallelism strategy
    if world_size > 1:
        if use_fsdp:
            parallel_mode = ParallelMode.FSDP
        else:
            parallel_mode = choose_parallel_strategy(config, world_size)
    else:
        parallel_mode = ParallelMode.NONE

    # Apply parallelism
    if parallel_mode == ParallelMode.FSDP:
        model = wrap_model_fsdp(
            model, local_rank,
            mixed_precision=True,
            cpu_offload=cpu_offload,
        )
        raw_model = model  # FSDP model is the raw model
    elif parallel_mode == ParallelMode.DDP:
        model = model.to(device)
        model = wrap_model_ddp(model, local_rank)
        raw_model = model.module
    else:
        model = model.to(device)
        raw_model = model

    if is_master:
        print(f"Using {parallel_mode.value} parallelism")
        print(f"Model params: {config.estimate_params()/1e6:.2f}M")

    # Optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        betas=(0.9, 0.99),
        device_type=device_type
    )

    # Mixed precision
    use_amp = device_type == 'cuda' and parallel_mode != ParallelMode.FSDP
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.autocast(device_type=device_type, dtype=dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # Training
    if is_master:
        print("\nTraining...")
        print("-" * 60)

    for iter_num in range(max_iters):
        lr = get_lr(iter_num, warmup_iters, max_iters, min_lr, max_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            if is_master:
                losses = estimate_loss(raw_model, train_data, val_data,
                                       eval_iters, micro_batch_size, block_size, device)
                print(f"Step {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(gradient_accumulation_steps):
            xb, yb = get_batch(train_data, micro_batch_size, block_size, device)

            if parallel_mode == ParallelMode.DDP:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with ctx:
                _, loss, _ = model(xb, yb)
                loss = loss / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    if is_master:
        print("-" * 60)
        print("Training complete!")

        # Test cache eviction with long generation
        print("\n" + "=" * 60)
        print("Generating with KV-cache eviction (max_cache=512)...")
        print("=" * 60 + "\n")

        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = raw_model.generate(
            context, max_new_tokens=1000,  # Long generation
            temperature=0.8, top_k=40,
            use_cache=True,
            max_cache_length=512,
            cache_eviction_strategy="attention_sink",
        )
        print(tokenizer.decode(generated[0].tolist()))

    if world_size > 1:
        cleanup_distributed()

    return raw_model, tokenizer


class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == '__main__':
    import sys

    use_fsdp = '--fsdp' in sys.argv
    cpu_offload = '--cpu_offload' in sys.argv

    grad_accum = 1
    for arg in sys.argv:
        if arg.startswith('--grad_accum='):
            grad_accum = int(arg.split('=')[1])

    demo(use_fsdp=use_fsdp, gradient_accumulation_steps=grad_accum, cpu_offload=cpu_offload)
