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
- KV-cache for fast autoregressive generation
- Multi-GPU training via DistributedDataParallel (DDP)
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- Attention masks for padded sequences
- Learning rate scheduling with warmup

Based on the architecture from "Attention Is All You Need" (Vaswani et al., 2017)
and "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure logging
logger = logging.getLogger(__name__)


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

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"


@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    During generation, we don't need to recompute K,V for previous tokens.
    We cache them and only compute K,V for the new token, then concatenate.

    This reduces generation complexity from O(n²) to O(n) per token.
    """

    key: torch.Tensor    # (batch, n_head, seq_len, head_dim)
    value: torch.Tensor  # (batch, n_head, seq_len, head_dim)

    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value and return the full cached tensors.

        Args:
            key: New key tensor (batch, n_head, 1, head_dim)
            value: New value tensor (batch, n_head, 1, head_dim)

        Returns:
            Updated (key, value) tensors with full sequence
        """
        self.key = torch.cat([self.key, key], dim=2)
        self.value = torch.cat([self.value, value], dim=2)
        return self.key, self.value

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self.key.size(2)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.

    Implements scaled dot-product attention with a causal mask to prevent
    tokens from attending to future positions. Uses Flash Attention when
    available for better performance.

    Supports:
    - KV-cache for efficient generation
    - Attention masks for padded sequences
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Combined Q, K, V projection (more efficient than separate projections)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check if Flash Attention is available
        self.use_flash = (
            config.use_flash_attention and
            hasattr(F, 'scaled_dot_product_attention')
        )

        if not self.use_flash:
            # Causal mask for manual attention (only needed without Flash Attention)
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
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: Optional mask (batch_size, seq_len) where 1=attend, 0=ignore
            kv_cache: Optional KV cache for generation
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output tensor, optional updated KV cache)
        """
        B, T, C = x.size()

        # Calculate Q, K, V in a single projection and split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Handle KV cache for generation
        new_cache = None
        if kv_cache is not None:
            # Append new K,V to cache and get full sequence
            k, v = kv_cache.update(k, v)

        if use_cache:
            new_cache = KVCache(key=k.clone(), value=v.clone())

        # Get sequence lengths for attention
        q_len = q.size(2)
        kv_len = k.size(2)

        if self.use_flash:
            # Build attention mask for Flash Attention
            attn_mask = None
            if attention_mask is not None:
                # Convert (B, kv_len) mask to (B, 1, q_len, kv_len) for broadcasting
                # Flash attention expects: True = attend, False = mask out
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(B, 1, q_len, kv_len)
                attn_mask = attn_mask.bool()

            # Use Flash Attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(kv_cache is None and attention_mask is None)  # Only use built-in causal for non-cached
            )
        else:
            # Manual attention implementation (fallback)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # Apply causal mask
            if kv_cache is None:
                # Standard causal mask during training
                att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            else:
                # During generation with cache, only mask future positions
                # Q is for position kv_len-1, K/V are for positions 0 to kv_len-1
                # No masking needed since we only have one query position attending to all past
                pass

            # Apply attention mask for padding
            if attention_mask is not None:
                # attention_mask: (B, kv_len) -> (B, 1, 1, kv_len)
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                att = att.masked_fill(mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, q_len, C)

        # Output projection with dropout
        y = self.resid_dropout(self.c_proj(y))

        return y, new_cache


class MLP(nn.Module):
    """
    Feed-forward network (MLP) for transformer blocks.

    Implements: FFN(x) = dropout(Linear(GELU(Linear(x))))
    Following GPT-2, the hidden dimension is 4x the embedding dimension.
    """

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
    """
    Transformer block with pre-normalization.

    Implements:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

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
        """Forward pass with optional KV cache support."""
        # Self-attention with residual connection
        attn_out, new_cache = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    """
    GPT Language Model.

    Architecture:
        1. Token embedding + positional embedding
        2. Dropout
        3. N transformer blocks
        4. Final layer normalization
        5. Linear projection to vocabulary (language modeling head)
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

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        logger.info(f"GPT model initialized with {self.get_num_params()/1e6:.2f}M parameters")

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
        """
        Forward pass through the GPT model.

        Args:
            idx: Input token indices (batch_size, seq_len)
            targets: Target token indices for loss computation (optional)
            attention_mask: Mask for padded sequences (batch_size, seq_len), 1=valid, 0=pad
            kv_caches: List of KV caches for each layer (for generation)
            use_cache: Whether to return updated KV caches

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            new_kv_caches: Updated KV caches if use_cache=True, else None
        """
        device = idx.device
        B, T = idx.size()

        # Determine position offset (for cached generation)
        pos_offset = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
            pos_offset = kv_caches[0].seq_len

        assert pos_offset + T <= self.config.block_size, \
            f"Sequence length {pos_offset + T} exceeds block size {self.config.block_size}"

        # Create position indices
        pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=device)

        # Get embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks with optional caching
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

        # Compute logits and loss
        if targets is not None:
            logits = self.lm_head(x)
            # Flatten for cross-entropy, handle attention mask for loss
            if attention_mask is not None:
                # Only compute loss on non-padded positions
                loss_mask = attention_mask.view(-1)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                # Set padded targets to ignore_index
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
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively with KV-cache support.

        Args:
            idx: Conditioning token indices (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (>0)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this probability threshold
            use_cache: Use KV-cache for faster generation (recommended)

        Returns:
            Generated token indices (batch_size, seq_len + max_new_tokens)
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0 or None, got {top_k}")
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        was_training = self.training
        self.eval()

        try:
            kv_caches = None

            for _ in range(max_new_tokens):
                # Determine input for this step
                if use_cache and kv_caches is not None:
                    # Only feed the last token when using cache
                    idx_cond = idx[:, -1:]
                else:
                    # First iteration or no cache: feed full sequence
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

                # Forward pass
                logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches, use_cache=use_cache)

                # Get logits for the last position and apply temperature
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

        finally:
            if was_training:
                self.train()

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with weight decay separation."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Try fused AdamW
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
        """Save model checkpoint."""
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
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        logger.info(f"Loaded checkpoint from {path}")
        return model, checkpoint


# ============================================================================
# Training utilities
# ============================================================================

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of training data."""
    if len(data) <= block_size:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than block_size ({block_size})"
        )
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
    """
    Create a padded batch from variable-length sequences.

    Args:
        sequences: List of token sequences (variable length)
        block_size: Maximum sequence length
        device: Target device
        pad_token_id: Token ID used for padding

    Returns:
        x: Input tensor (batch_size, max_len)
        y: Target tensor (batch_size, max_len)
        attention_mask: Mask tensor (batch_size, max_len), 1=valid, 0=pad
    """
    batch_size = len(sequences)
    max_len = min(block_size, max(len(s) for s in sequences))

    x = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    y = torch.full((batch_size, max_len), -1, dtype=torch.long)  # -1 for ignore_index
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        x[i, :seq_len] = torch.tensor(seq[:seq_len])
        if seq_len > 1:
            y[i, :seq_len - 1] = torch.tensor(seq[1:seq_len])
        attention_mask[i, :seq_len] = 1

    return x.to(device), y.to(device), attention_mask.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str
) -> dict:
    """Estimate loss on train and validation sets."""
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
    """Learning rate schedule with linear warmup and cosine decay."""
    if iter_num < warmup_iters:
        return max_lr * (iter_num + 1) / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# Multi-GPU Training Setup (DDP)
# ============================================================================

def setup_distributed():
    """
    Initialize distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size, is_master)
    """
    if not torch.distributed.is_initialized():
        # Check if we're in a distributed environment
        if 'RANK' in os.environ:
            torch.distributed.init_process_group(backend='nccl')
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            # Single GPU fallback
            rank = 0
            local_rank = 0
            world_size = 1
    else:
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = torch.distributed.get_world_size()

    is_master = rank == 0
    return rank, local_rank, world_size, is_master


def cleanup_distributed():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wrap_model_ddp(model: GPT, local_rank: int) -> nn.Module:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: The GPT model
        local_rank: Local GPU rank

    Returns:
        DDP-wrapped model
    """
    return DDP(model, device_ids=[local_rank])


# ============================================================================
# Tokenizers
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer with unknown character handling."""

    UNK_TOKEN = '<UNK>'
    PAD_TOKEN = '<PAD>'

    def __init__(self, text: str):
        chars = sorted(list(set(text)))

        # Special tokens
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
                       if i not in (self.pad_token_id,)])


class BPETokenizerWrapper:
    """
    Wrapper to integrate the cortical BPE tokenizer with nanoGPT.

    Usage:
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.common.filesystem import RealFileSystem

        bpe = BPETokenizer()
        bpe.learn_from_texts(texts)
        tokenizer = BPETokenizerWrapper(bpe)
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, bpe_tokenizer: Any):
        """
        Initialize wrapper around a cortical BPETokenizer.

        Args:
            bpe_tokenizer: Instance of cortical.cognitive.text_bridge.BPETokenizer
        """
        self.bpe = bpe_tokenizer

        # Build vocabulary mapping
        # Reserve first 4 indices for special tokens
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        for i, token in enumerate(self.special_tokens):
            self.stoi[token] = i
            self.itos[i] = token

        # Add BPE vocabulary
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
        """
        Encode text to token indices.

        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token

        Returns:
            List of token indices
        """
        # Use BPE tokenizer to get word tokens
        tokens = self.bpe.tokenize(text)

        # Convert to indices
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)

        for token in tokens:
            ids.append(self.stoi.get(token, self.unk_token_id))

        if add_eos:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Decode token indices to text.

        Args:
            indices: List of token indices
            skip_special: Skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []
        special_ids = set([self.pad_token_id, self.unk_token_id,
                          self.bos_token_id, self.eos_token_id])

        for idx in indices:
            if skip_special and idx in special_ids:
                continue
            token = self.itos.get(idx, self.UNK_TOKEN)
            tokens.append(token)

        return ' '.join(tokens)


# ============================================================================
# Demo with Multi-GPU Support
# ============================================================================

def demo(
    use_ddp: bool = False,
    gradient_accumulation_steps: int = 1,
):
    """
    Demonstrate nanoGPT training.

    Args:
        use_ddp: Enable multi-GPU training with DDP
        gradient_accumulation_steps: Number of steps to accumulate gradients

    To run multi-GPU:
        torchrun --nproc_per_node=4 nanogpt.py
    """
    import urllib.request

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup distributed if requested
    if use_ddp or 'RANK' in os.environ:
        rank, local_rank, world_size, is_master = setup_distributed()
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size, is_master = 0, 0, 1, True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    if is_master:
        print("=" * 60)
        print("nanoGPT Demo: Training a small language model")
        print(f"Using {world_size} GPU(s), gradient accumulation steps: {gradient_accumulation_steps}")
        print("=" * 60)

    # Hyperparameters
    micro_batch_size = 64
    batch_size = micro_batch_size * gradient_accumulation_steps * world_size
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_iters = 100
    lr_decay_iters = max_iters
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    grad_clip = 1.0

    if is_master:
        print(f"Effective batch size: {batch_size}")
        print(f"Using device: {device}")

    # Mixed precision
    use_amp = device_type == 'cuda'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.autocast(device_type=device_type, dtype=dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    # Get data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, 'input.txt')
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    if is_master and not os.path.exists(data_path):
        print("Downloading Shakespeare dataset...")
        try:
            urllib.request.urlretrieve(data_url, data_path)
        except Exception as e:
            print(f"Could not download: {e}, using sample text")
            sample = "To be, or not to be, that is the question.\n" * 1000
            with open(data_path, 'w') as f:
                f.write(sample)

    # Sync before loading data in multi-GPU
    if use_ddp:
        torch.distributed.barrier()

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    if is_master:
        print(f"Vocab size: {tokenizer.vocab_size}, Train: {len(train_data)}, Val: {len(val_data)}")

    # Create model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )

    model = GPT(config)
    model = model.to(device)

    if use_ddp:
        model = wrap_model_ddp(model, local_rank)

    # Get raw model for saving
    raw_model = model.module if use_ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        betas=(0.9, 0.99),
        device_type=device_type
    )

    # Training loop
    if is_master:
        print("\nStarting training...")
        print("-" * 60)

    best_val_loss = float('inf')

    for iter_num in range(max_iters):
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, min_lr, max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            if is_master:
                losses = estimate_loss(raw_model, train_data, val_data,
                                       eval_iters, micro_batch_size, block_size, device)
                print(f"Step {iter_num:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    raw_model.save_checkpoint(
                        os.path.join(data_dir, 'best_model.pt'),
                        optimizer=optimizer,
                        iter_num=iter_num,
                        best_val_loss=best_val_loss
                    )

        # Training step with gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(gradient_accumulation_steps):
            xb, yb = get_batch(train_data, micro_batch_size, block_size, device)

            # Sync gradients only on last micro step
            if use_ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with ctx:
                _, loss, _ = model(xb, yb)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

    if is_master:
        print("-" * 60)
        print("Training complete!")

        # Generate sample
        print("\n" + "=" * 60)
        print("Generating sample text (with KV-cache)...")
        print("=" * 60 + "\n")

        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = raw_model.generate(
            context, max_new_tokens=500,
            temperature=0.8, top_k=40, top_p=0.95,
            use_cache=True
        )
        print(tokenizer.decode(generated[0].tolist()))

    if use_ddp:
        cleanup_distributed()

    return raw_model, tokenizer


class nullcontext:
    """Context manager that does nothing."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == '__main__':
    import sys

    # Check for multi-GPU flag
    use_ddp = '--ddp' in sys.argv or 'RANK' in os.environ

    # Parse gradient accumulation
    grad_accum = 1
    for arg in sys.argv:
        if arg.startswith('--grad_accum='):
            grad_accum = int(arg.split('=')[1])

    demo(use_ddp=use_ddp, gradient_accumulation_steps=grad_accum)
