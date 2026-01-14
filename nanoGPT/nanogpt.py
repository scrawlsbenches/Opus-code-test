"""
nanoGPT - A minimal, single-file GPT implementation in PyTorch.

This implementation follows the GPT-2 architecture with:
- Learnable token and positional embeddings
- Pre-norm transformer blocks (LayerNorm before attention/FFN)
- Multi-head causal self-attention with scaled dot-product
- GELU-activated feed-forward networks
- Residual connections

Based on the architecture from "Attention Is All You Need" (Vaswani et al., 2017)
and "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.

    Implements scaled dot-product attention with a causal mask to prevent
    tokens from attending to future positions. Uses Flash Attention when
    available for better performance.
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
            # Registered as buffer (not a parameter) so it's moved with the model
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V in a single projection and split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # Use Flash Attention (PyTorch 2.0+)
            # Handles causal masking and scaling internally
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention implementation (fallback)
            # Scaled dot-product attention: att = (Q @ K^T) / sqrt(d_k)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # Apply causal mask: set future positions to -inf before softmax
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

            # Softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Apply attention to values
            y = att @ v

        # Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with dropout
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """
    Feed-forward network (MLP) for transformer blocks.

    Implements: FFN(x) = dropout(Linear(GELU(Linear(x))))
    Following GPT-2, the hidden dimension is 4x the embedding dimension.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # First linear: expand to 4x embedding dimension
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GELU activation (Gaussian Error Linear Unit)
        self.gelu = nn.GELU()
        # Second linear: project back to embedding dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
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

    Pre-norm (LayerNorm before sublayers) is used instead of post-norm
    for improved training stability.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Layer normalization (before attention and MLP)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Attention and MLP sublayers
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block."""
        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


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
            # Token embeddings: vocab_size -> n_embd
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings: block_size -> n_embd
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            # Dropout after embeddings
            'drop': nn.Dropout(config.dropout),
            # Stack of transformer blocks
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })

        # Language modeling head: project from embedding space to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        # This reduces parameters and often improves performance
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections
        # (per GPT-2 paper: scale by 1/sqrt(2*n_layer))
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Log number of parameters
        logger.info(f"GPT model initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 scheme."""
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
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude position embeddings from count
                          (they are sometimes not counted as "real" parameters)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the GPT model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for computing loss (optional)

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        device = idx.device
        B, T = idx.size()

        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (T,)

        # Get token and position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        # Combine embeddings and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Compute logits and optionally loss
        if targets is not None:
            # Training: compute logits for all positions and calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens
            )
        else:
            # Inference: only compute logits for the last position (more efficient)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Conditioning token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic)
            top_k: If set, only sample from the top k most likely tokens

        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        # Input validation
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0 or None, got {top_k}")

        # Set to eval mode to disable dropout
        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # Crop sequence to block_size if necessary
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

                # Get predictions
                logits, _ = self(idx_cond)

                # Focus on the last token's logits and apply temperature
                logits = logits[:, -1, :] / temperature

                # Optionally apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to the sequence
                idx = torch.cat((idx, idx_next), dim=1)
        finally:
            # Restore original training mode
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
        """
        Configure the optimizer with weight decay.

        Separates parameters into two groups:
        - Parameters that should be weight-decayed (weights of linear layers)
        - Parameters that should not be weight-decayed (biases, LayerNorm, embeddings)

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type for fused optimizer selection

        Returns:
            Configured AdamW optimizer
        """
        # Collect all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate into decay and no-decay groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"Decayed parameter tensors: {len(decay_params)}, totaling {num_decay_params:,} parameters")
        logger.info(f"Non-decayed parameter tensors: {len(nodecay_params)}, totaling {num_nodecay_params:,} parameters")

        # Use fused AdamW if available (faster on CUDA)
        use_fused = False
        if device_type == 'cuda':
            try:
                # Test if fused parameter is accepted
                test_opt = torch.optim.AdamW([torch.zeros(1)], fused=True)
                del test_opt
                use_fused = True
            except (TypeError, RuntimeError):
                pass

        extra_args = {'fused': True} if use_fused else {}

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        logger.info(f"Using fused AdamW: {use_fused}")

        return optimizer

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                        iter_num: int = 0, best_val_loss: float = float('inf'),
                        **extra_state):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            iter_num: Current iteration number
            best_val_loss: Best validation loss so far
            **extra_state: Additional state to save
        """
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
    def load_checkpoint(cls, path: str, device: str = 'cpu'
                        ) -> Tuple['GPT', dict]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model to

        Returns:
            Tuple of (model, checkpoint_dict)
        """
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
    """
    Generate a random batch of training data.

    Args:
        data: 1D tensor of token indices
        batch_size: Number of sequences in the batch
        block_size: Length of each sequence
        device: Device to place tensors on

    Returns:
        x: Input sequences of shape (batch_size, block_size)
        y: Target sequences of shape (batch_size, block_size)

    Raises:
        ValueError: If data is too small for the given block_size
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than block_size ({block_size}). "
            f"Need at least {block_size + 1} tokens."
        )

    # Random starting positions
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start + 1, (batch_size,))

    # Extract sequences
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


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
    """
    Estimate loss on train and validation sets.

    Args:
        model: The GPT model
        train_data: Training data tensor
        val_data: Validation data tensor
        eval_iters: Number of iterations to average over
        batch_size: Batch size for evaluation
        block_size: Sequence length
        device: Device to use

    Returns:
        Dictionary with 'train' and 'val' loss estimates
    """
    out = {}
    model.eval()

    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def get_lr(iter_num: int, warmup_iters: int, lr_decay_iters: int,
           min_lr: float, max_lr: float) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay.

    Args:
        iter_num: Current iteration
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations for cosine decay
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate

    Returns:
        Learning rate for current iteration
    """
    # Linear warmup
    if iter_num < warmup_iters:
        return max_lr * (iter_num + 1) / warmup_iters

    # After decay period, return minimum
    if iter_num > lr_decay_iters:
        return min_lr

    # Cosine decay
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# Simple character-level tokenizer for demo purposes
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer with unknown character handling."""

    # Special token for unknown characters
    UNK_TOKEN = '<UNK>'

    def __init__(self, text: str):
        """Initialize tokenizer from text corpus."""
        chars = sorted(list(set(text)))

        # Add UNK token at position 0
        self.itos = {0: self.UNK_TOKEN}
        self.stoi = {self.UNK_TOKEN: 0}

        # Add all characters from corpus
        for i, ch in enumerate(chars, start=1):
            self.itos[i] = ch
            self.stoi[ch] = i

        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> list:
        """
        Convert text to list of token indices.
        Unknown characters are mapped to UNK token.
        """
        unk_idx = self.stoi[self.UNK_TOKEN]
        return [self.stoi.get(c, unk_idx) for c in text]

    def decode(self, indices: list) -> str:
        """Convert list of token indices to text."""
        return ''.join([self.itos.get(i, self.UNK_TOKEN) for i in indices])


# ============================================================================
# Demo: Train a small GPT on Shakespeare
# ============================================================================

def demo():
    """
    Demonstrate nanoGPT by training on a small dataset.

    This demo:
    1. Downloads Shakespeare text (or uses sample text)
    2. Creates a character-level tokenizer
    3. Trains a small GPT model with mixed precision and LR scheduling
    4. Generates sample text
    """
    import urllib.request

    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("nanoGPT Demo: Training a small language model")
    print("=" * 60)

    # Hyperparameters for the demo (small model for quick training)
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    max_lr = 3e-4
    min_lr = 3e-5  # 10x smaller than max
    warmup_iters = 100
    lr_decay_iters = max_iters
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    grad_clip = 1.0

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    print(f"Using device: {device}")

    # Mixed precision settings
    use_amp = device_type == 'cuda'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.autocast(device_type=device_type, dtype=dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))
    print(f"Using mixed precision: {use_amp}" + (f" ({dtype})" if use_amp else ""))

    # Get training data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, 'input.txt')
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    if not os.path.exists(data_path):
        print("Downloading Shakespeare dataset...")
        try:
            urllib.request.urlretrieve(data_url, data_path)
            print("Download complete!")
        except Exception as e:
            print(f"Could not download data: {e}")
            print("Using sample text instead...")
            # Fallback sample text
            sample_text = """
            To be, or not to be, that is the question:
            Whether 'tis nobler in the mind to suffer
            The slings and arrows of outrageous fortune,
            Or to take arms against a sea of troubles
            And by opposing end them. To die-to sleep,
            No more; and by a sleep to say we end
            The heart-ache and the thousand natural shocks
            That flesh is heir to: 'tis a consummation
            Devoutly to be wish'd. To die, to sleep;
            To sleep, perchance to dream-ay, there's the rub:
            For in that sleep of death what dreams may come,
            When we have shuffled off this mortal coil,
            Must give us pause-there's the respect
            That makes calamity of so long life.
            """ * 100  # Repeat to have enough data
            with open(data_path, 'w') as f:
                f.write(sample_text)

    # Load and tokenize data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Dataset size: {len(text):,} characters")

    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Encode the full text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

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

    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        betas=(0.9, 0.99),
        device_type=device_type
    )

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    best_val_loss = float('inf')

    for iter_num in range(max_iters):
        # Update learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, min_lr, max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate periodically
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                eval_iters, batch_size, block_size, device
            )
            print(f"Step {iter_num:5d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, lr {lr:.2e}")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint_path = os.path.join(data_dir, 'best_model.pt')
                model.save_checkpoint(
                    checkpoint_path,
                    optimizer=optimizer,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss
                )

        # Get batch and compute loss with mixed precision
        xb, yb = get_batch(train_data, batch_size, block_size, device)

        with ctx:
            logits, loss = model(xb, yb)

        # Backpropagation with gradient scaling for mixed precision
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

    print("-" * 60)
    print("Training complete!")

    # Generate sample text
    print("\n" + "=" * 60)
    print("Generating sample text...")
    print("=" * 60 + "\n")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print(tokenizer.decode(generated[0].tolist()))

    return model, tokenizer


# Null context manager for non-CUDA devices
class nullcontext:
    """Context manager that does nothing (for non-CUDA devices)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == '__main__':
    demo()
