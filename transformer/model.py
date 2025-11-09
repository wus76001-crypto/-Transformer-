import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelArgs:
    vocab_size: int = 32000  # 词表大小
    block_size: int = 1024   # 单次可处理的最大序列长度
    n_embd: int = 512        # 词向量维度
    n_heads: int = 8         # 多注意力头数
    n_layer: int = 6         # encoder/decoder 的层数 L
    dropout: float = 0.1     # Dropout 比例


# -----------------------------
# Building Blocks
# -----------------------------
class LayerNorm(nn.Module):
    """Pre-LN: learnable affine on normalized last dim.
    Uses population variance (unbiased=False) for numerical stability.
    """

    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(ndim))
        self.b_2 = nn.Parameter(torch.zeros(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation.
    hidden_dim is typically 4 * n_embd.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention supporting self-attn and cross-attn.
    If is_causal=True, applies an upper-triangular mask sized to args.block_size.
    """

    def __init__(self, args: ModelArgs, is_causal: bool = False):
        super().__init__()
        assert args.n_embd % args.n_heads == 0, "n_embd must be divisible by n_heads"
        self.is_causal = is_causal
        self.n_heads = args.n_heads
        self.head_dim = args.n_embd // args.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.wq = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.n_embd, bias=False)

        # Causal mask (registered as non-persistent buffer so it doesn't count as a parameter)
        if is_causal:
            mask = torch.full((1, 1, args.block_size, args.block_size), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, n_heads*head_dim) -> (B, n_heads, T, head_dim)
        B, T, _ = x.size()
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, k_in: Optional[torch.Tensor] = None, v_in: Optional[torch.Tensor] = None,) -> torch.Tensor:
        # Self-attn: K,V from x; Cross-attn: K,V from encoder outputs
        k_src = x if k_in is None else k_in
        v_src = x if v_in is None else v_in

        q = self._shape(self.wq(x))  # (B, H, T_q, D)
        k = self._shape(self.wk(k_src))  # (B, H, T_k, D)
        v = self._shape(self.wv(v_src))  # (B, H, T_k, D)

        # Scaled dot-product attention
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_k)

        # Apply causal mask if needed (only on the last two dims)
        if self.is_causal:
            T_q, T_k = att.size(-2), att.size(-1)
            att = att + self.mask[:, :, :T_q, :T_k]

        # Softmax on last dim
        att = F.softmax(att.float(), dim=-1).type_as(att)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # (B, H, T_q, D)
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_heads * self.head_dim)
        y = self.resid_dropout(self.wo(y))
        return y


class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = LayerNorm(args.n_embd)
        self.attn = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        self.ffn = MLP(args.n_embd, 4 * args.n_embd, args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer block
        x = self.attn_norm(x)
        x = x + self.attn(x, x, x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Masked self-attention
        self.attn_norm_1 = LayerNorm(args.n_embd)
        self.mask_attn = MultiHeadAttention(args, is_causal=True)
        # Cross-attention
        self.attn_norm_2 = LayerNorm(args.n_embd)
        self.cross_attn = MultiHeadAttention(args, is_causal=False)
        # FFN
        self.ffn_norm = LayerNorm(args.n_embd)
        self.ffn = MLP(args.n_embd, 4 * args.n_embd, args.dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm_1(x)
        # Masked self-attention
        x = x + self.mask_attn(x)
        # Cross-attention (Q from decoder, K/V from encoder)
        x = self.attn_norm_2(x)
        x = x + self.cross_attn(x, enc_out, enc_out)
        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings.
    Stores pe as a buffer (non-persistent) of shape (1, block_size, n_embd).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2, dtype=torch.float) * (-(math.log(10000.0) / args.n_embd))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, C)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        T = x.size(1)
        return x + self.pe[:, :T]


# -----------------------------
# Stacks
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.final_norm = LayerNorm(args.n_embd)   # 新增

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)                  # 新增：统一输出分布


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


# -----------------------------
# Full Transformer (Encoder-Decoder)
# -----------------------------
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(args.vocab_size, args.n_embd),
                wpe=PositionalEncoding(args),
                drop=nn.Dropout(args.dropout),
                encoder=Encoder(args),
                decoder=Decoder(args),
            )
        )
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.size()
        # token + position
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        x = self.transformer.wpe(tok_emb)
        x = self.transformer.drop(x)

        # encode
        enc_out = self.transformer.encoder(x)
        # decode (here we feed the same sequence as a simple demo)
        x = self.transformer.decoder(x, enc_out)

        logits = self.lm_head(x)
        return logits


# -----------------------------
# Standalone attention utility (optional test helper)
# -----------------------------
@torch.no_grad()
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention for testing parity.
    Shapes: q,k,v -> (B, H, T, D)
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    att = torch.matmul(q, k.transpose(-2, -1)) * scale
    att = F.softmax(att, dim=-1)
    return torch.matmul(att, v)

