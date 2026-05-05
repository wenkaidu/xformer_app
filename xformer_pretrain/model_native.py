from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformer_pretrain.config import Tier1LiteConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE cos/sin for full head_dim (even split)."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)[None, None, :, :]
        sin = emb.sin().to(dtype)[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D]; cos/sin: [1, 1, S, D]
    return (x * cos) + (_rotate_half(x) * sin)


class Tier1LiteDecoderBlock(nn.Module):
    """Pre-norm: RMSNorm → GQA+RoPE+QK-norm → residual; RMSNorm → SwiGLU → residual."""

    def __init__(self, cfg: Tier1LiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        nh = cfg.num_attention_heads
        nkv = cfg.num_gqa_groups
        hd = d // nh
        assert nh % nkv == 0

        self.input_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.q_proj = nn.Linear(d, nh * hd, bias=False)
        self.k_proj = nn.Linear(d, nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, d, bias=False)
        self.rope = RotaryEmbedding(hd)

        self.post_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        ffn = cfg.ffn_hidden_size
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        d = cfg.hidden_size
        nh, nkv = cfg.num_attention_heads, cfg.num_gqa_groups
        hd = d // nh
        b, s, _ = x.shape

        h = self.input_norm(x)
        q = self.q_proj(h).view(b, s, nh, hd).transpose(1, 2)
        k = self.k_proj(h).view(b, s, nkv, hd).transpose(1, 2)
        v = self.v_proj(h).view(b, s, nkv, hd).transpose(1, 2)

        cos, sin = self.rope(s, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QK L2 norm (TE-style), after RoPE
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scale = 1.0 / math.sqrt(hd)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        attn = attn.transpose(1, 2).reshape(b, s, nh * hd)
        h = self.o_proj(attn)
        x = x + h

        h = self.post_norm(x)
        h = F.silu(self.gate(h)) * self.up(h)
        h = self.down(h)
        return x + h


class Tier1LiteDecoder(nn.Module):
    """Causal decoder: native PyTorch blocks (SwiGLU, GQA, RoPE, QK-norm). ROCm-safe."""

    def __init__(self, cfg: Tier1LiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Tier1LiteDecoderBlock(cfg) for _ in range(cfg.num_layers))
        self.final_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layernorm(x)
        return self.lm_head(x)
