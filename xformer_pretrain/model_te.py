from __future__ import annotations

import torch.nn as nn
from transformer_engine.pytorch import TransformerLayer
from transformer_engine.pytorch.attention import RotaryPositionEmbedding

from xformer_pretrain.config import Tier1LiteConfig


class Tier1LiteDecoderTE(nn.Module):
    """NVIDIA TE path: TransformerLayer × N with SwiGLU, GQA, RoPE, QK-norm."""

    def __init__(self, cfg: Tier1LiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=cfg.hidden_size,
                    ffn_hidden_size=cfg.ffn_hidden_size,
                    num_attention_heads=cfg.num_attention_heads,
                    num_gqa_groups=cfg.num_gqa_groups,
                    layernorm_epsilon=cfg.rms_norm_eps,
                    hidden_dropout=cfg.hidden_dropout,
                    attention_dropout=cfg.attn_dropout,
                    fuse_qkv_params=False,
                    normalization="RMSNorm",
                    activation="swiglu",
                    attn_input_format="bshd",
                    bias=False,
                    use_qk_norm=True,
                    layer_number=i + 1,
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.rope = RotaryPositionEmbedding(head_dim)
        self.final_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(input_ids)
        s = input_ids.shape[1]
        rope_emb = self.rope(s)
        for layer in self.layers:
            h = layer(
                h,
                attention_mask=None,
                self_attn_mask_type="causal",
                rotary_pos_emb=rope_emb,
            )
        h = self.final_layernorm(h)
        return self.lm_head(h)
