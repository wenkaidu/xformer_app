from dataclasses import dataclass


@dataclass(frozen=True)
class Tier1LiteConfig:
    """Small decoder stack: 4× TE layer, SwiGLU, GMQA (GQA), RoPE, QK-norm."""

    vocab_size: int = 8192
    hidden_size: int = 512
    ffn_hidden_size: int = 1408
    num_layers: int = 4
    num_attention_heads: int = 8
    num_gqa_groups: int = 2
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-5
    attn_dropout: float = 0.0
    hidden_dropout: float = 0.0
    # Variable-size MoE-style FFN (all_to_all + all_to_all_single); native backend only.
    use_moe_style_ffn: bool = False
    moe_num_experts: int = 16

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must divide num_attention_heads")
        if self.num_gqa_groups < 1 or self.num_gqa_groups > self.num_attention_heads:
            raise ValueError("num_gqa_groups must be in [1, num_attention_heads]")
        if self.num_attention_heads % self.num_gqa_groups != 0:
            raise ValueError("num_attention_heads must divide evenly by num_gqa_groups")
        if self.moe_num_experts < 1:
            raise ValueError("moe_num_experts must be >= 1")
