from __future__ import annotations

from typing import Literal

import torch.nn as nn

from xformer_pretrain.config import Tier1LiteConfig
from xformer_pretrain.model_native import Tier1LiteDecoder

BackendName = Literal["auto", "torch", "te"]

__all__ = ["Tier1LiteDecoder", "BackendName", "build_decoder", "resolve_backend"]


def resolve_backend(name: BackendName) -> Literal["torch", "te"]:
    if name == "torch":
        return "torch"
    if name == "te":
        try:
            import transformer_engine  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Backend 'te' requested but transformer_engine is not installed. "
                "On NVIDIA: pip install 'xformer-pretrain[te]'. On AMD/ROCm use --backend torch (default)."
            ) from e
        return "te"
    # auto
    try:
        import transformer_engine  # noqa: F401

        return "te"
    except ImportError:
        return "torch"


def build_decoder(cfg: Tier1LiteConfig, backend: BackendName = "auto") -> nn.Module:
    """Return native Tier1LiteDecoder or TE-backed module with the same forward API."""
    resolved = resolve_backend(backend)
    if resolved == "te":
        from xformer_pretrain.model_te import Tier1LiteDecoderTE

        return Tier1LiteDecoderTE(cfg)
    return Tier1LiteDecoder(cfg)
