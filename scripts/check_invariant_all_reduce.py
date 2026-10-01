#!/usr/bin/env python
"""Compare bf16 and fp32-accumulated all-reduce under different reduction orders.

Run the same all-reduce under several RCCL algorithms and channel counts; each
configuration sums the rank contributions in a different order. Print a CRC32 of
the bf16 result so a driver can check whether the bits moved.

    NCCL_ALGO=Tree torchrun --nproc_per_node=8 scripts/check_invariant_all_reduce.py
"""

from __future__ import annotations

import argparse
import os
import zlib

import torch
import torch.distributed as dist

from xformer_pretrain.invariant_collectives import position_invariant_all_reduce

# Per-rank magnitudes for the "wide" case: contributions span 1e-3 down to 1e-9,
# far more than the ~2**14 range where the fp32 sum of 8 bf16 values stays exact.
_WIDE_SCALES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-3)


def _crc(t: torch.Tensor) -> int:
    return zlib.crc32(t.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()) & 0xFFFFFFFF


def _contribution(case: str, numel: int, rank: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    x = torch.randn(numel, generator=g)
    scale = 0.01 if case == "uniform" else _WIDE_SCALES[rank % len(_WIDE_SCALES)]
    return (x * scale).to(torch.bfloat16).to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numel", type=int, default=1 << 20)
    p.add_argument("--cases", nargs="+", default=["uniform", "wide"])
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    for case in args.cases:
        src = _contribution(case, args.numel, rank, device)

        native = src.clone()
        dist.all_reduce(native)

        invariant = src.clone()
        position_invariant_all_reduce(invariant)

        if rank == 0:
            print(f"RESULT case={case} native_bf16={_crc(native):08x} invariant={_crc(invariant):08x}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
