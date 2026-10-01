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

from xformer_pretrain.invariant_collectives import (
    position_invariant_all_reduce,
    position_invariant_reduce_scatter_tensor,
)

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

    world = dist.get_world_size()
    for case in args.cases:
        src = _contribution(case, args.numel, rank, device)

        native = src.clone()
        dist.all_reduce(native)

        invariant = src.clone()
        position_invariant_all_reduce(invariant)

        # Reduce-scatter leaves each rank holding a different shard, so compare
        # the gathered shards to see the whole result.
        shard = torch.empty(args.numel // world, dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(shard, src.clone())
        native_rs = torch.empty_like(src)
        dist.all_gather_into_tensor(native_rs, shard)

        position_invariant_reduce_scatter_tensor(shard, src.clone())
        inv_rs = torch.empty_like(src)
        dist.all_gather_into_tensor(inv_rs, shard)

        if rank == 0:
            print(
                f"RESULT case={case}"
                f" ar_native={_crc(native):08x} ar_invariant={_crc(invariant):08x}"
                f" rs_native={_crc(native_rs):08x} rs_invariant={_crc(inv_rs):08x}",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
