from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from xformer_pretrain.config import Tier1LiteConfig
from xformer_pretrain.model import build_decoder, resolve_backend
from xformer_pretrain.model_native import Tier1LiteDecoderBlock

logger = logging.getLogger(__name__)


def _dist_env() -> tuple[int, int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    return rank, world_size, local_rank, local_world_size


def _init_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[rank{rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _process_group_backend() -> str:
    return os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")


def _setup_dist() -> tuple[int, int, int, int]:
    rank, world_size, local_rank, local_world_size = _dist_env()
    _init_logging(rank)
    if world_size > 1:
        dist.init_process_group(backend=_process_group_backend())
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, local_world_size


def _shard_mesh(world_size: int, local_world_size: int) -> tuple[ShardingStrategy, object | None]:
    if world_size <= 1:
        return ShardingStrategy.NO_SHARD, None
    num_nodes = world_size // local_world_size
    if num_nodes < 1 or world_size % local_world_size != 0:
        raise RuntimeError(
            f"WORLD_SIZE ({world_size}) must be a multiple of LOCAL_WORLD_SIZE ({local_world_size})."
        )
    mesh = init_device_mesh(
        "cuda",
        (num_nodes, local_world_size),
        mesh_dim_names=("replicate", "shard"),
    )
    if num_nodes > 1:
        logger.info("Using HSDP: HYBRID_SHARD mesh (replicate, shard) = (%s, %s)", num_nodes, local_world_size)
        return ShardingStrategy.HYBRID_SHARD, mesh
    logger.info("Single-node FSDP: HYBRID_SHARD with replicate=1 (equivalent to intra-node full shard).")
    return ShardingStrategy.HYBRID_SHARD, mesh


def _wrap_fsdp(
    model: nn.Module,
    strategy: ShardingStrategy,
    device_mesh: object | None,
    wrap_leaf: type[nn.Module] | tuple[type[nn.Module], ...],
) -> FSDP:
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    if isinstance(wrap_leaf, tuple):
        types = wrap_leaf
        lambda_fn = lambda m: isinstance(m, types)
    else:
        lambda_fn = lambda m: isinstance(m, wrap_leaf)
    auto_wrap = partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
    kwargs: dict = dict(
        use_orig_params=False,
        sync_module_states=True,
        mixed_precision=mp,
        auto_wrap_policy=auto_wrap,
    )
    if device_mesh is not None:
        kwargs["device_mesh"] = device_mesh
        kwargs["sharding_strategy"] = strategy
    else:
        kwargs["sharding_strategy"] = ShardingStrategy.NO_SHARD

    return FSDP(model, **kwargs)


def _synthetic_batch(
    micro_batch: int, seq_len: int, vocab_size: int, device: torch.device, seed: int
) -> torch.Tensor:
    # Draw on CPU with a CPU Generator, then move: avoids ROCm/PyTorch builds where
    # torch.Generator(device=cuda) still counts as CPU for randint(..., device=cuda).
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randint(0, vocab_size, (micro_batch, seq_len), generator=g)
    return x.to(device)


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
    shift_logits = logits[:, :-1].reshape(-1, vocab_size)
    shift_labels = input_ids[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tier1-lite decoder toy pretrain (HSDP / multi-node).")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--backend",
        choices=("auto", "torch", "te"),
        default="auto",
        help="auto: use TE if installed, else PyTorch (use torch on AMD/ROCm).",
    )
    args = p.parse_args(argv)

    rank, world_size, local_rank, local_world_size = _setup_dist()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    resolved = resolve_backend(args.backend)
    if rank == 0:
        logger.info("Decoder backend: %s (requested %s)", resolved, args.backend)

    cfg = Tier1LiteConfig()
    model = build_decoder(cfg, args.backend).to(device)

    strategy, mesh = _shard_mesh(world_size, local_world_size)
    if world_size > 1:
        if resolved == "te":
            import transformer_engine.pytorch as te

            wrap_leaf = te.TransformerLayer
        else:
            wrap_leaf = Tier1LiteDecoderBlock
        model = _wrap_fsdp(model, strategy, mesh, wrap_leaf)
    else:
        model = model.to(torch.bfloat16)

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step_seed = args.seed + rank * 9973
    for step in range(args.steps):
        loss_acc = 0.0
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            batch = _synthetic_batch(
                args.micro_batch, cfg.max_seq_len, cfg.vocab_size, device, step_seed + step
            )
            loss = train_step(model, batch, cfg.vocab_size) / args.grad_accum
            loss.backward()
            loss_acc += float(loss.detach()) * args.grad_accum
        opt.step()
        if rank == 0 and (step % 10 == 0 or step == args.steps - 1):
            logger.info("step %s loss=%.4f", step, loss_acc)

    if world_size > 1:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()
        if rank == 0:
            torch.save(state, "tier1_lite_rank0.pt")
            logger.info("Wrote tier1_lite_rank0.pt (rank-0 full state dict).")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
