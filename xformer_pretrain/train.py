from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
from functools import partial
from pathlib import Path

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
from torch.profiler import ProfilerActivity, profile, schedule as profiler_schedule, tensorboard_trace_handler

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


def _total_grad_l2_norm(model: nn.Module) -> float:
    """Global L2 norm of all gradients; FSDP-safe (collective — call on every rank)."""
    inf = float("inf")
    if isinstance(model, FSDP):
        return float(FSDP.clip_grad_norm_(model, max_norm=inf))
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=inf))


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
    p.add_argument(
        "--log-grad-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log global L2 grad norm with loss (FSDP: extra collective). Use --no-log-grad-norm to skip.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        metavar="N",
        help="Log loss (and grad norm if enabled) every N steps; always logs the last step.",
    )
    p.add_argument(
        "--checksum-collectives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "CRC32 each collective input and output buffer on every rank. "
            "Logs pg, seq, collective, and data size to --checksum-dir and stdout."
        ),
    )
    p.add_argument(
        "--checksum-dir",
        type=str,
        default="collective_checksums",
        metavar="DIR",
        help="Per-rank collective checksum logs (rank<N>.log). Used with --checksum-collectives.",
    )
    p.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PyTorch profiler on all ranks for one step (see --profile-step).",
    )
    p.add_argument(
        "--profile-step",
        type=int,
        default=50,
        metavar="S",
        help="Zero-indexed training step to record (schedule wait=S, active=1). Default 50.",
    )
    p.add_argument(
        "--profile-dir",
        type=str,
        default="profiler_trace",
        metavar="DIR",
        help="Output directory (shared FS recommended for multi-node). TensorBoard + Chrome traces.",
    )
    p.add_argument(
        "--moe",
        action="store_true",
        help="Native backend: SwiGLU FFN replaced by top-1 routed MoE with variable all_to_all / all_to_all_single.",
    )
    p.add_argument(
        "--moe-num-experts",
        type=int,
        default=16,
        metavar="E",
        help="Number of routed experts (logical); expert e maps to rank (e %% world_size).",
    )
    args = p.parse_args(argv)
    if args.log_every <= 0:
        p.error("--log-every must be >= 1")
    if args.profile_step < 0:
        p.error("--profile-step must be >= 0")

    rank, world_size, local_rank, local_world_size = _setup_dist()
    if args.checksum_collectives:
        from xformer_pretrain.collective_checksum import install_collective_checksum_hooks

        checksum_log = install_collective_checksum_hooks(log_dir=args.checksum_dir)
        if rank == 0:
            logger.info("Collective checksum hooks installed. Per-rank logs under %s", checksum_log.parent.resolve())
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    resolved = resolve_backend(args.backend)
    if args.moe and resolved == "te":
        p.error("--moe is implemented only on the native PyTorch decoder; use --backend torch.")
    if rank == 0:
        logger.info("Decoder backend: %s (requested %s)", resolved, args.backend)

    cfg = Tier1LiteConfig()
    if args.moe:
        cfg = dataclasses.replace(
            cfg,
            use_moe_style_ffn=True,
            moe_num_experts=args.moe_num_experts,
        )
        if rank == 0:
            logger.info("MoE-style FFN enabled (%s experts, map to ranks via %% world_size).", cfg.moe_num_experts)
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

    prof = None
    if args.profile:
        if args.profile_step >= args.steps:
            if rank == 0:
                logger.warning(
                    "--profile-step (%s) >= --steps (%s); profiler will never enter an active window.",
                    args.profile_step,
                    args.steps,
                )
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        tb = tensorboard_trace_handler(
            dir_name=str(profile_dir),
            worker_name=f"rank{rank}",
            use_gzip=True,
        )
        profile_step = args.profile_step

        # Kineto can only serialize a trace once: tensorboard_trace_handler already
        # persists it; do not also call export_chrome_trace on the same run.
        on_trace_ready = tb

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule(
                wait=profile_step,
                warmup=0,
                active=1,
                repeat=1,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        if rank == 0:
            logger.info(
                "Profiler enabled: active window covers training step index %s (0-based). "
                "Combined multi-rank view: tensorboard --logdir=%s "
                "(install torch_tb_profiler + tensorboard; PyTorch Profiler tab). "
                "Traces are *.pt.trace.json.gz per rank under that directory.",
                profile_step,
                profile_dir.resolve(),
            )

    try:
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
            log_this_step = (step % args.log_every == 0) or (step == args.steps - 1)
            grad_norm = 0.0
            if args.log_grad_norm and log_this_step:
                # FSDP: collective — same condition on every rank.
                grad_norm = _total_grad_l2_norm(model)
            opt.step()
            if rank == 0 and log_this_step:
                if args.log_grad_norm:
                    logger.info("step %s loss=%.6f grad_norm=%.6e", step, loss_acc, grad_norm)
                else:
                    logger.info("step %s loss=%.6f", step, loss_acc)
            if prof is not None:
                prof.step()
    finally:
        if prof is not None:
            prof.stop()

    if args.profile and world_size > 1:
        dist.barrier()
    if rank == 0 and args.profile and args.profile_step < args.steps:
        manifest = {
            "profile_step_zero_indexed": args.profile_step,
            "world_size": world_size,
            "tensorboard_cmd": f"tensorboard --logdir={Path(args.profile_dir).resolve()}",
            "tensorboard_note": "Install torch_tb_profiler and open the PyTorch Profiler tab for a single combined timeline of all ranks.",
            "trace_files_glob": str(Path(args.profile_dir) / "*.pt.trace.json.gz"),
        }
        with open(Path(args.profile_dir) / "profiler_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Wrote profiler_manifest.json under %s", Path(args.profile_dir).resolve())

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
