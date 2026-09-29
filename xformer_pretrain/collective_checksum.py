"""Checksum every torch.distributed collective buffer on each rank.

Each call logs process group, a per-group sequence number, the collective name,
and the payload size, plus CRC32 of the input and output bytes on this rank.
Sequence numbers start at 0 for each process group and follow issue order, which
matches the NCCL/RCCL opCount when that group uses a single communicator.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import zlib
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

_HOOKED = (
    "broadcast",
    "all_reduce",
    "reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all",
    "all_to_all_single",
    "barrier",
)


class _State:
    def __init__(self) -> None:
        self.installed = False
        self.rank = 0
        self.lock = threading.Lock()
        self.seq: dict[int, int] = {}
        self.pending: dict[int, list[_Record]] = {}
        self.log_path: Path | None = None
        self._fh: Any = None

    def write(self, line: str) -> None:
        # One write below PIPE_BUF so ranks do not tear lines on a pipe.
        os.write(sys.stdout.fileno(), (line + "\n").encode())
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()


_state = _State()


class _Record:
    def __init__(
        self,
        *,
        pg: str,
        pg_id: int,
        pg_rank: int,
        pg_size: int,
        pg_ranks: str,
        seq: int,
        collective: str,
        reduce_op: str,
        data_size: int,
        dtype: str,
        in_crc: int,
        in_bytes: int,
        out_tensors: list[torch.Tensor],
    ) -> None:
        self.pg = pg
        self.pg_id = pg_id
        self.pg_rank = pg_rank
        self.pg_size = pg_size
        self.pg_ranks = pg_ranks
        self.seq = seq
        self.collective = collective
        self.reduce_op = reduce_op
        self.data_size = data_size
        self.dtype = dtype
        self.in_crc = in_crc
        self.in_bytes = in_bytes
        self.out_tensors = out_tensors

    def emit(self) -> None:
        out_crc, out_bytes, _ = _crc_tensors(self.out_tensors)
        line = (
            f"COLL_CHECKSUM rank={_state.rank} pg={self.pg} pg_rank={self.pg_rank} "
            f"pg_size={self.pg_size} pg_ranks={self.pg_ranks} seq={self.seq} "
            f"collective={self.collective} reduce_op={self.reduce_op} "
            f"data_size={self.data_size} dtype={self.dtype} "
            f"in_crc={self.in_crc:08x} out_crc={out_crc:08x} "
            f"in_bytes={self.in_bytes} out_bytes={out_bytes}"
        )
        with _state.lock:
            _state.write(line)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    raw = t.detach()
    if raw.numel() == 0:
        return b""
    if raw.is_complex():
        raw = torch.view_as_real(raw)
    if raw.dtype == torch.bool:
        raw = raw.to(torch.uint8)
    # 0-dim tensors cannot be viewed as bytes; reshape to 1-D first.
    raw = raw.reshape(-1)
    if not raw.is_contiguous():
        raw = raw.contiguous()
    try:
        raw_u8 = raw.view(torch.uint8)
    except RuntimeError:
        raw_u8 = raw.clone().view(torch.uint8)
    return raw_u8.cpu().numpy().tobytes()


def _crc_tensors(tensors: list[torch.Tensor]) -> tuple[int, int, int]:
    crc = 0
    nbytes = 0
    numel = 0
    for t in tensors:
        crc = zlib.crc32(_tensor_bytes(t), crc) & 0xFFFFFFFF
        nbytes += t.numel() * t.element_size()
        numel += t.numel()
    return crc, nbytes, numel


def _fmt_ranks(ranks: list[int]) -> str:
    if not ranks:
        return "-"
    if ranks == list(range(ranks[0], ranks[0] + len(ranks))):
        if len(ranks) == 1:
            return str(ranks[0])
        return f"{ranks[0]}-{ranks[-1]}"
    return ",".join(str(r) for r in ranks)


def _resolve_group(group: Any) -> Any:
    if group is None:
        return c10d._get_default_group()
    return group


def _reduce_op_name(op: Any) -> str:
    if op is None:
        return "-"
    name = str(op)
    return name.split(".")[-1]


def _pg_fields(group: Any) -> tuple[str, int, int, int, str]:
    name = c10d._get_process_group_name(group)
    if not name or name == "None":
        group_name = getattr(group, "group_name", None)
        name = group_name() if callable(group_name) else str(id(group))
    pg_rank = int(group.rank())
    pg_size = int(group.size())
    try:
        ranks = list(c10d.get_process_group_ranks(group))
    except Exception:
        ranks = list(range(pg_size))
    return name, id(group), pg_rank, pg_size, _fmt_ranks(ranks)


def _next_seq(pg_id: int) -> int:
    seq = _state.seq.get(pg_id, 0)
    _state.seq[pg_id] = seq + 1
    return seq


def _in_coalesce(group: Any) -> bool:
    try:
        return group in c10d._world.pg_coalesce_state
    except Exception:
        return False


class _WaitProxy:
    """Finish the output checksum when an async collective is waited on."""

    def __init__(self, work: Any, done: Callable[[], None]) -> None:
        self._work = work
        self._done = done
        self._finished = False

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        result = self._work.wait(*args, **kwargs)
        self._finish()
        return result

    def synchronize(self) -> Any:
        result = self._work.synchronize()
        self._finish()
        return result

    def get_future(self) -> Any:
        fut = self._work.get_future()
        then = getattr(fut, "then", None)
        if then is None:
            return fut

        def _cb(f: Any) -> Any:
            self._finish()
            value = getattr(f, "value", None)
            return value() if callable(value) else f.wait()

        return then(_cb)

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._done()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


def _hook(
    orig: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    group: Any,
    async_op: bool,
    collective: str,
    in_tensors: list[torch.Tensor],
    out_tensors: list[torch.Tensor],
    data_size: int,
    dtype: str,
    reduce_op: str,
) -> Any:
    if c10d._rank_not_in_group(group):
        return orig(*args, **kwargs)

    resolved = _resolve_group(group)
    in_crc, in_bytes, _ = _crc_tensors(in_tensors)
    pg, pg_id, pg_rank, pg_size, pg_ranks = _pg_fields(resolved)
    with _state.lock:
        seq = _next_seq(pg_id)
    record = _Record(
        pg=pg,
        pg_id=pg_id,
        pg_rank=pg_rank,
        pg_size=pg_size,
        pg_ranks=pg_ranks,
        seq=seq,
        collective=collective,
        reduce_op=reduce_op,
        data_size=data_size,
        dtype=dtype,
        in_crc=in_crc,
        in_bytes=in_bytes,
        out_tensors=out_tensors,
    )

    if _in_coalesce(resolved):
        with _state.lock:
            _state.pending.setdefault(pg_id, []).append(record)
        return orig(*args, **kwargs)

    result = orig(*args, **kwargs)
    if async_op and result is not None:
        return _WaitProxy(result, record.emit)
    record.emit()
    return result


def _dtype_name(tensors: list[torch.Tensor]) -> str:
    if not tensors:
        return "-"
    names = {str(t.dtype) for t in tensors}
    if len(names) == 1:
        return next(iter(names))
    return "mixed"


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _wrap(name: str, orig: Callable[..., Any]) -> Callable[..., Any]:
    if name == "broadcast":

        def broadcast(tensor, src=None, group=None, async_op=False, group_src=None):
            return _hook(
                orig,
                (tensor,),
                {"src": src, "group": group, "async_op": async_op, "group_src": group_src},
                group=group,
                async_op=async_op,
                collective="broadcast",
                in_tensors=[tensor],
                out_tensors=[tensor],
                data_size=_nbytes(tensor),
                dtype=_dtype_name([tensor]),
                reduce_op="-",
            )

        return broadcast

    if name == "all_reduce":

        def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            return _hook(
                orig,
                (tensor,),
                {"op": op, "group": group, "async_op": async_op},
                group=group,
                async_op=async_op,
                collective="all_reduce",
                in_tensors=[tensor],
                out_tensors=[tensor],
                data_size=_nbytes(tensor),
                dtype=_dtype_name([tensor]),
                reduce_op=_reduce_op_name(op),
            )

        return all_reduce

    if name == "reduce":

        def reduce(tensor, dst=None, op=dist.ReduceOp.SUM, group=None, async_op=False, group_dst=None):
            return _hook(
                orig,
                (tensor,),
                {
                    "dst": dst,
                    "op": op,
                    "group": group,
                    "async_op": async_op,
                    "group_dst": group_dst,
                },
                group=group,
                async_op=async_op,
                collective="reduce",
                in_tensors=[tensor],
                out_tensors=[tensor],
                data_size=_nbytes(tensor),
                dtype=_dtype_name([tensor]),
                reduce_op=_reduce_op_name(op),
            )

        return reduce

    if name == "all_gather":

        def all_gather(tensor_list, tensor, group=None, async_op=False):
            return _hook(
                orig,
                (tensor_list, tensor),
                {"group": group, "async_op": async_op},
                group=group,
                async_op=async_op,
                collective="all_gather",
                in_tensors=[tensor],
                out_tensors=list(tensor_list),
                data_size=_nbytes(tensor),
                dtype=_dtype_name([tensor]),
                reduce_op="-",
            )

        return all_gather

    if name == "all_gather_into_tensor":

        def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
            return _hook(
                orig,
                (output_tensor, input_tensor),
                {"group": group, "async_op": async_op},
                group=group,
                async_op=async_op,
                collective="all_gather_into_tensor",
                in_tensors=[input_tensor],
                out_tensors=[output_tensor],
                data_size=_nbytes(input_tensor),
                dtype=_dtype_name([input_tensor]),
                reduce_op="-",
            )

        return all_gather_into_tensor

    if name == "reduce_scatter":

        def reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
            return _hook(
                orig,
                (output, input_list),
                {"op": op, "group": group, "async_op": async_op},
                group=group,
                async_op=async_op,
                collective="reduce_scatter",
                in_tensors=list(input_list),
                out_tensors=[output],
                data_size=_nbytes(output),
                dtype=_dtype_name([output]),
                reduce_op=_reduce_op_name(op),
            )

        return reduce_scatter

    if name == "reduce_scatter_tensor":

        def reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
            return _hook(
                orig,
                (output, input),
                {"op": op, "group": group, "async_op": async_op},
                group=group,
                async_op=async_op,
                collective="reduce_scatter_tensor",
                in_tensors=[input],
                out_tensors=[output],
                data_size=_nbytes(output),
                dtype=_dtype_name([output]),
                reduce_op=_reduce_op_name(op),
            )

        return reduce_scatter_tensor

    if name == "all_to_all":

        def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
            return _hook(
                orig,
                (output_tensor_list, input_tensor_list),
                {"group": group, "async_op": async_op},
                group=group,
                async_op=async_op,
                collective="all_to_all",
                in_tensors=list(input_tensor_list),
                out_tensors=list(output_tensor_list),
                data_size=sum(_nbytes(t) for t in input_tensor_list),
                dtype=_dtype_name(list(input_tensor_list)),
                reduce_op="-",
            )

        return all_to_all

    if name == "all_to_all_single":

        def all_to_all_single(
            output,
            input,
            output_split_sizes=None,
            input_split_sizes=None,
            group=None,
            async_op=False,
        ):
            return _hook(
                orig,
                (output, input),
                {
                    "output_split_sizes": output_split_sizes,
                    "input_split_sizes": input_split_sizes,
                    "group": group,
                    "async_op": async_op,
                },
                group=group,
                async_op=async_op,
                collective="all_to_all_single",
                in_tensors=[input],
                out_tensors=[output],
                data_size=_nbytes(input),
                dtype=_dtype_name([input]),
                reduce_op="-",
            )

        return all_to_all_single

    if name == "barrier":

        def barrier(group=None, async_op=False, device_ids=None):
            return _hook(
                orig,
                (),
                {"group": group, "async_op": async_op, "device_ids": device_ids},
                group=group,
                async_op=async_op,
                collective="barrier",
                in_tensors=[],
                out_tensors=[],
                data_size=0,
                dtype="-",
                reduce_op="-",
            )

        return barrier

    raise AssertionError(name)


def _flush_pending(pg_id: int) -> None:
    with _state.lock:
        records = _state.pending.pop(pg_id, [])
    for record in records:
        record.emit()


def _wrap_coalescing(orig: Callable[..., Any]) -> Callable[..., Any]:
    @contextlib.contextmanager
    def wrapped(group=None, device=None, async_ops=False):
        resolved = _resolve_group(group)
        pg_id = id(resolved)
        failed = False
        try:
            with orig(group, device, async_ops) as cm:
                yield cm
        except Exception:
            failed = True
            with _state.lock:
                _state.pending.pop(pg_id, None)
            raise
        if failed:
            return
        if async_ops:
            orig_wait = cm.wait

            def wait() -> None:
                orig_wait()
                _flush_pending(pg_id)

            cm.wait = wait  # type: ignore[method-assign]
            return
        _flush_pending(pg_id)

    return wrapped


def install_collective_checksum_hooks(log_dir: str = "collective_checksums") -> Path:
    """Patch torch.distributed collectives and return this rank's log path."""
    if _state.installed:
        assert _state.log_path is not None
        return _state.log_path

    rank = int(os.environ.get("RANK", "0"))
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rank{rank}.log"
    fh = path.open("w", encoding="utf-8")
    fh.write(
        "# COLL_CHECKSUM rank pg pg_rank pg_size pg_ranks seq collective reduce_op "
        "data_size dtype in_crc out_crc in_bytes out_bytes\n"
        "# data_size is the NCCL/RCCL count in bytes: the in-place buffer for "
        "all_reduce/broadcast/reduce, the packed tensor bytes for broadcast_coalesced, "
        "the per-rank input for all_gather/all_to_all, the per-rank output for "
        "reduce_scatter, and 0 for barrier.\n"
        "# in_crc/out_crc are CRC32 of this rank's raw input and output buffer bytes. "
        "seq is the per-process-group collective index, starting at 0.\n"
    )
    fh.flush()

    _state.rank = rank
    _state.log_path = path
    _state._fh = fh

    for name in _HOOKED:
        orig = getattr(c10d, name)
        wrapped = _wrap(name, orig)
        setattr(c10d, name, wrapped)
        if hasattr(dist, name):
            setattr(dist, name, wrapped)

    coalescing = _wrap_coalescing(c10d._coalescing_manager)
    c10d._coalescing_manager = coalescing
    if hasattr(dist, "_coalescing_manager"):
        dist._coalescing_manager = coalescing

    # FSDP param sync and some state-dict paths call this C++ API directly.
    # One call is one logical broadcast (it may pack several tensors into a bucket).
    broadcast_coalesced = dist._broadcast_coalesced

    def _broadcast_coalesced(process_group, tensors, buffer_size, src):
        tensor_list = list(tensors)
        return _hook(
            broadcast_coalesced,
            (process_group, tensors, buffer_size, src),
            {},
            group=process_group,
            async_op=False,
            collective="broadcast_coalesced",
            in_tensors=tensor_list,
            out_tensors=tensor_list,
            data_size=sum(_nbytes(t) for t in tensor_list),
            dtype=_dtype_name(tensor_list),
            reduce_op="-",
        )

    dist._broadcast_coalesced = _broadcast_coalesced

    _state.installed = True
    return path
