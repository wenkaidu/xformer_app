"""Position-invariant all-reduce for low-precision tensors.

Reducing bf16 buffers in bf16 makes the result depend on the order the rank
contributions are summed, so a different ring order, tree shape, or channel
count changes the output bits. Accumulating in fp32 and rounding once at the
end removes that dependence: as long as the per-element contributions span less
than ~2**14 in magnitude, the fp32 sum of 8 bf16 values is exact, and exact
arithmetic is associative. Wider spans can still differ -- see
``scripts/check_invariant_all_reduce.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

# Dtypes whose mantissa is too narrow to accumulate across ranks without
# the summation order showing up in the result.
_LOW_PRECISION = (torch.bfloat16, torch.float16)

_installed = False
_orig_all_reduce: Any = None


def _base_all_reduce(tensor: torch.Tensor, **kwargs: Any) -> Any:
    """The unwrapped all-reduce, whether or not the patch is installed."""
    if _installed:
        return _orig_all_reduce(tensor, **kwargs)
    return c10d.all_reduce(tensor, **kwargs)


class _DowncastWork:
    """Defer the fp32 -> low-precision rounding until the caller waits."""

    def __init__(self, work: Any, acc: torch.Tensor, out: torch.Tensor) -> None:
        self._work = work
        self._acc = acc
        self._out = out
        self._done = False

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        result = self._work.wait(*args, **kwargs)
        self._downcast()
        return result

    def synchronize(self) -> Any:
        result = self._work.synchronize()
        self._downcast()
        return result

    def get_future(self) -> Any:
        fut = self._work.get_future()
        then = getattr(fut, "then", None)
        if then is None:
            return fut

        def _cb(f: Any) -> Any:
            self._downcast()
            value = getattr(f, "value", None)
            return value() if callable(value) else f.wait()

        return then(_cb)

    def _downcast(self) -> None:
        if not self._done:
            self._done = True
            self._out.copy_(self._acc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


def position_invariant_all_reduce(
    tensor: torch.Tensor,
    op: Any = dist.ReduceOp.SUM,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    """All-reduce ``tensor`` so the result does not depend on reduction order.

    bf16/fp16 tensors are upcast to fp32, reduced, and rounded back in place.
    Other dtypes are passed straight through. Doubles the bytes on the wire and
    allocates an fp32 copy of ``tensor`` for the duration of the call.
    """
    if tensor.dtype not in _LOW_PRECISION:
        return _base_all_reduce(tensor, op=op, group=group, async_op=async_op)

    acc = tensor.float()
    work = _base_all_reduce(acc, op=op, group=group, async_op=async_op)
    if async_op and work is not None:
        return _DowncastWork(work, acc, tensor)
    tensor.copy_(acc)
    return work


def install_position_invariant_all_reduce() -> None:
    """Route every torch.distributed all-reduce through the fp32 path.

    Call after any other collective wrapper (such as the checksum hook) so the
    fp32 buffers are the ones those wrappers observe.
    """
    global _installed, _orig_all_reduce
    if _installed:
        return

    _orig_all_reduce = c10d.all_reduce
    c10d.all_reduce = position_invariant_all_reduce
    dist.all_reduce = position_invariant_all_reduce
    _installed = True
