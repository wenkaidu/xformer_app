"""Position-invariant reductions for low-precision tensors.

Reducing bf16 buffers in bf16 makes the result depend on the order the rank
contributions are summed, so a different ring order, tree shape, or channel
count changes the output bits. Accumulating in fp32 and rounding once at the
end removes that dependence: as long as the per-element contributions span less
than ~2**14 in magnitude, the fp32 sum of 8 bf16 values is exact, and exact
arithmetic is associative. Wider spans can still differ -- see
``scripts/check_invariant_all_reduce.py``.

Both all-reduce and reduce-scatter are covered. Under FSDP the gradient path is
reduce-scatter, so wrapping all-reduce alone leaves gradients order-dependent.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

# Dtypes whose mantissa is too narrow to accumulate across ranks without
# the summation order showing up in the result.
_LOW_PRECISION = (torch.bfloat16, torch.float16)

_ORIGINALS: dict[str, Any] = {}


def _base(name: str) -> Any:
    """The unwrapped collective, whether or not the patch is installed."""
    return _ORIGINALS.get(name) or getattr(c10d, name)


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


def _finish(work: Any, acc: torch.Tensor, out: torch.Tensor, async_op: bool) -> Any:
    if async_op and work is not None:
        return _DowncastWork(work, acc, out)
    out.copy_(acc)
    return work


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
    base = _base("all_reduce")
    if tensor.dtype not in _LOW_PRECISION:
        return base(tensor, op=op, group=group, async_op=async_op)

    acc = tensor.float()
    work = base(acc, op=op, group=group, async_op=async_op)
    return _finish(work, acc, tensor, async_op)


def position_invariant_reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    op: Any = dist.ReduceOp.SUM,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    """Reduce-scatter so the result does not depend on reduction order.

    This is FSDP's gradient path. The fp32 copy of ``input`` is the full
    unsharded buffer, so peak memory rises by its size for the duration.
    """
    base = _base("reduce_scatter_tensor")
    if input.dtype not in _LOW_PRECISION:
        return base(output, input, op=op, group=group, async_op=async_op)

    acc_in = input.float()
    acc_out = torch.empty_like(output, dtype=torch.float32)
    work = base(acc_out, acc_in, op=op, group=group, async_op=async_op)
    return _finish(work, acc_out, output, async_op)


def position_invariant_reduce_scatter(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op: Any = dist.ReduceOp.SUM,
    group: Any = None,
    async_op: bool = False,
) -> Any:
    """List-input form of :func:`position_invariant_reduce_scatter_tensor`."""
    base = _base("reduce_scatter")
    if output.dtype not in _LOW_PRECISION:
        return base(output, input_list, op=op, group=group, async_op=async_op)

    acc_in = [t.float() for t in input_list]
    acc_out = torch.empty_like(output, dtype=torch.float32)
    work = base(acc_out, acc_in, op=op, group=group, async_op=async_op)
    return _finish(work, acc_out, output, async_op)


_WRAPPERS = {
    "all_reduce": position_invariant_all_reduce,
    "reduce_scatter_tensor": position_invariant_reduce_scatter_tensor,
    "reduce_scatter": position_invariant_reduce_scatter,
}


def install_position_invariant_reductions() -> None:
    """Route every torch.distributed reduction through the fp32 path.

    Call after any other collective wrapper (such as the checksum hook) so the
    fp32 buffers are the ones those wrappers observe.
    """
    if _ORIGINALS:
        return

    for name, wrapper in _WRAPPERS.items():
        _ORIGINALS[name] = getattr(c10d, name)
        setattr(c10d, name, wrapper)
        setattr(dist, name, wrapper)
