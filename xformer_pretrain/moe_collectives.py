"""
MoE-style variable token dispatch / combine: all_to_all (counts) + all_to_all_single (payload).

Expert id maps to rank via expert_id % world_size. Counts differ per pair (variable-size / all-to-allv style).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _default_group():
    return dist.group.WORLD if dist.is_initialized() else None


def _exchange_per_rank_scalars(vec: torch.Tensor, group) -> torch.Tensor:
    """
    vec[d] is a value this rank sends to rank d (one scalar per destination).
    Returns recv[j] = value rank j sent to this rank (= vec[this_rank] computed on rank j).
    """
    ws = dist.get_world_size(group)
    assert vec.shape == (ws,)
    device = vec.device
    send_list = [vec[d : d + 1].contiguous() for d in range(ws)]
    recv_list = [torch.empty(1, dtype=vec.dtype, device=device) for _ in range(ws)]
    dist.all_to_all(recv_list, send_list, group=group)
    return torch.cat(recv_list, dim=0).view(ws)


def _alltoall_flat(
    send_buf: torch.Tensor,
    send_elems: list[int],
    recv_elems: list[int],
    group,
) -> torch.Tensor:
    recv_buf = torch.empty(sum(recv_elems), device=send_buf.device, dtype=send_buf.dtype)
    dist.all_to_all_single(
        recv_buf,
        send_buf.contiguous(),
        output_split_sizes=recv_elems,
        input_split_sizes=send_elems,
        group=group,
    )
    return recv_buf


def moe_dispatch(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    world_size: int,
    hidden: int,
    group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: [T, H], expert_ids: [T] -> dest = expert_id % world_size.

    Returns:
        x_expert: tokens received on this rank for local expert, [T_loc, H]
        send_counts: [ws] long — tokens this rank sends to each rank (for combine)
        orig_on_expert: [T_loc] long — original token index on source rank for each received row
    """
    device = x.device
    h = hidden
    t = x.shape[0]
    assert x.shape[1] == h and expert_ids.shape == (t,)

    dest = (expert_ids % world_size).to(torch.long)
    orig_idx = torch.arange(t, device=device, dtype=torch.long)

    order = torch.argsort(dest, stable=True)
    x_perm = x.index_select(0, order)
    orig_perm = orig_idx.index_select(0, order)
    dest_sorted = dest.index_select(0, order)

    send_counts = torch.bincount(dest_sorted, minlength=world_size)
    send_elems = [int(send_counts[d].item()) * h for d in range(world_size)]

    send_elem_vec = torch.tensor(send_elems, device=device, dtype=torch.long)
    recv_elem_vec = _exchange_per_rank_scalars(send_elem_vec, group)
    recv_elems = [int(recv_elem_vec[s].item()) for s in range(world_size)]

    send_buf = x_perm.reshape(-1)
    recv_flat = _alltoall_flat(send_buf, send_elems, recv_elems, group)

    recv_tokens_per_src = recv_elem_vec // int(h)
    send_tokens = [int(send_counts[d].item()) for d in range(world_size)]
    recv_orig_flat = _alltoall_flat(
        orig_perm.to(torch.float32),
        send_tokens,
        [int(recv_tokens_per_src[s].item()) for s in range(world_size)],
        group,
    ).to(torch.long)

    t_loc = recv_flat.numel() // h
    x_expert = recv_flat.view(t_loc, h)
    return x_expert, send_counts, recv_orig_flat.view(t_loc), recv_tokens_per_src.to(torch.long)


def moe_combine(
    y_expert: torch.Tensor,
    *,
    hidden: int,
    world_size: int,
    group,
    send_counts: torch.Tensor,
    orig_on_expert: torch.Tensor,
    recv_tokens_per_src: torch.Tensor,
) -> torch.Tensor:
    """Route expert outputs back to source ranks and restore original token order."""
    h = hidden
    device = y_expert.device
    ws = world_size
    t_in = y_expert.shape[0]
    assert y_expert.shape[1] == h and orig_on_expert.shape == (t_in,)
    assert recv_tokens_per_src.shape == (ws,)

    offsets = torch.cumsum(F.pad(recv_tokens_per_src, (1, 0)), dim=0)

    send_parts_y: list[torch.Tensor] = []
    send_parts_o: list[torch.Tensor] = []
    for s in range(ws):
        n = int(recv_tokens_per_src[s].item())
        sl = slice(int(offsets[s].item()), int(offsets[s].item()) + n)
        send_parts_y.append(y_expert[sl].reshape(-1))
        send_parts_o.append(orig_on_expert[sl].to(torch.float32))
    send_y = torch.cat(send_parts_y, dim=0)
    send_o = torch.cat(send_parts_o, dim=0)

    send_elems_combine = [int(recv_tokens_per_src[s].item()) * h for s in range(ws)]
    recv_elems_combine = _exchange_per_rank_scalars(
        torch.tensor(send_elems_combine, device=device, dtype=torch.long),
        group,
    )
    recv_elems_combine_list = [int(recv_elems_combine[j].item()) for j in range(ws)]

    recv_y = _alltoall_flat(send_y, send_elems_combine, recv_elems_combine_list, group)

    send_tokens_combine = [int(recv_tokens_per_src[s].item()) for s in range(ws)]
    recv_tokens_combine = _exchange_per_rank_scalars(
        torch.tensor(send_tokens_combine, device=device, dtype=torch.long),
        group,
    )
    recv_tokens_combine_list = [int(recv_tokens_combine[j].item()) for j in range(ws)]
    recv_o = _alltoall_flat(send_o, send_tokens_combine, recv_tokens_combine_list, group).to(torch.long)

    t_out = recv_y.numel() // h
    y_scattered = recv_y.view(t_out, h)
    order_back = torch.argsort(recv_o, stable=True)
    return y_scattered.index_select(0, order_back)


class MoEStyleFFN(nn.Module):
    """Top-1 router + variable all_to_all dispatch, shared SwiGLU expert, combine."""

    def __init__(self, hidden: int, ffn_dim: int, num_experts: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.up = nn.Linear(hidden, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, hidden, bias=False)

    def _expert_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def forward(self, h: torch.Tensor, group=None) -> torch.Tensor:
        b, s, d = h.shape
        assert d == self.hidden
        group = group or _default_group()
        x = h.reshape(b * s, d)
        logits = self.router(x)
        expert_ids = torch.argmax(logits, dim=-1).to(torch.long)

        if not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return self._expert_forward(x).view(b, s, d)

        ws = dist.get_world_size(group)
        x_expert, send_counts, orig_on_expert, recv_tok = moe_dispatch(
            x,
            expert_ids,
            world_size=ws,
            hidden=d,
            group=group,
        )
        y_expert = self._expert_forward(x_expert)
        y = moe_combine(
            y_expert,
            hidden=d,
            world_size=ws,
            group=group,
            send_counts=send_counts,
            orig_on_expert=orig_on_expert,
            recv_tokens_per_src=recv_tok,
        )
        return y.view(b, s, d)
