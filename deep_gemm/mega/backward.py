from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from .. import _C
from . import CombineOrderMode, RouteWeightMode


def _sort_bf16_pool_in_deepep_order(
    tensors: tuple[torch.Tensor, ...],
    token_src_metadata: torch.Tensor,
    valid_rows: torch.Tensor,
    expert_counts: torch.Tensor,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_ranks: int,
) -> None:
    """Sort valid expert rows into DeepEP's stable source-route order."""
    pool_rows = valid_rows.nonzero().flatten()
    if pool_rows.numel() <= 1:
        return
    expert_ids = torch.repeat_interleave(
        torch.arange(
            expert_counts.numel(),
            dtype=torch.long,
            device=expert_counts.device,
        ),
        expert_counts.to(torch.long),
    )
    if expert_ids.numel() != pool_rows.numel():
        raise RuntimeError(
            "valid BF16 pool rows do not match expert counts")
    metadata = token_src_metadata[pool_rows].long()
    routes_per_rank = num_max_tokens_per_rank * num_topk
    route_keys = (
        metadata[:, 0] * routes_per_rank +
        metadata[:, 1] * num_topk +
        metadata[:, 2]
    )
    order = torch.argsort(
        expert_ids * (num_ranks * routes_per_rank) + route_keys,
        stable=True,
    )
    ordered_rows = pool_rows[order]
    if torch.equal(ordered_rows, pool_rows):
        return

    seen_ptrs = set()
    for tensor in tensors:
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        tensor[pool_rows] = tensor[ordered_rows].clone()


def bf16_mega_moe_backward_dgrad(
    gate_up_output: torch.Tensor,
    grad_h_output: torch.Tensor,
    grad_gate_up_output: torch.Tensor,
    h_act_output: torch.Tensor,
    h_weighted_output: torch.Tensor,
    x_pool_output: torch.Tensor,
    grad_x_pool_output: Optional[torch.Tensor],
    grad_route_output: torch.Tensor,
    grad_ye: torch.Tensor,
    route_weights: torch.Tensor,
    w2_weights: torch.Tensor,
    w13_weights: torch.Tensor,
    expert_counts: torch.Tensor,
    grid_sync_counter: torch.Tensor,
    grad_y: torch.Tensor,
    sym_buffer: Any,
    activation_limit: float,
    block_m: int,
    activation: str = "swiglu",
    fast_math: bool = False,
    direct_remote_grad_x: bool = True,
    write_grad_x_pool: bool = True,
    clear_wgrad_padding: bool = True,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    grad_y_unweighted_output: Optional[torch.Tensor] = None,
    down_unweighted_output: Optional[torch.Tensor] = None,
    python_numerical_correction: bool = False,
    combine_order_mode: CombineOrderMode = CombineOrderMode.FIXED_TOPK,
) -> None:
    """Run BF16 reverse dispatch, dgrad, activation, and direct grad-x."""
    route_weight_mode = RouteWeightMode(route_weight_mode)
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if activation not in ("swiglu", "geglu"):
        raise ValueError(f"unsupported activation: {activation}")
    if not write_grad_x_pool and not direct_remote_grad_x:
        raise ValueError("grad-x requires a local or direct remote output")
    if grad_x_pool_output is None:
        if write_grad_x_pool:
            raise ValueError(
                "write_grad_x_pool requires grad_x_pool_output")
        grad_x_pool_output = grad_ye.new_empty(
            (0, grad_ye.size(1)))
    if python_numerical_correction and not write_grad_x_pool:
        raise ValueError(
            "Python numerical correction requires grad_x_pool_output")
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        if grad_y_unweighted_output is None:
            raise ValueError(
                "post_down requires grad_y_unweighted_output")
        if down_unweighted_output is None:
            raise ValueError(
                "post_down requires saved down_unweighted_output")
    else:
        if grad_y_unweighted_output is None:
            grad_y_unweighted_output = grad_ye
        if down_unweighted_output is None:
            down_unweighted_output = grad_ye
    if sym_buffer.group.size() > 1:
        # BLOCK_M determines the persistent grid shape and therefore every
        # grid/NVLink barrier's participant count. A rank with no source
        # tokens may choose a smaller local bucket than its peers; launching
        # those different specializations deadlocks at the first grid sync.
        # Canonicalize at the API boundary so every caller is safe.
        rank_uniform_block_m = torch.tensor(
            block_m, dtype=torch.int32, device=grad_y.device)
        dist.all_reduce(
            rank_uniform_block_m,
            op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        block_m = int(rank_uniform_block_m.item())
    num_tokens = grad_y.shape[0]
    backward_grad_y = sym_buffer.backward_grad_y
    if direct_remote_grad_x:
        # Kernel A reuses this region as [topk, token, hidden] direct-write
        # planes after every rank has pulled grad-y from plane zero. Clear all
        # planes so masked routes and repeated backward calls reduce as zero.
        backward_grad_y = torch.as_strided(
            backward_grad_y,
            size=(
                sym_buffer.num_topk,
                sym_buffer.num_max_tokens_per_rank,
                sym_buffer.hidden,
            ),
            stride=(
                sym_buffer.num_max_tokens_per_rank * sym_buffer.hidden,
                sym_buffer.hidden,
                1,
            ),
        )
        backward_grad_y.zero_()
        backward_grad_y[0, :num_tokens].copy_(
            grad_y.to(torch.bfloat16).contiguous()
        )
    else:
        backward_grad_y[:num_tokens].copy_(
            grad_y.to(torch.bfloat16).contiguous()
        )
    grad_route_output.zero_()
    _C.bf16_mega_moe_backward_dgrad(
        gate_up_output,
        grad_h_output,
        grad_gate_up_output,
        h_act_output,
        h_weighted_output,
        x_pool_output,
        grad_x_pool_output,
        grad_route_output,
        grad_ye,
        grad_y_unweighted_output,
        route_weights,
        w2_weights,
        w13_weights,
        expert_counts,
        grid_sync_counter,
        activation_limit,
        activation,
        fast_math,
        route_weight_mode.value,
        down_unweighted_output,
        block_m,
        direct_remote_grad_x,
        write_grad_x_pool,
        clear_wgrad_padding,
        sym_buffer.backward_grad_y,
        sym_buffer.x,
        sym_buffer.topk_weights,
        sym_buffer.token_src_metadata,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        combine_order_mode.value,
    )
    if not python_numerical_correction:
        return
    # FireTitan uses two native grouped GEMMs for W1/W3 dgrad and rounds each
    # result to BF16 before the in-place add. The fused W13 dgrad above uses
    # one FP32 accumulation across [gate | up], which is not bitwise
    # equivalent. Materialize the native boundary with the same grouped-MM
    # calls and offsets.
    valid_rows = torch.isfinite(gate_up_output[:, 0])
    row_indices = valid_rows.nonzero().flatten()
    intermediate_hidden = grad_gate_up_output.size(1) // 2
    if activation == "swiglu" and activation_limit == float("inf"):
        # Standard FireTitan grouped experts run BF16 SiLU followed by a
        # separate BF16 multiply. Reuse ATen's native BF16 forward/backward
        # operations so both materialization boundaries and SiLU's derivative
        # operation order match exactly.
        gate = gate_up_output[
            row_indices, :intermediate_hidden].contiguous()
        up = gate_up_output[
            row_indices, intermediate_hidden:].contiguous()
        native_silu = torch.nn.functional.silu(gate)
        native_h = native_silu * up
        native_grad_h = grad_h_output[row_indices]
        if route_weight_mode is RouteWeightMode.PRE_DOWN:
            native_grad_h = (
                native_grad_h.float() *
                route_weights[row_indices].float().unsqueeze(1)
            ).to(native_grad_h.dtype)
        native_grad_silu = native_grad_h * up
        native_grad_gate = torch.ops.aten.silu_backward(
            native_grad_silu, gate)
        native_grad_up = native_grad_h * native_silu
        grad_gate_up_output[row_indices] = torch.cat(
            (native_grad_gate, native_grad_up), dim=1)
        h_act_output[row_indices] = native_h
        h_weighted_output[row_indices] = (
            (
                native_h.float() *
                route_weights[row_indices].float().unsqueeze(1)
            ).to(native_h.dtype)
            if route_weight_mode is RouteWeightMode.PRE_DOWN
            else native_h
        )
    elif activation == "geglu":
        # GeGLU's tanh approximation is expressed as separate ATen FP32
        # operations in FireTitan. Preserve each materialization point here;
        # reassociating the polynomial or derivative in CUDA changes rare
        # BF16 round-to-nearest ties.
        gate_unclamped = gate_up_output[
            row_indices, :intermediate_hidden].contiguous()
        up_unclamped = gate_up_output[
            row_indices, intermediate_hidden:].contiguous()
        gate = torch.clamp(
            gate_unclamped, max=activation_limit).float()
        up = torch.clamp(
            up_unclamped,
            min=-activation_limit,
            max=activation_limit,
        ).float()
        alpha = 1.5957691216057308
        beta = 0.044715
        gate_sq = gate * gate
        z = (alpha * gate) * (1.0 + beta * gate_sq)
        dz = alpha * (1.0 + (3.0 * beta) * gate_sq)
        sig = torch.sigmoid(z)
        activated_gate = gate * sig
        native_h = (activated_gate * up).to(torch.bfloat16)
        native_grad_h = grad_h_output[row_indices]
        if route_weight_mode is RouteWeightMode.PRE_DOWN:
            native_grad_h = (
                native_grad_h.float() *
                route_weights[row_indices].float().unsqueeze(1)
            ).to(native_grad_h.dtype)
        activation_grad = (
            sig + gate * sig * (1.0 - sig) * dz)
        native_grad_gate = (
            native_grad_h.float() * up * activation_grad)
        native_grad_gate = torch.where(
            gate_unclamped.float() <= activation_limit,
            native_grad_gate,
            torch.zeros_like(native_grad_gate),
        ).to(torch.bfloat16)
        native_grad_up = (
            native_grad_h.float() * activated_gate)
        native_grad_up = torch.where(
            (up_unclamped.float() >= -activation_limit) &
            (up_unclamped.float() <= activation_limit),
            native_grad_up,
            torch.zeros_like(native_grad_up),
        ).to(torch.bfloat16)
        grad_gate_up_output[row_indices] = torch.cat(
            (native_grad_gate, native_grad_up), dim=1)
        h_act_output[row_indices] = native_h
        h_weighted_output[row_indices] = (
            (
                native_h.float() *
                route_weights[row_indices].float().unsqueeze(1)
            ).to(native_h.dtype)
            if route_weight_mode is RouteWeightMode.PRE_DOWN
            else native_h
        )
    offsets = expert_counts.cumsum(0).to(torch.int32)
    w1_weights = w13_weights[:, :intermediate_hidden].contiguous()
    w3_weights = w13_weights[:, intermediate_hidden:].contiguous()
    if row_indices.numel():
        native_grad_x = torch._grouped_mm(
            grad_gate_up_output[
                row_indices, :intermediate_hidden].contiguous(),
            w1_weights,
            offs=offsets,
        )
        native_grad_x.add_(torch._grouped_mm(
            grad_gate_up_output[
                row_indices, intermediate_hidden:].contiguous(),
            w3_weights,
            offs=offsets,
        ))
    else:
        native_grad_x = grad_x_pool_output.new_empty(
            (0, grad_x_pool_output.size(1)))
    grad_x_pool_output.zero_()
    grad_x_pool_output[row_indices] = native_grad_x

    # Match PyTorch's deterministic FP32 reduction tree for router gradients;
    # the fused kernel's scalar left fold has a different rounding order.
    if route_weight_mode is RouteWeightMode.POST_DOWN:
        route_lhs = grad_y_unweighted_output[row_indices]
        route_rhs = down_unweighted_output[row_indices]
    else:
        route_lhs = grad_h_output[row_indices]
        route_rhs = h_act_output[row_indices]
    native_grad_route = (
        route_lhs.float() * route_rhs.float()).sum(dim=1)
    grad_route_output.zero_()
    grad_route_output[row_indices] = native_grad_route.to(
        grad_route_output.dtype)

    if direct_remote_grad_x:
        metadata = sym_buffer.token_src_metadata[row_indices].long()
        native_grad_x_planes = torch.zeros(
            (
                sym_buffer.group.size(),
                sym_buffer.num_max_tokens_per_rank,
                sym_buffer.num_topk,
                grad_x_pool_output.size(1),
            ),
            dtype=grad_x_pool_output.dtype,
            device=grad_x_pool_output.device,
        )
        native_grad_x_planes[
            metadata[:, 0], metadata[:, 1], metadata[:, 2]
        ] = native_grad_x
        dist.all_reduce(native_grad_x_planes, group=sym_buffer.group)
        # Install the corrected values into the real slot-major symmetric
        # planes before the fused W13 combine. Reducing a shadow tensor after
        # combine masks destination/slot bugs and leaves direct-plane users
        # observing the fused W13 dgrad's different accumulation boundary.
        backward_grad_y.copy_(
            native_grad_x_planes[
                sym_buffer.group.rank()
            ].permute(1, 0, 2)
        )

    # MegaMoE's communication scheduler deliberately interleaves source
    # ranks, while DeepEP's native grouped wgrads consume stable
    # expert/rank/token/slot order. Canonicalize the retained pool before the
    # standalone wgrad kernels so their K traversal matches FireTitan.
    _sort_bf16_pool_in_deepep_order(
        (
            grad_gate_up_output,
            h_weighted_output,
            x_pool_output,
            grad_ye,
        ),
        sym_buffer.token_src_metadata,
        valid_rows,
        expert_counts,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        sym_buffer.group.size(),
    )


def fp8_fp4_mega_moe_backward_dgrad_swiglu(
    gate_up_output: torch.Tensor,
    grad_h_output: torch.Tensor,
    grad_gate_up_output: torch.Tensor,
    h_act_output: torch.Tensor,
    h_weighted_output: torch.Tensor,
    x_pool_output: torch.Tensor,
    grad_x_pool_output: torch.Tensor,
    l1_acts: torch.Tensor,
    l1_acts_sf: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    grad_ye: torch.Tensor,
    route_weights: torch.Tensor,
    w2_weights: Tuple[torch.Tensor, torch.Tensor],
    w2_dequant_scratch: torch.Tensor,
    w13_weights: Tuple[torch.Tensor, torch.Tensor],
    w13_dequant_scratch: torch.Tensor,
    expert_counts: torch.Tensor,
    grid_sync_counter: torch.Tensor,
    activation_limit: float,
    block_m: int,
    compute_w13_dgrad: bool = True,
    direct_remote_grad_x: bool = False,
    write_grad_x_pool: bool = True,
    clear_wgrad_padding: bool = False,
    sym_buffer: Optional[Any] = None,
    grad_y: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    token_src_metadata: Optional[torch.Tensor] = None,
) -> None:
    """Run the production L1 replay, dgrad/SwiGLU, and grad-x dispatch."""
    backward_grad_y = None
    backward_topk_weights = None
    backward_sym_buffer_ptrs = []
    backward_rank = 0
    num_max_tokens_per_rank = 0
    num_topk = 0
    if not write_grad_x_pool and not direct_remote_grad_x:
        raise ValueError("grad-x requires a local or direct remote output")
    if direct_remote_grad_x and not compute_w13_dgrad:
        raise ValueError("direct remote grad-x requires W13 dgrad")
    if direct_remote_grad_x and sym_buffer is None:
        raise ValueError("direct remote grad-x requires a symmetric buffer")
    if sym_buffer is not None:
        if grad_y is None or topk_weights is None or token_src_metadata is None:
            raise ValueError(
                "in-kernel EP dispatch requires grad_y, topk_weights, "
                "and token_src_metadata"
            )
        num_tokens = grad_y.shape[0]
        sym_buffer.backward_grad_y[:num_tokens].copy_(
            grad_y.to(torch.bfloat16).contiguous()
        )
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights.float().contiguous())
        backward_grad_y = sym_buffer.backward_grad_y
        backward_topk_weights = sym_buffer.topk_weights
        backward_sym_buffer_ptrs = sym_buffer.handle.buffer_ptrs
        backward_rank = sym_buffer.group.rank()
        num_max_tokens_per_rank = sym_buffer.num_max_tokens_per_rank
        num_topk = sym_buffer.num_topk

    _C.fp8_fp4_mega_moe_backward_dgrad_swiglu(
        gate_up_output,
        grad_h_output,
        grad_gate_up_output,
        h_act_output,
        h_weighted_output,
        x_pool_output,
        grad_x_pool_output,
        l1_acts,
        l1_acts_sf,
        l1_weights,
        grad_ye,
        route_weights,
        w2_weights,
        w2_dequant_scratch,
        w13_weights,
        w13_dequant_scratch,
        expert_counts,
        grid_sync_counter,
        activation_limit,
        compute_w13_dgrad,
        block_m,
        direct_remote_grad_x,
        write_grad_x_pool,
        clear_wgrad_padding,
        backward_grad_y,
        backward_topk_weights,
        token_src_metadata,
        backward_sym_buffer_ptrs,
        backward_rank,
        num_max_tokens_per_rank,
        num_topk,
    )


def bf16_mega_moe_backward_w2(
    grad_w2_output: torch.Tensor,
    grad_ye: torch.Tensor,
    h_weighted: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
) -> None:
    """Run standalone single-CTA BF16 W2 wgrad for local experts."""
    route_weight_mode = RouteWeightMode(route_weight_mode)
    _C.bf16_mega_moe_backward_w2(
        grad_w2_output,
        grad_ye,
        h_weighted,
        padded_expert_counts,
        route_weight_mode.value,
    )


def bf16_mega_moe_backward_w13(
    grad_w13_output: torch.Tensor,
    grad_gate_up: torch.Tensor,
    x_pool: torch.Tensor,
    padded_expert_counts: torch.Tensor,
) -> None:
    """Run standalone single-CTA BF16 combined W1/W3 wgrad."""
    _C.bf16_mega_moe_backward_w13(
        grad_w13_output,
        grad_gate_up,
        x_pool,
        padded_expert_counts,
    )


def bf16_mega_moe_backward_w2_combine(
    grad_w2_output: torch.Tensor,
    grad_ye: torch.Tensor,
    h_weighted: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    grad_x_output: torch.Tensor,
    sym_buffer: Any,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
) -> None:
    """Run W2 wgrad while protecting the direct-write receive planes."""
    route_weight_mode = RouteWeightMode(route_weight_mode)
    _C.bf16_mega_moe_backward_w2_combine(
        grad_w2_output,
        grad_ye,
        h_weighted,
        padded_expert_counts,
        grad_x_output,
        sym_buffer.backward_grad_y,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        route_weight_mode.value,
    )


def bf16_mega_moe_backward_w13_combine(
    grad_w13_output: torch.Tensor,
    grad_gate_up: torch.Tensor,
    x_pool: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    grad_x_output: torch.Tensor,
    sym_buffer: Any,
    combine_order_mode: CombineOrderMode =
    CombineOrderMode.FIXED_TOPK,
) -> None:
    """Run W13 wgrad while reducing direct-write grad-x planes."""
    combine_order_mode = CombineOrderMode(combine_order_mode)
    _C.bf16_mega_moe_backward_w13_combine(
        grad_w13_output,
        grad_gate_up,
        x_pool,
        padded_expert_counts,
        grad_x_output,
        sym_buffer.backward_grad_y,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_topk,
        sym_buffer.topk_idx,
        combine_order_mode.value,
    )
