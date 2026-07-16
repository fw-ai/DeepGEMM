from typing import Any, Optional, Tuple

import torch

from .. import _C


def bf16_mega_moe_backward_dgrad(
    gate_up_output: torch.Tensor,
    grad_h_output: torch.Tensor,
    grad_gate_up_output: torch.Tensor,
    h_act_output: torch.Tensor,
    h_weighted_output: torch.Tensor,
    x_pool_output: torch.Tensor,
    grad_x_pool_output: torch.Tensor,
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
) -> None:
    """Run BF16 reverse dispatch, dgrad, activation, and direct grad-x."""
    if activation not in ("swiglu", "geglu"):
        raise ValueError(f"unsupported activation: {activation}")
    if not write_grad_x_pool and not direct_remote_grad_x:
        raise ValueError("grad-x requires a local or direct remote output")
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
        route_weights,
        w2_weights,
        w13_weights,
        expert_counts,
        grid_sync_counter,
        activation_limit,
        activation,
        fast_math,
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
) -> None:
    """Run standalone single-CTA BF16 W2 wgrad for local experts."""
    _C.bf16_mega_moe_backward_w2(
        grad_w2_output,
        grad_ye,
        h_weighted,
        padded_expert_counts,
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
) -> None:
    """Run W2 wgrad while protecting the direct-write receive planes."""
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
    )


def bf16_mega_moe_backward_w13_combine(
    grad_w13_output: torch.Tensor,
    grad_gate_up: torch.Tensor,
    x_pool: torch.Tensor,
    padded_expert_counts: torch.Tensor,
    grad_x_output: torch.Tensor,
    sym_buffer: Any,
) -> None:
    """Run W13 wgrad while reducing direct-write grad-x planes."""
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
    )
