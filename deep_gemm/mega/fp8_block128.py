"""SM103-only distributed FP8-block128 MegaMoE training path.

The public operation owns routed-token quantization, transport, expert compute,
POST_DOWN combine, and the complete routed backward.  Weight tensors retain
GLM's E4M3 + FP32 128x128 block-scale contract; no BF16 weight dequantization,
MXFP4 transcode, or architecture fallback exists in this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Sequence

import torch
import torch.distributed as dist

from .. import _C


_BLOCK = 128
_PAD_ROWS = 128

REQUIRED_NATIVE_SYMBOLS = (
    "get_sm103_fp8_block128_capabilities",
    "sm103_fp8_block128_quantize",
    "sm103_fp8_block128_dequantize",
    "sm103_fp8_block128_grouped_gemm_nt",
    "sm103_fp8_block128_grouped_gemm_nn",
    "sm103_fp8_block128_swiglu_quantize",
    "sm103_fp8_block128_swiglu_backward",
    "sm103_fp8_block128_route_scale_quantize",
    "sm103_fp8_block128_post_down_combine",
    "sm103_fp8_block128_post_down_score_grad",
    "sm103_fp8_block128_route_sum",
    "k_grouped_bf16_gemm_tn_contiguous",
)

REQUIRED_PYTHON_SYMBOLS = (
    "fp8_block128_mega_moe",
    "transform_glm_w13_for_fp8_block128_mega_moe",
    "get_fp8_block128_mega_moe_capabilities",
)


def get_fp8_block128_mega_moe_capabilities() -> dict[str, Any]:
    """Return a non-launching, exact capability manifest for preflight."""
    native = dict(_C.get_sm103_fp8_block128_capabilities())
    missing = [name for name in REQUIRED_NATIVE_SYMBOLS if not hasattr(_C, name)]
    native.update(
        {
            "native_symbols": REQUIRED_NATIVE_SYMBOLS,
            "python_symbols": REQUIRED_PYTHON_SYMBOLS,
            "forward": not missing,
            "backward": not missing,
            "distributed_transport": "torch.distributed.all_to_all_single",
            "missing_symbols": tuple(missing),
        }
    )
    return native


def transform_glm_w13_for_fp8_block128_mega_moe(
    canonical_weight: torch.Tensor,
    canonical_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert canonical ``[gate, up]`` GLM storage to ``[up; gate]``.

    ``canonical_weight`` is ``[2E, H, D]`` and ``canonical_scale`` is
    ``[2E, H/128, D/128]``.  Returned tensors are contiguous
    ``[E, 2H, D]`` and ``[E, 2H/128, D/128]`` respectively.
    """
    if canonical_weight.ndim != 3 or canonical_scale.ndim != 3:
        raise ValueError("canonical W13 weight and scale must both be rank 3")
    if canonical_weight.shape[0] % 2:
        raise ValueError("canonical W13 must contain gate/up pairs")
    experts = canonical_weight.shape[0] // 2
    hidden, model_dim = canonical_weight.shape[1:]
    if hidden % _BLOCK or model_dim % _BLOCK:
        raise ValueError("W13 dimensions must be divisible by 128")
    expected_scale_shape = (experts * 2, hidden // _BLOCK, model_dim // _BLOCK)
    if tuple(canonical_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"canonical W13 scale shape must be {expected_scale_shape}, got {tuple(canonical_scale.shape)}"
        )
    # Canonical pair index 0 is gate and 1 is up.  The fused preactivation ABI
    # requires up first, followed by gate.
    pair_order = torch.tensor([1, 0], dtype=torch.int64, device=canonical_weight.device)
    active_weight = (
        canonical_weight.view(experts, 2, hidden, model_dim)
        .index_select(1, pair_order)
        .reshape(experts, hidden * 2, model_dim)
        .contiguous()
    )
    active_scale = (
        canonical_scale.view(
            experts, 2, hidden // _BLOCK, model_dim // _BLOCK
        )
        .index_select(1, pair_order)
        .reshape(experts, hidden * 2 // _BLOCK, model_dim // _BLOCK)
        .contiguous()
    )
    return active_weight, active_scale


def _active_w13_grad_to_canonical(active_grad: torch.Tensor) -> torch.Tensor:
    experts, doubled_hidden, model_dim = active_grad.shape
    hidden = doubled_hidden // 2
    up, gate = active_grad.view(experts, 2, hidden, model_dim).unbind(dim=1)
    return torch.stack((gate, up), dim=1).reshape(experts * 2, hidden, model_dim).contiguous()


@dataclass(frozen=True)
class _GroupState:
    group: Any
    rank: int
    world_size: int


def _resolve_group(group: Any) -> _GroupState:
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise RuntimeError("a process group was provided before torch.distributed initialization")
        return _GroupState(group=None, rank=0, world_size=1)
    return _GroupState(
        group=group,
        rank=dist.get_rank(group),
        world_size=dist.get_world_size(group),
    )


def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got rank {tensor.ndim}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_master: torch.Tensor,
    w2_master: torch.Tensor,
    group_state: _GroupState,
) -> tuple[int, int, int, int, int]:
    if not x.is_cuda:
        raise ValueError("FP8-block128 MegaMoE requires CUDA")
    if torch.cuda.get_device_capability(x.device) != (10, 3):
        capability = torch.cuda.get_device_capability(x.device)
        raise RuntimeError(
            f"FP8-block128 MegaMoE is SM103-only; runtime capability is {capability}. "
            "No fallback is available."
        )
    device = x.device
    _check_tensor(x, name="x", ndim=2, dtype=torch.bfloat16, device=device)
    _check_tensor(topk_ids, name="topk_ids", ndim=2, dtype=torch.int64, device=device)
    _check_tensor(topk_scores, name="topk_scores", ndim=2, dtype=torch.float32, device=device)
    _check_tensor(w13_weight, name="w13_weight", ndim=3, dtype=torch.float8_e4m3fn, device=device)
    _check_tensor(w13_scale, name="w13_scale", ndim=3, dtype=torch.float32, device=device)
    _check_tensor(w2_weight, name="w2_weight", ndim=3, dtype=torch.float8_e4m3fn, device=device)
    _check_tensor(w2_scale, name="w2_scale", ndim=3, dtype=torch.float32, device=device)
    _check_tensor(w13_master, name="w13_master", ndim=3, dtype=torch.bfloat16, device=device)
    _check_tensor(w2_master, name="w2_master", ndim=3, dtype=torch.bfloat16, device=device)

    tokens, model_dim = x.shape
    if model_dim % _BLOCK:
        raise ValueError("model dimension must be divisible by 128")
    if topk_ids.shape != topk_scores.shape or topk_ids.shape[0] != tokens:
        raise ValueError("top-k IDs/scores must have identical [tokens, top_k] shape")
    topk = topk_ids.shape[1]
    if topk <= 0:
        raise ValueError("top_k must be positive")

    local_experts, doubled_hidden, w13_k = w13_weight.shape
    if local_experts <= 0 or doubled_hidden % (2 * _BLOCK) or w13_k != model_dim:
        raise ValueError("W13 must have shape [local_experts, 2H, D] with D/H divisible by 128")
    hidden = doubled_hidden // 2
    if tuple(w13_scale.shape) != (
        local_experts,
        doubled_hidden // _BLOCK,
        model_dim // _BLOCK,
    ):
        raise ValueError("W13 scale shape does not match 128x128 weight blocks")
    if tuple(w2_weight.shape) != (local_experts, model_dim, hidden):
        raise ValueError("W2 must have shape [local_experts, D, H]")
    if tuple(w2_scale.shape) != (
        local_experts,
        model_dim // _BLOCK,
        hidden // _BLOCK,
    ):
        raise ValueError("W2 scale shape does not match 128x128 weight blocks")
    if tuple(w13_master.shape) != (local_experts * 2, hidden, model_dim):
        raise ValueError("canonical BF16 W13 master must have shape [2E, H, D]")
    if tuple(w2_master.shape) != tuple(w2_weight.shape):
        raise ValueError("BF16 W2 master shape must match W2")

    global_experts = local_experts * group_state.world_size
    if topk_ids.numel():
        minimum, maximum = torch.aminmax(topk_ids)
        if minimum.item() < 0 or maximum.item() >= global_experts:
            raise ValueError(
                f"top-k IDs must lie in [0, {global_experts}); got [{minimum.item()}, {maximum.item()}]"
            )
    return tokens, model_dim, hidden, local_experts, topk


def _exchange_counts(
    send_counts: Sequence[int], group_state: _GroupState, device: torch.device
) -> list[int]:
    if group_state.world_size == 1:
        return list(send_counts)
    send = torch.tensor(send_counts, device=device, dtype=torch.int64)
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group_state.group)
    return [int(value) for value in receive.cpu().tolist()]


def _all_to_all_rows(
    tensor: torch.Tensor,
    send_counts: Sequence[int],
    receive_counts: Sequence[int],
    group_state: _GroupState,
) -> torch.Tensor:
    if group_state.world_size == 1:
        return tensor
    output = torch.empty(
        (sum(receive_counts), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    source = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
    destination = output.view(torch.uint8) if output.dtype == torch.float8_e4m3fn else output
    dist.all_to_all_single(
        destination,
        source,
        output_split_sizes=list(receive_counts),
        input_split_sizes=list(send_counts),
        group=group_state.group,
    )
    return output


def _inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return inverse


def _padding_state(
    counts: Sequence[int], device: torch.device
) -> tuple[list[int], torch.Tensor]:
    padded_counts = [((count + _PAD_ROWS - 1) // _PAD_ROWS) * _PAD_ROWS if count else 0 for count in counts]
    total_actual = sum(counts)
    if total_actual == 0:
        return padded_counts, torch.empty(0, dtype=torch.int64, device=device)
    count_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
    padded_tensor = torch.tensor(padded_counts, dtype=torch.int64, device=device)
    group_ids = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int64, device=device), count_tensor
    )
    padding_before = torch.cumsum(padded_tensor - count_tensor, dim=0) - (
        padded_tensor - count_tensor
    )
    actual_to_padded = torch.arange(total_actual, dtype=torch.int64, device=device)
    actual_to_padded.add_(padding_before.index_select(0, group_ids))
    return padded_counts, actual_to_padded


def _pad_rows(
    tensor: torch.Tensor,
    actual_to_padded: torch.Tensor,
    padded_rows: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    output = torch.full(
        (padded_rows, *tensor.shape[1:]),
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if tensor.shape[0]:
        if tensor.dtype == torch.float8_e4m3fn:
            output.view(torch.uint8).index_copy_(
                0, actual_to_padded, tensor.view(torch.uint8)
            )
        else:
            output.index_copy_(0, actual_to_padded, tensor)
    return output


def _unpad_rows(tensor: torch.Tensor, actual_to_padded: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float8_e4m3fn:
        output = tensor.view(torch.uint8).index_select(0, actual_to_padded)
        return output.view(torch.float8_e4m3fn)
    return tensor.index_select(0, actual_to_padded)


def _bf16_grouped_wgrad(
    left: torch.Tensor,
    right: torch.Tensor,
    padded_counts: Sequence[int],
) -> torch.Tensor:
    output = torch.zeros(
        (len(padded_counts), left.shape[1], right.shape[1]),
        dtype=torch.bfloat16,
        device=left.device,
    )
    if left.shape[0] == 0:
        return output
    grouped_layout = torch.tensor(
        list(accumulate(padded_counts)), dtype=torch.int32, device=left.device
    )
    _C.k_grouped_bf16_gemm_tn_contiguous(
        left.contiguous(),
        right.contiguous(),
        output,
        None,
        grouped_layout,
        None,
        "mn",
        True,
    )
    return output


class _FP8Block128MegaMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_scores: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_master: torch.Tensor,
        w2_master: torch.Tensor,
        group: Any,
    ) -> torch.Tensor:
        group_state = _resolve_group(group)
        tokens, model_dim, hidden, local_experts, topk = _validate_inputs(
            x,
            topk_ids,
            topk_scores,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            w13_master,
            w2_master,
            group_state,
        )

        with torch.autograd.profiler.record_function(
            "sm103_fp8_block128_megamoe_forward"
        ):
            flat_ids = topk_ids.flatten()
            send_order = torch.argsort(flat_ids, stable=True)
            sorted_ids = flat_ids.index_select(0, send_order)
            destinations = torch.div(sorted_ids, local_experts, rounding_mode="floor")
            send_counts = [
                int(value)
                for value in torch.bincount(
                    destinations, minlength=group_state.world_size
                )
                .cpu()
                .tolist()
            ]
            receive_counts = _exchange_counts(send_counts, group_state, x.device)

            token_quantized, token_scales = _C.sm103_fp8_block128_quantize(x)
            sorted_tokens = torch.div(send_order, topk, rounding_mode="floor")
            send_activations = token_quantized.index_select(0, sorted_tokens)
            send_activation_scales = token_scales.index_select(0, sorted_tokens)
            receive_activations = _all_to_all_rows(
                send_activations, send_counts, receive_counts, group_state
            )
            receive_activation_scales = _all_to_all_rows(
                send_activation_scales, send_counts, receive_counts, group_state
            )
            receive_ids = _all_to_all_rows(
                sorted_ids, send_counts, receive_counts, group_state
            )

            local_ids = torch.remainder(receive_ids, local_experts)
            group_order = torch.argsort(local_ids, stable=True)
            ungroup_order = _inverse_permutation(group_order)
            grouped_activations = receive_activations.index_select(0, group_order)
            grouped_activation_scales = receive_activation_scales.index_select(
                0, group_order
            )
            grouped_local_ids = local_ids.index_select(0, group_order)
            actual_counts = [
                int(value)
                for value in torch.bincount(
                    grouped_local_ids, minlength=local_experts
                )
                .cpu()
                .tolist()
            ]
            padded_counts, actual_to_padded = _padding_state(
                actual_counts, x.device
            )
            padded_rows = sum(padded_counts)
            padded_activations = _pad_rows(
                grouped_activations,
                actual_to_padded,
                padded_rows,
                fill_value=0,
            )
            padded_activation_scales = _pad_rows(
                grouped_activation_scales,
                actual_to_padded,
                padded_rows,
                fill_value=1,
            )

            preactivation = _C.sm103_fp8_block128_grouped_gemm_nt(
                padded_activations,
                padded_activation_scales,
                w13_weight,
                w13_scale,
                padded_counts,
            )
            hidden_quantized, hidden_scales = (
                _C.sm103_fp8_block128_swiglu_quantize(preactivation)
            )
            routed_output_padded = _C.sm103_fp8_block128_grouped_gemm_nt(
                hidden_quantized,
                hidden_scales,
                w2_weight,
                w2_scale,
                padded_counts,
            )
            routed_output_grouped = _unpad_rows(
                routed_output_padded, actual_to_padded
            )
            routed_output_receive_order = routed_output_grouped.index_select(
                0, ungroup_order
            )
            routed_output_send_order = _all_to_all_rows(
                routed_output_receive_order,
                receive_counts,
                send_counts,
                group_state,
            )
            routed_output = torch.empty(
                (tokens * topk, model_dim),
                dtype=torch.bfloat16,
                device=x.device,
            )
            if routed_output.shape[0]:
                routed_output.index_copy_(
                    0, send_order, routed_output_send_order
                )
            output = _C.sm103_fp8_block128_post_down_combine(
                routed_output, topk_scores
            )

        ctx.group_state = group_state
        ctx.send_counts = send_counts
        ctx.receive_counts = receive_counts
        ctx.padded_counts = padded_counts
        ctx.tokens = tokens
        ctx.model_dim = model_dim
        ctx.hidden = hidden
        ctx.local_experts = local_experts
        ctx.topk = topk
        ctx.save_for_backward(
            topk_scores,
            send_order,
            group_order,
            ungroup_order,
            actual_to_padded,
            padded_activations,
            padded_activation_scales,
            preactivation,
            hidden_quantized,
            hidden_scales,
            routed_output,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            topk_scores,
            send_order,
            group_order,
            ungroup_order,
            actual_to_padded,
            padded_activations,
            padded_activation_scales,
            preactivation,
            hidden_quantized,
            hidden_scales,
            routed_output,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        with torch.autograd.profiler.record_function(
            "sm103_fp8_block128_megamoe_backward"
        ):
            grad_scores = _C.sm103_fp8_block128_post_down_score_grad(
                routed_output, grad_output, ctx.topk
            )
            grad_route_quantized_send, grad_route_scales_send = (
                _C.sm103_fp8_block128_route_scale_quantize(
                    grad_output, topk_scores, send_order
                )
            )
            grad_route_quantized_receive = _all_to_all_rows(
                grad_route_quantized_send,
                ctx.send_counts,
                ctx.receive_counts,
                ctx.group_state,
            )
            grad_route_scales_receive = _all_to_all_rows(
                grad_route_scales_send,
                ctx.send_counts,
                ctx.receive_counts,
                ctx.group_state,
            )
            grad_route_quantized_grouped = grad_route_quantized_receive.index_select(
                0, group_order
            )
            grad_route_scales_grouped = grad_route_scales_receive.index_select(
                0, group_order
            )
            padded_rows = sum(ctx.padded_counts)
            grad_route_quantized = _pad_rows(
                grad_route_quantized_grouped,
                actual_to_padded,
                padded_rows,
                fill_value=0,
            )
            grad_route_scales = _pad_rows(
                grad_route_scales_grouped,
                actual_to_padded,
                padded_rows,
                fill_value=1,
            )

            grad_hidden = _C.sm103_fp8_block128_grouped_gemm_nn(
                grad_route_quantized,
                grad_route_scales,
                w2_weight,
                w2_scale,
                ctx.padded_counts,
            )
            grad_preactivation = _C.sm103_fp8_block128_swiglu_backward(
                grad_hidden, preactivation
            )
            grad_preactivation_quantized, grad_preactivation_scales = (
                _C.sm103_fp8_block128_quantize(grad_preactivation)
            )
            grad_input_padded = _C.sm103_fp8_block128_grouped_gemm_nn(
                grad_preactivation_quantized,
                grad_preactivation_scales,
                w13_weight,
                w13_scale,
                ctx.padded_counts,
            )

            grad_route_dequantized = _C.sm103_fp8_block128_dequantize(
                grad_route_quantized, grad_route_scales
            )
            hidden_dequantized = _C.sm103_fp8_block128_dequantize(
                hidden_quantized, hidden_scales
            )
            grad_w2 = _bf16_grouped_wgrad(
                grad_route_dequantized,
                hidden_dequantized,
                ctx.padded_counts,
            )
            input_dequantized = _C.sm103_fp8_block128_dequantize(
                padded_activations, padded_activation_scales
            )
            grad_w13_active = _bf16_grouped_wgrad(
                grad_preactivation,
                input_dequantized,
                ctx.padded_counts,
            )
            grad_w13 = _active_w13_grad_to_canonical(grad_w13_active)

            grad_input_grouped = _unpad_rows(
                grad_input_padded, actual_to_padded
            )
            grad_input_receive_order = grad_input_grouped.index_select(
                0, ungroup_order
            )
            grad_input_send_order = _all_to_all_rows(
                grad_input_receive_order,
                ctx.receive_counts,
                ctx.send_counts,
                ctx.group_state,
            )
            grad_input_routes = torch.empty(
                (ctx.tokens * ctx.topk, ctx.model_dim),
                dtype=torch.bfloat16,
                device=grad_output.device,
            )
            if grad_input_routes.shape[0]:
                grad_input_routes.index_copy_(
                    0, send_order, grad_input_send_order
                )
            grad_input = _C.sm103_fp8_block128_route_sum(
                grad_input_routes, ctx.tokens, ctx.topk
            )

        return (
            grad_input,
            None,
            grad_scores,
            None,
            None,
            None,
            None,
            grad_w13,
            grad_w2,
            None,
        )


def fp8_block128_mega_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_master: torch.Tensor,
    w2_master: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """Run the complete SM103 FP8-block128 routed branch.

    W13 quantized tensors use active ``[up; gate]`` ordering while the BF16
    W13 master remains canonical interleaved ``[gate, up]`` storage.  Route
    scores are applied only after W2 and their gradients are accumulated in
    FP32.  ``group`` is the expert-parallel process group; no other token
    transport may wrap this operation.
    """
    return _FP8Block128MegaMoE.apply(
        x,
        topk_ids,
        topk_scores,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        w13_master,
        w2_master,
        group,
    )
