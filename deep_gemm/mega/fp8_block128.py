"""Exact-SM103 persistent FP8-block128 MegaMoE training path.

The operation owns route transport, expert compute, POST_DOWN score
application, remote combine, reverse transport, activation backward, and two
dedicated BF16-semantics weight-gradient kernels.  GLM's canonical E4M3 +
FP32 128x128 q/s storage is consumed directly; there is no W13 repack, public
workspace-size knob, composed DeepEP path, architecture fallback, or JIT
configuration matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .. import _C

_BLOCK = 128
_MODEL_DIM = 6144
_INTERMEDIATE = 2048
_GLOBAL_EXPERTS = 256
_TOPK = 8
_SUPPORTED_EP = (2, 16)

REQUIRED_NATIVE_SYMBOLS = (
    "get_sm103_fp8_block128_capabilities",
    "sm103_fp8_block128_persistent_workspace_info",
    "sm103_fp8_block128_prepare_persistent_inputs",
    "sm103_fp8_block128_persistent_forward",
    "sm103_fp8_block128_persistent_backward",
)

REQUIRED_PYTHON_SYMBOLS = (
    "fp8_block128_mega_moe",
    "transform_glm_w13_for_fp8_block128_mega_moe",
    "get_fp8_block128_mega_moe_capabilities",
)


def get_fp8_block128_mega_moe_capabilities() -> dict[str, Any]:
    """Return a non-launching exact capability manifest for preflight."""
    native = dict(_C.get_sm103_fp8_block128_capabilities())
    missing = [name for name in REQUIRED_NATIVE_SYMBOLS if not hasattr(_C, name)]
    try:
        import torch.distributed._symmetric_memory as symm_mem

        if not callable(getattr(symm_mem, "empty", None)) or not callable(
            getattr(symm_mem, "rendezvous", None)
        ):
            missing.append("torch.distributed._symmetric_memory")
    except (AttributeError, ImportError):
        missing.append("torch.distributed._symmetric_memory")
    native.update(
        {
            "native_symbols": REQUIRED_NATIVE_SYMBOLS,
            "python_symbols": REQUIRED_PYTHON_SYMBOLS,
            "forward": not missing,
            "backward": not missing,
            "distributed_transport": "persistent_symmetric_ring",
            "transport_layout": "ring_l1_l2_wave",
            "transport_scale_layout": "ue8m0_power2_group128",
            "transport_deterministic": True,
            "combine_reductions": 1,
            "wgrad_backend": "two_persistent_fused_dequant_bf16",
            "supported_ep": _SUPPORTED_EP,
            "missing_symbols": tuple(missing),
        }
    )
    return native


def transform_glm_w13_for_fp8_block128_mega_moe(
    canonical_weight: torch.Tensor,
    canonical_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return canonical ``[gate, up]`` storage without a copy.

    The persistent scheduler issues independent TMA offsets for gate and up
    and presents ``[up; gate]`` only as a logical MMA/output order.  Returning
    the original tensors is intentional and is part of the no-repack ABI.
    """
    if canonical_weight.ndim != 3 or canonical_scale.ndim != 3:
        raise ValueError("canonical W13 weight and scale must both be rank 3")
    if canonical_weight.shape[0] % 2:
        raise ValueError("canonical W13 must contain gate/up pairs")
    experts = canonical_weight.shape[0] // 2
    hidden, model_dim = canonical_weight.shape[1:]
    expected_scale = (experts * 2, hidden // _BLOCK, model_dim // _BLOCK)
    if hidden % _BLOCK or model_dim % _BLOCK:
        raise ValueError("canonical W13 dimensions must be divisible by 128")
    if tuple(canonical_scale.shape) != expected_scale:
        raise ValueError(
            f"canonical W13 scale shape must be {expected_scale}, "
            f"got {tuple(canonical_scale.shape)}"
        )
    if not canonical_weight.is_contiguous() or not canonical_scale.is_contiguous():
        raise ValueError("canonical W13 q/s must be contiguous")
    return canonical_weight, canonical_scale


@dataclass(frozen=True)
class _GroupState:
    group: Any
    rank: int
    world_size: int


def _resolve_group(group: Any) -> _GroupState:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("persistent MegaMoE requires initialized torch.distributed")
    resolved = dist.group.WORLD if group is None else group
    state = _GroupState(
        group=resolved,
        rank=dist.get_rank(resolved),
        world_size=dist.get_world_size(resolved),
    )
    if state.world_size not in _SUPPORTED_EP:
        raise RuntimeError(
            f"persistent GLM MegaMoE supports EP{_SUPPORTED_EP}, "
            f"got EP{state.world_size}"
        )
    return state


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


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _validate_master_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    local_shape: tuple[int, int, int],
    global_shape: tuple[int, int, int],
    device: torch.device,
    master_gradient_wrapper: Any,
) -> None:
    local = _local_tensor(tensor)
    _check_tensor(local, name=name, ndim=3, dtype=torch.bfloat16, device=device)
    if tuple(local.shape) == local_shape:
        return
    is_distributed = callable(getattr(tensor, "to_local", None))
    if not is_distributed or master_gradient_wrapper is None:
        raise ValueError(
            f"{name} must have resident shape {local_shape}, got {tuple(local.shape)}"
        )
    if tuple(tensor.shape) != global_shape:
        raise ValueError(
            f"{name} distributed logical shape must be {global_shape}, "
            f"got {tuple(tensor.shape)}"
        )
    if local.numel() == 0:
        raise ValueError(f"{name} distributed local shard must be resident")


def _validate_inputs(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_master: torch.Tensor,
    w2_master: torch.Tensor,
    w3_master: torch.Tensor,
    group_state: _GroupState,
    master_gradient_wrapper: Any,
) -> None:
    if not x.is_cuda:
        raise ValueError("FP8-block128 MegaMoE requires CUDA")
    capability = torch.cuda.get_device_capability(x.device)
    if capability != (10, 3):
        raise RuntimeError(
            f"FP8-block128 MegaMoE is SM103-only; runtime capability is "
            f"{capability}. No fallback is available."
        )
    if master_gradient_wrapper is not None and not callable(master_gradient_wrapper):
        raise TypeError("master_gradient_wrapper must be callable or None")
    device = x.device
    _check_tensor(x, name="x", ndim=2, dtype=torch.bfloat16, device=device)
    _check_tensor(topk_ids, name="topk_ids", ndim=2, dtype=torch.int64, device=device)
    _check_tensor(
        topk_scores,
        name="topk_scores",
        ndim=2,
        dtype=torch.float32,
        device=device,
    )
    _check_tensor(
        w13_weight,
        name="w13_weight",
        ndim=3,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _check_tensor(
        w13_scale,
        name="w13_scale",
        ndim=3,
        dtype=torch.float32,
        device=device,
    )
    _check_tensor(
        w2_weight,
        name="w2_weight",
        ndim=3,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _check_tensor(
        w2_scale,
        name="w2_scale",
        ndim=3,
        dtype=torch.float32,
        device=device,
    )
    tokens = x.shape[0]
    if tuple(x.shape[1:]) != (_MODEL_DIM,):
        raise ValueError(f"persistent GLM MegaMoE requires token width {_MODEL_DIM}")
    if tuple(topk_ids.shape) != (tokens, _TOPK) or topk_scores.shape != topk_ids.shape:
        raise ValueError(f"top-k IDs/scores must have shape [tokens, {_TOPK}]")

    local_experts = _GLOBAL_EXPERTS // group_state.world_size
    expected_w13 = (2 * local_experts, _INTERMEDIATE, _MODEL_DIM)
    expected_w13_scale = (2 * local_experts, _INTERMEDIATE // 128, _MODEL_DIM // 128)
    expected_w2 = (local_experts, _MODEL_DIM, _INTERMEDIATE)
    expected_w2_scale = (local_experts, _MODEL_DIM // 128, _INTERMEDIATE // 128)
    if tuple(w13_weight.shape) != expected_w13:
        raise ValueError(f"canonical W13 must have shape {expected_w13}")
    if tuple(w13_scale.shape) != expected_w13_scale:
        raise ValueError(f"canonical W13 scales must have shape {expected_w13_scale}")
    if tuple(w2_weight.shape) != expected_w2:
        raise ValueError(f"W2 must have shape {expected_w2}")
    if tuple(w2_scale.shape) != expected_w2_scale:
        raise ValueError(f"W2 scales must have shape {expected_w2_scale}")

    local_w13 = (local_experts, _INTERMEDIATE, _MODEL_DIM)
    global_w13 = (_GLOBAL_EXPERTS, _INTERMEDIATE, _MODEL_DIM)
    local_w2 = (local_experts, _MODEL_DIM, _INTERMEDIATE)
    global_w2 = (_GLOBAL_EXPERTS, _MODEL_DIM, _INTERMEDIATE)
    _validate_master_tensor(
        w1_master,
        name="w1_master",
        local_shape=local_w13,
        global_shape=global_w13,
        device=device,
        master_gradient_wrapper=master_gradient_wrapper,
    )
    _validate_master_tensor(
        w2_master,
        name="w2_master",
        local_shape=local_w2,
        global_shape=global_w2,
        device=device,
        master_gradient_wrapper=master_gradient_wrapper,
    )
    _validate_master_tensor(
        w3_master,
        name="w3_master",
        local_shape=local_w13,
        global_shape=global_w13,
        device=device,
        master_gradient_wrapper=master_gradient_wrapper,
    )
    if topk_ids.numel():
        torch._assert_async(
            ((topk_ids >= 0) & (topk_ids < _GLOBAL_EXPERTS)).all(),
            f"top-k IDs must lie in [0, {_GLOBAL_EXPERTS})",
        )


@dataclass(frozen=True)
class _PersistentBufferState:
    buffer: torch.Tensor
    handle: Any
    buffer_ptrs: tuple[int, ...]
    rank: int
    context_tokens_per_rank: int
    workspace_info: dict[str, Any]


_persistent_context_tokens_per_rank: dict[int, int] = {}
_persistent_buffers: dict[tuple[int, int], _PersistentBufferState] = {}


def _configure_fp8_block128_mega_moe_transport(
    group: Any,
    *,
    context_tokens_per_rank: int,
) -> None:
    """Install the model's existing ``seq_len / CP`` envelope once.

    This internal setup hook is not an operation flag.  FireTitan derives the
    value from its model context and CP degree; callers cannot tune route-pool
    capacity, and the hot path never resizes or runs a sizing collective.
    """
    if (
        isinstance(context_tokens_per_rank, bool)
        or not isinstance(context_tokens_per_rank, int)
        or context_tokens_per_rank < 1
    ):
        raise ValueError("context_tokens_per_rank must be a positive integer")
    group_state = _resolve_group(group)
    key = id(group_state.group)
    existing = _persistent_context_tokens_per_rank.get(key)
    if existing is not None and existing != context_tokens_per_rank:
        raise RuntimeError(
            "MegaMoE context/CP envelope changed after setup: "
            f"configured={context_tokens_per_rank}, existing={existing}"
        )
    if existing is None and any(buffer_key[0] == key for buffer_key in _persistent_buffers):
        raise RuntimeError("MegaMoE context must be installed before arena allocation")
    _persistent_context_tokens_per_rank.setdefault(key, context_tokens_per_rank)


def _get_persistent_buffer(
    group_state: _GroupState,
    *,
    device: torch.device,
    tokens: int,
) -> _PersistentBufferState:
    context_tokens = _persistent_context_tokens_per_rank.get(id(group_state.group))
    if context_tokens is None:
        raise RuntimeError("MegaMoE was not configured from the owning model context")
    if tokens > context_tokens:
        raise RuntimeError(
            "SM103 MegaMoE input exceeds the derived seq_len/CP envelope: "
            f"actual={tokens}, envelope={context_tokens}"
        )
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (id(group_state.group), device_index)
    cached = _persistent_buffers.get(key)
    if cached is not None:
        return cached

    import torch.distributed._symmetric_memory as symm_mem

    info = dict(
        _C.sm103_fp8_block128_persistent_workspace_info(
            group_state.world_size, context_tokens
        )
    )
    buffer = symm_mem.empty(
        int(info["num_bytes"]), dtype=torch.int8, device=device
    )
    handle = symm_mem.rendezvous(buffer, group=group_state.group)
    buffer.zero_()
    # One setup synchronization publishes the fixed arena before any layer can
    # use remote pointers.  There is no per-call capacity synchronization.
    dist.barrier(group=group_state.group, device_ids=[device_index])
    torch.cuda.synchronize(device)
    pointers = tuple(int(pointer) for pointer in handle.buffer_ptrs)
    if len(pointers) != group_state.world_size:
        raise RuntimeError("symmetric-memory rendezvous returned an invalid rank map")
    state = _PersistentBufferState(
        buffer=buffer,
        handle=handle,
        buffer_ptrs=pointers,
        rank=group_state.rank,
        context_tokens_per_rank=context_tokens,
        workspace_info=info,
    )
    _persistent_buffers[key] = state
    return state


class _FP8Block128MegaMoEPersistent(torch.autograd.Function):
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
        w1_master: torch.Tensor,
        w2_master: torch.Tensor,
        w3_master: torch.Tensor,
        group: Any,
        master_gradient_wrapper: Any,
    ) -> torch.Tensor:
        group_state = _resolve_group(group)
        _validate_inputs(
            x,
            topk_ids,
            topk_scores,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            w1_master,
            w2_master,
            w3_master,
            group_state,
            master_gradient_wrapper,
        )
        state = _get_persistent_buffer(
            group_state, device=x.device, tokens=x.shape[0]
        )
        with torch.autograd.profiler.record_function(
            "sm103_fp8_block128_megamoe_persistent_forward"
        ):
            _C.sm103_fp8_block128_prepare_persistent_inputs(
                state.buffer,
                x,
                topk_ids,
                topk_scores,
                group_state.world_size,
                state.context_tokens_per_rank,
            )
            output, expert_counts, token_src_metadata = (
                _C.sm103_fp8_block128_persistent_forward(
                    state.buffer,
                    list(state.buffer_ptrs),
                    state.rank,
                    state.context_tokens_per_rank,
                    x.shape[0],
                    w13_weight,
                    w13_scale,
                    w2_weight,
                    w2_scale,
                )
            )
        ctx.buffer_state = state
        ctx.master_gradient_wrapper = master_gradient_wrapper
        ctx.save_for_backward(
            x,
            topk_scores,
            expert_counts,
            token_src_metadata,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            x,
            topk_scores,
            expert_counts,
            token_src_metadata,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
        ) = ctx.saved_tensors
        state = ctx.buffer_state
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)
        with torch.autograd.profiler.record_function(
            "sm103_fp8_block128_megamoe_persistent_backward"
        ):
            grad_x, grad_scores, grad_w1, grad_w2, grad_w3 = (
                _C.sm103_fp8_block128_persistent_backward(
                    state.buffer,
                    list(state.buffer_ptrs),
                    state.rank,
                    state.context_tokens_per_rank,
                    x,
                    grad_output,
                    topk_scores,
                    expert_counts,
                    token_src_metadata,
                    w13_weight,
                    w13_scale,
                    w2_weight,
                    w2_scale,
                )
            )
        wrapper = ctx.master_gradient_wrapper
        if wrapper is not None:
            grad_w1, grad_w2, grad_w3 = wrapper(grad_w1, grad_w2, grad_w3)
        return (
            grad_x,
            None,
            grad_scores,
            None,
            None,
            None,
            None,
            grad_w1,
            grad_w2,
            grad_w3,
            None,
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
    w1_master: torch.Tensor,
    w2_master: torch.Tensor,
    w3_master: torch.Tensor,
    group: Any = None,
    master_gradient_wrapper: Any = None,
) -> torch.Tensor:
    """Run GLM's complete routed branch in the fixed SM103 pipeline.

    Workspace capacity comes only from the model's preinstalled
    ``seq_len / CP`` envelope.  The operation exposes no route-pool bound,
    performs no arena growth, and consumes canonical gate/up expert pairs
    directly with separate TMA offsets.
    """
    group_state = _resolve_group(group)
    return _FP8Block128MegaMoEPersistent.apply(
        x,
        topk_ids,
        topk_scores,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        w1_master,
        w2_master,
        w3_master,
        group_state.group,
        master_gradient_wrapper,
    )
