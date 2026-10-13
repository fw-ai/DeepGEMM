# Derived from sgl-project/DeepGEMM@ec226bcf400c11d62755137b916fae9ee359c684:
#   deep_gemm/mega/__init__.py
# Copyright (c) 2025 DeepSeek, with modifications by the sgl-project/DeepGEMM contributors.
# Licensed under the MIT License (see LICENSE at the repository root).
"""sgl-project/DeepGEMM's NVFP4 x NVFP4 MegaMoE (decode), as an additive fw-ai DeepGEMM module.

Drop-in for the subset of the ``sgl_deep_gemm`` API that Fireworks uses::

    import deep_gemm.mega_nvfp4 as deep_gemm
    from deep_gemm.mega_nvfp4 import transform_ue4m3_sf_into_required_layout

The kernels are sgl's SM100 MegaMoE and fused pre-dispatch, compiled from copies under
``deep_gemm/include/deep_gemm/mega_nvfp4/`` so the existing ``deep_gemm`` / ``deep_gemm.mega`` kernels
(and their JIT cache keys) are untouched. Like the separate ``sgl_deep_gemm`` library, this module keeps
its own SM count (``set_num_sms`` here and ``deep_gemm.set_num_sms`` do not affect each other) and never
launches with PDL.

Fireworks changes vs. sgl's ``deep_gemm/mega/__init__.py``: ``_C`` entry points renamed with ``nvfp4``;
``nvfp4_mega_moe`` calls the kernel directly instead of going through ``fp8_fp4_mega_moe``; not ported:
``fp8_fp4_mega_moe`` (other recipes), ``bf16_mega_moe``, ``transform_scales_for_mega_moe`` and
``weight_layout='trtllm'`` (TRT-LLM layout), and the SM90 entry points.
"""
import os
import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C
from .utils import transform_ue4m3_sf_into_required_layout

__all__ = [
    'SymmBuffer',
    'get_block_m_for_mega_moe',
    'get_symm_buffer_for_mega_moe',
    'mega_moe_pre_dispatch',
    'nvfp4_mega_moe',
    'set_num_sms',
    'transform_ue4m3_sf_into_required_layout',
    'transform_weights_for_mega_moe',
]


def set_num_sms(num_sms: int) -> None:
    """SM count for the NVFP4 MegaMoE kernels and buffer sizing only (``deep_gemm.set_num_sms`` is separate).

    Call it before creating the symmetric buffer: its ring sizing reads the SM count.
    """
    _C.set_num_sms_for_nvfp4_mega_moe(num_sms)


def _check_legacy_fp4_acts_env() -> None:
    """The fork's `DG_USE_FP4_ACTS` / `DG_USE_MXF4_KIND` env vars are superseded by
    `mma_type`, and are refused rather than ignored.

    They cannot be silently remapped. `DG_USE_FP4_ACTS` selected FP4 activations under
    `kind::mxf8f6f4` while leaving weights in the gran-8 *unpacked* gate/up interleave;
    the equivalent typed kinds (`mxf4xmxf4` / `nvfp4xnvfp4`) are dense `kind::mxf4` and
    need the gran-16 *packed* interleave from `_interleave_weights_packed_fp4`. Honouring
    the env var here would leave a caller who does not also pass `mma_type` to
    `transform_weights_for_mega_moe` handing packed-layout kernels unpacked weights —
    silently wrong results. Fail loudly instead.
    """
    for name in ('DG_USE_FP4_ACTS', 'DG_USE_MXF4_KIND'):
        if int(os.environ.get(name, '0')) != 0:
            raise RuntimeError(
                f'`{name}` is no longer supported; it is replaced by the symmetric '
                f"buffer's `mma_type`. Pass mma_type='mxf4xmxf4' (per-32 UE8M0 scales) "
                f"or 'nvfp4xnvfp4' (per-16 UE4M3) to get_symm_buffer_for_mega_moe(), and "
                f'the SAME mma_type to transform_weights_for_mega_moe() — the FP4 kinds '
                f'need the packed gate/up interleave, so weights transformed without it '
                f'would be laid out wrongly.'
            )


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_nvfp4_mega_moe())
        assert activation in ('swiglu', 'swigluoai', 'situ'), f'Unsupported activation `{activation}`'
        assert activation != 'situ' or mma_type == 'fp8xfp4', \
            '`situ` activation is supported only for `fp8xfp4` MegaMoE'
        _check_legacy_fp4_acts_env()
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_nvfp4_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        # NOTES: sgl's tvm-ffi build returns undefined views (no SF for BF16, no shared experts) as
        # empty byte tensors; pybind returns `None`, so map those to the same empty tensors here
        def _view(value):
            return value if value is not None else torch.empty((0, ), dtype=torch.int8, device=self.buffer.device)
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf,
         self.x_scales) = map(_view, slice_input_buffers(self.buffer))

        # Activation types (the C++ views already carry them: BF16, UInt8 for packed FP4, or FP8)
        activation_dtype = (torch.bfloat16 if mma_type == 'bf16xbf16' else
                            torch.uint8 if mma_type in ('mxf4xmxf4', 'nvfp4xnvfp4') else
                            torch.float8_e4m3fn)
        for name in ('x', 'shared_l1_acts', 'shared_l2_acts', 'l1_acts', 'l2_acts'):
            value = getattr(self, name)
            if value.numel():
                setattr(self, name, value.view(activation_dtype))

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _interleave_weights_packed_fp4(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    g, n, *rest = t.shape
    half = n // 2
    assert half % 16 == 0
    gate = t[:, :half].reshape(g, half // 16, 16, *rest)
    up = t[:, half:].reshape(g, half // 16, 16, *rest)
    result = torch.cat([gate[:, :, 0::2], up[:, :, 0::2], gate[:, :, 1::2], up[:, :, 1::2]], dim=2)
    result = torch.empty_like(t).copy_(result.reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu',
    mma_type: str = 'fp8xfp4'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation in ('swiglu', 'swigluoai', 'situ'), f'Unsupported activation `{activation}`'
    assert activation != 'situ' or (
        isinstance(l1_weights, tuple) and isinstance(l2_weights, tuple)
    ), '`situ` activation is supported only for FP8xFP4 MegaMoE weights'
    if isinstance(l1_weights, tuple):
        # FP8/MXFP4/NVFP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP.
        interleave = _interleave_weights_packed_fp4 if mma_type in ('mxf4xmxf4', 'nvfp4xnvfp4') \
            else _interleave_weights
        l1_w = interleave(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(interleave(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed


def nvfp4_mega_moe(y: torch.Tensor,
                   l1_weights: Tuple[torch.Tensor, torch.Tensor],
                   l2_weights: Tuple[torch.Tensor, torch.Tensor],
                   sym_buffer: SymmBuffer,
                   shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                   activation: str = 'swiglu',
                   activation_clamp: Optional[float] = None,
                   fast_math: bool = True,
                   use_x_scales: bool = False,
                   l1_alphas: Optional[torch.Tensor] = None,
                   l2_alphas: Optional[torch.Tensor] = None,
                   l2_act_scales: Optional[torch.Tensor] = None):
    # sgl's `fp8_fp4_mega_moe(..., recipe=(1, 1, 16))`
    if use_x_scales and sym_buffer.mma_type != 'nvfp4xnvfp4':
        raise ValueError(
            '`use_x_scales` is only supported for `nvfp4xnvfp4`; '
            f'got mma_type={sym_buffer.mma_type!r}'
        )

    (l1_weights_data, l1_weights_sf) = l1_weights
    (l2_weights_data, l2_weights_sf) = l2_weights
    _C.nvfp4_mega_moe(
        y,
        (l1_weights_data, l1_weights_sf),
        (l2_weights_data, l2_weights_sf),
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        (1, 1, 16), sym_buffer.mma_type,
        activation, activation_clamp,
        fast_math, use_x_scales, l1_alphas, l2_alphas, l2_act_scales,
    )


def get_block_m_for_mega_moe(num_ranks: int, num_experts: int,
                             num_max_tokens_per_rank: int, num_tokens: int,
                             num_topk: int, mma_type: str = 'fp8xfp4') -> int:
    """`BLOCK_M` the kernel will pick — callers need it to lay out shared-expert SFs."""
    return int(_C.get_block_m_for_nvfp4_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type))


def mega_moe_pre_dispatch(x: torch.Tensor,
                          topk_idx: torch.Tensor,
                          topk_weights: torch.Tensor,
                          buf_x: torch.Tensor,
                          buf_x_sf: torch.Tensor,
                          buf_topk_idx: torch.Tensor,
                          buf_topk_weights: torch.Tensor,
                          num_tokens: int,
                          group_size: int = 32,
                          mma_type: str = 'fp8xfp4',
                          buf_x_scales: Optional[torch.Tensor] = None,
                          expert_scales: Optional[torch.Tensor] = None) -> None:
    _C.nvfp4_mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, mma_type,
        buf_x_scales, expert_scales,
    )
