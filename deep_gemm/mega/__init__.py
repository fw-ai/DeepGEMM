import math
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


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_ring_tokens: int,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ring_tokens = num_ring_tokens

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_ring_tokens
        )
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

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

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
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 chunk_ratio: Union[float, str, None] = None) -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # Ring capacity. Without chunking, the ring must hold one wave
    # (num_min = num_ranks * num_max_tokens_per_rank), which blows up for large prefill
    # and clamps max seqlen. With `chunk_ratio`, the ring holds `ratio * tokens` and the
    # kernel runs multiple dispatch cycles (num_cycles = ceil(total_recv / num_ring_tokens)).
    #
    # `chunk_ratio='auto'` sizes the ring to the no-wrap minimum: `num_max * num_topk` (the
    # worst-case total_recv per rank = every token's topk experts land here). This avoids
    # ring-wrap stalls (the pull waiting on empty_count at each wrap) with minimal memory —
    # matching the un-chunked baseline's latency. A manually-set float ratio < topk yields a
    # ring smaller than total_recv, which wraps and costs ~2-3% latency (the memory/perf trade).
    no_wrap_ring_tokens = align(num_max_tokens_per_rank * num_topk,
                                _C.get_token_alignment_for_mega_moe())
    chunk_auto = isinstance(chunk_ratio, str) and chunk_ratio == 'auto'
    chunk_enabled = chunk_ratio is not None
    num_min_ring_tokens, num_max_ring_tokens = \
        _C.get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
    if chunk_enabled:
        if chunk_auto:
            # No-wrap minimum: ring holds the worst-case total_recv (num_max * topk). Clamp to
            # the device ring limits. This matches the un-chunked baseline's latency (no wrap)
            # with the smallest ring that achieves it.
            num_ring_tokens = min(no_wrap_ring_tokens, num_max_ring_tokens)
            num_ring_tokens = max(num_ring_tokens, num_min_ring_tokens)
        else:
            num_ring_tokens = align(int(math.ceil(chunk_ratio * num_max_tokens_per_rank)),
                                    _C.get_token_alignment_for_mega_moe())
            # Chunking permits num_ring_tokens < num_min; only clamp to the upper bound.
            num_ring_tokens = min(num_ring_tokens, num_max_ring_tokens)
            # Warn if the chosen ring is smaller than the no-wrap minimum: the ring will wrap
            # (num_cycles may still be 1 if ring >= num_min, but the pull stalls on empty_count
            # at each wrap -> ~2-3% latency). Use chunk_ratio >= num_topk (or 'auto') to avoid.
            if num_ring_tokens < no_wrap_ring_tokens:
                warnings.warn(
                    f'chunk_ratio={chunk_ratio} -> ring={num_ring_tokens} < no-wrap minimum '
                    f'{no_wrap_ring_tokens} (num_max*num_topk); the ring will wrap and incur a '
                    f'~2-3% latency stall. Use chunk_ratio>=num_topk ({num_topk}) or "auto" '
                    f'for no-wrap baseline-equivalent latency.',
                    UserWarning, stacklevel=3)
    elif num_max_tokens_per_rank >= 6144:
        # We assume must be prefill (decode cannot have such size)
        # We try to give ~8 GB budget (within V4 Pro config)
        # And batch size is mostly stable, to save buffer size, we use 1 expert per wave
        num_ring_tokens = align(768 * 1024, _C.get_token_alignment_for_mega_moe())
        num_ring_tokens = max(num_ring_tokens, num_min_ring_tokens)
        num_ring_tokens = min(num_ring_tokens, num_max_ring_tokens)
    else:
        # Otherwise, we must ensure, like for EP64, 4K decoding batch size,
        # the wave heuristics can select the best number of experts per wave
        # In this case, the budget is roughly ~18 GB
        num_ring_tokens = _C.get_ring_limit_for_mega_moe(
            align(4096, _C.get_token_alignment_for_mega_moe()), 432 // 72, 6, 72)[1]
        num_ring_tokens = max(num_ring_tokens, num_min_ring_tokens)
        num_ring_tokens = min(num_ring_tokens, num_max_ring_tokens)

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
        num_ring_tokens,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )

def mxfp4_mxfp4_mega_moe(y: torch.Tensor,
                         l1_weights: Tuple[torch.Tensor, torch.Tensor],
                         l2_weights: Tuple[torch.Tensor, torch.Tensor],
                         sym_buffer: SymmBuffer,
                         cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                         recipe: Tuple[int, int, int] = (1, 1, 32),
                         activation: str = 'swiglu',
                         activation_clamp: Optional[float] = None,
                         fast_math: bool = True):
    # Packed MXFP4 x MXFP4 mega MoE (both activations and weights are E2M1 + UE8M0).
    _C.mxfp4_mxfp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )

def nvfp4_nvfp4_mega_moe(y: torch.Tensor,
                         l1_weights: Tuple[torch.Tensor, torch.Tensor],
                         l2_weights: Tuple[torch.Tensor, torch.Tensor],
                         sym_buffer: SymmBuffer,
                         gate_alpha: torch.Tensor,
                         up_alpha: torch.Tensor,
                         l2_input_global_scale: torch.Tensor,
                         down_alpha: torch.Tensor,
                         cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                         recipe: Tuple[int, int, int] = (1, 1, 16),
                         activation: str = 'swiglu',
                         activation_clamp: Optional[float] = None,
                         fast_math: bool = True):
    # Packed NVFP4 x NVFP4 mega MoE (E2M1 data, E4M3 SF gran-16). Per-expert global scales
    # (TRT-LLM convention), each (num_experts_per_rank,) float32 device tensor:
    #   gate_alpha / up_alpha = 1/(l1_input_gs * gate|up_weight_gs)   (L1 acc -> real)
    #   l2_input_global_scale = L2-input per-expert global scale       (L1-output requant)
    #   down_alpha            = 1/(l2_input_gs * down_weight_gs)       (L2 acc -> real)
    _C.nvfp4_nvfp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        gate_alpha, up_alpha, l2_input_global_scale, down_alpha,
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )
