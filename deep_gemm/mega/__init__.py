import torch
import types
import warnings
from enum import Enum
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(
        f"Failed to load mega kernels, please check your PyTorch version: {exception}"
    )

from .. import _C


class RouteWeightMode(str, Enum):
    PRE_DOWN = 'pre_down'
    POST_DOWN = 'post_down'


class CombineOrderMode(str, Enum):
    FIXED_TOPK = 'fixed_topk'
    DEEPEP = 'deepep'
    DEEPEP_V1 = 'deepep_v1'


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_ring_tokens: int,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation in ('swiglu', 'geglu'), f'Unsupported activation: `{activation}`'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ring_tokens = num_ring_tokens

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = \
            _C.get_symm_buffer_size_for_mega_moe_v2(
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
        # `token_src_metadata` exposes the Workspace combine source mapping
        # (rank_idx, token_idx, topk_idx) per pool row, needed by the training
        # backward to scatter per-expert grads back to source tokens/top-k slots.
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
            self.token_src_metadata,
            self.backward_grad_y,
            self.backward_grad_route,
        ) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.token_src_metadata = None
        self.backward_grad_y = None
        self.backward_grad_route = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 num_ring_tokens: Optional[int] = None
                                 ) -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # To save buffer size, we enable ring buffer
    # TODO: move the wave concept into kernel and dynamically schedule
    # TODO: currently decoding may consume more memory than prefill
    # TODO: finer-grained wave
    num_min_ring_tokens, num_max_ring_tokens = \
        _C.get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
    if num_ring_tokens is not None:
        if num_ring_tokens % _C.get_token_alignment_for_mega_moe() != 0:
            raise ValueError(
                "num_ring_tokens must satisfy MegaMoE token alignment")
        if not (
            num_min_ring_tokens <= num_ring_tokens <= num_max_ring_tokens
        ):
            raise ValueError(
                "num_ring_tokens must be within the supported ring limits "
                f"[{num_min_ring_tokens}, {num_max_ring_tokens}]")
    elif num_max_tokens_per_rank >= 6144:
        # We assume must be prefill (decode cannot have such size)
        # We try to give ~8 GB budget (within V4 Pro config)
        # And batch size is mostly stable, to save buffer size, we use 1 expert per wave
        num_ring_tokens = align(768 * 1024, _C.get_token_alignment_for_mega_moe())
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
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    # Gate/up interleaving is independent of the gated-activation variant
    assert activation in ('swiglu', 'geglu'), f'Unsupported activation: `{activation}`'
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


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    saved_l1_preact: Optional[torch.Tensor] = None,
    route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
    saved_down_unweighted: Optional[torch.Tensor] = None,
    num_config_tokens: Optional[int] = None,
):
    """Run MXFP8/FP4 MegaMoE with an explicit route-weight boundary.

    POST_DOWN quantizes the unweighted gated activation for W2 and optionally
    saves each route's exact BF16 unweighted W2 output before applying its
    score at the remote combine write.
    """
    route_weight_mode = RouteWeightMode(route_weight_mode)
    has_explicit_config_tokens = num_config_tokens is not None
    if num_config_tokens is None:
        num_config_tokens = y.size(0)
    if (
        not has_explicit_config_tokens and
        sym_buffer.group.size() > 1
    ):
        # Every rank must instantiate the same persistent grid/barrier
        # specialization, including empty source ranks. Select from the
        # collective maximum source-token extent while retaining each rank's
        # local runtime token count.
        rank_uniform_num_tokens = torch.tensor(
            num_config_tokens, dtype=torch.int32, device=y.device)
        dist.all_reduce(
            rank_uniform_num_tokens,
            op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        num_config_tokens = int(rank_uniform_num_tokens.item())
    _C.fp8_fp4_mega_moe(
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
        recipe,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens,
        saved_l1_preact,
        route_weight_mode.value,
        saved_down_unweighted,
        num_config_tokens,
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  saved_l1_preact: Optional[torch.Tensor] = None,
                  route_weight_mode: RouteWeightMode = RouteWeightMode.PRE_DOWN,
                  saved_h_unweighted: Optional[torch.Tensor] = None,
                  saved_h_weighted: Optional[torch.Tensor] = None,
                  saved_down_unweighted: Optional[torch.Tensor] = None,
                  combine_order_mode: CombineOrderMode =
                  CombineOrderMode.FIXED_TOPK,
                  precomputed_route_counts: Optional[torch.Tensor] = None,
                  active_pool_rows: Optional[int] = None,
                  route_count_mismatch: Optional[torch.Tensor] = None,
                  num_config_tokens: Optional[int] = None,
                  saved_x: Optional[torch.Tensor] = None):
    """Run BF16 MegaMoE with an explicit route-weight boundary.

    The optional stage saves expose unweighted/weighted activation and W2
    output boundaries for strict parity checks. ``saved_down_unweighted`` is
    also used by post-down backward for the exact router gradient.

    Training callers may provide an exact local source-route histogram,
    rank-uniform ``active_pool_rows``, and a scalar mismatch flag to size saved
    pools from actual receive counts. The kernel publishes the precomputed
    counts, verifies them against its internal dispatch count, and sets the
    flag before any caller can accept a truncated result.
    """
    route_weight_mode = RouteWeightMode(route_weight_mode)
    combine_order_mode = CombineOrderMode(combine_order_mode)
    if (
        (saved_h_unweighted is None) !=
        (saved_h_weighted is None)
    ):
        raise ValueError(
            "both activation stage outputs must be provided together")
    active_plan = (
        precomputed_route_counts is not None,
        active_pool_rows is not None,
        route_count_mismatch is not None,
    )
    if any(active_plan) and not all(active_plan):
        raise ValueError(
            "precomputed_route_counts, active_pool_rows, and "
            "route_count_mismatch must be provided together")
    has_precomputed_config_tokens = num_config_tokens is not None
    if num_config_tokens is None:
        num_config_tokens = y.size(0)
    if (
        not has_precomputed_config_tokens
        and sym_buffer.group.size() > 1
    ):
        # The config selects BLOCK_M, which defines the persistent launch and
        # pool packing. Empty source ranks can still receive expert rows, so
        # every rank must select the config from the same source-token extent.
        rank_uniform_num_tokens = torch.tensor(
            num_config_tokens, dtype=torch.int32, device=y.device)
        dist.all_reduce(
            rank_uniform_num_tokens,
            op=dist.ReduceOp.MAX,
            group=sym_buffer.group)
        num_config_tokens = int(rank_uniform_num_tokens.item())
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
        sym_buffer.num_ring_tokens,
        saved_l1_preact,
        route_weight_mode.value,
        saved_h_unweighted,
        saved_h_weighted,
        saved_down_unweighted,
        num_config_tokens,
        combine_order_mode.value,
        precomputed_route_counts,
        active_pool_rows,
        route_count_mismatch,
        saved_x,
    )
