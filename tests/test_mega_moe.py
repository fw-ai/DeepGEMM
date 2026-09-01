import argparse
import copy
import hashlib
import json
import os
import random
import sys
import pytest
import torch
import torch.distributed as dist
from types import SimpleNamespace
from typing import Tuple

import deep_gemm
from deep_gemm.mega.backward import (
    _direct_grad_x_planes,
    _gather_backward_ranges,
    _normalize_backward_ranges,
    _stage_backward_ranges,
)
from deep_gemm.utils import (
    per_token_cast_to_fp4, per_token_cast_to_fp8,
    cast_back_from_fp4, unpack_ue8m0_from_int,
)
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff


def test_fp8_backward_canonicalizes_block_m_and_clears_padding(
    monkeypatch,
):
    class FakeGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 0

    captured = {}

    def fake_all_reduce(tensor, *, op, group):
        assert op == dist.ReduceOp.MAX
        assert isinstance(group, FakeGroup)
        tensor.fill_(96)

    def fake_backward(*args):
        captured["block_m"] = args[20]
        captured["clear_wgrad_padding"] = args[23]
        captured["activation"] = args[37:41]
        captured["build_residual_mxfp8_weights"] = args[42]
        captured["inline_wgrad"] = args[49:52]
        captured["gate_up_prepared"] = args[52]
        captured["wgrad_outputs"] = tuple(
            tensor.data_ptr() for tensor in args[53:55]
        )
        captured["backward_range_sizes"] = args[55]
        captured["mxfp8_three_term_wgrad"] = args[56]
        captured["staged_topk_weights"] = args[26].clone()
        args[27][0, 0] = 3.0
        args[27][1, 0] = 7.0

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        deep_gemm._C,
        "fp8_fp4_mega_moe_backward_dgrad_swiglu_v2",
        fake_backward,
    )
    monkeypatch.setattr(deep_gemm._C, "get_num_sms", lambda: 148)

    rows, hidden, intermediate = 2, 4, 4
    grad_y = torch.zeros(rows, hidden)
    topk_weights = torch.ones(rows, 1)
    symmetric_capacity = 4
    backward_grad_x_storage = torch.zeros(
        2 * symmetric_capacity, hidden, dtype=torch.bfloat16)
    sym_buffer = SimpleNamespace(
        group=FakeGroup(),
        backward_grad_y=backward_grad_x_storage[:symmetric_capacity],
        topk_weights=torch.zeros(symmetric_capacity, 1),
        backward_grad_route=torch.zeros(symmetric_capacity, 1),
        handle=SimpleNamespace(buffer_ptrs=[1, 2]),
        num_max_tokens_per_rank=symmetric_capacity,
        num_topk=2,
        hidden=hidden,
    )
    direct_planes = _direct_grad_x_planes(sym_buffer)
    assert direct_planes.shape == (2 * symmetric_capacity, hidden)
    assert direct_planes.data_ptr() == sym_buffer.backward_grad_y.data_ptr()
    pool_hidden = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    pool_intermediate = torch.zeros(
        rows, intermediate, dtype=torch.bfloat16)
    w2_wgrad_output = torch.zeros(
        1, hidden, intermediate, dtype=torch.bfloat16
    )
    w13_wgrad_output = torch.zeros(
        1, 2 * intermediate, hidden, dtype=torch.bfloat16
    )
    combined_grad_route_output = torch.empty_like(topk_weights)

    deep_gemm.fp8_fp4_mega_moe_backward_dgrad_swiglu(
        gate_up_output=torch.zeros(
            rows, 2 * intermediate, dtype=torch.bfloat16),
        grad_h_output=pool_intermediate,
        grad_gate_up_output=torch.zeros(
            rows, 2 * intermediate, dtype=torch.bfloat16),
        h_act_output=pool_intermediate.clone(),
        h_weighted_output=pool_intermediate.clone(),
        x_pool_output=pool_hidden,
        grad_x_pool_output=pool_hidden.clone(),
        l1_acts=torch.zeros(rows, hidden, dtype=torch.float8_e4m3fn),
        l1_acts_sf=torch.zeros(rows, 1, dtype=torch.int32),
        l1_weights=(
            torch.zeros(1, 2 * intermediate, hidden // 2, dtype=torch.int8),
            torch.zeros(1, 2 * intermediate, 1, dtype=torch.int32),
        ),
        grad_ye=pool_hidden.clone(),
        route_weights=torch.ones(rows, dtype=torch.bfloat16),
        down_unweighted_output=pool_hidden.clone(),
        w2_weights=(
            torch.zeros(1, hidden, intermediate // 2, dtype=torch.int8),
            torch.zeros(1, hidden, 1, dtype=torch.int32),
        ),
        w2_dequant_scratch=torch.zeros(
            1, hidden, intermediate, dtype=torch.bfloat16),
        w13_weights=(
            torch.zeros(1, 2 * intermediate, hidden // 2, dtype=torch.int8),
            torch.zeros(1, 2 * intermediate, 1, dtype=torch.int32),
        ),
        w13_dequant_scratch=torch.zeros(
            1, 2 * intermediate, hidden, dtype=torch.bfloat16),
        expert_counts=torch.tensor([rows], dtype=torch.int32),
        grid_sync_counter=torch.zeros(1, dtype=torch.int32),
        activation_limit=10.0,
        block_m=16,
        compute_w13_dgrad=False,
        sym_buffer=sym_buffer,
        grad_y=grad_y,
        topk_weights=topk_weights,
        token_src_metadata=torch.zeros(rows, 3, dtype=torch.int32),
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        inline_wgrad=True,
        accumulate_wgrad=True,
        combined_grad_route_output=combined_grad_route_output,
        wgrad_outputs=(w2_wgrad_output, w13_wgrad_output),
        gate_up_prepared=True,
        backward_range_sizes=(2, 2, 2, 2, 1, 0, 2, 0, 0, 0),
        mxfp8_three_term_wgrad=True,
    )

    staged_topk_weights = captured.pop("staged_topk_weights")
    assert torch.equal(
        staged_topk_weights,
        torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
    )
    assert captured == {
        "block_m": 96,
        "clear_wgrad_padding": True,
        "activation": ("situ", 4.0, 25.0, False),
        "build_residual_mxfp8_weights": False,
        "inline_wgrad": (True, True, None),
        "gate_up_prepared": True,
        "wgrad_outputs": (
            w2_wgrad_output.data_ptr(),
            w13_wgrad_output.data_ptr(),
        ),
        "backward_range_sizes": [2, 2, 2, 2, 1, 0, 2, 0, 0, 0],
        "mxfp8_three_term_wgrad": True,
    }
    assert combined_grad_route_output[:, 0].tolist() == [3.0, 7.0]


def test_multirange_backward_rejects_non_sm148_before_staging(monkeypatch):
    binding_called = False

    def unexpected_binding(*_args):
        nonlocal binding_called
        binding_called = True

    monkeypatch.setattr(deep_gemm._C, "get_num_sms", lambda: 132)
    monkeypatch.setattr(
        deep_gemm._C,
        "fp8_fp4_mega_moe_backward_dgrad_swiglu_v2",
        unexpected_binding,
    )

    # Deliberately omit every symmetric staging tensor. Reaching staging would
    # therefore raise AttributeError instead of the required fail-closed error.
    sym_buffer = SimpleNamespace(num_max_tokens_per_rank=4)
    with pytest.raises(ValueError, match="runtime num_sms=148"):
        deep_gemm.fp8_fp4_mega_moe_backward_dgrad_swiglu(
            gate_up_output=None,
            grad_h_output=None,
            grad_gate_up_output=None,
            h_act_output=None,
            h_weighted_output=None,
            x_pool_output=None,
            grad_x_pool_output=None,
            l1_acts=None,
            l1_acts_sf=None,
            l1_weights=(None, None),
            grad_ye=torch.zeros(2, 1),
            route_weights=None,
            w2_weights=(None, None),
            w2_dequant_scratch=None,
            w13_weights=(None, None),
            w13_dequant_scratch=None,
            expert_counts=None,
            grid_sync_counter=None,
            activation_limit=0.0,
            block_m=192,
            sym_buffer=sym_buffer,
            grad_y=torch.zeros(2, 1),
            topk_weights=torch.ones(2, 1),
            token_src_metadata=torch.zeros(2, 3, dtype=torch.int32),
            backward_range_sizes=(
                1, 2, 1, 1, 1,
                1, 2, 1, 1, 1,
            ),
        )

    assert not binding_called


def test_backward_range_staging_and_gather_unequal_active_counts():
    ranges = _normalize_backward_ranges(
        (3, 4, 8, 8, 2, 1, 3, 4, 4, 1),
        num_active_tokens=4,
        symmetric_token_capacity=7,
    )
    source = torch.tensor([[10], [11], [12], [20]], dtype=torch.int32)
    physical = torch.full((7, 1), -1, dtype=torch.int32)

    _stage_backward_ranges(physical, source, ranges)

    assert physical[:, 0].tolist() == [10, 11, 12, -1, 20, -1, -1]
    physical_route = torch.tensor(
        [[1.0], [2.0], [3.0], [-9.0], [4.0], [-9.0], [-9.0]]
    )
    logical_route = torch.empty(4, 1)
    _gather_backward_ranges(logical_route, physical_route, ranges)
    assert logical_route[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_backward_range_normalizer_accepts_three_and_rejects_four_ranges():
    three = _normalize_backward_ranges(
        (
            2, 3, 8, 8, 2,
            1, 2, 4, 4, 1,
            1, 2, 4, 4, 1,
        ),
        num_active_tokens=4,
        symmetric_token_capacity=7,
    )
    assert len(three) == 3
    assert tuple(record[0] for record in three) == (2, 1, 1)

    with pytest.raises(ValueError, match="at most 3 physical ranges"):
        _normalize_backward_ranges(
            (
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ),
            num_active_tokens=4,
            symmetric_token_capacity=4,
        )

    assert _normalize_backward_ranges(
        None, num_active_tokens=0, symmetric_token_capacity=0
    ) == ()
    assert _normalize_backward_ranges(
        (), num_active_tokens=0, symmetric_token_capacity=0
    ) == ()


@pytest.mark.parametrize(
    ("sizes", "message"),
    (
        ((3, 2, 8, 8, 2), "cannot exceed"),
        ((2, 4, 8, 8, 2), "active-token total"),
        ((4, 5, 8, 8, 2), "exactly cover"),
    ),
)
def test_backward_range_layout_rejects_invalid_totals(sizes, message):
    with pytest.raises(ValueError, match=message):
        _normalize_backward_ranges(
            sizes,
            num_active_tokens=4,
            symmetric_token_capacity=7,
        )


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


def _apply_gate_activation(gate: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == 'swiglu':
        # Match FireTitan's F.silu CUDA boundary. Evaluating this as
        # gate * sigmoid(gate) rounds the reciprocal separately and can move a
        # later BF16 route-weight product across a tie.
        return torch.nn.functional.silu(gate)
    if activation == 'geglu':
        alpha = 1.5957691216057308  # 2 * sqrt(2 / pi)
        beta = 0.044715
        gate_sq = gate * gate
        z = (alpha * gate) * (1.0 + beta * gate_sq)
        # 0.5 * (1 + tanh(t)) == sigmoid(2 * t).
        return gate * torch.sigmoid(z)
    raise ValueError(f'Unsupported activation: {activation}')


def _triton_route_dot(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Match DeepEP's FMA and butterfly order for one route per row."""
    assert lhs.shape == rhs.shape and lhs.ndim == 2
    if lhs.size(0) == 0:
        return torch.empty(
            (0,), dtype=torch.float32, device=lhs.device)
    hidden = lhs.size(1)
    block_h = 1
    while block_h < hidden and block_h < 8192:
        block_h <<= 1
    num_warps = min(32, max(4, block_h // 256))
    num_threads = num_warps * 32
    values_per_thread = block_h // num_threads
    assert values_per_thread in (2, 4, 8)

    if hidden < block_h:
        lhs = torch.nn.functional.pad(lhs, (0, block_h - hidden))
        rhs = torch.nn.functional.pad(rhs, (0, block_h - hidden))
    lhs_lanes = lhs.float().view(-1, values_per_thread, num_threads).transpose(1, 2)
    rhs_lanes = rhs.float().view(-1, values_per_thread, num_threads).transpose(1, 2)

    def fma(first: int, second: int) -> torch.Tensor:
        product = lhs_lanes[:, :, second] * rhs_lanes[:, :, second]
        return torch.addcmul(
            product,
            lhs_lanes[:, :, first],
            rhs_lanes[:, :, first])

    if values_per_thread == 2:
        lane_sums = fma(0, 1)
    elif values_per_thread == 4:
        lane_sums = fma(0, 2) + fma(1, 3)
    else:
        lane_sums = (
            fma(0, 2) + fma(4, 6) +
            (fma(1, 3) + fma(5, 7)))

    warp_lanes = lane_sums.view(-1, num_warps, 32)
    lane_indices = torch.arange(32, device=lhs.device)
    for offset in (16, 8, 4, 2, 1):
        warp_lanes = (
            warp_lanes +
            warp_lanes[:, :, lane_indices ^ offset])
    warp_sums = warp_lanes[:, :, 0]

    warp_indices = torch.arange(num_warps, device=lhs.device)
    offset = num_warps // 2
    while offset:
        warp_sums = warp_sums + warp_sums[:, warp_indices ^ offset]
        offset >>= 1
    return warp_sums[:, 0]


def _native_route_dot(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Match the fused backward kernel's four-column route reduction."""
    assert lhs.shape == rhs.shape and lhs.ndim == 2
    num_rows, hidden = lhs.shape
    assert hidden % 4 == 0
    if num_rows == 0:
        return torch.empty(
            (0,), dtype=torch.float32, device=lhs.device)

    input_pow2 = 1
    vectorized_columns = hidden // 4
    while input_pow2 < 512 and input_pow2 * 2 <= vectorized_columns:
        input_pow2 *= 2
    initial_group_threads = min(input_pow2, 32)
    output_pow2 = 1 << (max(num_rows, 1).bit_length() - 1)
    block_height = min(output_pow2, 512 // initial_group_threads)
    group_threads = min(input_pow2, 512 // block_height)
    assert group_threads >= 32 and group_threads & (group_threads - 1) == 0

    # CUDA assigns four adjacent columns to each lane, then advances by one
    # whole route group. Keep the four accumulators separate until every
    # column has been consumed.
    products = (lhs.float() * rhs.float()).view(
        num_rows, -1, group_threads, 4)
    lane_sums = torch.zeros(
        (num_rows, group_threads, 4),
        dtype=torch.float32,
        device=lhs.device)
    for chunk in products.unbind(dim=1):
        lane_sums = lane_sums + chunk
    thread_sums = (
        (lane_sums[:, :, 0] + lane_sums[:, :, 1]) +
        lane_sums[:, :, 2]
    ) + lane_sums[:, :, 3]

    # Groups wider than one warp first fold through shared memory. The first
    # warp then finishes with CUDA's shfl_down tree.
    offset = group_threads // 2
    while offset >= 32:
        thread_sums = torch.cat(
            (
                thread_sums[:, :offset] +
                thread_sums[:, offset:offset * 2],
                thread_sums[:, offset * 2:],
            ),
            dim=1)
        offset //= 2
    warp_sums = thread_sums[:, :32]
    for offset in (16, 8, 4, 2, 1):
        warp_sums = torch.cat(
            (
                warp_sums[:, :offset] +
                warp_sums[:, offset:offset * 2],
                warp_sums[:, offset * 2:],
            ),
            dim=1)
    return warp_sums[:, 0]


def _dequant_x_fp8(x_fp8: torch.Tensor, sf_packed: torch.Tensor, gran_k: int = 32) -> torch.Tensor:
    # FP8 (E4M3) activations with packed UE8M0 per-`gran_k` scale factors
    m, n = x_fp8.shape
    sf = unpack_ue8m0_from_int(sf_packed)[:, :n // gran_k]
    return (x_fp8.float().view(m, n // gran_k, gran_k) * sf.unsqueeze(2)).view(m, n)


def _dequant_weight_fp4(w_bf16: torch.Tensor, gran_k: int = 32) -> torch.Tensor:
    # Emulate the exact FP4 (E2M1) values the kernel consumes by round-tripping
    # each expert's weight through the same quantizer used to build the inputs.
    num_experts = w_bf16.size(0)
    out = torch.empty_like(w_bf16, dtype=torch.float32)
    for e in range(num_experts):
        packed, sf = per_token_cast_to_fp4(w_bf16[e], use_ue8m0=True, gran_k=gran_k)
        out[e] = cast_back_from_fp4(packed, sf, gran_k=gran_k)
    return out


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    is_bf16xbf16 = args.mma_type == 'bf16xbf16'
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = (
        0 if args.zero_tokens or args.zero_rank == rank_idx else
        max(
            0,
            args.num_max_tokens_per_rank -
            random.randint(0, args.num_max_removed_tokens))
        if args.num_tokens == 0 else args.num_tokens
    )
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        mma_type=args.mma_type,
        activation=args.activation,
        num_ring_tokens=args.num_ring_tokens,
    )
    assert buffer.backward_grad_y.data_ptr() % 128 == 0, (
        'MegaMoE combine storage must satisfy the K3 ring/TMA base ABI'
    )
    legacy_num_bytes, legacy_slicer = (
        deep_gemm._C.get_symm_buffer_size_for_mega_moe(
            num_ranks, num_experts,
            buffer.num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            args.mma_type, args.activation,
            buffer.num_ring_tokens))
    assert len(legacy_slicer(buffer.buffer)) == 10
    expanded_num_bytes, expanded_slicer = (
        deep_gemm._C.get_symm_buffer_size_for_mega_moe_v2(
            num_ranks, num_experts,
            buffer.num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            args.mma_type, args.activation,
            buffer.num_ring_tokens))
    legacy_buffer = buffer.buffer.narrow(0, 0, legacy_num_bytes)
    legacy_v2_slices = expanded_slicer(legacy_buffer)
    assert len(legacy_v2_slices) == 11
    assert legacy_v2_slices[-1] is None
    legacy_forward_buffer = copy.copy(buffer)
    legacy_forward_buffer.buffer = legacy_buffer
    (
        legacy_forward_buffer.x,
        legacy_forward_buffer.x_sf,
        legacy_forward_buffer.topk_idx,
        legacy_forward_buffer.topk_weights,
        legacy_forward_buffer.l1_acts,
        legacy_forward_buffer.l1_acts_sf,
        legacy_forward_buffer.l2_acts,
        legacy_forward_buffer.l2_acts_sf,
        legacy_forward_buffer.token_src_metadata,
        legacy_forward_buffer.backward_grad_y,
    ) = legacy_slicer(legacy_buffer)
    legacy_forward_buffer.backward_grad_route = None
    assert (
        legacy_num_bytes +
        buffer.backward_grad_route.nbytes ==
        expanded_num_bytes
    ), 'v2 must append the route plane without shifting legacy storage'

    # Cast weights into FP4
    def _cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, source_x_bf16, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global l1_weights_bf16, l2_weights_bf16
        global saved_l1_preact, saved_h_unweighted
        global saved_h_weighted, saved_down_unweighted
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        global precomputed_route_counts, active_pool_rows
        global pool_block_m, local_expert_counts
        global local_padded_pool_rows, destination_counts
        global route_count_mismatch, num_config_tokens
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        source_x_bf16 = x
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        # Keep BF16 originals for the self-contained numerical reference
        l1_weights_bf16, l2_weights_bf16 = l1_weights, l2_weights
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        if args.routing == 'balanced':
            token_ids = torch.arange(
                num_tokens, device='cuda', dtype=torch.long).unsqueeze(1)
            slots = torch.arange(
                num_topk, device='cuda', dtype=torch.long).unsqueeze(0)
            topk_idx = (token_ids * num_topk + slots) % num_experts
        elif args.routing == 'skew':
            token_ids = torch.arange(
                num_tokens, device='cuda', dtype=torch.long).unsqueeze(1)
            slots = torch.arange(
                num_topk, device='cuda', dtype=torch.long).unsqueeze(0)
            tail = token_ids >= max(1, (num_tokens * 7) // 8)
            topk_idx = torch.where(
                tail,
                (token_ids * num_topk + slots) % num_experts,
                slots % num_experts)
        elif args.routing == 'extreme':
            assert num_topk <= num_experts
            topk_idx = torch.arange(
                num_topk, device='cuda', dtype=torch.long
            ).expand(num_tokens, -1).clone()
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        source_route_counts = torch.bincount(
            topk_idx[(topk_idx >= 0) & (topk_idx < num_experts)],
            minlength=num_experts).to(torch.int32)
        global_route_counts = source_route_counts.clone()
        if num_ranks > 1:
            dist.all_reduce(global_route_counts, group=group)
        num_config_tokens_tensor = torch.tensor(
            num_tokens, dtype=torch.int32, device='cuda')
        if num_ranks > 1:
            dist.all_reduce(
                num_config_tokens_tensor,
                op=dist.ReduceOp.MAX,
                group=group)
        num_config_tokens = int(num_config_tokens_tensor.item())
        expected_tokens_per_expert = (
            num_config_tokens * num_ranks * num_topk /
            num_experts)
        if expected_tokens_per_expert <= 8.5:
            pool_block_m = 16
        elif expected_tokens_per_expert <= 16.5:
            pool_block_m = 32
        elif expected_tokens_per_expert <= 32.5:
            pool_block_m = 64
        elif expected_tokens_per_expert <= 64.5:
            pool_block_m = 96
        elif expected_tokens_per_expert <= 96.5:
            pool_block_m = 128
        else:
            pool_block_m = 192
        if args.expect_block_m:
            assert pool_block_m == args.expect_block_m, (
                f'expected BLOCK_M={args.expect_block_m}, '
                f'got {pool_block_m} from collective '
                f'num_config_tokens={num_config_tokens}')
        destination_counts = global_route_counts.view(
            num_ranks, num_experts_per_rank)
        local_expert_counts = destination_counts[rank_idx].clone()
        padded_destination_counts = (
            (destination_counts + pool_block_m - 1) //
            pool_block_m * pool_block_m)
        local_padded_pool_rows = int(
            padded_destination_counts[rank_idx].sum().item())

        precomputed_route_counts = None
        active_pool_rows = None
        route_count_mismatch = None
        if args.active_saved_pool:
            active_pool_rows = int(
                padded_destination_counts.sum(dim=1).max().item())
        if args.active_saved_pool and is_bf16xbf16:
            precomputed_route_counts = source_route_counts
            route_count_mismatch = torch.zeros(
                1, dtype=torch.int32, device='cuda')

        if not is_bf16xbf16:
            # FP8 path: cast inputs to FP8/FP4 with per-32 UE8M0 SF
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
            x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            l1_weights = _cast_weights_to_fp4(l1_weights)
            l2_weights = _cast_weights_to_fp4(l2_weights)

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                l1_weights, l2_weights, activation=args.activation))
        saved_l1_preact = None
        saved_h_unweighted = None
        saved_h_weighted = None
        saved_down_unweighted = None
        if (
            args.save_l1_preact or args.test_backward
        ):
            saved_l1_preact = torch.full(
                (
                    (active_pool_rows if is_bf16xbf16 else None) or
                    buffer.token_src_metadata.size(0),
                    2 * intermediate_hidden),
                float('nan'), dtype=torch.bfloat16, device='cuda')
        if (args.save_forward_stages and
                not args.benchmark_backward and
                is_bf16xbf16 and (
            args.save_l1_preact or args.test_backward
        )):
            saved_h_unweighted = torch.full(
                (
                    active_pool_rows or
                    buffer.token_src_metadata.size(0),
                    intermediate_hidden),
                float('nan'), dtype=torch.bfloat16, device='cuda')
            saved_h_weighted = torch.full_like(
                saved_h_unweighted, float('nan'))
        if (
            (args.save_forward_stages or
             args.route_weight_mode == 'post_down') and
            (
                args.save_l1_preact or args.test_backward
            )
        ):
            saved_down_unweighted = torch.full(
                (
                    active_pool_rows or
                    buffer.token_src_metadata.size(0),
                    hidden),
                float('nan'), dtype=torch.bfloat16, device='cuda')

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused(sym_buffer=buffer):
        if is_bf16xbf16:
            sym_buffer.x[:num_tokens].copy_(x)
        else:
            sym_buffer.x[:num_tokens].copy_(x[0])
            sym_buffer.x_sf[:num_tokens].copy_(x[1])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=sym_buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            saved_l1_preact=saved_l1_preact)
        kernel_kwargs.update(
            route_weight_mode=deep_gemm.RouteWeightMode(
                args.route_weight_mode),
            saved_down_unweighted=saved_down_unweighted)
        if is_bf16xbf16:
            kernel_kwargs.update(
                combine_order_mode=deep_gemm.CombineOrderMode(
                    args.combine_order_mode),
                saved_h_unweighted=saved_h_unweighted,
                saved_h_weighted=saved_h_weighted,
                precomputed_route_counts=precomputed_route_counts,
                active_pool_rows=active_pool_rows,
                route_count_mismatch=route_count_mismatch,
                num_config_tokens=num_config_tokens)
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        if route_count_mismatch is not None:
            if num_ranks > 1:
                dist.all_reduce(
                    route_count_mismatch,
                    op=dist.ReduceOp.MAX,
                    group=group)
            assert route_count_mismatch.item() == 0
        return y, cumulative_local_expert_recv_stats_fused

    def run_legacy_low_level_inference():
        assert not is_bf16xbf16
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        y = torch.empty(
            (num_tokens, hidden),
            dtype=torch.bfloat16,
            device='cuda')
        deep_gemm._C.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights,
            transformed_l2_weights,
            cumulative_local_expert_recv_stats_fused,
            buffer.buffer,
            buffer.handle.buffer_ptrs,
            buffer.group.rank(),
            buffer.num_max_tokens_per_rank,
            buffer.num_experts,
            buffer.num_topk,
            (1, 1, 32),
            args.activation,
            args.activation_clamp,
            None,
            None,
            bool(args.fast_math),
            buffer.num_ring_tokens,
            None,
            deep_gemm.RouteWeightMode.PRE_DOWN.value,
            None,
            num_config_tokens,
            32)
        return y

    def active_pool_route_rows(
        destination_rank: int = rank_idx,
    ) -> torch.Tensor:
        """Return logical route rows, excluding per-expert block padding."""
        rows = []
        pool_offset = 0
        counts = destination_counts[destination_rank]
        for count in counts.cpu().tolist():
            rows.extend(range(pool_offset, pool_offset + count))
            pool_offset += (
                (count + pool_block_m - 1) //
                pool_block_m * pool_block_m)
        if destination_rank == rank_idx:
            assert pool_offset == local_padded_pool_rows
        return torch.tensor(rows, dtype=torch.long, device='cuda')

    def check_fp8_fp4_predown_regression(
        explicit_y: torch.Tensor,
        stats_before: torch.Tensor,
    ) -> None:
        """Default and explicit PRE_DOWN must preserve every legacy bit."""
        explicit_l2_acts = buffer.l2_acts.clone()
        explicit_l2_acts_sf = buffer.l2_acts_sf.clone()
        explicit_stats = (
            cumulative_local_expert_recv_stats_fused.clone())
        if args.check_predown_golden:
            golden_config = (
                num_ranks, num_tokens, num_max_tokens_per_rank,
                hidden, intermediate_hidden, num_experts, num_topk,
                args.routing, args.activation, args.activation_clamp,
                bool(args.fast_math),
            )
            assert golden_config == (
                1, 64, 384, 1024, 1024, 8, 1,
                'balanced', 'swiglu', 10.0, True,
            ), f'PRE_DOWN golden requires the canonical config, got {golden_config}'

            def digest(tensor: torch.Tensor) -> str:
                payload = (
                    tensor.detach().contiguous().cpu()
                    .view(torch.uint8).numpy().tobytes())
                return hashlib.sha256(payload).hexdigest()

            active_rows = active_pool_route_rows()
            actual = {
                'y': digest(explicit_y),
                'l2_acts': digest(explicit_l2_acts[active_rows]),
            }
            expected = {
                # Canonical pre-POST_DOWN MXFP4 contract on SM100.
                'y': (
                    'cba343d11238766747367c6b3f285cb7a495f269'
                    'cd4491bb4fb76c7e22ea4522'),
                'l2_acts': (
                    '49d8335ec6159bd3b1fa009ea1590c22dc9df0f54'
                    'f04a962d3a4f6c86aff57b3'),
            }
            assert actual == expected, (
                'PRE_DOWN no longer matches the legacy golden: '
                f'{actual}')
        cumulative_local_expert_recv_stats_fused.copy_(
            stats_before)
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        default_y = torch.empty_like(explicit_y)
        deep_gemm.fp8_fp4_mega_moe(
            y=default_y,
            l1_weights=transformed_l1_weights,
            l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=(
                cumulative_local_expert_recv_stats_fused),
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            saved_l1_preact=saved_l1_preact)
        assert torch.equal(default_y, explicit_y), (
            'explicit PRE_DOWN changed the legacy final output')
        assert torch.equal(
            buffer.l2_acts, explicit_l2_acts), (
                'explicit PRE_DOWN changed quantized W2 inputs')
        assert torch.equal(
            buffer.l2_acts_sf, explicit_l2_acts_sf), (
                'explicit PRE_DOWN changed W2 input scales')
        assert torch.equal(
            cumulative_local_expert_recv_stats_fused,
            explicit_stats), (
                'explicit PRE_DOWN changed receive accounting')

    def check_fp8_fp4_postdown_route_semantics(
        fused_y: torch.Tensor,
        stats_before: torch.Tensor,
    ) -> None:
        assert saved_down_unweighted is not None
        if args.require_ring_wrap:
            assert local_padded_pool_rows > buffer.num_ring_tokens, (
                'ring-wrap test did not exceed the reusable ring: '
                f'{local_padded_pool_rows=} '
                f'{buffer.num_ring_tokens=}')
        first_down = saved_down_unweighted.clone()
        first_stats = (
            cumulative_local_expert_recv_stats_fused.clone())
        active_rows = active_pool_route_rows()
        assert torch.isfinite(
            first_down[active_rows].float()).all(), (
                'every routed row must be saved')
        if args.active_saved_pool:
            assert first_down.size(0) == active_pool_rows
            assert local_padded_pool_rows <= first_down.size(0)
            route_mask = torch.zeros(
                first_down.size(0), dtype=torch.bool, device='cuda')
            route_mask[active_rows] = True
            local_pool_mask = torch.arange(
                first_down.size(0), device='cuda'
            ) < local_padded_pool_rows
            padding_rows = (
                local_pool_mask & ~route_mask).nonzero().flatten()
            assert padding_rows.numel() == (
                local_padded_pool_rows -
                int(local_expert_counts.sum().item()))
            assert torch.isfinite(
                first_down[padding_rows].float()).all(), (
                    'saved pool must retain expert block padding')
            # Rank-uniform allocation may include a tail after this rank's
            # block-padded local pool. It is not a route and must stay
            # untouched; route validity comes from counts, never sentinels.
            if local_padded_pool_rows < first_down.size(0):
                assert torch.isnan(
                    first_down[local_padded_pool_rows:].float()).all()
        metadata = buffer.token_src_metadata[active_rows].long()

        def source_down_planes(
            down_pool: torch.Tensor,
        ) -> torch.Tensor:
            """Canonicalize a nondeterministically ordered expert pool."""
            if num_ranks == 1:
                all_down = [down_pool]
                all_metadata = [buffer.token_src_metadata]
            else:
                all_down = [torch.empty_like(down_pool)
                            for _ in range(num_ranks)]
                all_metadata = [
                    torch.empty_like(buffer.token_src_metadata)
                    for _ in range(num_ranks)]
                dist.all_gather(all_down, down_pool, group=group)
                dist.all_gather(
                    all_metadata, buffer.token_src_metadata, group=group)
            source_planes = torch.zeros(
                (num_tokens, num_topk, hidden),
                dtype=torch.bfloat16, device='cuda')
            for destination_rank in range(num_ranks):
                destination_rows = active_pool_route_rows(
                    destination_rank)
                destination_metadata = all_metadata[
                    destination_rank][destination_rows].long()
                local_mask = (
                    destination_metadata[:, 0] == rank_idx)
                local_metadata = destination_metadata[local_mask]
                source_planes[
                    local_metadata[:, 1],
                    local_metadata[:, 2]] = all_down[destination_rank][
                        destination_rows[local_mask]]
            return source_planes

        def expected_output(
            source_planes: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            # The production combine applies the FP32 route score to the BF16
            # W2 boundary with one FMA per slot, accumulates slots in order,
            # then rounds the combined token once. This is also K3's native
            # non-DeepEP combine contract.
            expected = torch.zeros(
                (num_tokens, hidden),
                dtype=torch.float32, device='cuda')
            for slot in range(num_topk):
                expected = torch.addcmul(
                    expected,
                    source_planes[:, slot].float(),
                    weights[:, slot].float().unsqueeze(1))
            return expected.to(torch.bfloat16)

        first_source_down = source_down_planes(first_down)
        assert torch.equal(
            fused_y,
            expected_output(first_source_down, topk_weights)), (
                'POST_DOWN must multiply the BF16 down output at '
                'the remote combine write')

        changed_weights = torch.where(
            topk_idx >= 0,
            topk_weights * -0.375 + 0.125,
            torch.zeros_like(topk_weights))
        second_down = torch.full_like(
            saved_down_unweighted, float('nan'))
        cumulative_local_expert_recv_stats_fused.copy_(
            stats_before)
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(
            changed_weights)
        changed_y = torch.empty_like(fused_y)
        deep_gemm.fp8_fp4_mega_moe(
            y=changed_y,
            l1_weights=transformed_l1_weights,
            l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=(
                cumulative_local_expert_recv_stats_fused),
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            saved_l1_preact=saved_l1_preact,
            route_weight_mode=(
                deep_gemm.RouteWeightMode.POST_DOWN),
            saved_down_unweighted=second_down)
        assert torch.isfinite(
            second_down[active_rows].float()).all()
        second_source_down = source_down_planes(second_down)
        down_delta = second_source_down.float() - first_source_down.float()
        assert torch.equal(second_source_down, first_source_down), (
            'POST_DOWN W2 quantization must not depend on route scores: '
            f'max_abs={down_delta.abs().max().item():.6g}, '
            f'changed={down_delta.ne(0).sum().item()}/{down_delta.numel()}, '
            f'cosine={torch.nn.functional.cosine_similarity(second_source_down.float().flatten(), first_source_down.float().flatten(), dim=0).item():.9f}')
        assert torch.equal(
            changed_y,
            expected_output(second_source_down, changed_weights)), (
                'changed route scores must be applied after the '
                'saved BF16 down boundary')
        assert torch.equal(
            cumulative_local_expert_recv_stats_fused,
            first_stats), (
                'POST_DOWN changed receive accounting')
        saved_down_unweighted.copy_(first_down)

    # Self-contained PyTorch reference. Distributed routing/combine is modeled
    # for BF16; the quantized reference remains single-rank.
    def gather_rank_padded(
        local_tensor: torch.Tensor, fill_value: float
    ) -> torch.Tensor:
        padded_shape = (
            buffer.num_max_tokens_per_rank, *local_tensor.shape[1:])
        padded = torch.full(
            padded_shape, fill_value,
            dtype=local_tensor.dtype, device=local_tensor.device)
        padded[:local_tensor.size(0)].copy_(local_tensor)
        if num_ranks == 1:
            return padded.unsqueeze(0)
        gathered = [torch.empty_like(padded) for _ in range(num_ranks)]
        dist.all_gather(gathered, padded, group=group)
        return torch.stack(gathered)

    reference_stages = {}
    native_route_layout = {}

    def combine_source_planes(
        source_planes: torch.Tensor,
        source_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce local [token, slot, hidden] planes in the selected order."""
        if args.combine_order_mode == 'fixed_topk':
            assert source_weights is None
            combined = torch.zeros(
                (num_tokens, source_planes.size(-1)),
                dtype=torch.float32,
                device=source_planes.device)
            for slot in range(num_topk):
                combined += source_planes[:, slot].float()
            return combined.to(torch.bfloat16)

        valid = topk_idx[:num_tokens] >= 0
        route_ranks = torch.div(
            topk_idx[:num_tokens].clamp_min(0),
            num_experts_per_rank,
            rounding_mode='floor')
        combined = torch.zeros(
            (num_tokens, source_planes.size(-1)),
            dtype=torch.float32,
            device=source_planes.device)
        if args.combine_order_mode == 'deepep_v1':
            for master_rank in range(num_ranks):
                rank_partial = torch.zeros_like(combined)
                for slot in range(num_topk):
                    in_rank = (
                        valid[:, slot] &
                        (route_ranks[:, slot] == master_rank)
                    )
                    slot_values = source_planes[:, slot].float()
                    selected_values = torch.where(
                        in_rank.unsqueeze(1),
                        slot_values,
                        0.0)
                    if source_weights is None:
                        rank_partial += selected_values
                    else:
                        # Match the fused post-down reducer's single FP32 FMA.
                        rank_partial = torch.addcmul(
                            rank_partial,
                            selected_values,
                            source_weights[
                                :, slot].float().unsqueeze(1))
                combined += rank_partial.to(torch.bfloat16).float()
            return combined.to(torch.bfloat16)
        for master_slot in range(num_topk):
            master_rank = route_ranks[:, master_slot]
            is_last_for_rank = valid[:, master_slot].clone()
            for later_slot in range(
                    master_slot + 1, num_topk):
                is_last_for_rank &= ~(
                    valid[:, later_slot] &
                    (route_ranks[:, later_slot] ==
                     master_rank))
            rank_partial = torch.zeros_like(combined)
            for slot in range(num_topk):
                in_rank = (
                    is_last_for_rank &
                    valid[:, slot] &
                    (route_ranks[:, slot] == master_rank))
                slot_values = source_planes[:, slot].float()
                selected_values = torch.where(
                    in_rank.unsqueeze(1),
                    slot_values,
                    0.0)
                if source_weights is None:
                    rank_partial += selected_values
                else:
                    rank_partial = torch.addcmul(
                        rank_partial,
                        selected_values,
                        source_weights[
                            :, slot].float().unsqueeze(1))
            combined += rank_partial.to(torch.bfloat16).float()
        return combined.to(torch.bfloat16)

    def native_bf16_grouped_mm(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        offsets: torch.Tensor,
        *,
        rhs_needs_transpose: bool = True,
        lhs_needs_contiguous: bool = True,
    ) -> torch.Tensor:
        mat2 = rhs.transpose(-2, -1) if rhs_needs_transpose else rhs
        mat1 = lhs.contiguous() if lhs_needs_contiguous else lhs
        if not offsets.numel() or int(offsets[-1].item()) == 0:
            if lhs_needs_contiguous:
                output_columns = (
                    rhs.size(-2)
                    if rhs_needs_transpose
                    else rhs.size(-1))
                return lhs.new_zeros((0, output_columns))
            return lhs.new_zeros(
                (offsets.numel(), lhs.size(0), rhs.size(1)))
        return torch._grouped_mm(mat1, mat2, offs=offsets)

    def build_native_route_layout(
        all_topk_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_expert_start = rank_idx * num_experts_per_rank
        flat_experts = all_topk_idx.reshape(-1)
        valid = (
            (flat_experts >= local_expert_start) &
            (flat_experts < local_expert_start + num_experts_per_rank))
        flat_positions = valid.nonzero().flatten()
        local_experts = (
            flat_experts[flat_positions] - local_expert_start)
        # FireTitan DeepEP's permutation is expert-major and stable within an
        # expert: source token order first, then top-k slot order.
        order = torch.argsort(local_experts, stable=True)
        flat_positions = flat_positions[order]
        local_experts = local_experts[order]
        routes_per_rank = all_topk_idx.size(1) * num_topk
        source_ranks = flat_positions // routes_per_rank
        rank_positions = flat_positions % routes_per_rank
        source_tokens = rank_positions // num_topk
        source_slots = rank_positions % num_topk
        counts = torch.bincount(
            local_experts, minlength=num_experts_per_rank).to(torch.int32)
        return {
            'local_experts': local_experts,
            'source_ranks': source_ranks,
            'source_tokens': source_tokens,
            'source_slots': source_slots,
            'counts': counts,
            'offsets': counts.cumsum(0).to(torch.int32),
        }

    def native_to_pool_permutation(
        metadata: torch.Tensor,
        layout: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        routes_per_rank = (
            buffer.num_max_tokens_per_rank * num_topk)
        actual_keys = (
            metadata[:, 0] * routes_per_rank +
            metadata[:, 1] * num_topk +
            metadata[:, 2])
        native_keys = (
            layout['source_ranks'] * routes_per_rank +
            layout['source_tokens'] * num_topk +
            layout['source_slots'])
        positions_by_key = torch.full(
            (num_ranks * routes_per_rank,),
            -1, dtype=torch.long, device='cuda')
        positions_by_key[actual_keys] = torch.arange(
            actual_keys.numel(), dtype=torch.long, device='cuda')
        permutation = positions_by_key[native_keys]
        assert (permutation >= 0).all(), (
            'MegaMoE pool is missing a native DeepEP route')
        assert torch.unique(permutation).numel() == permutation.numel(), (
            'MegaMoE pool contains duplicate route metadata')
        return permutation

    def run_native_bf16_reference() -> torch.Tensor:
        all_topk_idx = gather_rank_padded(topk_idx, -1)
        all_topk_weights = gather_rank_padded(topk_weights, 0)
        all_x = gather_rank_padded(x, 0)
        layout = build_native_route_layout(all_topk_idx)
        native_route_layout.clear()
        native_route_layout.update(layout)

        source_ranks = layout['source_ranks']
        source_tokens = layout['source_tokens']
        source_slots = layout['source_slots']
        counts = layout['counts']
        offsets = layout['offsets']
        if source_ranks.numel():
            assert int(source_ranks.min().item()) >= 0
            assert int(source_ranks.max().item()) < all_x.size(0), (
                f'{source_ranks.max().item()=} {all_x.shape=}')
            assert int(source_tokens.min().item()) >= 0
            assert int(source_tokens.max().item()) < all_x.size(1), (
                f'{source_tokens.max().item()=} {all_x.shape=}')
        x_pool = all_x[source_ranks, source_tokens]
        route_weights = all_topk_weights[
            source_ranks, source_tokens, source_slots].float()

        w1_weights = l1_weights_bf16[
            :, :intermediate_hidden].contiguous()
        w3_weights = l1_weights_bf16[
            :, intermediate_hidden:].contiguous()
        gate = native_bf16_grouped_mm(
            x_pool, w1_weights, offsets)
        up = native_bf16_grouped_mm(
            x_pool, w3_weights, offsets)
        w13 = torch.cat((gate, up), dim=1)
        clamp = float(args.activation_clamp)
        gate_clamped = torch.clamp(gate, max=clamp)
        up_clamped = torch.clamp(up, min=-clamp, max=clamp)
        if (
            args.activation == 'swiglu' and
            clamp == float('inf')
        ):
            # Standard FireTitan grouped experts materialize F.silu's BF16
            # output before the separate BF16 multiply by W3(x).
            h = torch.nn.functional.silu(gate_clamped) * up_clamped
            h_for_weight = h.float()
        else:
            h_for_weight = (
                _apply_gate_activation(
                    gate_clamped.float(), args.activation) *
                up_clamped.float())
            h = h_for_weight.to(torch.bfloat16)
        h_weighted = (
            h_for_weight * route_weights.unsqueeze(1)
        ).to(torch.bfloat16)
        w2_input = (
            h_weighted
            if args.route_weight_mode == 'pre_down'
            else h)
        down = native_bf16_grouped_mm(
            w2_input, l2_weights_bf16, offsets)
        route_output = (
            down
            if args.route_weight_mode == 'pre_down'
            else (
                down.float() * route_weights.unsqueeze(1)
            ).to(torch.bfloat16))

        reference_stages.clear()
        reference_stages.update({
            'w13': w13,
            'h': h,
            'h_weighted': h_weighted,
            'w2_input': w2_input,
            'down': down,
            'route_output': route_output,
            'route_weights': route_weights,
        })

        # Match the native slot reducer: each top-k slot is accumulated in
        # ascending slot order with FP32 accumulation, then rounded to BF16.
        route_planes = torch.zeros(
            (
                num_ranks,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
            ),
            dtype=torch.float32,
            device='cuda')
        combine_route_output = (
            down
            if (
                args.combine_order_mode != 'fixed_topk' and
                args.route_weight_mode == 'post_down')
            else route_output)
        route_planes[
            source_ranks, source_tokens, source_slots
        ] = combine_route_output.float()
        if num_ranks > 1:
            dist.all_reduce(route_planes, group=group)
        slot_combined = combine_source_planes(
            route_planes[rank_idx, :num_tokens],
            (
                all_topk_weights[
                    rank_idx, :num_tokens]
                if (
                    args.combine_order_mode != 'fixed_topk' and
                    args.route_weight_mode == 'post_down')
                else None
            ))

        # The non-EP DSV4 bridge accumulates already expert-sorted output in
        # increasing expert-id order. This is intentionally only local: EP
        # uses DeepEP's combine rather than this eager expert loop.
        expert_combined = None
        if num_ranks == 1:
            expert_combined_fp32 = torch.zeros(
                (num_tokens, hidden),
                dtype=torch.float32,
                device='cuda')
            start = 0
            for count in counts.tolist():
                end = start + int(count)
                if end > start:
                    expert_combined_fp32[
                        source_tokens[start:end]
                    ] += route_output[start:end].float()
                start = end
            expert_combined = expert_combined_fp32.to(torch.bfloat16)

        reference_stages['final_slot'] = slot_combined
        reference_stages['final_expert'] = expert_combined
        return slot_combined

    def run_reference():
        assert is_bf16xbf16 or num_ranks == 1, (
            'Distributed numerical reference only supports BF16')
        if is_bf16xbf16:
            return run_native_bf16_reference()
        clamp = float(args.activation_clamp)
        x_deq = _dequant_x_fp8(
            x[0][:num_tokens], x[1][:num_tokens])
        l1_w = _dequant_weight_fp4(l1_weights_bf16)
        l2_w = _dequant_weight_fp4(l2_weights_bf16)
        num_reference_experts = num_experts_per_rank

        y = torch.zeros((num_tokens, hidden), dtype=torch.float32, device='cuda')
        for slot in range(num_topk):
            expert_idx = topk_idx[:num_tokens, slot]
            weight = topk_weights[:num_tokens, slot].float()
            for e in range(num_reference_experts):
                mask = expert_idx == e
                if not bool(mask.any()):
                    continue
                xt = x_deq[mask]

                # L1 GEMM then split into gate/up (first/second half of the output)
                acc1 = xt @ l1_w[e].t()
                gate = acc1[:, :intermediate_hidden].to(torch.bfloat16)
                up = acc1[:, intermediate_hidden:].to(torch.bfloat16)
                if clamp != float('inf'):
                    gate = torch.clamp(gate, max=clamp)
                    up = torch.clamp(up, min=-clamp, max=clamp)

                # Unweighted gated activation.
                act = (
                    _apply_gate_activation(
                        gate.float(), args.activation) *
                    up.float())

                # Requantize to FP8 (per-32 UE8M0), matching the
                # quantized kernel's L1 output.
                if args.route_weight_mode == 'pre_down':
                    act = act * weight[mask].unsqueeze(1)
                act_fp8, act_sf = per_token_cast_to_fp8(
                    act, use_ue8m0=True, gran_k=32)
                n_groups = intermediate_hidden // 32
                act_deq = (
                    act_fp8.float().view(-1, n_groups, 32) *
                    act_sf[:, :n_groups].unsqueeze(2)
                ).view(-1, intermediate_hidden)
                down = (
                    act_deq @ l2_w[e].t()
                ).to(torch.bfloat16)
                route_output = (
                    down
                    if args.route_weight_mode == 'pre_down'
                    else (
                        down.float() *
                        weight[mask].unsqueeze(1)
                    ).to(torch.bfloat16))

                # L2 GEMM, accumulate across the top-k experts
                y[mask] += route_output.float()
        return y.to(torch.bfloat16)

    def bf16_ulp_distance(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        def ordered_bits(tensor: torch.Tensor) -> torch.Tensor:
            bits = tensor.view(torch.int16).to(torch.int32) & 0xffff
            magnitude = bits & 0x7fff
            return torch.where(
                (bits & 0x8000) != 0,
                0x8000 - magnitude,
                0x8000 + magnitude)

        return (
            ordered_bits(lhs) - ordered_bits(rhs)
        ).abs()

    def report_bf16_parity(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> int:
        assert actual.shape == expected.shape, (
            f'{name}: {actual.shape=} != {expected.shape=}')
        mismatch_count = int((actual != expected).sum().item())
        if actual.numel():
            abs_error = (actual.float() - expected.float()).abs()
            max_abs = float(abs_error.max().item())
            denominator = expected.float().abs().clamp_min(
                torch.finfo(torch.float32).tiny)
            max_relative = float(
                (abs_error / denominator).max().item())
            max_ulp = int(
                bf16_ulp_distance(actual, expected).max().item())
        else:
            max_abs = max_relative = 0.0
            max_ulp = 0
        mismatch_detail = ''
        if mismatch_count:
            mismatch = actual != expected
            flat_mismatch_indices = (
                mismatch.flatten().nonzero().flatten())
            mismatch_ulps = bf16_ulp_distance(
                actual[mismatch], expected[mismatch])
            ulps, ulp_counts = torch.unique(
                mismatch_ulps, return_counts=True)
            top_count = min(12, ulp_counts.numel())
            top_indices = torch.topk(
                ulp_counts, top_count).indices
            ulp_histogram = ','.join(
                f'{int(ulp)}:{int(count)}'
                for ulp, count in zip(
                    ulps[top_indices].cpu().tolist(),
                    ulp_counts[top_indices].cpu().tolist()))
            first_flat = int(flat_mismatch_indices[0].item())
            first_actual = float(actual.flatten()[first_flat].float())
            first_expected = float(
                expected.flatten()[first_flat].float())
            coordinate_detail = ''
            if actual.ndim == 2:
                columns = actual.size(1)
                coordinate_detail = ', first_coords=' + ','.join(
                    f'({int(flat_idx) // columns},'
                    f'{int(flat_idx) % columns})'
                    for flat_idx in
                    flat_mismatch_indices[:12].cpu().tolist())
                row_mismatches = mismatch.sum(dim=1)
                mismatch_rows = row_mismatches.nonzero().flatten()
                coordinate_detail += ', row_counts={' + ','.join(
                    f'{int(row)}:{int(row_mismatches[row])}'
                    for row in mismatch_rows[:12].cpu().tolist()
                ) + '}'
            mismatch_detail = (
                f', top_ulp_hist={{{ulp_histogram}}}, '
                f'unique_ulps={ulps.numel()}, '
                f'first_flat={first_flat}, '
                f'first_actual={first_actual:.9g}, '
                f'first_expected={first_expected:.9g}'
                f'{coordinate_detail}')
        # Keep passing output rank-0-only, but never hide a mismatch from a
        # nonzero rank. Rank-local failures otherwise surface only as a generic
        # spawn assertion without the tensor location or ULP histogram.
        dist_print(
            f' > rank {rank_idx} {name}: mismatches={mismatch_count}/'
            f'{actual.numel()}, max_abs={max_abs:.9g}, '
            f'max_rel={max_relative:.9g}, max_ulp={max_ulp}'
            f'{mismatch_detail}',
            once_in_node=mismatch_count == 0)
        return mismatch_count

    def report_forward_stage_parity(
        fused_y: torch.Tensor,
        ref_y: torch.Tensor,
    ) -> None:
        final_mismatches = report_bf16_parity(
            'Stage final', fused_y, ref_y)
        if reference_stages['final_expert'] is not None:
            report_bf16_parity(
                'Stage final (native expert order)',
                fused_y,
                reference_stages['final_expert'])
        if saved_l1_preact is None:
            assert final_mismatches == 0, (
                'native grouped-MM final output must be bitwise equal')
            return

        valid_rows = torch.isfinite(
            saved_l1_preact.float()).all(dim=1)
        pool_indices = valid_rows.nonzero().flatten()
        metadata = buffer.token_src_metadata[pool_indices].long()
        native_to_pool = native_to_pool_permutation(
            metadata, native_route_layout)
        native_pool_indices = pool_indices[native_to_pool]
        pool_order_mismatches = int(
            (native_to_pool != torch.arange(
                native_to_pool.numel(),
                dtype=torch.long,
                device=native_to_pool.device)).sum().item())
        dist_print(
            f' > Pool-vs-DeepEP order mismatches='
            f'{pool_order_mismatches}/{native_to_pool.numel()}',
            once_in_node=True)
        activation_stage = (
            'h_weighted'
            if args.route_weight_mode == 'pre_down'
            else 'w2_input')
        actual_stages = {
            'w13': saved_l1_preact[native_pool_indices],
        }
        if saved_h_unweighted is not None:
            actual_stages['h'] = (
                saved_h_unweighted[native_pool_indices])
            actual_stages[activation_stage] = (
                saved_h_weighted[native_pool_indices]
                if args.route_weight_mode == 'pre_down'
                else saved_h_unweighted[native_pool_indices])
        if saved_down_unweighted is not None:
            actual_stages['down'] = (
                saved_down_unweighted[native_pool_indices])
        first_difference = (
            'final' if final_mismatches else None)
        for stage in actual_stages:
            mismatch_count = report_bf16_parity(
                f'Stage {stage}',
                actual_stages[stage],
                reference_stages[stage])
            if stage == 'h' and mismatch_count:
                mismatch = (
                    actual_stages[stage] != reference_stages[stage])
                first_flat = int(
                    mismatch.flatten().nonzero()[0].item())
                route_row = first_flat // intermediate_hidden
                hidden_col = first_flat % intermediate_hidden
                gate = reference_stages['w13'][
                    route_row, hidden_col].float()
                up = reference_stages['w13'][
                    route_row,
                    intermediate_hidden + hidden_col].float()
                alpha = 1.5957691216057308
                beta = 0.044715
                gate_sq = gate * gate
                z = (alpha * gate) * (1.0 + beta * gate_sq)
                explicit = (
                    gate / (1.0 + torch.exp(-z)) * up
                ).to(torch.bfloat16)
                print(
                    f' > rank {rank_idx} first h mismatch inputs: '
                    f'row={route_row}, col={hidden_col}, '
                    f'gate={float(gate):.9g}, up={float(up):.9g}, '
                    f'z={float(z):.9g}, '
                    f'explicit={float(explicit.float()):.9g}',
                    flush=True)
            if stage == 'h_weighted' and mismatch_count:
                mismatch = (
                    actual_stages[stage] != reference_stages[stage])
                first_flat = int(
                    mismatch.flatten().nonzero()[0].item())
                route_row = first_flat // intermediate_hidden
                hidden_col = first_flat % intermediate_hidden
                physical_row = int(
                    native_pool_indices[route_row].item())
                source = buffer.token_src_metadata[
                    physical_row].tolist()
                actual_h = float(
                    actual_stages['h'][
                        route_row, hidden_col].float())
                actual_weighted = float(
                    actual_stages['h_weighted'][
                        route_row, hidden_col].float())
                expected_weighted = float(
                    reference_stages['h_weighted'][
                        route_row, hidden_col].float())
                expected_weight = float(
                    reference_stages['route_weights'][
                        route_row].float())
                observed_weight = (
                    actual_weighted / actual_h
                    if actual_h else float('nan'))
                print(
                    f' > rank {rank_idx} first h_weighted mismatch inputs: '
                    f'native_row={route_row}, '
                    f'physical_row={physical_row}, col={hidden_col}, '
                    f'source={source}, h={actual_h:.9g}, '
                    f'expected_weight={expected_weight:.9g}, '
                    f'observed_weight={observed_weight:.9g}, '
                    f'actual={actual_weighted:.9g}, '
                    f'expected={expected_weighted:.9g}',
                    flush=True)
            if mismatch_count and first_difference is None:
                first_difference = stage
        dist_print(
            f' > First differing stage: {first_difference}',
            once_in_node=True)
        assert first_difference is None, (
            f'first native grouped-MM difference: {first_difference}')

    def check_bf16_parity(fused_y: torch.Tensor, ref_y: torch.Tensor):
        report_forward_stage_parity(fused_y, ref_y)
        assert torch.isfinite(fused_y.float()).all()
        assert torch.isfinite(ref_y.float()).all()

    def check_native_forward_repeatability(
        ref_y: torch.Tensor,
    ) -> None:
        first_stages = dict(reference_stages)
        repeated_y = run_native_bf16_reference()
        mismatch_count = report_bf16_parity(
            'Native repeat final', repeated_y, ref_y)
        assert mismatch_count == 0, (
            'native final output varied across identical runs')
        for stage in (
            'w13', 'h', 'h_weighted', 'w2_input',
            'down', 'route_output'
        ):
            mismatch_count = report_bf16_parity(
                f'Native repeat {stage}',
                reference_stages[stage],
                first_stages[stage])
            assert mismatch_count == 0, (
                f'native {stage} varied across identical runs')

        expected_stats = cumulative_local_expert_recv_stats_baseline.clone()
        all_topk_idx = gather_rank_padded(topk_idx, -1)
        local_expert_start = rank_idx * num_experts_per_rank
        valid_experts = all_topk_idx[
            (all_topk_idx >= local_expert_start) &
            (all_topk_idx < local_expert_start + num_experts_per_rank)
        ] - local_expert_start
        if valid_experts.numel():
            expected_stats += torch.bincount(
                valid_experts, minlength=num_experts_per_rank).to(torch.int)
        assert torch.equal(
            cumulative_local_expert_recv_stats_fused, expected_stats)

        if saved_l1_preact is not None:
            finite = torch.isfinite(saved_l1_preact.float())
            valid_rows = finite.all(dim=1)
            assert torch.equal(valid_rows, finite.any(dim=1)), (
                'saved pre-clamp rows must be either complete or untouched')
            assert valid_rows.sum().item() == valid_experts.numel()

    def run_bf16_backward_test():
        backward_base_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        config_num_tokens = torch.tensor(
            num_tokens, dtype=torch.int32, device='cuda')
        if num_ranks > 1:
            dist.all_reduce(
                config_num_tokens,
                op=dist.ReduceOp.MAX,
                group=group)
        expected_tokens_per_expert = (
            int(config_num_tokens.item()) *
            num_ranks * num_topk / num_experts)
        if expected_tokens_per_expert <= 8.5:
            block_m = 16
        elif expected_tokens_per_expert <= 16.5:
            block_m = 32
        elif expected_tokens_per_expert <= 32.5:
            block_m = 64
        elif expected_tokens_per_expert <= 64.5:
            block_m = 96
        elif expected_tokens_per_expert <= 96.5:
            block_m = 128
        else:
            block_m = 192

        all_topk_idx = gather_rank_padded(topk_idx, -1)
        all_topk_weights = gather_rank_padded(topk_weights, 0)
        local_expert_start = rank_idx * num_experts_per_rank
        local_route_mask = (
            (all_topk_idx >= local_expert_start) &
            (all_topk_idx <
             local_expert_start + num_experts_per_rank))
        local_expert_ids = (
            all_topk_idx[local_route_mask] - local_expert_start)
        expert_counts = torch.bincount(
            local_expert_ids,
            minlength=num_experts_per_rank).to(torch.int)
        padded_expert_counts = (
            (expert_counts + block_m - 1) // block_m * block_m)
        wgrad_pool_rows = int(padded_expert_counts.sum().item())

        pool_rows = saved_l1_preact.size(0)
        phase_ordered_memory = (
            args.backward_memory_mode == 'phase_ordered')
        reference_gate_up = (
            saved_l1_preact.clone()
            if phase_ordered_memory
            else saved_l1_preact)
        reference_down_unweighted = (
            saved_down_unweighted.clone()
            if phase_ordered_memory and
            args.route_weight_mode == 'post_down' and
            saved_down_unweighted is not None
            else saved_down_unweighted)
        grad_y = torch.randn(
            (num_tokens, hidden),
            dtype=torch.bfloat16, device='cuda')
        all_grad_y = gather_rank_padded(grad_y, 0)
        all_x = gather_rank_padded(x, 0)

        grad_ye = (
            saved_down_unweighted
            if phase_ordered_memory and
            args.route_weight_mode == 'post_down'
            else torch.zeros(
                (pool_rows, hidden),
                dtype=torch.bfloat16, device='cuda'))
        grad_y_unweighted = (
            grad_ye
            if args.route_weight_mode == 'pre_down'
            else torch.zeros_like(grad_ye))
        if (
            phase_ordered_memory and
            args.route_weight_mode == 'post_down'
        ):
            if intermediate_hidden > hidden:
                raise ValueError(
                    'phase_ordered post_down requires '
                    'intermediate_hidden <= hidden')
            grad_h = grad_y_unweighted.view(-1)[
                :pool_rows * intermediate_hidden
            ].view(pool_rows, intermediate_hidden)
            h_act = grad_h
            h_weighted = grad_h
        else:
            grad_h = torch.zeros(
                (pool_rows, intermediate_hidden),
                dtype=torch.bfloat16, device='cuda')
            h_act = torch.zeros_like(grad_h)
            h_weighted = (
                h_act
                if phase_ordered_memory
                else torch.zeros_like(grad_h))
            if (
                not phase_ordered_memory and
                args.route_weight_mode == 'post_down'
            ):
                h_act = h_weighted
        grad_gate_up = (
            saved_l1_preact
            if phase_ordered_memory
            else torch.zeros(
                (pool_rows, 2 * intermediate_hidden),
                dtype=torch.bfloat16, device='cuda'))
        x_pool = torch.zeros_like(grad_ye)
        grad_x_pool = (
            torch.zeros_like(grad_ye)
            if args.write_grad_x_pool
            else torch.empty(
                (0, hidden),
                dtype=torch.bfloat16,
                device='cuda'))
        route_weights_pool = torch.zeros(
            pool_rows, dtype=torch.float, device='cuda')
        grad_route_pool = torch.zeros_like(route_weights_pool)
        num_grid_states = (
            num_experts_per_rank *
            ((hidden // 64) *
             (intermediate_hidden // 128) +
             ((2 * intermediate_hidden) // 64) *
             (hidden // 128)) +
            2)
        grid_sync_counter = torch.zeros(
            num_grid_states, dtype=torch.int, device='cuda')
        metadata_before_backward = buffer.token_src_metadata.clone()

        def run_backward_dgrad():
            if (
                num_ranks > 1 and
                args.combine_order_mode == 'fixed_topk'
            ):
                # Repeated-call regression: every slot starts stale. Kernel A
                # must wait for remote grad-y pulls, clear all fixed-slot
                # planes, then publish only the current valid routes.
                torch.as_strided(
                    buffer.backward_grad_y,
                    size=(
                        num_topk,
                        buffer.num_max_tokens_per_rank,
                        hidden,
                    ),
                    stride=(
                        buffer.num_max_tokens_per_rank * hidden,
                        hidden,
                        1,
                    ),
                ).fill_(7)
            deep_gemm.bf16_mega_moe_backward_dgrad(
                gate_up_output=saved_l1_preact,
                grad_h_output=grad_h,
                grad_gate_up_output=grad_gate_up,
                h_act_output=h_act,
                h_weighted_output=h_weighted,
                x_pool_output=x_pool,
                grad_x_pool_output=grad_x_pool,
                grad_route_output=grad_route_pool,
                grad_ye=grad_ye,
                route_weights=route_weights_pool,
                w2_weights=l2_weights_bf16,
                w13_weights=l1_weights_bf16,
                expert_counts=expert_counts,
                grid_sync_counter=grid_sync_counter,
                grad_y=grad_y,
                sym_buffer=buffer,
                activation_limit=float(args.activation_clamp),
                block_m=block_m,
                activation=args.activation,
                fast_math=bool(args.fast_math),
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode),
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted,
                direct_remote_grad_x=num_ranks > 1,
                write_grad_x_pool=args.write_grad_x_pool,
                clear_wgrad_padding=True,
                python_numerical_correction=(
                    args.python_numerical_correction),
                combine_order_mode=args.combine_order_mode,
                memory_mode=args.backward_memory_mode)

        if args.benchmark_backward:
            for _ in range(args.backward_warmup):
                if phase_ordered_memory:
                    saved_l1_preact.copy_(reference_gate_up)
                    if args.route_weight_mode == 'post_down':
                        saved_down_unweighted.copy_(
                            reference_down_unweighted)
                run_backward_dgrad()
            dist.barrier()
            elapsed_ms = 0.0
            for _ in range(args.backward_iterations):
                if phase_ordered_memory:
                    saved_l1_preact.copy_(reference_gate_up)
                    if args.route_weight_mode == 'post_down':
                        saved_down_unweighted.copy_(
                            reference_down_unweighted)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_backward_dgrad()
                end.record()
                end.synchronize()
                elapsed_ms += start.elapsed_time(end)
            backward_ms = torch.tensor(
                elapsed_ms / args.backward_iterations,
                dtype=torch.float64,
                device='cuda')
            dist.all_reduce(
                backward_ms, op=dist.ReduceOp.MAX, group=group)
            pool_tensors = {
                'saved_gate_up': saved_l1_preact,
                'saved_down_unweighted': saved_down_unweighted,
                'grad_ye': grad_ye,
                'grad_y_unweighted': grad_y_unweighted,
                'grad_h': grad_h,
                'grad_gate_up': grad_gate_up,
                'h_act': h_act,
                'h_weighted': h_weighted,
                'x_pool': x_pool,
                'grad_x_pool': grad_x_pool,
                'route_weights': route_weights_pool,
                'grad_route': grad_route_pool,
            }
            seen_ptrs = set()
            logical_bytes = {}
            for name, tensor in pool_tensors.items():
                if tensor is None:
                    logical_bytes[name] = 0
                    continue
                is_alias = tensor.data_ptr() in seen_ptrs
                logical_bytes[name] = (
                    0 if is_alias else tensor.nbytes)
                seen_ptrs.add(tensor.data_ptr())
            benchmark_result = {
                'backward_dgrad_ms': float(
                    backward_ms.item()),
                'write_grad_x_pool':
                    args.write_grad_x_pool,
                'python_numerical_correction':
                    args.python_numerical_correction,
                'pool_rows': pool_rows,
                'pool_tensor_gib': {
                    name: value / 2 ** 30
                    for name, value in
                    logical_bytes.items()
                },
                'pool_total_gib':
                    sum(logical_bytes.values()) / 2 ** 30,
                'backward_incremental_allocated_gib': (
                    torch.cuda.memory_allocated() -
                    backward_base_allocated) / 2 ** 30,
                'backward_peak_incremental_gib': (
                    torch.cuda.max_memory_allocated() -
                    backward_base_allocated) / 2 ** 30,
            }
            dist_print(
                json.dumps(
                    benchmark_result, indent=2),
                once_in_node=True)
            return

        run_backward_dgrad()
        actual_direct_grad_route = (
            buffer.backward_grad_route[:num_tokens].clone())
        actual_direct_grad_x_planes = None
        if num_ranks > 1:
            actual_direct_grad_x_planes = torch.as_strided(
                buffer.backward_grad_y,
                size=(
                    num_topk,
                    buffer.num_max_tokens_per_rank,
                    hidden,
                ),
                stride=(
                    buffer.num_max_tokens_per_rank * hidden,
                    hidden,
                    1,
                ),
            ).clone()
        assert torch.equal(
            buffer.token_src_metadata,
            metadata_before_backward), (
                'backward must preserve forward source-route metadata')
        valid_rows = torch.isfinite(
            reference_gate_up.float()).all(dim=1)
        metadata = buffer.token_src_metadata[:pool_rows][valid_rows].long()
        actual_pool_indices = valid_rows.nonzero().flatten()

        ref_grad_ye = torch.zeros_like(grad_ye)
        ref_grad_y_unweighted = torch.zeros_like(
            grad_y_unweighted)
        ref_grad_h = torch.zeros_like(grad_h)
        ref_grad_gate_up = torch.zeros_like(grad_gate_up)
        ref_h_act = torch.zeros_like(h_act)
        ref_h_weighted = torch.zeros_like(h_weighted)
        ref_x_pool = torch.zeros_like(x_pool)
        ref_grad_x_pool = torch.zeros(
            (pool_rows, hidden),
            dtype=torch.bfloat16,
            device='cuda')
        ref_route_weights = torch.zeros_like(route_weights_pool)
        ref_grad_route = torch.zeros_like(grad_route_pool)

        layout = native_route_layout
        assert torch.equal(layout['counts'], expert_counts)
        native_to_pool = native_to_pool_permutation(metadata, layout)
        pool_indices = actual_pool_indices[native_to_pool]
        source_ranks = layout['source_ranks']
        source_tokens = layout['source_tokens']
        source_slots = layout['source_slots']
        offsets = layout['offsets']
        xe = all_x[source_ranks, source_tokens]
        gye = all_grad_y[source_ranks, source_tokens]
        route = all_topk_weights[
            source_ranks, source_tokens, source_slots]
        preact = reference_gate_up[pool_indices]
        gate = preact[:, :intermediate_hidden]
        up = preact[:, intermediate_hidden:]
        clamp = float(args.activation_clamp)
        gate_clamped = torch.clamp(gate, max=clamp)
        up_clamped = torch.clamp(
            up, min=-clamp, max=clamp)
        gate_fp32 = gate_clamped.float()
        up_fp32 = up_clamped.float()
        standard_bf16_swiglu = (
            args.activation == 'swiglu' and
            clamp == float('inf')
        )
        if args.activation == 'geglu':
            alpha = 1.5957691216057308
            beta = 0.044715
            gate_sq = gate_fp32 * gate_fp32
            z = (
                alpha * gate_fp32 *
                (1.0 + beta * gate_sq))
            dz = alpha * (
                1.0 + (3.0 * beta) * gate_sq)
        else:
            z = gate_fp32
            dz = torch.ones_like(gate_fp32)
        sig = torch.sigmoid(z)
        activated_gate = gate_fp32 * sig
        h = (
            torch.nn.functional.silu(gate_clamped) * up_clamped
            if standard_bf16_swiglu
            else (activated_gate * up_fp32).to(torch.bfloat16)
        )
        if args.route_weight_mode == 'pre_down':
            w2_input = (
                h.float() * route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            grad_w2_input = gye
            grad_h_w2 = native_bf16_grouped_mm(
                gye,
                l2_weights_bf16,
                offsets,
                rhs_needs_transpose=False)
            gh = (
                grad_h_w2.float() *
                route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            grad_route = _native_route_dot(grad_h_w2, h)
        else:
            w2_input = h
            grad_w2_input = (
                gye.float() * route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            grad_h_w2 = native_bf16_grouped_mm(
                grad_w2_input,
                l2_weights_bf16,
                offsets,
                rhs_needs_transpose=False)
            gh = grad_h_w2
            down = native_bf16_grouped_mm(
                h, l2_weights_bf16, offsets)
            grad_route = _triton_route_dot(gye, down)
            assert torch.equal(
                reference_down_unweighted[pool_indices], down)
        if standard_bf16_swiglu:
            silu_gate = torch.nn.functional.silu(gate_clamped)
            grad_silu = gh * up_clamped
            grad_gate_bf16 = torch.ops.aten.silu_backward(
                grad_silu, gate_clamped)
            grad_up_bf16 = gh * silu_gate
        else:
            activation_grad = (
                sig + gate_fp32 * sig * (1 - sig) * dz)
            grad_gate = (
                gh.float() * up_fp32 * activation_grad)
            grad_gate = torch.where(
                gate.float() <= clamp,
                grad_gate,
                torch.zeros_like(grad_gate))
            grad_up = gh.float() * activated_gate
            grad_up = torch.where(
                (up.float() >= -clamp) &
                (up.float() <= clamp),
                grad_up,
                torch.zeros_like(grad_up))
            grad_gate_bf16 = grad_gate.to(torch.bfloat16)
            grad_up_bf16 = grad_up.to(torch.bfloat16)
        grad_gu = torch.cat(
            (grad_gate_bf16, grad_up_bf16), dim=1)
        w1_weights = l1_weights_bf16[
            :, :intermediate_hidden].contiguous()
        w3_weights = l1_weights_bf16[
            :, intermediate_hidden:].contiguous()
        grad_xe = native_bf16_grouped_mm(
            grad_gate_bf16,
            w1_weights,
            offsets,
            rhs_needs_transpose=False)
        grad_xe.add_(native_bf16_grouped_mm(
            grad_up_bf16,
            w3_weights,
            offsets,
            rhs_needs_transpose=False))

        ref_grad_ye[actual_pool_indices] = grad_w2_input
        ref_grad_y_unweighted[pool_indices] = gye
        ref_grad_h[pool_indices] = grad_h_w2
        ref_grad_gate_up[actual_pool_indices] = grad_gu
        ref_h_act[pool_indices] = h
        ref_h_weighted[actual_pool_indices] = w2_input
        ref_x_pool[actual_pool_indices] = xe
        ref_grad_x_pool[pool_indices] = grad_xe
        ref_route_weights[pool_indices] = route
        ref_grad_route[pool_indices] = grad_route

        gradient_mismatches = {}
        gradient_similarities = {}

        def assert_gradient_close(
            actual: torch.Tensor,
            expected: torch.Tensor,
            name: str,
        ):
            finite = torch.isfinite(actual.float())
            if not finite.all():
                nonfinite = (~finite).nonzero()
                dist_print(
                    f' > rank {rank_idx} {name}: '
                    f'nonfinite={nonfinite.size(0)}, '
                    f'first={nonfinite[:12].tolist()}',
                    once_in_node=False)
            assert finite.all(), name
            if actual.dtype == torch.bfloat16:
                mismatch_count = report_bf16_parity(
                    name, actual, expected)
            else:
                mismatch_count = int(
                    (actual != expected).sum().item())
                max_abs = (
                    float(
                        (actual.float() - expected.float())
                        .abs().max().item())
                    if actual.numel() else 0.0)
                dist_print(
                    f' > {name}: mismatches={mismatch_count}/'
                    f'{actual.numel()}, max_abs={max_abs:.9g}',
                    once_in_node=True)
                if mismatch_count:
                    first = (actual != expected).nonzero()[0]
                    index = tuple(first.tolist())
                    dist_print(
                        f' > {name}: first={index}, '
                        f'actual={float(actual[index]):.9g}, '
                        f'expected={float(expected[index]):.9g}',
                        once_in_node=True)
            if mismatch_count:
                gradient_mismatches[name] = mismatch_count
                similarity = 1.0 - float(calc_diff(actual, expected))
                gradient_similarities[name] = similarity
                dist_print(
                    f' > {name}: symmetric cosine similarity='
                    f'{similarity:.9f}',
                    once_in_node=False)

        def wgrad_observable(tensor: torch.Tensor) -> torch.Tensor:
            # Phase-ordered outputs alias forward scratch. Kernel B consumes
            # only expert-count-rounded rows; the bucket tail is intentionally
            # left untouched and is no longer a logical tensor after reuse.
            return (
                tensor[:wgrad_pool_rows]
                if phase_ordered_memory
                else tensor
            )

        assert_gradient_close(
            wgrad_observable(grad_ye),
            wgrad_observable(ref_grad_ye),
            'grad_ye')
        if not (
            phase_ordered_memory and
            args.route_weight_mode == 'post_down'
        ):
            assert_gradient_close(
                grad_y_unweighted, ref_grad_y_unweighted,
                'grad_y_unweighted')
            assert_gradient_close(grad_h, ref_grad_h, 'grad_h')
        assert_gradient_close(
            wgrad_observable(grad_gate_up),
            wgrad_observable(ref_grad_gate_up),
            'grad_gate_up')
        if (
            args.activation == 'geglu' and
            not torch.equal(
                wgrad_observable(grad_gate_up),
                wgrad_observable(ref_grad_gate_up))
        ):
            mismatch_rows = (
                wgrad_observable(grad_gate_up) !=
                wgrad_observable(ref_grad_gate_up)
            ).nonzero()[:8]
            mismatch_details = []
            for pool_row, col in mismatch_rows.tolist():
                mismatch_details.append({
                    'pool_row': pool_row,
                    'col': col,
                    'actual': float(
                        grad_gate_up[pool_row, col]),
                    'expected': float(
                        ref_grad_gate_up[pool_row, col]),
                    'valid_row': bool(
                        torch.isfinite(
                            reference_gate_up[
                                pool_row, 0])),
                })
            print(
                f' > rank {rank_idx} GeGLU WGrad-pool mismatch details: '
                f'{mismatch_details}; '
                f'expert_counts={expert_counts.tolist()}',
                flush=True)
        if not (
            phase_ordered_memory and
            args.route_weight_mode == 'pre_down'
        ):
            assert_gradient_close(h_act, ref_h_act, 'h_act')
        assert_gradient_close(
            wgrad_observable(h_weighted),
            wgrad_observable(ref_h_weighted),
            'h_weighted')
        assert_gradient_close(x_pool, ref_x_pool, 'x_pool')
        if args.write_grad_x_pool:
            assert_gradient_close(
                grad_x_pool, ref_grad_x_pool,
                'grad_x_pool')
        assert_gradient_close(
            route_weights_pool, ref_route_weights,
            'route_weights_pool')
        assert_gradient_close(
            grad_route_pool, ref_grad_route,
            'grad_route_pool')
        if (
            args.activation == 'geglu' and
            not torch.equal(
                grad_gate_up[
                    actual_pool_indices,
                    :intermediate_hidden],
                grad_gate_bf16,
            )
        ):
            manual_sig = 1.0 / (1.0 + torch.exp(-z))
            manual_activation_grad = (
                manual_sig +
                gate_fp32 * manual_sig *
                (1.0 - manual_sig) * dz)
            manual_grad_gate = (
                gh.float() * up_fp32 *
                manual_activation_grad)
            manual_grad_gate = torch.where(
                gate.float() <= clamp,
                manual_grad_gate,
                torch.zeros_like(manual_grad_gate),
            ).to(torch.bfloat16)
            actual_grad_gate = grad_gate_up[
                actual_pool_indices,
                :intermediate_hidden]
            mismatch = actual_grad_gate != grad_gate_bf16
            mismatch_rows = mismatch.nonzero()[:8]
            mismatch_details = []
            for compact_row, col in mismatch_rows.tolist():
                mismatch_details.append({
                    'pool_row': int(
                        actual_pool_indices[compact_row]),
                    'compact_row': compact_row,
                    'col': col,
                    'actual': float(
                        actual_grad_gate[compact_row, col]),
                    'expected': float(
                        grad_gate_bf16[compact_row, col]),
                    'manual_sigmoid': float(
                        manual_grad_gate[compact_row, col]),
                    'gate': float(
                        gate_fp32[compact_row, col]),
                    'up': float(
                        up_fp32[compact_row, col]),
                    'grad_h': float(
                        gh[compact_row, col]),
                    'z': float(z[compact_row, col]),
                    'sigmoid': float(sig[compact_row, col]),
                    'manual_sig': float(
                        manual_sig[compact_row, col]),
                    'dz': float(dz[compact_row, col]),
                })
            print(
                f' > rank {rank_idx} GeGLU grad-gate mismatch details: '
                f'{mismatch_details}',
                flush=True)
        repeated_grad_h = native_bf16_grouped_mm(
            grad_w2_input,
            l2_weights_bf16,
            offsets,
            rhs_needs_transpose=False)
        repeated_grad_x = native_bf16_grouped_mm(
            grad_gate_bf16,
            w1_weights,
            offsets,
            rhs_needs_transpose=False)
        repeated_grad_x.add_(native_bf16_grouped_mm(
            grad_up_bf16,
            w3_weights,
            offsets,
            rhs_needs_transpose=False))
        if args.route_weight_mode == 'pre_down':
            repeated_grad_route = _native_route_dot(
                repeated_grad_h, h)
        else:
            repeated_down = native_bf16_grouped_mm(
                h, l2_weights_bf16, offsets)
            repeated_grad_route = _triton_route_dot(
                gye, repeated_down)
        assert_gradient_close(
            repeated_grad_h, grad_h_w2,
            'native_repeat_grad_h')
        assert_gradient_close(
            repeated_grad_x, grad_xe,
            'native_repeat_grad_x')
        assert_gradient_close(
            repeated_grad_route, grad_route,
            'native_repeat_grad_route')

        grad_w2 = torch.zeros_like(l2_weights_bf16)
        grad_w13 = torch.zeros_like(l1_weights_bf16)
        grad_x_combined = None
        grad_x_standalone = None
        if num_ranks > 1:
            grad_x_combined = torch.zeros(
                (num_tokens, hidden),
                dtype=torch.bfloat16, device='cuda')
            deep_gemm.bf16_mega_moe_backward_w2_combine(
                grad_w2, grad_ye, h_weighted,
                padded_expert_counts, block_m, grad_x_combined,
                buffer,
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode))
            deep_gemm.bf16_mega_moe_backward_w13_combine(
                grad_w13, grad_gate_up, x_pool,
                padded_expert_counts, block_m, grad_x_combined,
                buffer,
                combine_order_mode=deep_gemm.CombineOrderMode(
                    args.combine_order_mode))
            grad_x_standalone = torch.empty_like(grad_x_combined)
            deep_gemm.mega_moe_backward_combine_grad_x(
                grad_x_standalone,
                buffer,
                combine_order_mode=deep_gemm.CombineOrderMode(
                    args.combine_order_mode))
        else:
            deep_gemm.bf16_mega_moe_backward_w2(
                grad_w2, grad_ye, h_weighted,
                padded_expert_counts, block_m,
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode))
            deep_gemm.bf16_mega_moe_backward_w13(
                grad_w13, grad_gate_up, x_pool,
                padded_expert_counts, block_m)
        compact_grad_ye = ref_grad_ye[actual_pool_indices]
        compact_h_weighted = ref_h_weighted[actual_pool_indices]
        compact_grad_gate = ref_grad_gate_up[
            actual_pool_indices, :intermediate_hidden]
        compact_grad_up = ref_grad_gate_up[
            actual_pool_indices, intermediate_hidden:]
        compact_x = ref_x_pool[actual_pool_indices]
        # These are the exact native FireTitan transposed-LHS grouped-MM
        # calls. They preserve the native K-group boundaries and BF16 output
        # rounding; no per-expert Python matmul participates in parity.
        ref_grad_w2 = native_bf16_grouped_mm(
            compact_grad_ye.t(),
            compact_h_weighted,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        ref_grad_w1 = native_bf16_grouped_mm(
            compact_grad_gate.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        ref_grad_w3 = native_bf16_grouped_mm(
            compact_grad_up.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        inactive = (expert_counts == 0).view(-1, 1, 1)
        ref_grad_w2.masked_fill_(inactive, 0)
        ref_grad_w1.masked_fill_(inactive, 0)
        ref_grad_w3.masked_fill_(inactive, 0)
        ref_grad_w13 = torch.cat(
            (ref_grad_w1, ref_grad_w3), dim=1)
        repeated_grad_w2 = native_bf16_grouped_mm(
            compact_grad_ye.t(),
            compact_h_weighted,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        repeated_grad_w1 = native_bf16_grouped_mm(
            compact_grad_gate.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        repeated_grad_w3 = native_bf16_grouped_mm(
            compact_grad_up.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        repeated_grad_w2.masked_fill_(inactive, 0)
        repeated_grad_w1.masked_fill_(inactive, 0)
        repeated_grad_w3.masked_fill_(inactive, 0)
        assert_gradient_close(
            repeated_grad_w2, ref_grad_w2,
            'native_repeat_grad_w2')
        assert_gradient_close(
            repeated_grad_w1, ref_grad_w1,
            'native_repeat_grad_w1')
        assert_gradient_close(
            repeated_grad_w3, ref_grad_w3,
            'native_repeat_grad_w3')
        assert_gradient_close(
            grad_w2, ref_grad_w2, 'grad_w2')
        assert_gradient_close(
            grad_w13, ref_grad_w13, 'grad_w13')

        ref_route_planes = torch.zeros(
            (num_ranks, buffer.num_max_tokens_per_rank, num_topk),
            dtype=torch.float, device='cuda')
        ref_route_planes[
            source_ranks, source_tokens, source_slots
        ] = ref_grad_route[pool_indices]
        if num_ranks > 1:
            dist.all_reduce(ref_route_planes, group=group)
        assert_gradient_close(
            actual_direct_grad_route,
            ref_route_planes[rank_idx, :num_tokens],
            'grad_route_source')

        if num_ranks > 1:
            ref_grad_x_planes = torch.zeros(
                (num_ranks, buffer.num_max_tokens_per_rank,
                 num_topk, hidden),
                dtype=torch.bfloat16, device='cuda')
            ref_grad_x_planes[
                source_ranks, source_tokens, source_slots
            ] = ref_grad_x_pool[pool_indices]
            dist.all_reduce(ref_grad_x_planes, group=group)
            assert actual_direct_grad_x_planes is not None
            direct_expected = ref_grad_x_planes[
                rank_idx, :num_tokens
            ].permute(1, 0, 2)
            if not torch.equal(
                actual_direct_grad_x_planes[:, :num_tokens],
                direct_expected,
            ):
                cross_slot_mismatches = [
                    [
                        int((
                            actual_direct_grad_x_planes[
                                actual_slot, :num_tokens
                            ] != direct_expected[expected_slot]
                        ).sum().item())
                        for expected_slot in range(num_topk)
                    ]
                    for actual_slot in range(num_topk)
                ]
                print(
                    f' > rank {rank_idx} grad_x source slot cross-mismatches: '
                    f'{cross_slot_mismatches}',
                    flush=True)
            assert_gradient_close(
                actual_direct_grad_x_planes[
                    :, :num_tokens
                ].permute(1, 0, 2),
                ref_grad_x_planes[
                    rank_idx, :num_tokens
                ],
                'grad_x_source_planes')
            ref_grad_x = combine_source_planes(
                ref_grad_x_planes[rank_idx, :num_tokens])
            assert_gradient_close(
                grad_x_combined, ref_grad_x,
                'grad_x_combined')
            assert_gradient_close(
                grad_x_standalone, ref_grad_x,
                'grad_x_standalone')

        if (
            args.backward_memory_mode == 'legacy' and
            args.write_grad_x_pool
        ):
            # Raw positional ABI regression. Legacy bindings have no
            # dedicated route-gradient plane and therefore must compute only
            # the expert-major grad_route_output. In particular, they must
            # never reinterpret immutable source route weights as the direct
            # publication destination.
            source_route_weights_before = (
                buffer.topk_weights.clone())
            buffer.backward_grad_route.fill_(123.0)
            dedicated_route_plane_before = (
                buffer.backward_grad_route.clone())

            if args.route_weight_mode == 'post_down':
                deep_gemm._C.bf16_mega_moe_backward_post_down_prelude(
                    grad_y_unweighted,
                    grad_ye,
                    x_pool,
                    route_weights_pool,
                    grad_route_pool,
                    saved_down_unweighted,
                    expert_counts,
                    buffer.backward_grad_y,
                    buffer.x,
                    buffer.topk_weights,
                    buffer.token_src_metadata,
                    buffer.handle.buffer_ptrs,
                    buffer.group.rank(),
                    buffer.num_topk,
                    block_m,
                    args.combine_order_mode,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    256,
                )
                torch.cuda.synchronize()
                buffer.group.barrier()
                assert torch.equal(
                    buffer.topk_weights,
                    source_route_weights_before,
                ), 'legacy prelude overwrote source route weights'
                assert torch.equal(
                    buffer.backward_grad_route,
                    dedicated_route_plane_before,
                ), 'legacy prelude unexpectedly published route gradients'

            buffer.backward_grad_y[:num_tokens].copy_(grad_y)
            buffer.group.barrier()
            deep_gemm._C.bf16_mega_moe_backward_dgrad(
                saved_l1_preact,
                grad_h,
                grad_gate_up,
                h_act,
                h_weighted,
                x_pool,
                grad_x_pool,
                grad_route_pool,
                grad_ye,
                grad_y_unweighted,
                route_weights_pool,
                l2_weights_bf16,
                l1_weights_bf16,
                expert_counts,
                grid_sync_counter,
                float(args.activation_clamp),
                args.activation,
                bool(args.fast_math),
                args.route_weight_mode,
                saved_down_unweighted,
                block_m,
                False,
                True,
                True,
                buffer.backward_grad_y,
                buffer.x,
                buffer.topk_weights,
                buffer.token_src_metadata,
                buffer.handle.buffer_ptrs,
                buffer.group.rank(),
                buffer.num_max_tokens_per_rank,
                buffer.num_topk,
                args.combine_order_mode,
                'legacy',
                None,
            )
            torch.cuda.synchronize()
            buffer.group.barrier()
            assert torch.equal(
                buffer.topk_weights,
                source_route_weights_before,
            ), 'legacy dgrad overwrote source route weights'
            assert torch.equal(
                buffer.backward_grad_route,
                dedicated_route_plane_before,
            ), 'legacy dgrad unexpectedly published route gradients'

        similarity_failures = {
            name: similarity
            for name, similarity in gradient_similarities.items()
            if similarity < 0.999
        }
        assert not similarity_failures, (
            'native FireTitan gradient similarity below 0.999: '
            f'{similarity_failures}; mismatches={gradient_mismatches}')

    def run_fp8_fp4_route_backward_test():
        assert args.activation == 'swiglu'
        assert saved_l1_preact is not None
        pool_rows = saved_l1_preact.size(0)
        block_m = pool_block_m
        active_rows = active_pool_route_rows()
        expert_counts = local_expert_counts
        metadata = buffer.token_src_metadata[active_rows].long()
        route_weights = torch.zeros(
            pool_rows, dtype=torch.float32, device='cuda')
        all_topk_weights = gather_rank_padded(topk_weights, 0)
        exact_route_weights = all_topk_weights[
            metadata[:, 0], metadata[:, 1], metadata[:, 2]]
        assert torch.any(
            exact_route_weights !=
            exact_route_weights.to(torch.bfloat16).float()), (
                'test route scores must exercise non-BF16 FP32 bits')
        route_weights[active_rows] = exact_route_weights
        source_grad_y = torch.randn(
            (num_tokens, hidden),
            dtype=torch.bfloat16, device='cuda')
        all_source_grad_y = gather_rank_padded(source_grad_y, 0)
        grad_y_unweighted = torch.zeros(
            (pool_rows, hidden),
            dtype=torch.bfloat16, device='cuda')
        grad_y_unweighted[active_rows] = all_source_grad_y[
            metadata[:, 0], metadata[:, 1]]

        def cast_backward_weight(
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            packed = torch.empty(
                (*weight.shape[:-1], weight.size(-1) // 2),
                dtype=torch.int8, device='cuda')
            scales = torch.empty(
                (*weight.shape[:-1], weight.size(-1) // 32),
                dtype=torch.float32, device='cuda')
            for expert in range(weight.size(0)):
                packed[expert], scales[expert] = (
                    per_token_cast_to_fp4(
                        weight[expert],
                        use_ue8m0=True, gran_k=32))
            return packed.view(torch.float8_e4m3fn), scales

        backward_l1_weights = cast_backward_weight(
            l1_weights_bf16)
        backward_l2_weights = cast_backward_weight(
            l2_weights_bf16)
        w13_weights = (
            backward_l1_weights[0].view(
                2 * num_experts_per_rank,
                intermediate_hidden, hidden // 2),
            backward_l1_weights[1].view(
                2 * num_experts_per_rank,
                intermediate_hidden, hidden // 32),
        )
        num_grid_states = (
            num_experts_per_rank *
            ((hidden // 64) *
             (intermediate_hidden // 128) +
             ((2 * intermediate_hidden) // 64) *
             (hidden // 128)) +
            2)

        def allocate_outputs(grad_ye_input=None):
            return {
                'grad_h': torch.zeros(
                    (pool_rows, intermediate_hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'grad_gate_up': torch.zeros_like(
                    saved_l1_preact),
                'h_act': torch.zeros(
                    (pool_rows, intermediate_hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'h_weighted': torch.zeros(
                    (pool_rows, intermediate_hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'x_pool': torch.zeros(
                    (pool_rows, hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'grad_x_pool': torch.zeros(
                    (pool_rows, hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'grad_ye': (
                    torch.zeros(
                        (pool_rows, hidden),
                        dtype=torch.bfloat16, device='cuda')
                    if grad_ye_input is None
                    else grad_ye_input.clone()),
                'w2_scratch': torch.empty(
                    (
                        num_experts_per_rank, hidden,
                        intermediate_hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'w13_scratch': torch.empty(
                    (
                        num_experts_per_rank,
                        2 * intermediate_hidden, hidden),
                    dtype=torch.bfloat16, device='cuda'),
                'grid': torch.zeros(
                    num_grid_states,
                    dtype=torch.int32, device='cuda'),
            }

        def run_backward(
            outputs,
            compute_w13_dgrad=True,
            direct_remote_grad_x=None,
            write_grad_x_pool=True,
            **route_kwargs,
        ):
            if direct_remote_grad_x is None:
                direct_remote_grad_x = num_ranks > 1
            if (
                num_ranks > 1 and
                args.combine_order_mode == 'fixed_topk'
            ):
                torch.as_strided(
                    buffer.backward_grad_y,
                    size=(
                        num_topk,
                        buffer.num_max_tokens_per_rank,
                        hidden,
                    ),
                    stride=(
                        buffer.num_max_tokens_per_rank * hidden,
                        hidden,
                        1,
                    ),
                ).fill_(7)
            deep_gemm.fp8_fp4_mega_moe_backward_dgrad_swiglu(
                gate_up_output=outputs.get(
                    'gate_up_input', saved_l1_preact),
                grad_h_output=outputs['grad_h'],
                grad_gate_up_output=outputs['grad_gate_up'],
                h_act_output=outputs['h_act'],
                h_weighted_output=outputs['h_weighted'],
                x_pool_output=outputs['x_pool'],
                grad_x_pool_output=outputs['grad_x_pool'],
                l1_acts=buffer.l1_acts[:pool_rows],
                l1_acts_sf=buffer.l1_acts_sf[
                    : (pool_rows // block_m) *
                    ((block_m + 127) // 128 * 128)
                ],
                l1_weights=transformed_l1_weights,
                grad_ye=outputs['grad_ye'],
                route_weights=route_weights,
                w2_weights=backward_l2_weights,
                w2_dequant_scratch=outputs['w2_scratch'],
                w13_weights=w13_weights,
                w13_dequant_scratch=outputs['w13_scratch'],
                expert_counts=expert_counts,
                grid_sync_counter=outputs['grid'],
                activation_limit=float(args.activation_clamp),
                block_m=block_m,
                compute_w13_dgrad=compute_w13_dgrad,
                direct_remote_grad_x=direct_remote_grad_x,
                write_grad_x_pool=write_grad_x_pool,
                clear_wgrad_padding=True,
                sym_buffer=buffer,
                grad_y=source_grad_y,
                topk_weights=topk_weights,
                token_src_metadata=buffer.token_src_metadata,
                **route_kwargs)

        def assert_direct_route_plane(expected_grad_route):
            expected_route_planes = torch.zeros(
                (
                    num_ranks,
                    buffer.num_max_tokens_per_rank,
                    num_topk,
                ),
                dtype=torch.float32,
                device='cuda',
            )
            expected_route_planes[
                metadata[:, 0], metadata[:, 1], metadata[:, 2]
            ] = expected_grad_route
            if num_ranks > 1:
                dist.all_reduce(expected_route_planes, group=group)
            torch.testing.assert_close(
                buffer.backward_grad_route[:num_tokens],
                expected_route_planes[rank_idx, :num_tokens],
                rtol=2e-6,
                atol=2e-4,
            )

        if args.route_weight_mode == 'post_down':
            assert saved_down_unweighted is not None
            grad_route = torch.zeros(
                pool_rows, dtype=torch.float32, device='cuda')
            outputs = allocate_outputs()
            run_backward(
                outputs,
                route_weight_mode=deep_gemm.RouteWeightMode.POST_DOWN,
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted,
                grad_route_output=grad_route)
            prepared_outputs = allocate_outputs()
            prepared_grad_route = torch.zeros_like(grad_route)
            run_backward(
                prepared_outputs,
                route_weight_mode=deep_gemm.RouteWeightMode.POST_DOWN,
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted,
                grad_route_output=prepared_grad_route,
                gate_up_prepared=True)
            for name in (
                'grad_h', 'grad_gate_up', 'h_act', 'h_weighted',
                'x_pool', 'grad_x_pool', 'grad_ye',
                'w2_scratch', 'w13_scratch',
            ):
                torch.testing.assert_close(
                    prepared_outputs[name], outputs[name], rtol=0, atol=0)
            torch.testing.assert_close(
                prepared_grad_route, grad_route, rtol=0, atol=0)
            dist_print(
                ' > Prepared gate/up backward matches replay bitwise',
                once_in_node=True)

            # Regression: prepared gate/up and dGate/dUp may share storage.
            # Kernel A must not publish conventional gradient columns until
            # every N-tile CTA has retired its interleaved preactivation reads.
            aliased_outputs = allocate_outputs()
            aliased_gate_up = saved_l1_preact.clone()
            aliased_outputs['gate_up_input'] = aliased_gate_up
            aliased_outputs['grad_gate_up'] = aliased_gate_up
            aliased_grad_route = torch.zeros_like(grad_route)
            run_backward(
                aliased_outputs,
                route_weight_mode=deep_gemm.RouteWeightMode.POST_DOWN,
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted,
                grad_route_output=aliased_grad_route,
                gate_up_prepared=True)
            observable_pool_rows = int(
                (((expert_counts + block_m - 1) // block_m) * block_m)
                .sum().item()
            )
            for name in (
                'grad_h', 'grad_gate_up', 'h_act', 'h_weighted',
                'x_pool', 'grad_x_pool', 'grad_ye',
                'w2_scratch', 'w13_scratch',
            ):
                actual = aliased_outputs[name]
                expected = prepared_outputs[name]
                if name == 'grad_gate_up':
                    # W13 consumes only expert-block-padded rows. The shared
                    # rank-uniform capacity tail remains dead forward scratch.
                    actual = actual[:observable_pool_rows]
                    expected = expected[:observable_pool_rows]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                aliased_grad_route, prepared_grad_route,
                rtol=0, atol=0)
            dist_print(
                ' > In-place prepared gate/up gradient reuse matches '
                'separate pools bitwise',
                once_in_node=True)

            # POST_DOWN never reads grad_h after the activation epilogue: its
            # route dot uses grad-y and saved down. Verify that SiTU output and
            # W2-wgrad input can overwrite the retired dgrad pool exactly.
            h_aliased_outputs = allocate_outputs()
            h_aliased_outputs['h_act'] = h_aliased_outputs['grad_h']
            h_aliased_outputs['h_weighted'] = (
                h_aliased_outputs['grad_h']
            )
            h_aliased_grad_route = torch.zeros_like(grad_route)
            run_backward(
                h_aliased_outputs,
                route_weight_mode=deep_gemm.RouteWeightMode.POST_DOWN,
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted,
                grad_route_output=h_aliased_grad_route,
                gate_up_prepared=True)
            for name in (
                'grad_gate_up', 'h_act', 'h_weighted', 'x_pool',
                'grad_x_pool', 'grad_ye', 'w2_scratch', 'w13_scratch',
            ):
                torch.testing.assert_close(
                    h_aliased_outputs[name], prepared_outputs[name],
                    rtol=0, atol=0)
            torch.testing.assert_close(
                h_aliased_grad_route, prepared_grad_route,
                rtol=0, atol=0)
            dist_print(
                ' > In-place POST_DOWN grad-h/SiTU reuse matches separate '
                'pools bitwise',
                once_in_node=True)
            assert torch.equal(
                route_weights[active_rows], exact_route_weights), (
                    'MXFP4 backward must preserve FP32 route scores exactly')
            expected_grad_down = (
                grad_y_unweighted[active_rows].float() *
                route_weights[active_rows].float().unsqueeze(1)
            ).to(torch.bfloat16)
            assert torch.equal(
                outputs['grad_ye'][active_rows],
                expected_grad_down), (
                    'POST_DOWN grad_down must be BF16(score * grad_y)')
            assert torch.equal(
                outputs['h_weighted'][active_rows],
                outputs['h_act'][active_rows]), (
                    'POST_DOWN W2 wgrad must consume unweighted h')
            expected_grad_route = (
                grad_y_unweighted[active_rows].float() *
                saved_down_unweighted[active_rows].float()
            ).sum(dim=1)
            torch.testing.assert_close(
                grad_route[active_rows],
                expected_grad_route,
                rtol=2e-6, atol=2e-4)
            assert_direct_route_plane(expected_grad_route)
            combine_outputs = outputs
        else:
            # The explicit mode must instantiate the same PRE_DOWN kernel and
            # preserve all historical output bits.
            initial_grad_ye = torch.randn(
                (pool_rows, hidden),
                dtype=torch.bfloat16, device='cuda')
            default_outputs = allocate_outputs(initial_grad_ye)
            explicit_outputs = allocate_outputs(initial_grad_ye)
            run_backward(default_outputs)
            run_backward(
                explicit_outputs,
                route_weight_mode=(
                    deep_gemm.RouteWeightMode.PRE_DOWN))
            for name in (
                'grad_h', 'grad_gate_up', 'h_act',
                'h_weighted', 'x_pool', 'grad_x_pool',
                'grad_ye',
            ):
                assert torch.equal(
                    default_outputs[name],
                    explicit_outputs[name]), (
                        f'explicit PRE_DOWN changed {name}')
            expected_grad_route = _native_route_dot(
                explicit_outputs['grad_h'][active_rows],
                explicit_outputs['h_act'][active_rows])
            assert_direct_route_plane(expected_grad_route)
            combine_outputs = explicit_outputs

        # K3 training keeps W2/W13 wgrad in Kernel A so its UMMA work can
        # overlap the cross-rank receive-plane reduction.  Compare that
        # persistent phase against the standalone wgrad kernels before using
        # it in FireTitan's full-parameter path.
        padded_expert_counts = (
            (expert_counts + block_m - 1) // block_m * block_m
        )
        expected_w2 = torch.zeros_like(
            combine_outputs['w2_scratch'])
        expected_w13 = torch.zeros_like(
            combine_outputs['w13_scratch'])
        expected_combined = None
        if num_ranks > 1:
            expected_combined = torch.empty(
                (num_tokens, hidden),
                dtype=torch.bfloat16,
                device='cuda')
            deep_gemm.bf16_mega_moe_backward_w2_combine(
                expected_w2,
                combine_outputs['grad_ye'],
                combine_outputs['h_weighted'],
                padded_expert_counts,
                block_m,
                expected_combined,
                buffer,
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode))
            deep_gemm.bf16_mega_moe_backward_w13_combine(
                expected_w13,
                combine_outputs['grad_gate_up'],
                combine_outputs['x_pool'],
                padded_expert_counts,
                block_m,
                expected_combined,
                buffer,
                combine_order_mode=deep_gemm.CombineOrderMode(
                    args.combine_order_mode))
        else:
            deep_gemm.bf16_mega_moe_backward_w2(
                expected_w2,
                combine_outputs['grad_ye'],
                combine_outputs['h_weighted'],
                padded_expert_counts,
                block_m,
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode))
            deep_gemm.bf16_mega_moe_backward_w13(
                expected_w13,
                combine_outputs['grad_gate_up'],
                combine_outputs['x_pool'],
                padded_expert_counts,
                block_m)

        inline_outputs = allocate_outputs()
        inline_combined = (
            torch.empty_like(expected_combined)
            if expected_combined is not None else None
        )
        inline_kwargs = {
            'route_weight_mode': deep_gemm.RouteWeightMode(
                args.route_weight_mode),
            'inline_wgrad': True,
            'combined_grad_x_output': inline_combined,
            'gate_up_prepared': True,
        }
        if args.route_weight_mode == 'post_down':
            inline_kwargs.update(
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted,
                grad_route_output=torch.zeros(
                    pool_rows, dtype=torch.float32, device='cuda'))
        run_backward(inline_outputs, **inline_kwargs)
        for name in (
            'grad_h', 'grad_gate_up', 'h_act', 'h_weighted',
            'x_pool', 'grad_x_pool', 'grad_ye',
        ):
            torch.testing.assert_close(
                inline_outputs[name], combine_outputs[name], rtol=0, atol=0)
        for name, actual, expected in (
            ('inline_w2', inline_outputs['w2_scratch'], expected_w2),
            ('inline_w13', inline_outputs['w13_scratch'], expected_w13),
        ):
            cosine = torch.nn.functional.cosine_similarity(
                actual.reshape(-1).double(),
                expected.reshape(-1).double(),
                dim=0)
            assert float(cosine) > 0.99999, (
                f'{name} cosine {float(cosine):.9f} did not clear 0.99999')
        if expected_combined is not None:
            assert inline_combined is not None
            assert torch.equal(inline_combined, expected_combined)
        dist_print(
            ' > Inline UMMA/TMA wgrad matches standalone wgrad',
            once_in_node=True)

        if num_ranks > 1 and args.route_weight_mode == 'post_down':
            # K3 retains the exact BF16 source rows for native W13 wgrad.
            # Validate the memory-reuse path against the same exact-source
            # specialization with separate grad-y and x-pool allocations.
            exact_outputs = allocate_outputs()
            exact_combined = torch.empty_like(expected_combined)
            exact_grad_route = torch.zeros(
                pool_rows, dtype=torch.float32, device='cuda')
            exact_kwargs = {
                'route_weight_mode': deep_gemm.RouteWeightMode.POST_DOWN,
                'grad_y_unweighted_output': grad_y_unweighted,
                'down_unweighted_output': saved_down_unweighted,
                'grad_route_output': exact_grad_route,
                'inline_wgrad': True,
                'combined_grad_x_output': exact_combined,
                'gate_up_prepared': True,
                'source_x_bf16': source_x_bf16,
            }
            run_backward(exact_outputs, **exact_kwargs)
            all_source_x_bf16 = gather_rank_padded(
                source_x_bf16, 0)
            expected_exact_x = all_source_x_bf16[
                metadata[:, 0], metadata[:, 1]]
            torch.testing.assert_close(
                exact_outputs['x_pool'][active_rows],
                expected_exact_x,
                rtol=0,
                atol=0,
            )

            aliased_outputs = allocate_outputs()
            aliased_outputs['x_pool'] = torch.zeros_like(
                grad_y_unweighted)
            aliased_combined = torch.empty_like(expected_combined)
            aliased_grad_route = torch.zeros_like(exact_grad_route)
            aliased_kwargs = dict(exact_kwargs)
            aliased_kwargs.update(
                grad_y_unweighted_output=aliased_outputs['x_pool'],
                grad_route_output=aliased_grad_route,
                combined_grad_x_output=aliased_combined,
            )
            run_backward(aliased_outputs, **aliased_kwargs)

            for name in (
                'grad_h', 'grad_gate_up', 'h_act', 'h_weighted',
                'x_pool', 'grad_x_pool', 'grad_ye',
                'w2_scratch', 'w13_scratch',
            ):
                torch.testing.assert_close(
                    aliased_outputs[name], exact_outputs[name],
                    rtol=0, atol=0)
            torch.testing.assert_close(
                aliased_grad_route, exact_grad_route, rtol=0, atol=0)
            torch.testing.assert_close(
                aliased_combined, exact_combined, rtol=0, atol=0)
            dist_print(
                ' > Late exact-X TMA pool reuse matches separate pools bitwise',
                once_in_node=True)

        if args.benchmark_inline_wgrad:
            baseline_kwargs = {
                'route_weight_mode': deep_gemm.RouteWeightMode(
                    args.route_weight_mode),
                'gate_up_prepared': True,
            }
            if args.route_weight_mode == 'post_down':
                baseline_kwargs.update(
                    grad_y_unweighted_output=grad_y_unweighted,
                    down_unweighted_output=saved_down_unweighted,
                    grad_route_output=torch.zeros(
                        pool_rows, dtype=torch.float32, device='cuda'))

            def run_standalone_backward():
                run_backward(combine_outputs, **baseline_kwargs)
                if num_ranks > 1:
                    deep_gemm.bf16_mega_moe_backward_w2_combine(
                        expected_w2,
                        combine_outputs['grad_ye'],
                        combine_outputs['h_weighted'],
                        padded_expert_counts,
                        block_m,
                        expected_combined,
                        buffer,
                        route_weight_mode=deep_gemm.RouteWeightMode(
                            args.route_weight_mode))
                    deep_gemm.bf16_mega_moe_backward_w13_combine(
                        expected_w13,
                        combine_outputs['grad_gate_up'],
                        combine_outputs['x_pool'],
                        padded_expert_counts,
                        block_m,
                        expected_combined,
                        buffer,
                        combine_order_mode=deep_gemm.CombineOrderMode(
                            args.combine_order_mode))
                else:
                    deep_gemm.bf16_mega_moe_backward_w2(
                        expected_w2,
                        combine_outputs['grad_ye'],
                        combine_outputs['h_weighted'],
                        padded_expert_counts,
                        block_m,
                        route_weight_mode=deep_gemm.RouteWeightMode(
                            args.route_weight_mode))
                    deep_gemm.bf16_mega_moe_backward_w13(
                        expected_w13,
                        combine_outputs['grad_gate_up'],
                        combine_outputs['x_pool'],
                        padded_expert_counts,
                        block_m)

            def measure_backward(fn):
                for _ in range(args.backward_warmup):
                    fn()
                torch.cuda.synchronize()
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.backward_iterations):
                    fn()
                stop.record()
                stop.synchronize()
                elapsed_ms = start.elapsed_time(stop) / args.backward_iterations
                rank_max_ms = torch.tensor(
                    elapsed_ms, dtype=torch.float64, device='cuda')
                dist.all_reduce(rank_max_ms, op=dist.ReduceOp.MAX, group=group)
                return float(rank_max_ms)

            standalone_ms = measure_backward(run_standalone_backward)
            inline_ms = measure_backward(
                lambda: run_backward(inline_outputs, **inline_kwargs))
            dist_print(
                f' > BACKWARD standalone dgrad+wgrad: {standalone_ms:.3f} ms',
                once_in_node=True)
            dist_print(
                f' > BACKWARD inline dgrad+wgrad: {inline_ms:.3f} ms',
                once_in_node=True)
            dist_print(
                f' > BACKWARD inline speedup: {standalone_ms / inline_ms:.3f}x',
                once_in_node=True)
            return

        # Shared BF16/MXFP4 combine-only coverage, independent of wgrad.
        direct_planes = torch.as_strided(
            buffer.backward_grad_y,
            size=(
                num_topk,
                buffer.num_max_tokens_per_rank,
                hidden,
            ),
            stride=(
                buffer.num_max_tokens_per_rank * hidden,
                hidden,
                1,
            ),
        )
        expected_grad_x_planes = torch.zeros(
            (
                num_ranks,
                buffer.num_max_tokens_per_rank,
                num_topk,
                hidden,
            ),
            dtype=torch.bfloat16,
            device='cuda',
        )
        expected_grad_x_planes[
            metadata[:, 0], metadata[:, 1], metadata[:, 2]
        ] = combine_outputs['grad_x_pool'][active_rows]
        if num_ranks > 1:
            dist.all_reduce(expected_grad_x_planes, group=group)
        else:
            direct_planes.copy_(
                expected_grad_x_planes[0].permute(1, 0, 2))
        assert torch.equal(
            direct_planes[:, :num_tokens].permute(1, 0, 2),
            expected_grad_x_planes[rank_idx, :num_tokens])
        combined_grad_x = torch.empty(
            (num_tokens, hidden),
            dtype=torch.bfloat16, device='cuda')
        deep_gemm.mega_moe_backward_combine_grad_x(
            combined_grad_x,
            buffer,
            combine_order_mode=deep_gemm.CombineOrderMode(
                args.combine_order_mode))
        expected_combined_grad_x = combine_source_planes(
            expected_grad_x_planes[rank_idx, :num_tokens])
        assert torch.equal(
            combined_grad_x, expected_combined_grad_x)
        # Route-gradient production is a W2-side operation and must not force
        # W13 dgrad or a grad-x destination.
        route_only_outputs = allocate_outputs(
            combine_outputs['grad_ye'])
        route_only_grad = torch.zeros(
            pool_rows, dtype=torch.float32, device='cuda')
        route_only_kwargs = {
            'route_weight_mode': deep_gemm.RouteWeightMode(
                args.route_weight_mode),
            'grad_route_output': route_only_grad,
        }
        if args.route_weight_mode == 'post_down':
            route_only_kwargs.update(
                grad_y_unweighted_output=grad_y_unweighted,
                down_unweighted_output=saved_down_unweighted)
        run_backward(
            route_only_outputs,
            compute_w13_dgrad=False,
            direct_remote_grad_x=False,
            write_grad_x_pool=False,
            **route_only_kwargs)
        torch.testing.assert_close(
            route_only_grad[active_rows],
            expected_grad_route,
            rtol=2e-6,
            atol=2e-4,
        )

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    try:
        ep_buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
            num_topk=num_topk, use_fp8_dispatch=True,
            explicitly_destroy=True,
            allow_multiple_reduction=False,
            num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
        ) if is_legacy_loaded else None
    except Exception as ex:
        dist_print(f'Failed to create legacy EP buffer: {ex}, skip baseline', once_in_node=True)
        ep_buffer, is_legacy_loaded = None, False

    # Baseline params differ by mma type
    run_baseline = None
    if is_legacy_loaded:
        if is_bf16xbf16:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
            gemm_kwargs = {'compiled_dims': '', 'use_psum_layout': True}
            swiglu_kwargs = {'round_scale': False, 'ue8m0_scale': False, 'output_bf16': True}
            get_num_tokens = lambda recv_x: recv_x.size(0)
        else:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False,
                               'do_expand': True, 'use_tma_aligned_col_major_sf': True}
            gemm_fn = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
            gemm_kwargs = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
            swiglu_kwargs = {'round_scale': True, 'ue8m0_scale': True, 'output_bf16': False}
            get_num_tokens = lambda recv_x: recv_x[0].size(0)

        def run_baseline():
            # Dispatch
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                num_experts=num_experts, expert_alignment=alignment,
                **dispatch_kwargs)
            num_recv_tokens = get_num_tokens(recv_x)

            # L1 GEMM
            l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
            gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # SwiGLU
            swiglu_result = tilelang_ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights,
                avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True,
                clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                **swiglu_kwargs)
            l1_y = swiglu_result[-1] if is_bf16xbf16 else swiglu_result

            # L2 GEMM
            l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Combine
            return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # The legacy TileLang baseline is bitwise-identical only for the original
    # SwiGLU/PRE_DOWN contract. POST_DOWN deliberately moves the route multiply
    # across W2 quantization and is validated by the self-contained reference.
    use_legacy_baseline = (
        is_legacy_loaded and
        args.activation == 'swiglu' and
        args.route_weight_mode == 'pre_down'
    )
    # The self-contained reference models the native BF16 boundaries and the
    # quantized FP8/FP4 boundaries, but only local routing/combine.
    use_numerical_reference = is_bf16xbf16 or num_ranks == 1
    ran_correctness = False

    if args.benchmark_backward:
        if not is_bf16xbf16:
            raise ValueError(
                '--benchmark-backward requires bf16xbf16')
        create_inputs()
        run_fused()
        run_bf16_backward_test()
        ran_correctness = True

    # noinspection PyBroadException
    if (not args.benchmark_backward and
            use_legacy_baseline and num_correctness_tests > 0):
        dist_print('Running correctness tests (bitwise vs legacy baseline):', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_results = run_fused()
            for fused_result, baseline_result in zip(
                fused_results, run_baseline()
            ):
                assert torch.equal(fused_result, baseline_result)
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
        ran_correctness = True

    if (not args.benchmark_backward and
            use_numerical_reference and num_correctness_tests > 0):
        # FP8/FP4 quantization across two GEMMs bounds the achievable similarity
        max_diff = 0.05
        dist_print(f'Running correctness tests ({args.activation}, numerical reference):', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            stats_before = (
                cumulative_local_expert_recv_stats_fused.clone())
            fused_y, _ = run_fused()
            if (
                not is_bf16xbf16 and
                args.route_weight_mode == 'pre_down'
            ):
                check_fp8_fp4_predown_regression(
                    fused_y, stats_before)
            elif (
                not is_bf16xbf16 and
                args.route_weight_mode == 'post_down'
            ):
                check_fp8_fp4_postdown_route_semantics(
                    fused_y, stats_before)
            ref_y = run_reference()
            if is_bf16xbf16:
                check_native_forward_repeatability(ref_y)
                check_bf16_parity(fused_y, ref_y)
                if args.test_backward:
                    run_bf16_backward_test()
                diff = calc_diff(fused_y, ref_y)
            else:
                diff = calc_diff(fused_y, ref_y)
                assert diff < max_diff, f'{args.activation} diff too large: {diff:.5f} >= {max_diff}'
                if args.test_backward:
                    run_fp8_fp4_route_backward_test()
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed (diff {diff:.5f})', once_in_node=True)
        dist_print(once_in_node=True)
        ran_correctness = True

    if (
        not args.benchmark_backward and
        not is_bf16xbf16 and
        num_ranks > 1 and
        args.route_weight_mode == 'post_down' and
        num_correctness_tests > 0
    ):
        dist_print(
            'Running distributed POST_DOWN route-boundary tests:',
            once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            stats_before = (
                cumulative_local_expert_recv_stats_fused.clone())
            fused_y, _ = run_fused()
            check_fp8_fp4_postdown_route_semantics(
                fused_y, stats_before)
            if (
                (i + 1) % 100 == 0 or
                i == num_correctness_tests - 1
            ):
                dist_print(
                    f' > Distributed route test '
                    f'#{i + 1}/{num_correctness_tests} passed',
                    once_in_node=True)
        dist_print(once_in_node=True)
        ran_correctness = True

    if (
        args.benchmark_inline_wgrad and
        not is_bf16xbf16 and
        num_ranks > 1
    ):
        create_inputs()
        run_fused()
        run_fp8_fp4_route_backward_test()
        ran_correctness = True

    if (
        not is_bf16xbf16 and
        args.route_weight_mode == 'pre_down' and
        num_correctness_tests > 0
    ):
        create_inputs()
        cumulative_local_expert_recv_stats_fused.zero_()
        full_inference_y, _ = run_fused()
        cumulative_local_expert_recv_stats_fused.zero_()
        legacy_buffer_y, _ = run_fused(legacy_forward_buffer)
        assert torch.equal(full_inference_y, legacy_buffer_y)
        cumulative_local_expert_recv_stats_fused.zero_()
        legacy_low_level_y = run_legacy_low_level_inference()
        assert torch.equal(full_inference_y, legacy_low_level_y)
        ran_correctness = True

    if not ran_correctness:
        create_inputs()

    if args.correctness_only:
        dist.barrier()
        buffer.destroy()
        ep_buffer.destroy() if ep_buffer else None
        dist.destroy_process_group()
        return

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    act_elem_size, weight_elem_size = (2, 2) if is_bf16xbf16 else (1, 0.5)
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * weight_elem_size   # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * weight_elem_size     # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                  # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                     # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                     # L2 acts read
        + num_recv_tokens * hidden * 2                                              # L2 output write (always BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument(
        '--num-ring-tokens',
        type=int,
        default=None,
        help='Override ring capacity to exercise multi-wave ring reuse')
    parser.add_argument(
        '--require-ring-wrap',
        action='store_true',
        help='Fail unless the active logical pool exceeds the ring capacity')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--zero-tokens', action='store_true', help='Launch a rank with exactly zero tokens')
    parser.add_argument('--zero-rank', type=int, default=-1, help='Rank that launches with zero tokens')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation', type=str, default='swiglu', choices=['swiglu', 'geglu'], help='Gated activation type')
    parser.add_argument('--route-weight-mode', type=str, default='pre_down', choices=['pre_down', 'post_down'], help='Location of the BF16 route-weight boundary')
    parser.add_argument('--combine-order-mode', type=str, default='fixed_topk', choices=['fixed_topk', 'deepep', 'deepep_v1'], help='Top-k combine reduction and BF16 materialization order')
    parser.add_argument(
        '--backward-memory-mode',
        choices=['legacy', 'phase_ordered'],
        default='legacy',
        help='Use validated destructive BF16 backward pool aliases')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument(
        '--expect-block-m',
        type=int,
        default=0,
        choices=[0, 16, 32, 64, 96, 128, 192],
        help='Assert the rank-uniform forward/backward BLOCK_M selection')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--routing', choices=['random', 'balanced', 'skew', 'extreme'], default='random')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4', help='MMA type: fp8xfp4 or bf16xbf16')
    parser.add_argument('--save-l1-preact', action='store_true', help='Validate optional exact BF16 W13 output')
    parser.add_argument(
        '--active-saved-pool',
        action='store_true',
        help='Size BF16 saved pools from an exact route-count exchange')
    parser.add_argument(
        '--no-save-forward-stages',
        dest='save_forward_stages',
        action='store_false',
        help='Match production training: save preactivation only (and post-down output when needed)')
    parser.set_defaults(save_forward_stages=True)
    parser.add_argument('--test-backward', action='store_true', help='Validate BF16 dgrad and grouped wgrad')
    parser.add_argument(
        '--python-numerical-correction',
        action='store_true',
        help='Debug only: replace raw BF16 CUDA backward outputs with native operations')
    parser.add_argument(
        '--no-write-grad-x-pool',
        dest='write_grad_x_pool',
        action='store_false',
        help='Use direct source planes without allocating the local grad-x pool')
    parser.set_defaults(write_grad_x_pool=True)
    parser.add_argument(
        '--benchmark-backward',
        action='store_true',
        help='Benchmark raw BF16 dgrad and skip backward reference construction')
    parser.add_argument(
        '--benchmark-inline-wgrad',
        action='store_true',
        help='Compare complete standalone and inline FP8/FP4 backward waves')
    parser.add_argument(
        '--backward-warmup', type=int, default=3)
    parser.add_argument(
        '--backward-iterations', type=int, default=10)

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--correctness-only', action='store_true', help='Exit after correctness validation')
    parser.add_argument(
        '--check-predown-golden',
        action='store_true',
        help='Check canonical MXFP4 PRE_DOWN output against legacy digests')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
