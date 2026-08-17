"""Native MegaMoE side-LoRA correctness and EP benchmark driver.

This intentionally exercises the dedicated kernels. PyTorch matmuls are used
only to construct a numerical reference after the native call has completed.
"""

import argparse
import inspect
import json
import math

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import (
    cast_back_from_fp4, per_token_cast_to_fp4, per_token_cast_to_fp8,
    unpack_ue8m0_from_int)
from deep_gemm.utils.dist import init_dist


def _block_m(tokens: int, ranks: int, topk: int, experts: int) -> int:
    expected = tokens * ranks * topk / experts
    if expected <= 8.5:
        return 16
    if expected <= 16.5:
        return 32
    if expected <= 32.5:
        return 64
    if expected <= 64.5:
        return 96
    if expected <= 96.5:
        return 128
    return 192


def _active_rows(counts: torch.Tensor, padded: torch.Tensor) -> torch.Tensor:
    rows = []
    offset = 0
    for count, capacity in zip(counts.cpu().tolist(), padded.cpu().tolist()):
        rows.extend(range(offset, offset + count))
        offset += capacity
    return torch.tensor(rows, dtype=torch.long, device=counts.device)


def _make_routing(
    tokens: int,
    topk: int,
    experts: int,
    rank: int,
    ranks: int,
    scenario: str,
    masked_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic, duplicate-free routes for edge-case coverage."""
    token_ids = torch.arange(tokens, device="cuda").unsqueeze(1)
    slots = torch.arange(topk, device="cuda").unsqueeze(0)
    global_token_ids = rank * tokens + token_ids
    if scenario == "balanced":
        topk_idx = (global_token_ids * topk + slots) % experts
    elif scenario == "skewed":
        # Keep most traffic on the first experts while retaining distinct
        # top-k destinations for every token.
        hot_experts = max(topk, min(experts, 2 * topk))
        topk_idx = (global_token_ids + slots) % hot_experts
    elif scenario == "empty_experts":
        # Only the first top-k experts receive traffic. This exercises zero
        # count experts and padding without violating the no-duplicate rule.
        topk_idx = slots.expand(tokens, -1).clone()
    elif scenario == "remote":
        # Rotate the balanced assignment so EP ranks send to remote owners.
        topk_idx = (
            global_token_ids * topk + slots + experts // ranks) % experts
    else:
        raise ValueError(f"unsupported routing scenario: {scenario}")

    route_ids = global_token_ids * topk + slots
    topk_weights = (0.125 + (route_ids % 7).float() / 10).contiguous()
    if masked_ratio:
        period = max(2, round(1.0 / masked_ratio))
        mask = route_ids.remainder(period) == 0
        topk_idx.masked_fill_(mask, -1)
        topk_weights.masked_fill_(mask, 0.0)
    return topk_idx.contiguous(), topk_weights


def _counts(
    topk_idx: torch.Tensor,
    experts: int,
    local_start: int,
    local_experts: int,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = topk_idx >= 0
    source_counts = torch.bincount(
        topk_idx[valid], minlength=experts).to(torch.int32)
    global_counts = source_counts.clone()
    dist.all_reduce(global_counts, group=group)
    return (
        source_counts,
        global_counts[local_start:local_start + local_experts].contiguous(),
    )


def _all_gather_equal(tensor: torch.Tensor, group) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.stack(gathered)


def _rank_uniform_max(value: int, group) -> int:
    value_tensor = torch.tensor(value, dtype=torch.int32, device="cuda")
    dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX, group=group)
    return int(value_tensor.item())


def _scatter_to_sources(
    rows: torch.Tensor,
    metadata: torch.Tensor,
    ranks: int,
    tokens: int,
    group,
) -> torch.Tensor:
    planes = torch.zeros(
        (ranks * tokens, *rows.shape[1:]),
        dtype=rows.dtype, device=rows.device)
    source_rows = metadata[:, 0] * tokens + metadata[:, 1]
    planes.index_add_(0, source_rows, rows)
    dist.all_reduce(planes, group=group)
    return planes.view(ranks, tokens, *rows.shape[1:])


def _scatter_route_grads(
    rows: torch.Tensor,
    metadata: torch.Tensor,
    ranks: int,
    tokens: int,
    topk: int,
    group,
) -> torch.Tensor:
    flat = torch.zeros(
        ranks * tokens * topk, dtype=rows.dtype, device=rows.device)
    indices = (
        (metadata[:, 0] * tokens + metadata[:, 1]) * topk +
        metadata[:, 2])
    flat.index_add_(0, indices, rows)
    dist.all_reduce(flat, group=group)
    return flat.view(ranks, tokens, topk)


def _apply_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
    activation_limit: float,
) -> torch.Tensor:
    gate_clamped = torch.clamp(gate, max=activation_limit)
    up_clamped = torch.clamp(
        up, min=-activation_limit, max=activation_limit)
    if activation == "swiglu":
        activated = torch.nn.functional.silu(gate_clamped)
    elif activation == "geglu":
        alpha = 1.5957691216057308
        beta = 0.044715
        gate_sq = gate_clamped * gate_clamped
        activated = gate_clamped * torch.sigmoid(
            (alpha * gate_clamped) * (1.0 + beta * gate_sq))
    else:
        raise ValueError(f"unsupported activation: {activation}")
    if activation == "swiglu" and math.isinf(activation_limit):
        return activated * up_clamped
    return (activated.float() * up_clamped.float()).to(torch.bfloat16)


def _cast_fp4(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, n, k = weights.shape
    packed = torch.empty((groups, n, k // 2), dtype=torch.int8, device="cuda")
    scales = torch.empty((groups, n, k // 32), dtype=torch.float32, device="cuda")
    for group in range(groups):
        packed[group], scales[group] = per_token_cast_to_fp4(
            weights[group], use_ue8m0=True, gran_k=32)
    scales = deep_gemm.transform_sf_into_required_layout(
        scales, n, k, (1, 32), groups)
    return packed, scales


def _cast_fp4_backward(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.empty(
        (*weights.shape[:-1], weights.size(-1) // 2),
        dtype=torch.int8, device="cuda")
    scales = torch.empty(
        (*weights.shape[:-1], weights.size(-1) // 32),
        dtype=torch.float32, device="cuda")
    for expert in range(weights.size(0)):
        packed[expert], scales[expert] = per_token_cast_to_fp4(
            weights[expert], use_ue8m0=True, gran_k=32)
    return packed.view(torch.float8_e4m3fn), scales


def _adapters(experts: int, hidden: int, intermediate: int):
    rank = 128
    return (
        (torch.randn(hidden, rank, device="cuda", dtype=torch.bfloat16) * 0.02),
        (torch.randn(experts, rank, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02),
        (torch.randn(hidden, rank, device="cuda", dtype=torch.bfloat16) * 0.02),
        (torch.randn(experts, rank, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02),
        (torch.randn(experts, intermediate, rank, device="cuda", dtype=torch.bfloat16) * 0.02),
        (torch.randn(rank, hidden, device="cuda", dtype=torch.bfloat16) * 0.02),
    )


def _dequant_fp4(weights: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(weights, dtype=torch.float32)
    for expert in range(weights.size(0)):
        packed, scales = per_token_cast_to_fp4(
            weights[expert], use_ue8m0=True, gran_k=32)
        output[expert] = cast_back_from_fp4(packed, scales, gran_k=32)
    return output


def _dequant_fp8(x: torch.Tensor, packed_scales: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    scales = unpack_ue8m0_from_int(packed_scales)[:, :width // 32]
    return (
        x.float().view(rows, width // 32, 32) * scales.unsqueeze(2)
    ).reshape(rows, width)


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if expected.numel() == 0:
        return 0.0
    if float(expected.detach().float().norm()) == 0.0:
        return float(actual.detach().float().norm())
    return float(calc_diff(actual.detach().float(), expected.detach().float()))


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.detach().float() - expected.detach().float()
    return float(delta.abs().max()) if delta.numel() else 0.0


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    delta = actual_f - expected_f
    actual_sq = actual_f.square().sum().double()
    expected_sq = expected_f.square().sum().double()
    if float(expected_sq) == 0.0:
        actual_norm = math.sqrt(float(actual_sq))
        return {
            "relative_l2": actual_norm,
            "cosine_similarity": 1.0 if actual_norm == 0.0 else 0.0,
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        }
    return {
        "relative_l2": math.sqrt(
            float(delta.square().sum().double() / expected_sq)
        ),
        "cosine_similarity": float(
            (actual_f * expected_f).sum().double()
            / torch.sqrt(actual_sq * expected_sq)
        ),
        "max_abs": float(delta.abs().max()),
    }


def _mx_interleave_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    width = left.size(1)
    return torch.stack(
        (left.view(-1, width // 8, 8), right.view(-1, width // 8, 8)),
        dim=2).reshape(-1, 2 * width)


def _mx_deinterleave_pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = value.size(1) // 2
    chunks = value.view(-1, width // 8, 2, 8)
    return chunks[:, :, 0].reshape(-1, width), chunks[:, :, 1].reshape(-1, width)


def test_side_lora_backward_rejects_unsupported_post_down() -> None:
    """Do not silently run a numerically invalid backward boundary."""
    calls = (
        (
            deep_gemm.bf16_mega_moe_side_lora_backward,
            dict(
                gate_up_output=None, saved_h=None,
                saved_down_unweighted=None, q13=None, q2=None,
                side_lora=None, w2_weights=None, w13_weights=None,
                expert_counts=None, padded_expert_counts=None,
                grad_y=None, sym_buffer=None, block_m=16,
            ),
        ),
        (
            deep_gemm.fp8_fp4_mega_moe_side_lora_backward,
            dict(
                gate_up_output=None, saved_h=None,
                saved_down_unweighted=None, q13=None, q2=None,
                side_lora=None, l1_acts=None, l1_acts_sf=None,
                l1_weights=None, w13_weights=None, w2_weights=None,
                w13_dequant_scratch=None, w2_dequant_scratch=None,
                expert_counts=None, padded_expert_counts=None,
                grad_y=None, sym_buffer=None, block_m=16,
            ),
        ),
    )
    for function, keywords in calls:
        try:
            function(
                **keywords,
                route_weight_mode=deep_gemm.RouteWeightMode.POST_DOWN)
        except NotImplementedError as error:
            assert "pre_down" in str(error)
        else:
            raise AssertionError(
                f"{function.__name__} accepted unsupported post_down")


def test_mxfp4_side_lora_backward_signature_and_unknown_activation() -> None:
    signature = inspect.signature(
        deep_gemm.fp8_fp4_mega_moe_side_lora_backward)
    assert signature.parameters["activation"].default == "swiglu"
    assert signature.parameters["fast_math"].default is False
    try:
        deep_gemm.fp8_fp4_mega_moe_side_lora_backward(
            gate_up_output=None, saved_h=None,
            saved_down_unweighted=None, q13=None, q2=None,
            side_lora=None, l1_acts=None, l1_acts_sf=None,
            l1_weights=None, w13_weights=None, w2_weights=None,
            w13_dequant_scratch=None, w2_dequant_scratch=None,
            expert_counts=None, padded_expert_counts=None,
            grad_y=None, sym_buffer=None, block_m=16,
            activation="relu",
        )
    except ValueError as error:
        assert "unsupported activation" in str(error)
    else:
        raise AssertionError(
            "MXFP4 side-LoRA backward accepted an unknown activation")


def test_side_lora_transform_validates_shared_layout() -> None:
    hidden, intermediate, experts, rank = 256, 128, 4, 128
    side_lora = (
        torch.empty(hidden, rank, dtype=torch.bfloat16),
        torch.empty(experts, rank, intermediate, dtype=torch.bfloat16),
        torch.empty(hidden, rank, dtype=torch.bfloat16),
        torch.empty(experts, rank, intermediate, dtype=torch.bfloat16),
        torch.empty(experts, intermediate, rank, dtype=torch.bfloat16),
        torch.empty(rank, hidden, dtype=torch.bfloat16),
    )
    transformed = deep_gemm.transform_side_lora_for_mega_moe(side_lora)
    assert [tuple(tensor.shape) for tensor in transformed] == [
        (rank, hidden),
        (experts, intermediate, rank),
        (rank, hidden),
        (experts, intermediate, rank),
        (experts, rank, intermediate),
        (hidden, rank),
    ]

    invalid_b2 = (*side_lora[:-1], side_lora[-1][:, :-1])
    try:
        deep_gemm.transform_side_lora_for_mega_moe(invalid_b2)
    except ValueError as error:
        assert "B2" in str(error)
    else:
        raise AssertionError("accepted an inconsistent shared B2 shape")


def run_bf16_correctness(local_rank: int, world: int, args) -> None:
    rank, ranks, group = init_dist(local_rank, world)
    torch.manual_seed(1234 + rank)
    tokens, hidden, intermediate = args.tokens, args.hidden, args.intermediate
    experts, topk = args.experts, args.topk
    local_experts = experts // ranks
    block_m = _block_m(tokens, ranks, topk, experts)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, experts, tokens, topk, hidden, intermediate,
        mma_type="bf16xbf16", activation=args.activation)

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(local_experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(local_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    transformed_w13, transformed_w2 = deep_gemm.transform_weights_for_mega_moe(w13, w2)
    adapters = _adapters(local_experts, hidden, intermediate)
    side = deep_gemm.transform_side_lora_for_mega_moe(adapters)

    topk_idx, topk_weights = _make_routing(
        tokens, topk, experts, rank, ranks,
        args.routing, args.masked_ratio)
    route_counts, counts = _counts(
        topk_idx, experts, rank * local_experts, local_experts, group)
    padded = ((counts + block_m - 1) // block_m * block_m).to(torch.int32)
    local_pool_rows = int(padded.sum().item())
    pool_rows = _rank_uniform_max(local_pool_rows, group)
    active = _active_rows(counts, padded)

    saved_gate_up = torch.full((pool_rows, 2 * intermediate), float("nan"), device="cuda", dtype=torch.bfloat16)
    saved_h = torch.full((pool_rows, intermediate), float("nan"), device="cuda", dtype=torch.bfloat16)
    saved_h_weighted = torch.full_like(saved_h, float("nan"))
    saved_down = torch.full((pool_rows, hidden), float("nan"), device="cuda", dtype=torch.bfloat16)
    saved_x = torch.full_like(saved_down, float("nan"))
    q13 = torch.empty((pool_rows, 2, 128), device="cuda", dtype=torch.bfloat16)
    q2 = torch.empty((pool_rows, 128), device="cuda", dtype=torch.bfloat16)
    ready = torch.zeros((4, buffer.num_ring_tokens // 8), device="cuda", dtype=torch.int32)
    mismatch = torch.zeros(1, device="cuda", dtype=torch.int32)

    buffer.x[:tokens].copy_(x)
    buffer.topk_idx[:tokens].copy_(topk_idx)
    buffer.topk_weights[:tokens].copy_(topk_weights)
    y = torch.empty_like(x)
    deep_gemm.bf16_mega_moe_side_lora(
        y, transformed_w13, transformed_w2, buffer,
        saved_l1_preact=saved_gate_up,
        saved_h_unweighted=saved_h,
        saved_h_weighted=saved_h_weighted,
        saved_down_unweighted=saved_down,
        saved_x=saved_x,
        precomputed_route_counts=route_counts,
        active_pool_rows=pool_rows,
        route_count_mismatch=mismatch,
        num_config_tokens=tokens,
        side_lora=side,
        side_lora_scale=args.scale,
        side_lora_scratch=(q13, q2, ready),
        activation=args.activation,
        activation_clamp=args.activation_limit,
        route_weight_mode=args.route_weight_mode,
        fast_math=False)
    if args.default_side_lora_scratch:
        assert ready.numel() == 4 * buffer.num_ring_tokens // 8
    torch.cuda.synchronize()
    assert mismatch.item() == 0
    base_preservation = None

    # Build an autograd reference on the exact dispatched rows and preserve
    # the same BF16 materialization boundaries as the native epilogues.
    metadata = buffer.token_src_metadata[active].long()
    x_ref = saved_x[active].detach().clone().requires_grad_(True)
    grad_y = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    all_grad_y = _all_gather_equal(grad_y, group)
    all_route_weights = _all_gather_equal(topk_weights, group)
    route_ref = all_route_weights[
        metadata[:, 0], metadata[:, 1], metadata[:, 2]
    ].to(torch.bfloat16).detach().clone().requires_grad_(True)
    adapter_ref = tuple(t.detach().clone().requires_grad_(True) for t in adapters)
    output_rows, down_rows = [], []
    gate_rows, up_rows, h_rows, h_weighted_rows = [], [], [], []
    q1_rows, q3_rows, q2_rows = [], [], []
    cursor = 0
    for expert, count in enumerate(counts.cpu().tolist()):
        xe = x_ref[cursor:cursor + count]
        a1, b1, a3, b3, a2, b2 = (
            adapter_ref[0], adapter_ref[1][expert],
            adapter_ref[2], adapter_ref[3][expert],
            adapter_ref[4][expert], adapter_ref[5])
        gate_base = (xe @ w13[expert, :intermediate].t()).to(torch.bfloat16)
        up_base = (xe @ w13[expert, intermediate:].t()).to(torch.bfloat16)
        q1 = (xe @ a1).to(torch.bfloat16)
        q3 = (xe @ a3).to(torch.bfloat16)
        gate = torch.add(gate_base, (q1 @ b1).to(torch.bfloat16), alpha=args.scale)
        up = torch.add(up_base, (q3 @ b3).to(torch.bfloat16), alpha=args.scale)
        h = _apply_activation(
            gate, up, args.activation, args.activation_limit)
        routes = route_ref[cursor:cursor + count]
        h_weighted = (
            h.float() * routes.float().unsqueeze(1)).to(torch.bfloat16)
        h_for_w2 = (
            h_weighted
            if args.route_weight_mode == "pre_down"
            else h.to(torch.bfloat16))
        q_down = (h_for_w2 @ a2).to(torch.bfloat16)
        down_base = (h_for_w2 @ w2[expert].t()).to(torch.bfloat16)
        down = torch.add(down_base, (q_down @ b2).to(torch.bfloat16), alpha=args.scale)
        down_rows.append(down)
        output_rows.append(
            down if args.route_weight_mode == "pre_down" else
            (down.float() * routes.float().unsqueeze(1)).to(torch.bfloat16))
        gate_rows.append(gate)
        up_rows.append(up)
        h_rows.append(h)
        h_weighted_rows.append(h_weighted)
        q1_rows.append(q1)
        q3_rows.append(q3)
        q2_rows.append(q_down)
        cursor += count
    route_output = torch.cat(output_rows)
    down_unweighted_ref = torch.cat(down_rows)
    route_grad = all_grad_y[metadata[:, 0], metadata[:, 1]]
    (route_output.float() * route_grad.float()).sum().backward()

    forward_diff = _relative(
        saved_gate_up[active],
        torch.cat((torch.cat(gate_rows), torch.cat(up_rows)), dim=1))
    h_diff = max(
        _relative(saved_h[active], torch.cat(h_rows)),
        _relative(saved_h_weighted[active], torch.cat(h_weighted_rows)))
    down_diff = _relative(saved_down[active], down_unweighted_ref)
    expected_y = _scatter_to_sources(
        route_output, metadata, ranks, tokens, group)[rank]
    output_diff = _relative(y, expected_y)
    q_diff = max(
        _relative(q13[active, 0], torch.cat(q1_rows)),
        _relative(q13[active, 1], torch.cat(q3_rows)),
        _relative(q2[active], torch.cat(q2_rows)),
    )
    result = deep_gemm.bf16_mega_moe_side_lora_backward(
        saved_gate_up,
        saved_h_weighted if args.route_weight_mode == "pre_down" else saved_h,
        saved_down, q13, q2, side,
        w2, transformed_w13, counts, padded, grad_y, buffer,
        block_m, activation_limit=args.activation_limit,
        activation=args.activation, fast_math=False,
        route_weight_mode=args.route_weight_mode,
        side_lora_scale=args.scale, direct_remote_grad_x=ranks > 1)
    torch.cuda.synchronize()
    grad_diffs = [
        _relative(actual, expected.grad)
        for actual, expected in zip(result.grad_side_lora, adapter_ref)
    ]
    grad_accuracy = [
        _accuracy(actual, expected.grad)
        for actual, expected in zip(result.grad_side_lora, adapter_ref)
    ]
    grad_x_diff = _relative(result.grad_x_pool[active], x_ref.grad)
    grad_x_accuracy = _accuracy(result.grad_x_pool[active], x_ref.grad)
    # Validate the three shared native contractions on the kernel's own saved
    # forward/backward boundaries. The separate autograd comparison above is
    # still useful, but at tiny route counts its BF16 reduction order can
    # dominate cosine even when the fused contraction itself is exact.
    native_boundary_shared_grad_refs = [
        (saved_x[active].float().t() @ result.t13[active, 0].float()
         * args.scale).bfloat16(),
        (saved_x[active].float().t() @ result.t13[active, 1].float()
         * args.scale).bfloat16(),
        (q2[active].float().t()
         @ (route_grad.float() *
            (route_ref.detach().float().unsqueeze(1)
             if args.route_weight_mode == "post_down" else 1.0))
         * args.scale).bfloat16(),
    ]
    native_boundary_shared_adapter_accuracy = [
        _accuracy(actual, expected)
        for actual, expected in zip(
            (
                result.grad_side_lora[0],
                result.grad_side_lora[2],
                result.grad_side_lora[5],
            ),
            native_boundary_shared_grad_refs,
            strict=True,
        )
    ]
    source_grad_x = _scatter_to_sources(
        x_ref.grad.to(torch.bfloat16), metadata, ranks, tokens, group)[rank]
    combined_grad_x_diff = (
        _relative(result.grad_x, source_grad_x) if ranks > 1 else 0.0)
    route_grad_diff = _relative(result.grad_route[active], route_ref.grad)
    route_grad_max_abs = _max_abs_diff(
        result.grad_route[active], route_ref.grad)
    source_route_grad = _scatter_route_grads(
        route_ref.grad, metadata, ranks, tokens, topk, group)[rank]
    combined_route_grad_diff = _relative(
        buffer.backward_grad_route[:tokens, :topk], source_route_grad)
    combined_route_grad_max_abs = _max_abs_diff(
        buffer.backward_grad_route[:tokens, :topk], source_route_grad)
    route_grad_close = (
        route_grad_diff < 0.03 or route_grad_max_abs < 0.01)
    combined_route_grad_close = (
        combined_route_grad_diff < 0.03 or
        combined_route_grad_max_abs < 0.01)
    diagnostics = {
        "actual_grad_norms": [
            float(t.float().norm()) for t in result.grad_side_lora],
        "reference_grad_norms": [
            float(t.grad.float().norm()) for t in adapter_ref],
        "t13_norm": float(result.t13[active].float().norm()),
        "t2_norm": float(result.t2[active].float().norm()),
    }
    if rank == 0 and (
        max(grad_diffs) >= 0.03 or grad_x_diff >= 0.03 or
        combined_grad_x_diff >= 0.03 or not route_grad_close or
        not combined_route_grad_close
    ):
        print(json.dumps({
            "adapter_grad_diffs": grad_diffs,
            "grad_x_diff": grad_x_diff,
            "combined_grad_x_diff": combined_grad_x_diff,
            "route_grad_diff": route_grad_diff,
            "route_grad_max_abs": route_grad_max_abs,
            "combined_route_grad_diff": combined_route_grad_diff,
            "combined_route_grad_max_abs": combined_route_grad_max_abs,
            "zero_scale_base_preservation": base_preservation,
            **diagnostics,
        }, indent=2), flush=True)
    forward_tolerance = 0.04 if args.scale == 0.0 else 0.02
    assert forward_diff < forward_tolerance, forward_diff
    assert h_diff < forward_tolerance, h_diff
    assert down_diff < forward_tolerance, down_diff
    if args.scale != 0.0:
        assert output_diff < forward_tolerance, output_diff
    assert q_diff < forward_tolerance, q_diff
    assert max(grad_diffs) < 0.03, grad_diffs
    assert min(
        metric["cosine_similarity"]
        for metric in native_boundary_shared_adapter_accuracy
    ) > 0.9999, native_boundary_shared_adapter_accuracy
    assert grad_x_diff < 0.03, grad_x_diff
    assert combined_grad_x_diff < 0.03, combined_grad_x_diff
    assert route_grad_close, (route_grad_diff, route_grad_max_abs)
    assert combined_route_grad_close, (
        combined_route_grad_diff, combined_route_grad_max_abs)
    if args.scale == 0.0:
        base_y = torch.empty_like(y)
        buffer.x[:tokens].copy_(x)
        buffer.topk_idx[:tokens].copy_(topk_idx)
        buffer.topk_weights[:tokens].copy_(topk_weights)
        deep_gemm.bf16_mega_moe(
            base_y, transformed_w13, transformed_w2, buffer,
            activation=args.activation,
            activation_clamp=args.activation_limit,
            route_weight_mode=args.route_weight_mode,
            num_config_tokens=tokens, fast_math=False)
        torch.cuda.synchronize()
        base_preservation = _accuracy(y, base_y)
        assert base_preservation["relative_l2"] == 0.0, base_preservation
    if rank == 0:
        print(json.dumps({
            "mode": "bf16", "ranks": ranks, "tokens_per_rank": tokens,
            "routing": args.routing,
            "masked_ratio": args.masked_ratio,
            "activation": args.activation,
            "activation_limit": args.activation_limit,
            "route_weight_mode": args.route_weight_mode,
            "pool_rows": pool_rows, "block_m": block_m,
            "forward_diff": forward_diff, "activation_diff": h_diff,
            "down_diff": down_diff, "output_diff": output_diff,
            "side_q_diff": q_diff, "adapter_grad_diffs": grad_diffs,
            "adapter_grad_accuracy": grad_accuracy,
            "native_boundary_shared_adapter_accuracy": (
                native_boundary_shared_adapter_accuracy
            ),
            "grad_x_diff": grad_x_diff,
            "grad_x_accuracy": grad_x_accuracy,
            "combined_grad_x_diff": combined_grad_x_diff,
            "route_grad_diff": route_grad_diff,
            "route_grad_max_abs": route_grad_max_abs,
            "combined_route_grad_diff": combined_route_grad_diff,
            "combined_route_grad_max_abs": combined_route_grad_max_abs,
            "zero_scale_base_preservation": base_preservation,
            **diagnostics,
        }, indent=2), flush=True)
    buffer.destroy()
    dist.destroy_process_group()


def run_mxfp4_correctness(local_rank: int, world: int, args) -> None:
    rank, ranks, group = init_dist(local_rank, world)
    torch.manual_seed(4321 + rank)
    tokens, hidden, intermediate = args.tokens, args.hidden, args.intermediate
    experts, topk = args.experts, args.topk
    local_experts = experts // ranks
    block_m = _block_m(tokens, ranks, topk, experts)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, experts, tokens, topk, hidden, intermediate,
        mma_type="fp8xfp4", activation=args.activation)

    x_bf16 = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    w13_bf16 = torch.randn(local_experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2_bf16 = torch.randn(local_experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02
    w13 = _cast_fp4(w13_bf16)
    w2 = _cast_fp4(w2_bf16)
    transformed_w13, transformed_w2 = deep_gemm.transform_weights_for_mega_moe(w13, w2)
    backward_w13_raw = _cast_fp4_backward(w13_bf16)
    backward_w2 = _cast_fp4_backward(w2_bf16)
    backward_w13 = (
        backward_w13_raw[0].view(
            2 * local_experts, intermediate, hidden // 2),
        backward_w13_raw[1].view(
            2 * local_experts, intermediate, hidden // 32),
    )
    adapters = _adapters(local_experts, hidden, intermediate)
    side = deep_gemm.transform_side_lora_for_mega_moe(adapters)
    topk_idx, topk_weights = _make_routing(
        tokens, topk, experts, rank, ranks,
        args.routing, args.masked_ratio)
    _, counts = _counts(
        topk_idx, experts, rank * local_experts, local_experts, group)
    padded = ((counts + block_m - 1) // block_m * block_m).to(torch.int32)
    local_pool_rows = int(padded.sum().item())
    pool_rows = _rank_uniform_max(local_pool_rows, group)
    active = _active_rows(counts, padded)

    saved_gate_up = torch.full((pool_rows, 2 * intermediate), float("nan"), device="cuda", dtype=torch.bfloat16)
    saved_h = torch.full((pool_rows, intermediate), float("nan"), device="cuda", dtype=torch.bfloat16)
    saved_down = torch.full((pool_rows, hidden), float("nan"), device="cuda", dtype=torch.bfloat16)
    saved_x = torch.full_like(saved_down, float("nan"))
    q13 = torch.empty((pool_rows, 2, 128), device="cuda", dtype=torch.bfloat16)
    q2 = torch.empty((pool_rows, 128), device="cuda", dtype=torch.bfloat16)
    ready = torch.zeros(4 * buffer.num_ring_tokens // 8, device="cuda", dtype=torch.int32)
    buffer.x[:tokens].copy_(x_fp8[0])
    buffer.x_sf[:tokens].copy_(x_fp8[1])
    buffer.topk_idx[:tokens].copy_(topk_idx)
    buffer.topk_weights[:tokens].copy_(topk_weights)
    y = torch.empty_like(x_bf16)
    q13, q2, ready = deep_gemm.fp8_fp4_mega_moe_side_lora(
        y, transformed_w13, transformed_w2, buffer,
        side_lora_input=x_bf16, side_lora=side,
        saved_x=saved_x, saved_h_unweighted=saved_h,
        saved_l1_preact=saved_gate_up,
        saved_down_unweighted=saved_down,
        num_config_tokens=tokens, side_lora_scale=args.scale,
        side_lora_scratch=(
            None if args.default_side_lora_scratch
            else (q13, q2, ready)
        ),
        activation_clamp=args.activation_limit,
        route_weight_mode=args.route_weight_mode,
        fast_math=False)
    torch.cuda.synchronize()
    base_preservation = None

    metadata = buffer.token_src_metadata[active].long()
    x_deq = _dequant_fp8(*x_fp8)
    all_x_deq = _all_gather_equal(x_deq, group)
    all_route_weights = _all_gather_equal(topk_weights, group)
    x_base_ref = all_x_deq[
        metadata[:, 0], metadata[:, 1]
    ].detach().clone().requires_grad_(True)
    x_side_ref = saved_x[active].detach().clone().requires_grad_(True)
    adapter_ref = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in adapters)
    grad_y = torch.randn_like(y)
    all_grad_y = _all_gather_equal(grad_y, group)
    route_ref = all_route_weights[
        metadata[:, 0], metadata[:, 1], metadata[:, 2]
    ].to(torch.bfloat16).detach().clone().requires_grad_(True)
    w13_deq = _dequant_fp4(w13_bf16)
    w2_deq = _dequant_fp4(w2_bf16)
    gate_rows, up_rows = [], []
    h_unweighted_rows, h_for_w2_rows = [], []
    down_rows, down_unweighted_rows = [], []
    q1_rows, q3_rows, q2_rows = [], [], []
    cursor = 0
    for expert, count in enumerate(counts.cpu().tolist()):
        xe_side = x_side_ref[cursor:cursor + count]
        xe_base = x_base_ref[cursor:cursor + count]
        a1, b1, a3, b3, a2, b2 = (
            adapter_ref[0], adapter_ref[1][expert],
            adapter_ref[2], adapter_ref[3][expert],
            adapter_ref[4][expert], adapter_ref[5])
        q1 = (xe_side @ a1).to(torch.bfloat16)
        q3 = (xe_side @ a3).to(torch.bfloat16)
        gate = torch.add(
            (xe_base @ w13_deq[expert, :intermediate].t()).to(torch.bfloat16),
            (q1 @ b1).to(torch.bfloat16), alpha=args.scale)
        up = torch.add(
            (xe_base @ w13_deq[expert, intermediate:].t()).to(torch.bfloat16),
            (q3 @ b3).to(torch.bfloat16), alpha=args.scale)
        h_unweighted = _apply_activation(
            gate, up, args.activation, args.activation_limit)
        route = route_ref[cursor:cursor + count]
        h_weighted = (
            h_unweighted.float() * route.float().unsqueeze(1)
        ).to(torch.bfloat16)
        h = (
            h_weighted
            if args.route_weight_mode == "pre_down"
            else h_unweighted.to(torch.bfloat16))
        q_down = (h @ a2).to(torch.bfloat16)
        down = torch.add(
            (h.float() @ w2_deq[expert].t()).to(torch.bfloat16),
            (q_down @ b2).to(torch.bfloat16), alpha=args.scale)
        gate_rows.append(gate); up_rows.append(up)
        h_unweighted_rows.append(h_unweighted)
        h_for_w2_rows.append(h)
        route_output = (
            down if args.route_weight_mode == "pre_down" else
            (down.float() * route.float().unsqueeze(1)).to(torch.bfloat16))
        down_unweighted_rows.append(down)
        down_rows.append(route_output); q1_rows.append(q1); q3_rows.append(q3)
        q2_rows.append(q_down)
        gate.retain_grad(); up.retain_grad(); h_unweighted.retain_grad(); h.retain_grad()
        q1.retain_grad(); q3.retain_grad(); q_down.retain_grad()
        cursor += count
    gate_ref = _mx_interleave_pair(torch.cat(gate_rows), torch.cat(up_rows))
    down_ref = torch.cat(down_rows)
    expected_y = _scatter_to_sources(
        down_ref, metadata, ranks, tokens, group)[rank]
    route_grad_y = all_grad_y[metadata[:, 0], metadata[:, 1]]
    (down_ref.float() * route_grad_y.float()).sum().backward()
    # The production backward is free to phase-reuse forward-only saved
    # storage. Preserve the native forward boundary for post-run diagnostics.
    saved_gate_up_before_backward = saved_gate_up.clone()
    forward_diffs = {
        "gate_up": _relative(saved_gate_up[active], gate_ref),
        "h": _relative(saved_h[active], torch.cat(h_for_w2_rows)),
        "down": _relative(
            saved_down[active], torch.cat(down_unweighted_rows)),
        "output": _relative(y, expected_y),
        "q": max(
            _relative(q13[active, 0], torch.cat(q1_rows)),
            _relative(q13[active, 1], torch.cat(q3_rows)),
            _relative(q2[active], torch.cat(q2_rows))),
    }
    w13_dequant_scratch = torch.empty(
        (local_experts, 2 * intermediate, hidden),
        device="cuda", dtype=torch.bfloat16)
    w2_dequant_scratch = torch.empty(
        (local_experts, hidden, intermediate),
        device="cuda", dtype=torch.bfloat16)
    if args.check_short_saved_down:
        try:
            deep_gemm.fp8_fp4_mega_moe_side_lora_backward(
                saved_gate_up, saved_h, saved_down[:-1], q13, q2, side,
                buffer.l1_acts[:pool_rows], buffer.l1_acts_sf,
                transformed_w13, backward_w13, backward_w2,
                w13_dequant_scratch, w2_dequant_scratch,
                counts, padded, grad_y, buffer, block_m,
                activation_limit=args.activation_limit,
                activation=args.activation,
                fast_math=False,
                route_weight_mode=args.route_weight_mode,
                side_lora_scale=args.scale,
                direct_remote_grad_x=ranks > 1)
        except ValueError as error:
            assert "full route pool" in str(error), str(error)
        else:
            raise AssertionError(
                "short saved_down_unweighted was not rejected")
    result = deep_gemm.fp8_fp4_mega_moe_side_lora_backward(
        saved_gate_up, saved_h, saved_down, q13, q2, side,
        buffer.l1_acts[:pool_rows], buffer.l1_acts_sf,
        transformed_w13, backward_w13, backward_w2,
        w13_dequant_scratch, w2_dequant_scratch,
        counts, padded, grad_y, buffer, block_m,
        activation_limit=args.activation_limit,
        activation=args.activation,
        fast_math=False,
        route_weight_mode=args.route_weight_mode,
        side_lora_scale=args.scale, direct_remote_grad_x=ranks > 1)
    torch.cuda.synchronize()

    expected_grads = [tensor.grad for tensor in adapter_ref]
    expected_grad_x = (
        x_base_ref.grad.float() + x_side_ref.grad.float()
    ).to(torch.bfloat16)
    grad_diffs = [
        _relative(actual, expected)
        for actual, expected in zip(result.grad_side_lora, expected_grads)]
    grad_x_diff = _relative(result.grad_x_pool[active], expected_grad_x)
    source_grad_x = _scatter_to_sources(
        expected_grad_x, metadata, ranks, tokens, group)[rank]
    combined_grad_x_diff = (
        _relative(result.grad_x, source_grad_x) if ranks > 1 else 0.0)
    route_grad_diff = _relative(result.grad_route[active], route_ref.grad)
    source_route_grad = _scatter_route_grads(
        route_ref.grad, metadata, ranks, tokens, topk, group)[rank]
    combined_route_grad_diff = _relative(
        buffer.backward_grad_route[:tokens, :topk], source_route_grad)
    grad_accuracy = [
        _accuracy(actual, expected)
        for actual, expected in zip(result.grad_side_lora, expected_grads)
    ]
    grad_x_accuracy = _accuracy(result.grad_x_pool[active], expected_grad_x)
    backward_boundary_accuracy = {
        "grad_h": _accuracy(
            result.grad_h[active],
            torch.cat([tensor.grad for tensor in h_for_w2_rows]),
        ),
        "grad_gate_up": _accuracy(
            result.grad_gate_up[active],
            torch.cat(
                (
                    torch.cat([tensor.grad for tensor in gate_rows]),
                    torch.cat([tensor.grad for tensor in up_rows]),
                ),
                dim=1,
            ),
        ),
        "t2": _accuracy(
            result.t2[active].float() * args.scale,
            torch.cat([tensor.grad for tensor in q2_rows]),
        ),
        "t1": _accuracy(
            result.t13[active, 0].float() * args.scale,
            torch.cat([tensor.grad for tensor in q1_rows]),
        ),
        "t3": _accuracy(
            result.t13[active, 1].float() * args.scale,
            torch.cat([tensor.grad for tensor in q3_rows]),
        ),
    }
    # Diagnose the exact fused gated-activation boundary independently of
    # either base dgrad kernel. This reference starts from the native W2+LoRA
    # grad-h and the native forward's saved gate/up values, so discrepancies are
    # owned by the activation-backward phase itself.
    saved_gate, saved_up = _mx_deinterleave_pair(
        saved_gate_up_before_backward[active])
    active_route = route_ref.detach()
    grad_h_for_activation = result.grad_h[active]
    if args.route_weight_mode == "pre_down":
        grad_h_for_activation = (
            grad_h_for_activation.float() *
            active_route.float().unsqueeze(1)).to(torch.bfloat16)
    native_gate = saved_gate.detach().clone().requires_grad_(True)
    native_up = saved_up.detach().clone().requires_grad_(True)
    native_h = _apply_activation(
        native_gate, native_up, args.activation, args.activation_limit)
    (native_h.float() * grad_h_for_activation.float()).sum().backward()
    exact_grad_gate = native_gate.grad
    exact_grad_up = native_up.grad
    backward_boundary_accuracy["grad_gate_up_from_native_boundaries"] = _accuracy(
        result.grad_gate_up[active],
        torch.cat((exact_grad_gate, exact_grad_up), dim=1),
    )
    autograd_grad_gate = torch.cat([tensor.grad for tensor in gate_rows])
    autograd_grad_up = torch.cat([tensor.grad for tensor in up_rows])
    backward_boundary_accuracy["native_boundary_vs_autograd_gate"] = _accuracy(
        exact_grad_gate, autograd_grad_gate)
    backward_boundary_accuracy["native_boundary_vs_autograd_up"] = _accuracy(
        exact_grad_up, autograd_grad_up)
    backward_boundary_accuracy["kernel_vs_native_boundary_gate"] = _accuracy(
        result.grad_gate_up[active, :intermediate], exact_grad_gate)
    backward_boundary_accuracy["kernel_vs_native_boundary_up"] = _accuracy(
        result.grad_gate_up[active, intermediate:], exact_grad_up)
    backward_boundary_accuracy["native_vs_autograd_grad_h_unweighted"] = _accuracy(
        grad_h_for_activation,
        torch.cat([tensor.grad for tensor in h_unweighted_rows]),
    )
    backward_boundary_accuracy["saved_vs_autograd_gate_value"] = _accuracy(
        saved_gate, torch.cat(gate_rows))
    backward_boundary_accuracy["saved_vs_autograd_up_value"] = _accuracy(
        saved_up, torch.cat(up_rows))
    backward_boundary_accuracy["saved_vs_autograd_activation_value"] = _accuracy(
        native_h, _apply_activation(
            torch.cat(gate_rows), torch.cat(up_rows),
            args.activation, args.activation_limit))
    backward_boundary_accuracy["saved_gate_up_after_backward_reuse"] = _accuracy(
        saved_gate_up[active], saved_gate_up_before_backward[active])
    # Validate the six native wgrad contractions on the kernel's own exact
    # forward/backward boundaries. This removes unrelated MXFP4 base-GEMM
    # ordering drift from the adapter-gradient check.
    native_boundary_grad_refs = [
        (x_side_ref.detach().float().t() @ result.t13[active, 0].float()
         * args.scale).bfloat16(),
        [],
        (x_side_ref.detach().float().t() @ result.t13[active, 1].float()
         * args.scale).bfloat16(),
        [],
        [],
        (q2[active].float().t()
         @ (route_grad_y.float() *
            (active_route.float().unsqueeze(1)
             if args.route_weight_mode == "post_down" else 1.0))
         * args.scale).bfloat16(),
    ]
    cursor = 0
    for expert, count in enumerate(counts.cpu().tolist()):
        rows = slice(cursor, cursor + count)
        native_boundary_grad_refs[1].append(
            (q13[active][rows, 0].float().t()
             @ result.grad_gate_up[active][rows, :intermediate].float()
             * args.scale).bfloat16())
        native_boundary_grad_refs[3].append(
            (q13[active][rows, 1].float().t()
             @ result.grad_gate_up[active][rows, intermediate:].float()
             * args.scale).bfloat16())
        native_boundary_grad_refs[4].append(
            (saved_h[active][rows].float().t()
             @ result.t2[active][rows].float()
             * args.scale).bfloat16())
        cursor += count
    native_boundary_grad_refs[1] = torch.stack(native_boundary_grad_refs[1])
    native_boundary_grad_refs[3] = torch.stack(native_boundary_grad_refs[3])
    native_boundary_grad_refs[4] = torch.stack(native_boundary_grad_refs[4])
    native_boundary_adapter_accuracy = [
        _accuracy(actual, expected)
        for actual, expected in zip(
            result.grad_side_lora, native_boundary_grad_refs, strict=True)
    ]
    assert max(forward_diffs.values()) < 0.04, forward_diffs
    assert max(grad_diffs) < 0.04, grad_diffs
    assert min(
        metric["cosine_similarity"] for metric in grad_accuracy
    ) > 0.9999, grad_accuracy
    assert grad_x_diff < 0.04, grad_x_diff
    assert grad_x_accuracy["cosine_similarity"] > 0.9999, grad_x_accuracy
    assert combined_grad_x_diff < 0.04, combined_grad_x_diff
    assert route_grad_diff < 0.04, route_grad_diff
    assert combined_route_grad_diff < 0.04, combined_route_grad_diff
    assert backward_boundary_accuracy[
        "saved_gate_up_after_backward_reuse"]["relative_l2"] == 0.0
    assert backward_boundary_accuracy[
        "grad_gate_up_from_native_boundaries"]["cosine_similarity"] > 0.99999
    assert min(
        metric["cosine_similarity"]
        for metric in native_boundary_adapter_accuracy
    ) > 0.9999, native_boundary_adapter_accuracy
    if args.scale == 0.0:
        base_y = torch.empty_like(y)
        buffer.x[:tokens].copy_(x_fp8[0])
        buffer.x_sf[:tokens].copy_(x_fp8[1])
        buffer.topk_idx[:tokens].copy_(topk_idx)
        buffer.topk_weights[:tokens].copy_(topk_weights)
        deep_gemm.fp8_fp4_mega_moe(
            base_y, transformed_w13, transformed_w2, buffer,
            activation_clamp=args.activation_limit,
            route_weight_mode=args.route_weight_mode,
            num_config_tokens=tokens, fast_math=False)
        torch.cuda.synchronize()
        base_preservation = _accuracy(y, base_y)
        assert base_preservation["relative_l2"] == 0.0, base_preservation
    if rank == 0:
        print(json.dumps({
            "mode": "mxfp4", "ranks": ranks,
            "tokens_per_rank": tokens, "pool_rows": pool_rows,
            "routing": args.routing,
            "masked_ratio": args.masked_ratio,
            "activation_limit": args.activation_limit,
            "route_weight_mode": args.route_weight_mode,
            "block_m": block_m, "forward_diffs": forward_diffs,
            "adapter_grad_diffs": grad_diffs,
            "grad_x_diff": grad_x_diff,
            "combined_grad_x_diff": combined_grad_x_diff,
            "route_grad_diff": route_grad_diff,
            "combined_route_grad_diff": combined_route_grad_diff,
            "zero_scale_base_preservation": base_preservation,
            "adapter_grad_accuracy": grad_accuracy,
            "grad_x_accuracy": grad_x_accuracy,
            "backward_boundary_accuracy": backward_boundary_accuracy,
            "native_boundary_adapter_accuracy": native_boundary_adapter_accuracy,
        }, indent=2), flush=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument(
        "--routing",
        choices=("balanced", "skewed", "empty_experts", "remote"),
        default="balanced")
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument(
        "--activation", choices=("swiglu", "geglu"), default="swiglu")
    parser.add_argument("--activation-limit", type=float, default=float("inf"))
    parser.add_argument(
        "--route-weight-mode", choices=("pre_down", "post_down"),
        default="pre_down")
    parser.add_argument("--mode", choices=("bf16", "mxfp4"), default="bf16")
    parser.add_argument("--default-side-lora-scratch", action="store_true")
    parser.add_argument("--check-short-saved-down", action="store_true")
    args = parser.parse_args()
    if args.experts % args.num_processes:
        parser.error("experts must be divisible by num-processes")
    if not 1 <= args.topk <= args.experts:
        parser.error("topk must be in [1, experts]")
    if not 0.0 <= args.masked_ratio < 1.0:
        parser.error("masked-ratio must be in [0, 1)")
    if args.activation_limit < 0:
        parser.error("activation-limit must be non-negative")
    torch.multiprocessing.spawn(
        run_bf16_correctness if args.mode == "bf16" else run_mxfp4_correctness,
        args=(args.num_processes, args),
        nprocs=args.num_processes)
