import re

import pytest
import torch

import deep_gemm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103-only primitive test",
)


def _reference_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    grouped = x.float().view(rows, columns // 128, 128)
    scales = grouped.abs().amax(dim=-1) / 448.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = (grouped / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).view_as(x)
    return quantized, scales


def _assert_quantized_close(
    actual_quantized: torch.Tensor,
    actual_scales: torch.Tensor,
    expected_quantized: torch.Tensor,
    expected_scales: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual_scales, expected_scales, rtol=2e-6, atol=1e-8)
    if actual_quantized.numel() == 0:
        return
    # CUDA's native FP8 constructor and PyTorch's vectorized cast can choose
    # adjacent E4M3 bins at exact midpoints. Bound that difference explicitly
    # and compare the represented (dequantized) values rather than requiring
    # implementation-specific byte identity.
    mismatch_ratio = (
        actual_quantized.view(torch.uint8) != expected_quantized.view(torch.uint8)
    ).float().mean()
    assert mismatch_ratio <= 0.03
    columns = actual_quantized.size(1)
    actual = (
        actual_quantized.float().view(-1, columns // 128, 128)
        * actual_scales.unsqueeze(-1)
    ).flatten(1)
    expected = (
        expected_quantized.float().view(-1, columns // 128, 128)
        * expected_scales.unsqueeze(-1)
    ).flatten(1)
    torch.testing.assert_close(actual, expected, rtol=0.13, atol=0.02)


def _blockwise_weight_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rows, columns = weight.shape
    blocks = (
        weight.float()
        .view(groups, rows // 128, 128, columns // 128, 128)
        .permute(0, 1, 3, 2, 4)
    )
    scales = blocks.abs().amax(dim=(-1, -2)) / 448.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = (
        (blocks / scales[..., None, None])
        .to(torch.float8_e4m3fn)
        .permute(0, 1, 3, 2, 4)
        .reshape_as(weight)
        .contiguous()
    )
    return quantized, scales.contiguous()


def _blockwise_weight_dequantize(
    quantized: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    groups, rows, columns = quantized.shape
    return (
        quantized.float()
        .view(groups, rows // 128, 128, columns // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .mul(scales[..., None, None])
        .permute(0, 1, 3, 2, 4)
        .reshape(groups, rows, columns)
    )


def test_build_provenance_and_capabilities_are_fail_closed() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", deep_gemm.__git_commit__)
    capabilities = deep_gemm._C.get_sm103_fp8_block128_capabilities()
    assert capabilities["architecture"] == "sm103"
    assert capabilities["compute_capability"] == (10, 3)
    assert capabilities["activation_dtype"] == "float8_e4m3fn"
    assert capabilities["weight_dtype"] == "float8_e4m3fn"
    assert capabilities["scale_dtype"] == "float32"
    assert capabilities["activation_group_k"] == 128
    assert capabilities["weight_block_m"] == 128
    assert capabilities["weight_block_k"] == 128
    assert capabilities["route_score_placement"] == "post_down"
    assert capabilities["fallback"] is None


@pytest.mark.parametrize("rows", [0, 1, 5, 128, 129])
def test_quantize_and_dequantize_match_reference(rows: int) -> None:
    torch.manual_seed(1000 + rows)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    quantized, scales = deep_gemm._C.sm103_fp8_block128_quantize(x)
    expected_quantized, expected_scales = _reference_quantize(x)
    _assert_quantized_close(quantized, scales, expected_quantized, expected_scales)

    dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(quantized, scales)
    expected_dequantized = (
        quantized.float().view(rows, 2, 128) * scales.unsqueeze(-1)
    ).view_as(x).to(torch.bfloat16)
    assert torch.equal(dequantized, expected_dequantized)


@pytest.mark.parametrize("rows", [0, 3, 129])
def test_swiglu_forward_and_backward_match_reference(rows: int) -> None:
    torch.manual_seed(2000 + rows)
    preactivation = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    quantized, scales = deep_gemm._C.sm103_fp8_block128_swiglu_quantize(preactivation)
    up, gate = preactivation.float().chunk(2, dim=-1)
    expected_quantized, expected_scales = _reference_quantize(torch.nn.functional.silu(gate) * up)
    _assert_quantized_close(quantized, scales, expected_quantized, expected_scales)

    grad_output = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    actual_grad = deep_gemm._C.sm103_fp8_block128_swiglu_backward(grad_output, preactivation)
    preactivation_ref = preactivation.float().detach().requires_grad_(True)
    up_ref, gate_ref = preactivation_ref.chunk(2, dim=-1)
    (torch.nn.functional.silu(gate_ref) * up_ref).backward(grad_output.float())
    torch.testing.assert_close(
        actual_grad.float(), preactivation_ref.grad.to(torch.bfloat16).float(), rtol=3e-2, atol=2e-2
    )
    canonical_grad = deep_gemm._C.sm103_fp8_block128_swiglu_backward_canonical(
        grad_output, preactivation
    )
    active_up_grad, active_gate_grad = preactivation_ref.grad.chunk(2, dim=-1)
    expected_canonical = torch.cat((active_gate_grad, active_up_grad), dim=-1)
    torch.testing.assert_close(
        canonical_grad.float(),
        expected_canonical.to(torch.bfloat16).float(),
        rtol=3e-2,
        atol=2e-2,
    )


def test_post_down_combine_score_grad_and_route_sum() -> None:
    torch.manual_seed(3000)
    tokens, topk, hidden = 7, 8, 256
    route_output = torch.randn(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    route_scores = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
    grad_output = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    actual = deep_gemm._C.sm103_fp8_block128_post_down_combine(route_output, route_scores)
    expected = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    for route in range(topk):
        expected.add_(route_output.view(tokens, topk, hidden)[:, route].float() * route_scores[:, route, None])
    assert torch.equal(actual, expected.to(torch.bfloat16))

    actual_score_grad = deep_gemm._C.sm103_fp8_block128_post_down_score_grad(
        route_output, grad_output, topk
    )
    expected_score_grad = (
        route_output.view(tokens, topk, hidden).float() * grad_output[:, None].float()
    ).sum(dim=-1)
    torch.testing.assert_close(actual_score_grad, expected_score_grad, rtol=2e-5, atol=2e-4)

    actual_sum = deep_gemm._C.sm103_fp8_block128_route_sum(route_output, tokens, topk)
    expected_sum = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    for route in range(topk):
        expected_sum.add_(route_output.view(tokens, topk, hidden)[:, route].float())
    assert torch.equal(actual_sum, expected_sum.to(torch.bfloat16))


def test_route_scale_quantize_uses_flat_route_order() -> None:
    torch.manual_seed(4000)
    tokens, topk, hidden = 5, 3, 256
    grad_output = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    route_scores = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
    route_order = torch.randperm(tokens * topk, device="cuda", dtype=torch.int64)

    quantized, scales = deep_gemm._C.sm103_fp8_block128_route_scale_quantize(
        grad_output, route_scores, route_order
    )
    flat = route_order.cpu().tolist()
    expanded = torch.stack(
        [grad_output[route // topk].float() * route_scores.view(-1)[route] for route in flat]
    )
    expected_quantized, expected_scales = _reference_quantize(expanded)
    _assert_quantized_close(quantized, scales, expected_quantized, expected_scales)


@pytest.mark.parametrize("weight_is_k_by_n", [False, True])
def test_grouped_fp8_block128_gemm_matches_dequantized_reference(
    weight_is_k_by_n: bool,
) -> None:
    torch.manual_seed(5000 + int(weight_is_k_by_n))
    counts = [4, 8, 0, 12]
    rows, groups, k, n = sum(counts), len(counts), 256, 256
    activations_bf16 = (
        torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    activations, activation_scales = deep_gemm._C.sm103_fp8_block128_quantize(
        activations_bf16
    )
    physical_shape = (groups, k, n) if weight_is_k_by_n else (groups, n, k)
    weights_bf16 = (
        torch.randn(*physical_shape, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    weights, weight_scales = _blockwise_weight_quantize(weights_bf16)

    symbol = (
        deep_gemm._C.sm103_fp8_block128_grouped_gemm_nn
        if weight_is_k_by_n
        else deep_gemm._C.sm103_fp8_block128_grouped_gemm_nt
    )
    actual = symbol(
        activations, activation_scales, weights, weight_scales, counts
    )

    activations_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(
        activations, activation_scales
    ).float()
    weights_dequantized = _blockwise_weight_dequantize(weights, weight_scales)
    expected_parts = []
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            expert_weight = weights_dequantized[expert]
            expected_parts.append(
                activations_dequantized[offset : offset + count]
                @ (expert_weight if weight_is_k_by_n else expert_weight.t())
            )
        offset += count
    expected = torch.cat(expected_parts).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.08)


def test_grouped_fp8_block128_gemm_covers_zero_routes_and_rejects_unpadded_groups() -> None:
    activations = torch.empty(0, 256, device="cuda", dtype=torch.float8_e4m3fn)
    activation_scales = torch.empty(0, 2, device="cuda", dtype=torch.float32)
    weights_bf16 = torch.randn(3, 256, 256, device="cuda", dtype=torch.bfloat16)
    weights, weight_scales = _blockwise_weight_quantize(weights_bf16)
    output = deep_gemm._C.sm103_fp8_block128_grouped_gemm_nt(
        activations, activation_scales, weights, weight_scales, [0, 0, 0]
    )
    assert output.shape == (0, 256)
    assert output.dtype == torch.bfloat16

    one_row = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    one_q, one_s = deep_gemm._C.sm103_fp8_block128_quantize(one_row)
    with pytest.raises(RuntimeError, match="multiple of four"):
        deep_gemm._C.sm103_fp8_block128_grouped_gemm_nt(
            one_q, one_s, weights, weight_scales, [1, 0, 0]
        )


def test_canonical_w13_gemm_presents_up_then_gate_without_weight_copy() -> None:
    torch.manual_seed(5500)
    counts = [4, 0, 8]
    experts, hidden, model_dim = len(counts), 128, 256
    activations_bf16 = torch.randn(
        sum(counts), model_dim, device="cuda", dtype=torch.bfloat16
    )
    activations, activation_scales = deep_gemm._C.sm103_fp8_block128_quantize(
        activations_bf16
    )
    canonical_bf16 = torch.randn(
        experts * 2, hidden, model_dim, device="cuda", dtype=torch.bfloat16
    )
    canonical, canonical_scales = _blockwise_weight_quantize(canonical_bf16)
    actual = deep_gemm._C.sm103_fp8_block128_grouped_w13_gemm_nt_canonical(
        activations,
        activation_scales,
        canonical,
        canonical_scales,
        counts,
    )

    x_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(
        activations, activation_scales
    ).float()
    w_dequantized = _blockwise_weight_dequantize(canonical, canonical_scales)
    expected_parts = []
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            gate = x_dequantized[offset : offset + count] @ w_dequantized[2 * expert].t()
            up = x_dequantized[offset : offset + count] @ w_dequantized[2 * expert + 1].t()
            expected_parts.append(torch.cat((up, gate), dim=-1))
        offset += count
    expected = torch.cat(expected_parts).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.08)


def test_k_grouped_bf16_wgrad_supports_no_accumulator_and_empty_expert() -> None:
    torch.manual_seed(6000)
    counts = [128, 0, 128]
    total, m, n = sum(counts), 256, 256
    left = torch.randn(total, m, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(total, n, device="cuda", dtype=torch.bfloat16)
    output = torch.zeros(len(counts), m, n, device="cuda", dtype=torch.bfloat16)
    grouped_layout = torch.tensor(
        [128, 128, 256], device="cuda", dtype=torch.int32
    )

    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        left,
        right,
        output,
        None,
        grouped_layout,
        None,
        use_psum_layout=True,
    )
    torch.testing.assert_close(
        output[0].float(), (left[:128].t() @ right[:128]).to(torch.bfloat16).float(),
        rtol=2e-2, atol=0.2,
    )
    assert torch.count_nonzero(output[1]) == 0
    torch.testing.assert_close(
        output[2].float(), (left[128:].t() @ right[128:]).to(torch.bfloat16).float(),
        rtol=2e-2, atol=0.2,
    )
