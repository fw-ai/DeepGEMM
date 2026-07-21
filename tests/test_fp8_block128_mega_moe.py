import pytest
import torch

import deep_gemm
from deep_gemm.mega.fp8_block128 import _validate_master_tensor


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103-only MegaMoE test",
)


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


def _make_case(tokens: int, *, all_to_one: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(7000 + tokens + int(all_to_one))
    experts, model_dim, hidden, topk = 4, 256, 128, 2
    x = (torch.randn(tokens, model_dim, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    w1_master = (
        torch.randn(experts, hidden, model_dim, device="cuda", dtype=torch.bfloat16) * 0.05
    ).requires_grad_()
    w3_master = (
        torch.randn(experts, hidden, model_dim, device="cuda", dtype=torch.bfloat16) * 0.05
    ).requires_grad_()
    w2_master = (
        torch.randn(experts, model_dim, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    ).requires_grad_()
    canonical_w13 = torch.stack((w1_master.detach(), w3_master.detach()), dim=1).flatten(0, 1)
    w13_q, w13_s = _blockwise_weight_quantize(
        canonical_w13
    )
    w2_q, w2_s = _blockwise_weight_quantize(w2_master.detach())
    if all_to_one:
        topk_ids = torch.zeros(tokens, topk, device="cuda", dtype=torch.int64)
    else:
        route = torch.arange(tokens * topk, device="cuda", dtype=torch.int64)
        topk_ids = torch.remainder(route * 3 + 1, experts).view(tokens, topk)
    scores = torch.sigmoid(torch.randn(tokens, topk, device="cuda", dtype=torch.float32))
    scores = (scores / scores.sum(dim=-1, keepdim=True) * 2.5).detach().requires_grad_()
    return {
        "x": x,
        "ids": topk_ids.contiguous(),
        "scores": scores.contiguous(),
        "w13_q": w13_q,
        "w13_s": w13_s,
        "w2_q": w2_q,
        "w2_s": w2_s,
        "w1_master": w1_master,
        "w2_master": w2_master,
        "w3_master": w3_master,
    }


def _forward_reference(case: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    x_q, x_s = deep_gemm._C.sm103_fp8_block128_quantize(case["x"].detach())
    x_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(x_q, x_s).float()
    canonical_w13 = _blockwise_weight_dequantize(case["w13_q"], case["w13_s"])
    experts = case["w2_q"].shape[0]
    hidden = case["w2_q"].shape[2]
    model_dim = case["w2_q"].shape[1]
    canonical_pairs = canonical_w13.view(experts, 2, hidden, model_dim)
    w13 = torch.stack((canonical_pairs[:, 1], canonical_pairs[:, 0]), dim=1).reshape(
        experts, hidden * 2, model_dim
    )
    w2 = _blockwise_weight_dequantize(case["w2_q"], case["w2_s"])
    flat_ids = case["ids"].flatten()
    route_x = x_dequantized.repeat_interleave(case["ids"].shape[1], dim=0)
    preactivation = torch.bmm(
        w13.index_select(0, flat_ids), route_x.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)
    hidden_q, hidden_s = deep_gemm._C.sm103_fp8_block128_swiglu_quantize(preactivation)
    hidden = deep_gemm._C.sm103_fp8_block128_dequantize(hidden_q, hidden_s).float()
    route_output = torch.bmm(
        w2.index_select(0, flat_ids), hidden.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)
    output = deep_gemm._C.sm103_fp8_block128_post_down_combine(
        route_output, case["scores"]
    )
    return output, route_output


def _ste_reference_backward(
    case: dict[str, torch.Tensor], upstream: torch.Tensor
) -> dict[str, torch.Tensor]:
    x = case["x"].detach().clone().requires_grad_()
    scores = case["scores"].detach().clone().requires_grad_()
    w1_master = case["w1_master"].detach().clone().requires_grad_()
    w2_master = case["w2_master"].detach().clone().requires_grad_()
    w3_master = case["w3_master"].detach().clone().requires_grad_()
    experts, model_dim, hidden = w2_master.shape

    x_q, x_s = deep_gemm._C.sm103_fp8_block128_quantize(x.detach())
    x_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(x_q, x_s).float()
    x_effective = x.float() + (x_dequantized - x.float()).detach()
    w13_active_master = torch.stack((w3_master, w1_master), dim=1).reshape(
        experts, hidden * 2, model_dim
    )
    canonical_w13_dequantized = _blockwise_weight_dequantize(
        case["w13_q"], case["w13_s"]
    ).view(experts, 2, hidden, model_dim)
    w13_effective = w13_active_master.float() + (
        torch.stack(
            (canonical_w13_dequantized[:, 1], canonical_w13_dequantized[:, 0]),
            dim=1,
        ).reshape(experts, hidden * 2, model_dim)
        - w13_active_master.float()
    ).detach()
    w2_dequantized = _blockwise_weight_dequantize(case["w2_q"], case["w2_s"])
    w2_effective = w2_master.float() + (
        w2_dequantized - w2_master.float()
    ).detach()

    flat_ids = case["ids"].flatten()
    route_x = x_effective.repeat_interleave(case["ids"].shape[1], dim=0)
    preactivation = torch.bmm(
        w13_effective.index_select(0, flat_ids), route_x.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)
    up, gate = preactivation.float().chunk(2, dim=-1)
    hidden_raw = torch.nn.functional.silu(gate) * up
    hidden_q, hidden_s = deep_gemm._C.sm103_fp8_block128_swiglu_quantize(
        preactivation.detach()
    )
    hidden_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(
        hidden_q, hidden_s
    ).float()
    hidden_effective = hidden_raw + (hidden_dequantized - hidden_raw).detach()
    route_output = torch.bmm(
        w2_effective.index_select(0, flat_ids), hidden_effective.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)
    tokens, topk = case["ids"].shape
    output_float = torch.zeros(tokens, model_dim, device="cuda", dtype=torch.float32)
    route_output_view = route_output.view(tokens, topk, model_dim)
    for route in range(topk):
        output_float = output_float + route_output_view[:, route].float() * scores[:, route, None]
    output = output_float.to(torch.bfloat16)
    output.backward(upstream)
    return {
        "output": output.detach(),
        "x_grad": x.grad.detach(),
        "score_grad": scores.grad.detach(),
        "w1_grad": w1_master.grad.detach(),
        "w2_grad": w2_master.grad.detach(),
        "w3_grad": w3_master.grad.detach(),
    }


def _normalized_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    return float(
        1
        - 2
        * (actual_f @ expected_f)
        / (actual_f.square().sum() + expected_f.square().sum() + 1e-12)
    )


class _DistributedMasterFixture:
    def __init__(
        self,
        local: torch.Tensor,
        logical_shape: tuple[int, int, int],
    ) -> None:
        self._local = local
        self.shape = logical_shape

    def to_local(self) -> torch.Tensor:
        return self._local


def test_distributed_master_validation_accepts_resident_efsdp_shard_with_wrapper() -> None:
    local = torch.empty(1, 128, 256, device="cuda", dtype=torch.bfloat16)
    master = _DistributedMasterFixture(local, (4, 128, 256))

    _validate_master_tensor(
        master,
        name="w1_master",
        local_shape=(2, 128, 256),
        global_shape=(4, 128, 256),
        device=local.device,
        master_gradient_wrapper=lambda *_grads: None,
    )


def test_distributed_master_validation_requires_gradient_wrapper() -> None:
    local = torch.empty(1, 128, 256, device="cuda", dtype=torch.bfloat16)
    master = _DistributedMasterFixture(local, (4, 128, 256))

    with pytest.raises(ValueError, match="resident shape"):
        _validate_master_tensor(
            master,
            name="w1_master",
            local_shape=(2, 128, 256),
            global_shape=(4, 128, 256),
            device=local.device,
            master_gradient_wrapper=None,
        )


@pytest.mark.parametrize(
    ("tokens", "all_to_one"),
    [(1, True), (63, False), (64, True), (65, False)],
)
def test_single_rank_forward_matches_reference_across_padding_boundaries(
    tokens: int, all_to_one: bool
) -> None:
    case = _make_case(tokens, all_to_one=all_to_one)
    actual = deep_gemm.fp8_block128_mega_moe(
        case["x"],
        case["ids"],
        case["scores"],
        case["w13_q"],
        case["w13_s"],
        case["w2_q"],
        case["w2_s"],
        case["w1_master"],
        case["w2_master"],
        case["w3_master"],
    )
    expected, _ = _forward_reference(case)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.08)


def test_glm_w13_transform_is_gate_up_to_up_gate() -> None:
    gate = torch.full((128, 256), 3, device="cuda", dtype=torch.float8_e4m3fn)
    up = torch.full((128, 256), 7, device="cuda", dtype=torch.float8_e4m3fn)
    gate_second = torch.full_like(gate, 6)
    up_second = torch.full_like(up, 14)
    canonical = torch.stack((gate, up, gate_second, up_second))
    scales = torch.arange(8, device="cuda", dtype=torch.float32).view(4, 1, 2)
    active, active_scales = deep_gemm.transform_glm_w13_for_fp8_block128_mega_moe(
        canonical, scales
    )
    assert torch.equal(active[0, :128], up)
    assert torch.equal(active[0, 128:], gate)
    assert torch.equal(active[1, :128], up_second)
    assert torch.equal(active[1, 128:], gate_second)
    assert torch.equal(active_scales[0, 0], scales[1, 0])
    assert torch.equal(active_scales[0, 1], scales[0, 0])


def test_single_rank_backward_returns_input_score_and_canonical_master_grads() -> None:
    case = _make_case(9, all_to_one=False)
    upstream = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    expected = _ste_reference_backward(case, upstream)
    output = deep_gemm.fp8_block128_mega_moe(
        case["x"],
        case["ids"],
        case["scores"],
        case["w13_q"],
        case["w13_s"],
        case["w2_q"],
        case["w2_s"],
        case["w1_master"],
        case["w2_master"],
        case["w3_master"],
    )
    output.backward(upstream)

    torch.testing.assert_close(
        output.float(), expected["output"].float(), rtol=0.08, atol=0.08
    )
    torch.testing.assert_close(
        case["scores"].grad, expected["score_grad"], rtol=3e-4, atol=3e-3
    )
    assert case["x"].grad.shape == case["x"].shape
    assert case["x"].grad.dtype == torch.bfloat16
    assert case["w1_master"].grad.shape == case["w1_master"].shape
    assert case["w1_master"].grad.dtype == torch.bfloat16
    assert case["w2_master"].grad.shape == case["w2_master"].shape
    assert case["w2_master"].grad.dtype == torch.bfloat16
    assert case["w3_master"].grad.shape == case["w3_master"].shape
    assert case["w3_master"].grad.dtype == torch.bfloat16
    assert _normalized_difference(case["x"].grad, expected["x_grad"]) < 0.12
    assert _normalized_difference(case["w1_master"].grad, expected["w1_grad"]) < 0.15
    assert _normalized_difference(case["w2_master"].grad, expected["w2_grad"]) < 0.12
    assert _normalized_difference(case["w3_master"].grad, expected["w3_grad"]) < 0.15
