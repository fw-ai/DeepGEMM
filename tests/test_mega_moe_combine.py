"""Local gradient-combine coverage without launching distributed collectives."""

import pytest
import torch

import deep_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Blackwell GPU")


def reference(planes: torch.Tensor, ids: torch.Tensor, num_ranks: int, local_experts: int, mode: str) -> torch.Tensor:
    tokens, hidden = ids.shape[0], planes.shape[-1]
    slots = ids.shape[1]
    values = planes[:, :tokens].float()
    result = torch.zeros((tokens, hidden), device=planes.device, dtype=torch.float32)
    if mode == "fixed_topk":
        for slot in range(slots):
            result = result + values[slot]
        return result.bfloat16()

    owners = torch.where(ids >= 0, ids // local_experts, -1)
    partials = []
    for rank in range(num_ranks):
        partial = torch.zeros_like(result)
        for slot in range(slots):
            partial = partial + torch.where(owners[:, slot, None] == rank, values[slot], 0.0)
        partials.append(partial.bfloat16().float())

    if mode == "deepep_v1":
        for partial in partials:
            result = result + partial
    else:
        # DeepEPv2 orders the rounded owner partials by their last slot.
        owner_partials = torch.stack(partials)
        token_rows = torch.arange(tokens, device=planes.device)
        for slot in range(slots):
            owner = owners[:, slot]
            last = (owner >= 0) & ~(owners[:, slot + 1 :] == owner[:, None]).any(dim=1)
            partial = owner_partials[owner.clamp_min(0), token_rows]
            result = result + torch.where(last[:, None], partial, 0.0)
    return result.bfloat16()


@pytest.mark.parametrize("mode", ["fixed_topk", "deepep", "deepep_v1"])
@pytest.mark.parametrize("num_ranks", [1, 2, 4])
@pytest.mark.parametrize(
    "tokens,hidden,topk,masked",
    [
        (0, 3584, 8, False),
        (1, 16, 1, False),
        (7, 512, 8, False),
        (37, 3584, 8, True),
        (65, 3584, 32, False),
        (513, 3584, 8, False),
    ],
)
def test_local_combine_launch_shapes(
    mode: str, num_ranks: int, tokens: int, hidden: int, topk: int, masked: bool
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(6142)
    local_experts = 64
    capacity = tokens + 13
    ids = torch.rand((tokens, num_ranks * local_experts), device=device, generator=generator).topk(topk).indices
    if topk > 1:
        ids[:, -1] = -1
    if masked:
        ids.fill_(-1)
    planes = torch.randn((topk, capacity, hidden), device=device, dtype=torch.bfloat16, generator=generator)
    planes[:, tokens:] = float("nan")
    invalid = ids.T[..., None]
    planes[:, :tokens] = torch.where(invalid < 0, 0.0 if mode == "fixed_topk" else float("nan"), planes[:, :tokens])
    expected = reference(planes, ids, num_ranks, local_experts, mode)
    actual = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    for _ in range(2):
        deep_gemm._C.mega_moe_backward_combine_grad_x(
            actual,
            planes.flatten(0, 1),
            capacity,
            topk,
            num_ranks,
            local_experts,
            None if mode == "fixed_topk" else ids,
            mode,
        )
        assert torch.equal(actual, expected)
