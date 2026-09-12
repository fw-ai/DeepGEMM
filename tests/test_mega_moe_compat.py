"""CPU regressions for the upstream sync's Python and native API boundary."""

from types import SimpleNamespace

import pytest
import torch

import deep_gemm


def _buffer(num_ranks=1):
    return SimpleNamespace(
        group=SimpleNamespace(size=lambda: num_ranks, rank=lambda: 0),
        buffer=object(), handle=SimpleNamespace(buffer_ptrs=[1] * num_ranks),
        num_max_tokens_per_rank=1920, num_experts=4, num_topk=2,
        num_ring_tokens=1920,
    )


def test_native_training_bindings_are_registered():
    for name in (
        "fp8_fp4_mega_moe_backward_dgrad_swiglu_v2",
        "bf16_mega_moe_backward_dgrad_v2",
        "bf16_mega_moe_backward_post_down_prelude_v2",
        "bf16_mega_moe_backward_w13_combine",
        "bf16_mega_moe_backward_w2_combine",
        "get_symm_buffer_size_for_mega_moe_v2",
        "get_symm_buffer_size_for_mega_moe_v3",
        "bf16_mega_gate", "mega_mhc", "fp8_fp4_sparse_mqa_logits",
    ):
        assert callable(getattr(deep_gemm._C, name))


@pytest.mark.parametrize("kind", ["bf16", "fp8_fp4"])
def test_inference_does_not_sync_config_tokens_to_host(monkeypatch, kind):
    """Default inference must remain usable inside upstream's CUDA graphs."""
    def unexpected_collective(*args, **kwargs):
        pytest.fail("inference performed a host-synchronized config collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_collective)
    captured = []
    monkeypatch.setattr(deep_gemm._C, f"{kind}_mega_moe", lambda *args: captured.append(args))
    shared_l1, shared_l2 = object(), object()
    getattr(deep_gemm, f"{kind}_mega_moe")(
        torch.empty(2, 4), object(), object(), _buffer(2),
        shared_l1_weights=shared_l1, shared_l2_weights=shared_l2,
    )
    assert captured[0][-2:] == (shared_l1, shared_l2)


def test_fp8_training_keeps_rank_uniform_pool_config(monkeypatch):
    def maximum(tensor, *, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        tensor.fill_(17)

    monkeypatch.setattr(torch.distributed, "all_reduce", maximum)
    captured = []
    monkeypatch.setattr(deep_gemm._C, "fp8_fp4_mega_moe", lambda *args: captured.append(args))
    saved = torch.empty(32, 8)
    deep_gemm.fp8_fp4_mega_moe(
        torch.empty(2, 4), object(), object(), _buffer(2),
        saved_l1_preact=saved,
    )
    assert captured[0][15] is saved
    assert captured[0][20] == 17


def test_symm_buffer_keeps_ring_argument_and_exposes_shared_views(monkeypatch):
    group = SimpleNamespace(size=lambda: 1)
    base = SimpleNamespace(buffer=torch.empty(16, dtype=torch.int8), handle=object(), group=group)
    views = [torch.full((2, 4), i) for i in range(15)]
    captured = []

    def size_and_slicer(*args):
        captured.append(args)
        return 16, lambda buffer: views

    monkeypatch.setattr(deep_gemm._C, "get_token_alignment_for_mega_moe", lambda: 1920)
    monkeypatch.setattr(deep_gemm._C, "get_symm_buffer_size_for_mega_moe_v3", size_and_slicer)
    buffer = deep_gemm.SymmBuffer(
        group, 4, 128, 2, 4, 4, 1920, "bf16xbf16", "geglu",
        base=base, num_shared_experts=1,
    )
    assert captured[0][-2:] == (1920, 1)
    assert buffer.token_src_metadata is views[8]
    assert buffer.backward_grad_route is views[10]
    assert buffer.shared_l1_acts is views[11]
    assert buffer.shared_l2_acts_sf is views[14]
