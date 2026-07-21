"""Non-launching provenance and capability checks for the SM103 backend."""

from __future__ import annotations

import re

import deep_gemm


def test_fp8_block128_capability_manifest_is_exact_and_fail_closed() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", deep_gemm.__git_commit__)

    capabilities = deep_gemm.get_fp8_block128_mega_moe_capabilities()
    assert capabilities["name"] == "fp8_block128_mega_moe"
    assert capabilities["architecture"] == "sm103"
    assert capabilities["compute_capability"] == (10, 3)
    assert capabilities["activation_dtype"] == "float8_e4m3fn"
    assert capabilities["weight_dtype"] == "float8_e4m3fn"
    assert capabilities["scale_dtype"] == "float32"
    assert capabilities["activation_group_k"] == 128
    assert capabilities["weight_block_m"] == 128
    assert capabilities["weight_block_k"] == 128
    assert capabilities["route_score_placement"] == "post_down"
    assert capabilities["distributed_transport"] == "deep_ep.ElasticBuffer.expanded"
    assert capabilities["transport_layout"] == "expert_aligned_128"
    assert capabilities["transport_scale_layout"] == "row_major_fp32_group128"
    assert capabilities["transport_deterministic"] is True
    assert capabilities["combine_reductions"] == 1
    assert capabilities["wgrad_backend"] == "sm103_companion_grouped_bf16"
    assert capabilities["forward"] is True
    assert capabilities["backward"] is True
    assert capabilities["missing_symbols"] == ()
    assert capabilities["fallback"] is None

    for symbol in capabilities["native_symbols"]:
        assert hasattr(deep_gemm._C, symbol)
    for symbol in capabilities["python_symbols"]:
        assert hasattr(deep_gemm, symbol)
