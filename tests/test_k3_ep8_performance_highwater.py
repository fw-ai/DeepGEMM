"""Regression tests for the immutable Kimi-K3 EP=8 performance ledger."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path


def _repository_root() -> Path:
    """Return the DeepGEMM repository containing this test."""
    return Path(__file__).resolve().parents[1]


def _load_ledger() -> dict:
    """Load the checked-in performance ledger as a JSON object."""
    path = _repository_root() / "docs/k3_ep8_performance_highwater.json"
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_validator() -> dict:
    """Load the receipt validator without requiring scripts to be a package."""
    return runpy.run_path(
        str(_repository_root() / "scripts/validate_k3_ep8_receipts.py"),
        run_name="k3_receipt_validator",
    )


def _phase_result_from_ledger(avg_tokens_per_rank: int = 65536) -> dict:
    """Build a minimal phase receipt from the immutable high-water values."""
    ledger = _load_ledger()
    phases = _receipt_validator()["_protected_phases"](
        ledger, avg_tokens_per_rank
    )
    result = {}
    for phase, result_key in {
        "forward": "forward_ms",
        "backward": "backward_ms",
        "forward_backward": "forward_backward_ms",
    }.items():
        record = phases[phase]
        result[result_key] = {
            "median": {
                "native_deepep_v2": record["native_deepep_v2"],
                "megamoe_mok": record["candidate"],
            },
            "speedup": record["native_deepep_v2"] / record["candidate"],
        }
    return result


def test_fast_milestone_source_bytes_match_the_ledger() -> None:
    """Prevent experiments from silently replacing the recovered fast source."""
    root = _repository_root()
    identity = _load_ledger()["source_identity"]
    files = {
        "backward_parent_header_sha256": (
            "deep_gemm/include/deep_gemm/impls/"
            "sm100_fp8_fp4_mega_moe_backward.cuh"
        ),
        "bf16_gemm_body_sha256": (
            "deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh"
        ),
        "scheduler_sha256": "deep_gemm/include/deep_gemm/scheduler/gemm.cuh",
        "range_provider_sha256": (
            "deep_gemm/include/deep_gemm/scheduler/"
            "external_k_grouped_range.hpp"
        ),
        "jit_host_sha256": (
            "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
        ),
        "wgrad_header_sha256": (
            "deep_gemm/include/deep_gemm/impls/"
            "sm100_mxfp8_three_term_grouped_wgrad.cuh"
        ),
    }
    for ledger_key, relative_path in files.items():
        assert _sha256(root / relative_path) == identity[ledger_key]


def test_64k_phase_highwater_is_the_fastest_retained_receipt() -> None:
    """Keep the 139.209-ms backward result ahead of the regressed rollback."""
    phases = _load_ledger()["phase_highwater_ms"]["65536"]
    expected = {
        "forward": (106.712, 46.681758880615234),
        "backward": (190.4432601928711, 139.2091522216797),
        "forward_backward": (294.2889862060547, 186.5465316772461),
    }
    for phase, (native_ms, candidate_ms) in expected.items():
        assert phases[phase]["native_deepep_v2"] == native_ms
        assert phases[phase]["candidate"] == candidate_ms
        observed_speedup = native_ms / candidate_ms
        assert abs(observed_speedup - phases[phase]["speedup"]) < 1e-8

    # The discarded rollback must never fit inside the allowed 3% noise band.
    maximum_noise = _load_ledger()["promotion"][
        "maximum_latency_noise_fraction"
    ]
    assert 159.20633697509766 > (
        phases["backward"]["candidate"] * (1 + maximum_noise)
    )
    assert 207.20146942138672 > (
        phases["forward_backward"]["candidate"] * (1 + maximum_noise)
    )


def test_numeric_qualification_covers_all_outputs_and_gradients() -> None:
    """Require the six retained FP64 cosines to exceed the strict floor."""
    ledger = _load_ledger()
    cosines = ledger["qualified_cosines"]
    assert set(cosines) == {"output", "dX", "dRoute", "dW1", "dW2", "dW3"}
    assert all(
        value > ledger["cosine_floor_exclusive"] for value in cosines.values()
    )


def test_historical_short_length_frontier_is_immutable_and_qualified() -> None:
    """Retain every 4K--32K GPU winner as a latency regression ceiling."""
    ledger = _load_ledger()
    frontier = ledger["historical_regression_frontier"]
    expected_candidate_ms = {
        "4096": (5.0423359870910645, 20.046367645263672, 24.013887405395508),
        "8192": (8.091168403625488, 25.515743255615234, 32.21945571899414),
        "16384": (13.440735816955566, 44.304447174072266, 56.119232177734375),
        "32768": (24.800159454345703, 83.12163543701172, 106.23721313476562),
    }
    assert set(frontier) == {"semantics", *expected_candidate_ms}
    for length, expected in expected_candidate_ms.items():
        record = frontier[length]
        phases = record["phases"]
        observed = tuple(
            phases[phase]["candidate"]
            for phase in ("forward", "backward", "forward_backward")
        )
        assert observed == expected
        assert all(
            value > ledger["cosine_floor_exclusive"]
            for value in record["qualified_cosines"].values()
        )
        assert all(
            len(digest) == 64 for digest in record["receipts"].values()
        )

    four_k = frontier["4096"]
    assert four_k["source_status"] == (
        "raw_receipt_without_same_run_source_provenance"
    )
    assert "source_artifacts" not in four_k
    replay_seed = four_k["reconstructed_replay_seed"]
    assert replay_seed["relationship"] == (
        "post_hoc_reconstruction_not_same_run_identity"
    )
    assert len(replay_seed["deep_gemm_commit"]) == 40


def test_unpromoted_4k_perf_winner_remains_a_mandatory_latency_target() -> None:
    """Preserve the 2.068x BWD result without mislabeling it promotable."""
    ledger = _load_ledger()
    record = ledger["observed_unpromoted_performance_frontier"]["4096"]
    assert record["source_status"] == (
        "paired_raw_receipts_without_source_commit"
    )
    assert set(record["promotion_blockers"]) == {
        "missing_same_run_source_identity",
        "forward_backward_peak_allocated_exceeds_native_by_more_than_noise_ceiling",
    }
    assert record["phases"]["backward"] == {
        "native_deepep_v2": 33.61590576171875,
        "candidate": 16.252992630004883,
        "speedup": 2.068290223651486,
    }
    assert record["phases"]["forward_backward"] == {
        "native_deepep_v2": 41.53744125366211,
        "candidate": 20.16431999206543,
        "speedup": 2.05994753455643,
    }
    peaks = record["same_process_peak_allocated_bytes"][
        "forward_backward"
    ]
    assert peaks["candidate"] - peaks["native_deepep_v2"] == 89183744
    assert peaks["candidate"] - peaks["native_deepep_v2"] > ledger[
        "promotion"
    ]["maximum_peak_memory_noise_bytes"]
    assert all(
        value > ledger["cosine_floor_exclusive"]
        for value in record["qualified_cosines"].values()
    )
    assert all(len(digest) == 64 for digest in record["receipts"].values())

    protected = _receipt_validator()["_protected_phases"](ledger, 4096)
    assert protected["forward"]["candidate"] == 5.0423359870910645
    assert protected["backward"]["candidate"] == 16.252992630004883
    assert protected["forward_backward"]["candidate"] == 20.16431999206543


def test_replay_identity_covers_benchmark_integration_and_runtime() -> None:
    """Require every non-DeepGEMM replay component to have a pinned identity."""
    identity = _load_ledger()["replay_identity"]
    assert set(identity) == {
        "benchmark_sha256",
        "firetitan_commit",
        "megamoe_mok_sha256",
        "native_mxfp4_megamoe_sha256",
        "native_quant_experts_sha256",
        "mxfp4_quant_sha256",
        "deepep_v2_sha256",
        "runtime_deep_gemm_init_sha256",
        "runtime_mega_init_sha256",
        "runtime_mega_backward_sha256",
        "runtime_extension_sha256",
    }
    assert len(identity["firetitan_commit"]) == 40
    assert all(
        len(value) == 64
        for key, value in identity.items()
        if key != "firetitan_commit"
    )


def test_receipt_validator_accepts_the_protected_phase_highwater() -> None:
    """Prove that the immutable high-water satisfies its own promotion rules."""
    failures = []
    metrics = _receipt_validator()["validate_phase_result"](
        _phase_result_from_ledger(), _load_ledger(), failures
    )
    assert not failures
    assert metrics["backward"]["candidate_ms"] == 139.2091522216797


def test_receipt_validator_rejects_the_slower_rollback() -> None:
    """Prevent the 159/207-ms rollback from becoming the optimization base."""
    result = _phase_result_from_ledger()
    for result_key, candidate_ms in {
        "backward_ms": 159.20633697509766,
        "forward_backward_ms": 207.20146942138672,
    }.items():
        result[result_key]["median"]["megamoe_mok"] = candidate_ms
        result[result_key]["speedup"] = (
            result[result_key]["median"]["native_deepep_v2"] / candidate_ms
        )
    failures = []
    _receipt_validator()["validate_phase_result"](
        result, _load_ledger(), failures
    )
    assert any("backward candidate" in failure for failure in failures)
    assert any("forward_backward candidate" in failure for failure in failures)


def test_receipt_validator_enforces_the_4k_winner() -> None:
    """Reject a candidate that starts below any per-phase 4K winner."""
    validator = _receipt_validator()
    ledger = _load_ledger()
    result = _phase_result_from_ledger(4096)
    failures = []
    validator["validate_phase_result"](
        result, ledger, failures, avg_tokens_per_rank=4096
    )
    assert not failures

    result["backward_ms"]["median"]["megamoe_mok"] = 20.046367645263672
    result["backward_ms"]["speedup"] = (
        result["backward_ms"]["median"]["native_deepep_v2"]
        / result["backward_ms"]["median"]["megamoe_mok"]
    )
    failures = []
    validator["validate_phase_result"](
        result, ledger, failures, avg_tokens_per_rank=4096
    )
    assert any("backward candidate" in failure for failure in failures)
