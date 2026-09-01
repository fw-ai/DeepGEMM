#!/usr/bin/env python3
"""Validate Kimi-K3 EP=8 latency, numerics, provenance, and memory receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _phase_keys() -> dict[str, str]:
    """Return the benchmark result key for every protected phase."""
    return {
        "forward": "forward_ms",
        "backward": "backward_ms",
        "forward_backward": "forward_backward_ms",
    }


def _benchmark_arms() -> tuple[str, str]:
    """Return baseline and candidate arm names in comparison order."""
    return "native_deepep_v2", "megamoe_mok"


def _protected_phases(
    ledger: dict[str, Any], avg_tokens_per_rank: int
) -> dict[str, Any]:
    """Return the fastest observed ceiling for every protected phase.

    A numerically qualified observation can define a latency target before it
    is promotable as a complete milestone. Keep that target while provenance
    and memory remain independent mandatory gates; otherwise a slower replay
    could erase useful performance simply because the faster run was missing
    one of those separate receipts.
    """
    key = str(avg_tokens_per_rank)
    source_bound = ledger["phase_highwater_ms"].get(key)
    historical = ledger["historical_regression_frontier"].get(key)
    observed = ledger.get(
        "observed_unpromoted_performance_frontier", {}
    ).get(key)
    candidates = [
        phases
        for phases in (
            source_bound,
            historical["phases"] if historical is not None else None,
            observed["phases"] if observed is not None else None,
        )
        if phases is not None
    ]
    if not candidates:
        raise KeyError(f"no protected latency frontier for {avg_tokens_per_rank}")
    return {
        phase: min(
            (candidate[phase] for candidate in candidates),
            key=lambda record: record["candidate"],
        )
        for phase in _phase_keys()
    }


def load_tagged_json(path: Path, tag: str) -> list[dict[str, Any]]:
    """Return every JSON payload following the requested tag in a log."""
    marker = tag + " "
    records = []
    for line in path.read_text(errors="replace").splitlines():
        offset = line.find(marker)
        if offset >= 0:
            records.append(json.loads(line[offset + len(marker) :]))
    return records


def _require(condition: bool, message: str, failures: list[str]) -> None:
    """Append message when a promotion invariant is false."""
    if not condition:
        failures.append(message)


def validate_provenance(
    provenance: dict[str, Any],
    ledger: dict[str, Any],
    failures: list[str],
) -> None:
    """Require the replay to use the protected source and integration bytes."""
    source = ledger["source_identity"]
    replay = ledger["replay_identity"]
    expected = {
        "benchmark": replay["benchmark_sha256"],
        "megamoe_mok": replay["megamoe_mok_sha256"],
        "native_mxfp4_megamoe": replay["native_mxfp4_megamoe_sha256"],
        "deepep_v2": replay["deepep_v2_sha256"],
        "deep_gemm": replay["runtime_deep_gemm_init_sha256"],
        "deep_gemm_mega_init": replay["runtime_mega_init_sha256"],
        "deep_gemm_mega_backward": replay["runtime_mega_backward_sha256"],
        "deep_gemm_extension": replay["runtime_extension_sha256"],
        "backward_parent_header": source["backward_parent_header_sha256"],
        "wgrad_header": source["wgrad_header_sha256"],
        "scheduler_header": source["scheduler_sha256"],
    }
    for name, expected_sha in expected.items():
        observed_sha = provenance.get(name, {}).get("sha256")
        _require(
            observed_sha == expected_sha,
            f"provenance mismatch for {name}: {observed_sha} != {expected_sha}",
            failures,
        )

    environment = provenance.get("environment", {})
    expected_environment = {
        "K3_MOK_REQUIRED_WORLD": "8",
        "K3_MOK_AVG_TOKENS_PER_RANK": "65536",
        "K3_MOK_CHUNK_TOKENS": "131072",
        "K3_MOK_REQUIRED_BACKWARD_RANGES": "1",
        "K3_MOK_VERIFY_NUMERICS": "1",
        "K3_MOK_VERIFY_NUMERICS_ALL_LENGTHS": "1",
        "K3_MOK_GRAD_SCOPE": "full",
        "K3_MOK_BENCH_ARMS": "native_deepep_v2,megamoe_mok",
    }
    for name, expected_value in expected_environment.items():
        observed_value = environment.get(name)
        _require(
            observed_value == expected_value,
            f"environment mismatch for {name}: {observed_value} != {expected_value}",
            failures,
        )


def validate_phase_result(
    result: dict[str, Any],
    ledger: dict[str, Any],
    failures: list[str],
    avg_tokens_per_rank: int = 65536,
) -> dict[str, dict[str, float]]:
    """Validate phase speedups and reject latency above the protected high-water."""
    protected = _protected_phases(ledger, avg_tokens_per_rank)
    noise = ledger["promotion"]["maximum_latency_noise_fraction"]
    arms = _benchmark_arms()
    metrics: dict[str, dict[str, float]] = {}
    for phase, result_key in _phase_keys().items():
        medians = result.get(result_key, {}).get("median", {})
        native_ms = medians.get(arms[0])
        candidate_ms = medians.get(arms[1])
        if not isinstance(native_ms, (int, float)) or not isinstance(
            candidate_ms, (int, float)
        ):
            failures.append(f"missing {phase} medians for both benchmark arms")
            continue
        speedup = native_ms / candidate_ms
        reported_speedup = result[result_key].get("speedup")
        _require(
            isinstance(reported_speedup, (int, float))
            and math.isclose(speedup, reported_speedup, rel_tol=1e-12),
            f"{phase} reported speedup does not match its medians",
            failures,
        )
        ceiling_ms = protected[phase]["candidate"] * (1 + noise)
        _require(
            candidate_ms <= ceiling_ms,
            (
                f"{phase} candidate {candidate_ms:.6f} ms exceeds protected "
                f"{ceiling_ms:.6f}-ms ceiling"
            ),
            failures,
        )
        metrics[phase] = {
            "native_ms": float(native_ms),
            "candidate_ms": float(candidate_ms),
            "speedup": float(speedup),
            "protected_candidate_ms": float(protected[phase]["candidate"]),
        }

    if set(metrics) == set(_phase_keys()):
        targets = ledger["required_speedup"]["short_range_through_64k"]
        separate_target = (
            metrics["forward"]["speedup"] > targets["forward_exclusive"]
            and metrics["backward"]["speedup"] > targets["backward_exclusive"]
        )
        combined_target = (
            metrics["forward_backward"]["speedup"]
            > targets["forward_backward_alternative_exclusive"]
        )
        _require(
            separate_target or combined_target,
            (
                f"{avg_tokens_per_rank}-token speedup target failed for both "
                "the separate and combined gates"
            ),
            failures,
        )
    return metrics


def validate_numerics(
    records: list[dict[str, Any]],
    ledger: dict[str, Any],
    failures: list[str],
) -> dict[str, float]:
    """Require one full-size record with six finite cosines above the floor."""
    contract = ledger["measurement_contract"]
    expected_global_tokens = 65536 * contract["ep"]
    matching = [
        record
        for record in records
        if record.get("global_tokens") == expected_global_tokens
    ]
    _require(bool(matching), "missing 64K/rank production numerics record", failures)
    if not matching:
        return {}
    record = matching[-1]
    cosines = record.get("min_cosines", {})
    expected_names = {"output", "dX", "dRoute", "dW1", "dW2", "dW3"}
    _require(set(cosines) == expected_names, "numerics must cover all six tensors", failures)
    floor = ledger["cosine_floor_exclusive"]
    for name in expected_names:
        value = cosines.get(name)
        _require(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value > floor,
            f"{name} cosine {value} does not exceed {floor}",
            failures,
        )
        counts = record.get("nonfinite_counts", {}).get(name, {})
        _require(
            counts.get("reference") == 0 and counts.get("candidate") == 0,
            f"{name} contains nonfinite reference or candidate values",
            failures,
        )
    return {
        name: float(value)
        for name, value in cosines.items()
        if isinstance(value, (int, float))
    }


def nvml_peak_bytes(path: Path) -> int:
    """Return the largest per-GPU used-memory sample from an NVML CSV receipt."""
    peak_mib = 0
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.rsplit(",", 2)]
        if len(fields) != 3:
            continue
        try:
            peak_mib = max(peak_mib, int(fields[2]))
        except ValueError:
            continue
    if peak_mib == 0:
        raise ValueError(f"no NVML memory samples found in {path}")
    return peak_mib * 1024 * 1024


def validate_memory_receipts(
    specs: dict[tuple[str, str], Path],
    ledger: dict[str, Any],
    failures: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compare phase-isolated allocator and device-wide peaks across both arms."""
    benchmark_arms = _benchmark_arms()
    metrics: dict[str, dict[str, dict[str, int]]] = {}
    for phase in _phase_keys():
        metrics[phase] = {}
        for arm in benchmark_arms:
            path = specs.get((phase, arm))
            if path is None or not path.is_file():
                failures.append(f"missing fresh-memory log for {phase}:{arm}")
                continue
            summaries = load_tagged_json(path, "K3_MOK_PHASE_BENCH_SUMMARY")
            results = load_tagged_json(path, "K3_MOK_PHASE_BENCH")
            if len(summaries) != 1 or len(results) != 1:
                failures.append(f"invalid fresh-memory receipt count for {phase}:{arm}")
                continue
            summary = summaries[0]
            result = results[0]
            _require(
                summary.get("fresh_memory_phase") == phase,
                f"fresh-memory phase mismatch for {phase}:{arm}",
                failures,
            )
            _require(
                summary.get("arms") == [arm],
                f"fresh-memory arm mismatch for {phase}:{arm}",
                failures,
            )
            record = result.get("peak_memory_bytes", {}).get(phase, {}).get(arm, {})
            try:
                allocated = int(
                    record["max_peak_allocated_record"]["peak_allocated"]
                )
                reserved = int(record["max_peak_reserved_record"]["peak_reserved"])
                device = nvml_peak_bytes(Path(str(path) + ".nvml.csv"))
            except (KeyError, TypeError, ValueError) as error:
                failures.append(f"invalid memory receipt for {phase}:{arm}: {error}")
                continue
            metrics[phase][arm] = {
                "allocator_peak_allocated_bytes": allocated,
                "allocator_peak_reserved_bytes": reserved,
                "device_peak_used_bytes": device,
            }

    tolerance = ledger["promotion"]["maximum_peak_memory_noise_bytes"]
    for phase, arms in metrics.items():
        if set(arms) != set(benchmark_arms):
            continue
        for metric in (
            "allocator_peak_allocated_bytes",
            "allocator_peak_reserved_bytes",
            "device_peak_used_bytes",
        ):
            native = arms[benchmark_arms[0]][metric]
            candidate = arms[benchmark_arms[1]][metric]
            _require(
                candidate <= native + tolerance,
                (
                    f"{phase} {metric} regressed by {candidate - native} bytes; "
                    f"noise ceiling is {tolerance}"
                ),
                failures,
            )
    return metrics


def parse_memory_specs(values: list[str]) -> dict[tuple[str, str], Path]:
    """Parse repeated PHASE:ARM:PATH command-line memory receipts."""
    specs: dict[tuple[str, str], Path] = {}
    for value in values:
        phase, arm, raw_path = value.split(":", 2)
        if phase not in _phase_keys() or arm not in _benchmark_arms():
            raise ValueError(f"invalid memory receipt selector: {value}")
        specs[(phase, arm)] = Path(raw_path)
    return specs


def main() -> int:
    """Validate a complete promotion receipt and print one machine-readable report."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-log", required=True, type=Path)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=root / "docs/k3_ep8_performance_highwater.json",
    )
    parser.add_argument(
        "--memory-log",
        action="append",
        default=[],
        metavar="PHASE:ARM:PATH",
        help="repeat for all three phases and both arms",
    )
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    failures: list[str] = []
    provenances = load_tagged_json(args.performance_log, "K3_MOK_RUN_PROVENANCE")
    results = load_tagged_json(args.performance_log, "K3_MOK_PHASE_BENCH")
    summaries = load_tagged_json(args.performance_log, "K3_MOK_PHASE_BENCH_SUMMARY")
    numerics = load_tagged_json(args.performance_log, "K3_MOK_PRODUCTION_NUMERICS")

    _require(len(provenances) == 1, "expected one run provenance record", failures)
    _require(len(results) == 1, "expected one 64K phase result", failures)
    _require(len(summaries) == 1, "expected one phase summary", failures)
    if provenances:
        validate_provenance(provenances[-1], ledger, failures)
    if summaries:
        summary = summaries[-1]
        contract = ledger["measurement_contract"]
        _require(summary.get("ep") == contract["ep"], "EP must be 8", failures)
        _require(
            summary.get("requested_avg_tokens_per_rank") == [65536],
            "receipt must contain exactly 64K tokens/rank",
            failures,
        )
        _require(
            summary.get("grad_scope") == contract["gradient_scope"],
            "receipt must contain full gradients",
            failures,
        )

    phase_metrics: dict[str, dict[str, float]] = {}
    if results:
        result = results[-1]
        _require(
            result.get("rank_tokens") == ledger["measurement_contract"]["rank_tokens_64k"],
            "rank-local token vector does not prove the protected true-varlen case",
            failures,
        )
        _require(
            result.get("candidate_num_ranges") == 1,
            "candidate must use one backward range",
            failures,
        )
        _require(
            result.get("candidate_backend")
            == "clustered_deepgemm_shape_selected_k3_wgrad",
            "unexpected candidate backward backend",
            failures,
        )
        phase_metrics = validate_phase_result(result, ledger, failures)

    cosine_metrics = validate_numerics(numerics, ledger, failures)
    try:
        memory_specs = parse_memory_specs(args.memory_log)
    except ValueError as error:
        failures.append(str(error))
        memory_specs = {}
    memory_metrics = validate_memory_receipts(memory_specs, ledger, failures)
    report = {
        "status": "pass" if not failures else "fail",
        "performance_log": str(args.performance_log),
        "phase_metrics": phase_metrics,
        "cosines": cosine_metrics,
        "memory_metrics": memory_metrics,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
