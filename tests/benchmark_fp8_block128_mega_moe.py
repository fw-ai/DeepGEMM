"""Structured SM103 performance smoke for FP8-block128 MegaMoE.

The defaults are intentionally small enough for routine companion validation.
Use GLM-5.2 dimensions explicitly for integration evidence, for example::

    python tests/benchmark_fp8_block128_mega_moe.py \
      --tokens 15625 --experts 16 --model-dim 6144 --hidden 2048 --topk 8

The process must be launched on an otherwise idle SM103 GPU.  The benchmark
prints one JSON object so callers can archive the exact dimensions and metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch

import deep_gemm


def _blockwise_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _elapsed_ms(operation: Callable[[], None], iterations: int) -> list[float]:
    measurements: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        measurements.append(float(start.elapsed_time(end)))
    return measurements


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "maximum_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the performance benchmark requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(
            f"the performance benchmark is SM103-only; runtime capability is {capability}"
        )
    for name in ("model_dim", "hidden"):
        if getattr(args, name) <= 0 or getattr(args, name) % 128:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive multiple of 128")
    if args.tokens <= 0 or args.experts <= 0 or args.topk <= 0:
        raise ValueError("tokens, experts, and topk must be positive")
    if args.topk > args.experts:
        raise ValueError("topk cannot exceed experts")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    torch.manual_seed(20260721)
    device = torch.device("cuda")
    x = (
        torch.randn(args.tokens, args.model_dim, device=device, dtype=torch.bfloat16)
        * 0.02
    ).requires_grad_()
    w1_master = (
        torch.randn(
            args.experts,
            args.hidden,
            args.model_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    ).requires_grad_()
    w3_master = (
        torch.randn(
            args.experts,
            args.hidden,
            args.model_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    ).requires_grad_()
    w2_master = (
        torch.randn(
            args.experts,
            args.model_dim,
            args.hidden,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    ).requires_grad_()
    canonical_w13 = torch.stack(
        (w1_master.detach(), w3_master.detach()), dim=1
    ).flatten(0, 1)
    w13_q, w13_s = _blockwise_quantize(
        canonical_w13
    )
    w2_q, w2_s = _blockwise_quantize(w2_master.detach())
    routes = torch.arange(args.tokens * args.topk, device=device, dtype=torch.int64)
    topk_ids = torch.remainder(routes * 17 + 3, args.experts).view(
        args.tokens, args.topk
    )
    raw_scores = torch.sigmoid(
        torch.randn(args.tokens, args.topk, device=device, dtype=torch.float32)
    )
    scores = (
        raw_scores / raw_scores.sum(dim=-1, keepdim=True) * 2.5
    ).detach().requires_grad_()
    upstream = torch.randn_like(x)

    latest_output: torch.Tensor | None = None

    def forward() -> None:
        nonlocal latest_output
        latest_output = deep_gemm.fp8_block128_mega_moe(
            x,
            topk_ids,
            scores,
            w13_q,
            w13_s,
            w2_q,
            w2_s,
            w1_master,
            w2_master,
            w3_master,
        )

    def forward_backward() -> None:
        forward()
        assert latest_output is not None
        latest_output.backward(upstream)
        x.grad = None
        scores.grad = None
        w1_master.grad = None
        w2_master.grad = None
        w3_master.grad = None

    for _ in range(args.warmup):
        forward_backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    forward_times = _elapsed_ms(forward, args.iterations)
    # Do not charge destruction of the final forward-only autograd graph to
    # the first complete forward/backward sample.
    latest_output = None
    torch.cuda.synchronize()
    forward_backward_times = _elapsed_ms(forward_backward, args.iterations)
    wall_seconds = time.monotonic() - started

    result = {
        "schema_version": 1,
        "backend": "fp8_block128_mega_moe",
        "architecture": "sm103",
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
        "deep_gemm_git_commit": deep_gemm.__git_commit__,
        "dimensions": {
            "tokens": args.tokens,
            "experts": args.experts,
            "topk": args.topk,
            "model_dim": args.model_dim,
            "hidden": args.hidden,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "forward": _summary(forward_times),
        "forward_backward": _summary(forward_backward_times),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": wall_seconds,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
