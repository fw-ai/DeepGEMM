# DeepGEMM packed-FP4 benchmarks

Performance benchmarks for the packed-FP4 (MXFP4 / NVFP4) GEMM and mega-MoE
kernels on SM100 (Blackwell). Three scripts, each measuring a different axis:

| Script | Axis | What it compares | GPUs |
|---|---|---|---|
| `bench_packed_fp4.py` | **FP4 format** | MXFP4 vs NVFP4 (vs `fp8xfp4`) within DeepGEMM | 1 |
| `bench_flashinfer_vs_deepgemm.py` | **framework (single box)** | DeepGEMM mega-MoE vs FlashInfer NVFP4 MoE backends | 1 |
| `bench_ep_multi_gpu.py` | **scale (multi-GPU EP)** | DeepGEMM native fused EP vs FlashInfer + NCCL `all_reduce` combine | N (2/4/8) |

`bench_flashinfer_vs_deepgemm.py` also has a `--breakdown` mode that prints a
per-kernel device-time breakdown (launches/iter + us/iter per kernel) for each
backend, so you can see how many kernels each launches and where the time goes.

## Prerequisites

- **SM100 (Blackwell)** GPU — the packed-FP4 kernels are SM100-only.
- **DeepGEMM built in-place** from the repo root:
  ```bash
  git submodule update --init --recursive   # cutlass + fmt
  ./develop.sh                               # builds deep_gemm._C and symlinks the .so
  ```
- **FlashInfer** (`pip install flashinfer`) for the two vs-benchmarks. Some
  FlashInfer MoE backends are sensitive to the installed version (see
  *Compatibility notes* below).
- Tests are imported by the benchmarks for shared quantization helpers, so the
  repo root must be on the path (the scripts add `tests/` to `sys.path` themselves).

## How to run

### 1. `bench_packed_fp4.py` — FP4 format comparison (single device)

DeepGEMM's own packed-FP4 paths: standalone GEMM (`mxfp4_gemm_nt` /
`nvfp4_gemm_nt`) and single-rank mega-MoE (`mxfp4_mxfp4_mega_moe` /
`nvfp4_nvfp4_mega_moe`), with `fp8_fp4_mega_moe` as the FP8×FP4 baseline.

```bash
python benchmarks/bench_packed_fp4.py
```

Output: a GEMM table (mxfp4 vs nvfp4 us + TFLOPS) and a MoE table
(fp8xfp4 vs mxfp4 vs nvfp4 us, with nv/fp8 and nv/mx ratios).

### 2. `bench_flashinfer_vs_deepgemm.py` — DeepGEMM vs FlashInfer (single device)

Same MoE problem (identical shapes + routing) for every backend, apples-to-apples.
Backends: DeepGEMM `fp8_fp4_mega_moe` (dg_fp8), DeepGEMM `nvfp4_nvfp4_mega_moe`
(dg_nvfp4), FlashInfer `cute_dsl_fused_moe_nvfp4` (fi_cutedsl), FlashInfer
`cutlass_fused_moe` (fi_cutlass), FlashInfer trtllm-gen (fi_trtllm). Timing is
end-to-end device time via CUDA-graph replay (suffix `g` = graph, `e` = eager).

```bash
python benchmarks/bench_flashinfer_vs_deepgemm.py            # timing table
python benchmarks/bench_flashinfer_vs_deepgemm.py --breakdown  # per-kernel breakdown
```

### 3. `bench_ep_multi_gpu.py` — multi-GPU expert-parallel EP

`total_experts=512` sharded across N GPUs; each shard holds `num_tokens` local
tokens; top-k routing. DeepGEMM's native fused EP (dispatch + grouped GEMM +
combine via symmetric memory, timed end-to-end with comm internal) vs FlashInfer
single-device kernel + manual NCCL `all_reduce` combine (timed as
barrier → moe → all_reduce). Reports worst-rank average latency.

```bash
python benchmarks/bench_ep_multi_gpu.py --num-processes 2 --num-tokens 32
python benchmarks/bench_ep_multi_gpu.py --num-processes 4 --num-tokens 32
python benchmarks/bench_ep_multi_gpu.py --num-processes 8 --num-tokens 32
```

Flags: `--num-processes` (world size), `--num-tokens` (per-rank tokens for the
DeepGEMM dispatch model), `--inter` (intermediate size; `fp8xfp4` needs a
multiple of 512, e.g. 2304 is nvfp4-only), `--nccl-algo` (force `NCCL_ALGO`,
e.g. `NVLS`, for the combine `all_reduce`).

## Results

Environment: **NVIDIA B200 (SM100)**, torch `2.9.0a0+145a3a7bda.nv25.10`,
flashinfer `0.6.11`, single B200 for the single-device benches, 2× B200 for EP.
Captured 2026-07-06 on branch `mxfp4-mxfp4-mega-moe`. Times are kernel-only
device time (CUDA graph / profiler); lower is better.

### bench_packed_fp4 — standalone packed-FP4 GEMM (2-CTA de-risk kernel)

| M | N | K | mxfp4 us | mxfp4 TFLOPS | nvfp4 us | nvfp4 TFLOPS | nv/mx |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 4096 | 4096 | 59.9 | 2293 | 69.8 | 1968 | 1.16× |
| 4096 | 4096 | 8192 | 119.0 | 2309 | 134.3 | 2047 | 1.13× |
| 8192 | 8192 | 8192 | 456.6 | 2408 | 507.4 | 2167 | 1.11× |
| 2048 | 4096 | 16384 | 140.5 | 1957 | 153.0 | 1797 | 1.09× |

MXFP4 is ~1.1–1.16× faster than NVFP4 on the standalone GEMM (same FP4 tensor-core
path; MXFP4's UE8M0 gran-32 SFs are cheaper than NVFP4's E4M3 gran-16 + global scale).

### bench_packed_fp4 — single-rank mega-MoE (fp8xfp4 vs mxfp4 vs nvfp4)

| tokens | experts | topk | hidden | inter | fp8xfp4 us | mxfp4 us | nvfp4 us | nv/fp8 | nv/mx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 2 | 2048 | 2048 | 39.7 | 40.9 | 45.3 | 1.14× | 1.11× |
| 512 | 8 | 2 | 2048 | 2048 | 47.5 | 51.4 | 54.5 | 1.15× | 1.06× |
| 1024 | 32 | 4 | 4096 | 1536 | 124.6 | 117.2 | 122.4 | 0.98× | 1.04× |

fp8xfp4 (FP8 E4M3 activations × FP4 weights) and the FP4×FP4 variants are within
~15%; which wins depends on shape (the 1024/32/4 shape is compute-bound enough
that the FP4-activation paths pull ahead of fp8xfp4).

### bench_flashinfer_vs_deepgemm — single-device DeepGEMM vs FlashInfer (device us, CUDA graph)

| tokens | experts | topk | hidden | inter | dg_fp8 | dg_nvfp4 | fi_cutedsl | fi_cutlass | fi_trtllm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 4 | 2048 | 2048 | 63.7 | 69.1 | n/a | 110.1 | n/a |
| 512 | 32 | 4 | 2048 | 2048 | 77.7 | 84.4 | n/a | 107.5 | n/a |
| 1024 | 256 | 8 | 7168 | 2560 | 1347.7 | 1322.7 | n/a | 2242.7 | n/a |
| 32 | 256 | 16 | 4608 | 2304 | n/a | 604.2 | n/a | 1127.5 | n/a |

DeepGEMM's fused mega kernel is ~1.6–1.9× faster than FlashInfer's
`cutlass_fused_moe` on these shapes. `dg_fp8` is `n/a` for `inter=2304`
(fp8xfp4 needs `inter % 512 == 0`); that shape is nvfp4-only.

### bench_ep_multi_gpu — multi-GPU EP (world=2, total_experts=512, 32 tokens/rank, hidden=4608 inter=2560 top_k=16)

| kernel | moe us | allreduce us | total us |
|---|---:|---:|---:|
| DeepGEMM fp8xfp4 mega (fused EP) | 721.9 | fused | 721.9 |
| DeepGEMM nvfp4 mega (fused EP) | 742.6 | fused | 742.6 |
| FlashInfer nvfp4 cutlass (GeGLU) | 1252.8 | 14.4 | 1271.0 |

DeepGEMM's fused EP (dispatch + GEMM + combine via symmetric memory, comm
internal) is ~1.7× faster than FlashInfer `cutlass_fused_moe` + NCCL
`all_reduce` combine at world=2. The combine `all_reduce` itself is only
~14 µs here — the gap is the MoE kernel, not the combine.

### `--breakdown` example (`bench_flashinfer_vs_deepgemm.py --breakdown`)

For shape `tokens=32 experts=256 top_k=16 hidden=4608 inter=2304`:

```
===== DeepGEMM nvfp4 mega (fused EP) =====
  1 distinct kernels, 1 launches/iter, total device 604.2 us/iter
   #/it    us/it  kernel
      1    604.2  void deep_gemm::sm100_mxfp4_mxfp4_mega_moe_impl<...>

===== FlashInfer nvfp4 (cutlass) =====
  8 distinct kernels, 9 launches/iter, total device 1127.5 us/iter
   #/it    us/it  kernel
      2  1058.4  cutlass::device_kernel<...GemmUniversal...>
      1    36.8  tensorrt_llm::finalizeMoeRoutingKernel<bf16, ...>
      1    15.1  tensorrt_llm::blockExpertPrefixSumKernel<256>
      ...
```

DeepGEMM launches **1** fused kernel; FlashInfer cutlass launches **9** across
8 distinct kernels (two GEMMs + finalize + prefix-sum/expand/activation helpers).

## Compatibility notes

- The FlashInfer `cute_dsl` and `trtllm-gen` MoE backends are sensitive to the
  installed FlashInfer version. On flashinfer `0.6.11` they fail with
  `Module has no function 'flashinfer_moe_output_memset_inplace_bf16'` /
  trtllm argument-mismatch errors; the `cutlass` backend works. The DeepGEMM
  paths are unaffected. (The benches catch and report these per-backend errors
  rather than aborting the sweep.)
- `fp8xfp4` requires `inter % 512 == 0`; `inter = 2304` is nvfp4-only.
- All single-device benches use CUDA-graph replay for timing; the suffix `g`
  (graph) or `e` (eager) is printed next to each number.
