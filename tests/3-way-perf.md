# DeepGEMM mega-MoE vs FlashInfer NVFP4 MoE — performance log

Device-time comparison of DeepGEMM's fused mega-MoE kernels against FlashInfer's
three NVFP4 MoE backends (CuteDSL, CUTLASS, TRT-LLM-gen) on **NVIDIA B200**.

- **HW/SW:** 8× NVIDIA B200 (single node, NVSwitch), torch 2.9, flashinfer 0.6.11
- **All times are pure GPU device time** read from profiler traces (kernel self-time,
  no CPU/launch time). Multi-GPU numbers are **averaged across ranks**.
- Scripts: `tests/bench_flashinfer_vs_deepgemm.py` (single device),
  `tests/bench_ep_multi_gpu.py` (expert-parallel), `tests/bench_kernel_breakdown.py`
  (per-kernel), `tests/fi_trtllm.py` (+ `tests/_fi_vendor/` harness).

## Backends

| backend | kernel(s) | activation | combine |
|---|---|---|---|
| DeepGEMM fp8×fp4 mega | 1 fused kernel (dispatch+GEMMs+act+combine) | SwiGLU only | fused (symmetric memory) |
| DeepGEMM nvfp4 mega | 1 fused kernel | SwiGLU only | fused (symmetric memory) |
| FlashInfer cute_dsl | moe_sort + 2 grouped GEMMs + finalize | SwiGLU only | external all_reduce |
| FlashInfer cutlass | grouped GEMM ×2 + ~6 helper kernels | SwiGLU / **GeGLU** | external all_reduce |
| FlashInfer trtllm-gen | 2 bmm GEMMs + routing + finalize | SwiGLU / **GeGLU** | external all_reduce |

Notes:
- `fp8×fp4` (gran-32 scale factors) requires `intermediate_size % 512 == 0`, so it
  cannot run `inter=2304` (NVFP4-only, gran-16).
- GeGLU vs SwiGLU is perf-neutral (the gating activation is a ~few-µs elementwise op
  dwarfed by the two GEMMs); only cutlass/trtllm expose a non-SwiGLU activation.

---

## 1. Single device (device µs, CUDA-graph replay)

| tok | exp | top_k | hidden | inter | dg_fp8 | dg_nvfp4 | fi_cutedsl | fi_cutlass | fi_trtllm |
|----:|----:|------:|-------:|------:|-------:|---------:|-----------:|-----------:|----------:|
| 128 | 32 | 4 | 2048 | 2048 | 65.6 | 70.6 | 57.5 | 109.6 | **57.4** |
| 512 | 32 | 4 | 2048 | 2048 | 80.1 | 86.3 | **60.6** | 106.3 | 68.9 |
| 1024 | 256 | 8 | 7168 | 2560 | 1327.4 | 1322.6 | **1254.1** | 2367.6 | 1526.6 |
| 32 | 256 | 16 | 4608 | 2304 | n/a | 599.6 | 602.0 | 1248.0 | **575.3** |

- cute_dsl and trtllm-gen are the fastest FlashInfer backends; **cutlass is ~1.6–2× slower**.
- At the large compute-bound config, DeepGEMM ≈ cute_dsl; cutlass far behind.

---

## 2. Expert-parallel (EP) — the target workload

Shape: `hidden=4608, inter=2304` (GeGLU MoE: gate/up `[M_e,4608]×[4608,2304]`,
down `[M_e,2304]×[2304,4608]`), `total_experts=512`, `top_k=16`.

EP model:
- **DeepGEMM mega** — native dispatch: `num_tokens`/rank, fused dispatch+GEMM+combine.
- **FlashInfer** — replicated input + expert-shard + all_reduce: every GPU holds the
  full global batch (`num_tokens × world`), owns `512/world` experts (via
  `local_expert_offset`/`ep_rank`), computes only its slice, then `all_reduce` sums
  the partial `[global_tokens, hidden]` outputs.
- Both strategies process the same global tokens and the same per-expert load (`m_e`).
- Combine `all_reduce` uses **NVLS** (NVLink-SHARP multicast) via an NCCL
  symmetric-registered buffer: `ncclSymDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_bf16`.
- `sum` = moe device time (steps: dispatch+slice+moe+combine, all inside the FlashInfer
  kernel) **+** NVLS all_reduce. DeepGEMM total = its single fused kernel.

Activation: cutlass & trtllm-gen run **GeGLU**; cute_dsl & DeepGEMM run SwiGLU
(only option) — perf-equivalent.

### num_tokens = 48 per rank (a2a path); FlashInfer global = 48 × world

**4 GPUs** — 128 experts/gpu, global batch 192, m_e≈6:
| backend | act | moe | allreduce (NVLS) | total |
|---|---|---:|---:|---:|
| DeepGEMM nvfp4 mega | SwiGLU | 393.9 | fused | 393.9 |
| FlashInfer cute_dsl | SwiGLU | 367.4 | 17.0 | 384.4 |
| FlashInfer cutlass | GeGLU | 693.6 | 17.0 | 710.5 |
| FlashInfer trtllm-gen | GeGLU | 358.2 | 17.0 | **375.2** |

**8 GPUs** — 64 experts/gpu, global batch 384, m_e≈12:
| backend | act | moe | allreduce (NVLS) | total |
|---|---|---:|---:|---:|
| DeepGEMM nvfp4 mega | SwiGLU | 251.8 | fused | 251.8 |
| FlashInfer cute_dsl | SwiGLU | 199.5 | 37.4 | **236.9** |
| FlashInfer cutlass | GeGLU | 394.9 | 37.4 | 432.3 |
| FlashInfer trtllm-gen | GeGLU | 318.2 | 37.4 | 355.6 |

### num_tokens = 192 per rank (= 48 × 4); FlashInfer global = 192 × world

**4 GPUs** — 128 experts/gpu, global batch 768, m_e≈24:
| backend | act | moe | allreduce (NVLS) | total |
|---|---|---:|---:|---:|
| DeepGEMM nvfp4 mega | SwiGLU | 440.7 | fused | 440.7 |
| FlashInfer cute_dsl | SwiGLU | 378.6 | 36.0 | **414.6** |
| FlashInfer cutlass | GeGLU | 733.0 | 36.0 | 769.0 |
| FlashInfer trtllm-gen | GeGLU | 381.5 | 36.0 | 417.6 |

**8 GPUs** — 64 experts/gpu, global batch 1536, m_e≈48:
| backend | act | moe | allreduce (NVLS) | total |
|---|---|---:|---:|---:|
| DeepGEMM nvfp4 mega | SwiGLU | 287.4 | fused | 287.4 |
| FlashInfer cute_dsl | SwiGLU | 213.1 | 57.5 | **270.6** |
| FlashInfer cutlass | GeGLU | 415.2 | 57.5 | 472.7 |
| FlashInfer trtllm-gen | GeGLU | 240.9 | 57.5 | 298.4 |

---

## 3. Per-kernel device-time breakdown

Single GPU, representative per-GPU EP shape: 32 tokens × 128 local experts, top-16,
hidden=4608, inter=2304 (m_e=4).

### DeepGEMM nvfp4 mega — 1 kernel, 355.2 µs
| #/it | µs/it | kernel |
|--:|--:|---|
| 1 | 355.2 | `sm100_mxfp4_mxfp4_mega_moe_impl<…>` (dispatch + 2 GEMMs + SwiGLU + combine fused) |

### FlashInfer cute_dsl — 5 kernels, 354.2 µs
| #/it | µs/it | kernel |
|--:|--:|---|
| 1 | 226.1 | gemm1: gather + grouped GEMM + SwiGLU |
| 1 | 116.4 | gemm2: grouped GEMM + finalize |
| 1 | 6.6 | routing |
| 1 | 2.7 | copy |
| 1 | 2.5 | fill |

### FlashInfer trtllm-gen — 4 kernels, 341.9 µs
| #/it | µs/it | kernel |
|--:|--:|---|
| 1 | 210.9 | gemm1 `bmm_E2m1_E2m1E2m1_Fp32_…` |
| 1 | 111.9 | gemm2 `bmm_Bfloat16_E2m1E2m1_…` |
| 1 | 10.6 | finalize |
| 1 | 8.5 | routing |

### FlashInfer cutlass — 8 kernels (9 launches), 704.3 µs
| #/it | µs/it | kernel |
|--:|--:|---|
| 2 | 629.9 | grouped GEMM `cutlass::device_kernel<GemmUniversal…>` (gemm1+gemm2) |
| 1 | 36.6 | finalizeMoeRouting |
| 1 | 14.9 | blockExpertPrefixSum |
| 1 | 7.5 | doActivation |
| 1 | 6.5 | expandInputRows |
| 1 | 4.6 | computeStridesTmaWarpSpecialized |
| 1 | 2.3 | mergeExpertPrefixSum |
| 1 | 2.0 | globalExpertPrefixSum |

All backends are ~95% GEMM; gemm1 (hidden→2·inter + gate act) ≈ 2× gemm2.

---

## 4. NVLS all_reduce combine

The combine is a single NCCL **NVLS** kernel
(`ncclSymDevKernel_AllReduce_RSxLDMC_AGxSTMC_sum_bf16`), enabled by allocating the
buffer from NCCL's symmetric/multicast allocator
(`torch.cuda.MemPool(backend.mem_allocator)` + `register_mem_pool(pool, symm=True)`).
Without this registration NCCL falls back to `RING_LL` even with `NCCL_ALGO=NVLS`.

NVLS vs RING+LL (sleep-aligned device µs):
| GPUs | payload | RING+LL | NVLS |
|--:|--:|--:|--:|
| 4 | 1152 KiB | 23.0 | **16.3** |
| 8 | 2304 KiB | 60.5 | **~31** |

NVLS combine scales with the replicated global batch: 17→36 µs (4 GPU), 37→58 µs
(8 GPU) as tokens grow from 48→192/rank.

---

## Key findings

1. **Backend ranking (device time):** trtllm-gen ≈ cute_dsl (fastest) > DeepGEMM mega
   ≈ them; **cutlass is consistently ~1.7–2× slower**, entirely due to its grouped-GEMM
   kernel (the helper kernels are minor).
2. **DeepGEMM fusion vs FlashInfer + NVLS combine:** DeepGEMM folds the combine into a
   single kernel (no separate collective). FlashInfer's fast backends + NVLS all_reduce
   reach comparable or slightly better totals at these small/medium batches (e.g. 8-GPU,
   48 tok: cute_dsl 236.9 vs DeepGEMM 251.8 µs).
3. **GeGLU vs SwiGLU:** perf-neutral.
4. **NVLS matters:** ~1.4–1.9× faster combine than RING+LL and scales better; required a
   symmetric-registered buffer to engage.
5. **Measurement:** per-kernel device time is read directly from traces. Collective
   (all_reduce) timing needs launch alignment, otherwise the NCCL kernel absorbs
   cross-rank skew (a single profiling window over moe+all_reduce over-counts the
   combine — use moe + separately-aligned all_reduce instead).
