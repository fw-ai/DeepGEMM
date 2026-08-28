# Kimi-K3 MegaMoE ready-BF16 backward high-water

## Status

This document freezes the reproducible EP=8 Kimi-K3 routed-MoE kernel
high-water recovered on 2026-08-28. It is the baseline for subsequent backward
optimization. A candidate must beat this state without regressing numerical
parity, shorter sequence lengths, or peak memory.

The design follows DeepGEMM's official
[MegaMoE implementation](https://github.com/deepseek-ai/DeepGEMM#mega-moe):
dispatch, expert GEMMs, and combine use symmetric memory and persistent kernel
roles so NVLink communication can overlap TMA/UMMA work. K3 changes the math
and saved-state contract for full-parameter training.

This milestone deliberately does **not** enable the experimental three-range
exact-MXFP8 epilogue ring. That experiment changed the selector and descriptor
ABI and regressed 64K/rank backward from about 160 ms to multiple seconds. The
verified three-range path is the ready-BF16 suffix described below.

## Mathematical contract

The operator aligns with training `native_mxfp4`, rather than inference
MegaMoE's default group-32 FP8 arithmetic. For a routed token:

```text
gate = BF16(Q128(x) @ MXFP4(W1))
up   = BF16(Q128(x) @ MXFP4(W3))
h    = BF16(4*tanh(gate/4)*sigmoid(gate) * 25*tanh(up/25))
down = BF16(Q128(h) @ MXFP4(W2))
y    = BF16(sum_topk(FP32(route_weight) * FP32(down)))
```

Important consequences:

- route weights are applied after W2;
- SiTU uses `beta=4` and `linear_beta=25`;
- native BF16 rounding points are retained;
- forward saves exact BF16 gate/up and unweighted down values; and
- backward computes output, dX, dRoute, dW1, dW2, and dW3 with the native
  fixed-top-k reduction order.

All promotion gates compute cosine in FP64 and require every tensor to exceed
`0.99999`, with no nonfinite values.

## EP=8 variable-length schedule

The 64K/rank validation distribution assigns unequal local lengths with a
2/3--4/3 spread. The maximum rank therefore has 87,381 tokens. Every rank uses
the same rank-uniform three-launch schedule:

```text
range 0: [0,      32768)
range 1: [32768, 65536)
range 2: [65536, global_rank_max)
```

Ranks with fewer tokens enter empty suffix slices rather than skipping a
launch. This keeps device-side symmetric-memory barriers identical across all
eight ranks. `K3BackwardRangeSet` carries the active token count, capacity,
value rows, scale rows, and per-expert prefixes for each physical range.

The current short-window ABI accepts at most three physical ranges. Inputs
above the 64K/rank envelope select the production persistent MoK owner until a
larger bounded-window ABI is separately validated.

## Persistent backward ownership

Each bounded backward launch uses the clustered DeepGEMM K3 kernel with direct
remote dX publication and in-kernel wgrad. The relevant role split is:

| Role | Responsibility |
|---|---|
| Reverse-dispatch roles | Read symmetric peer state and construct routed grad-output work |
| W13 dgrad roles | Compute expert dX contributions and publish direct remote grad-X planes |
| Ready-BF16 wgrad roles | Consume saved exact BF16 operands and execute dW2/dW13 TMA/UMMA work |
| Reduction roles | Perform fixed-top-k FP32 slot-order accumulation and one BF16 dX rounding |
| Barrier roles | Advance generation-tagged EP publication/combine epochs |

There is no Python collective or extra host-side synchronization between those
roles. Communication ownership remains inside the kernel and uses the same
symmetric workspace established by MegaMoE forward.

The verified generated specialization has these properties:

```text
direct_remote_grad_x = true
inline_wgrad = true
multi_range_backward = true
DG_EXPERIMENTAL_K3_READY_WGRAD = 1
DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS = 8
DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS = 32
mxfp8_three_term_wgrad = false
exact_two_range = false
exact_epilogue_ring = false
```

For two physical ranges, FireTitan may request the independently validated
exact-MXFP8 wgrad specialization. For three ranges it sets
`mxfp8_three_term_wgrad = false`, retaining the ready-BF16 high-water.

## Allocation-free descriptors

The ready-BF16 three-range specialization reads exact source X, saved gate/up,
and saved down values directly. Its nominal FP8 activation and scale arguments
exist only so the common host API can build TensorMaps; the selected device
path does not consume new backing allocations.

FireTitan constructs those descriptors by aliasing existing live storage:

- the FP8 descriptor aliases the first byte-half of the contiguous BF16
  saved-down arena; and
- the column-major int32 readiness descriptor aliases a bounded prefix of
  MegaMoE's compile-dead symmetric L1 activation ring.

The views allocate and copy nothing. Bounds and contiguity are checked before
the DeepGEMM call. No device global variable or process-global mode flag is
introduced.

## Two-warp terminal dX reduction

At the three-range 64K/rank shape, two warps saturate the terminal fixed-top-k
reduction. The milestone keeps reduction warps 8 and 9 active and leaves the
other two eligible roles idle. This reduces global-memory issue contention
with concurrent dW13 UMMA/TMA traffic. The behavior is a compile-time kernel
constant, not a global runtime setting.

## Reproducibility manifest

The recovered source and integration hashes are:

| Artifact | SHA256 |
|---|---|
| FireTitan `megamoe_mok.py` | `0a93bea178db9c5f6604b82b657b50ca37e042b0597d3f27e9ffdd03af0a493e` |
| DeepGEMM JIT generator | `328395f351b25c4b2a5b218a83c6551cc2bc2207eec33d079323a936b45799f5` |
| Parent backward CUDA header | `494bb2e9986f1c1e240a6e4a183f2f1feb87c1c1a96638af6edce7dc689205ab` |
| Multi-range descriptor ABI | `9ab232f84110ff19443d553ee5ac361ed22fbe53ea533fe68b09048a9e5b2c90` |
| Benchmark runner | `7d69a7f34dca01b0d4177a573b365027a121d12dd2e149a765cebed412184636` |
| Generated CUDA source | `063b16a5df3ffcb800823933f39b2b9ffa0b6daf30eb985dda97b6db1f3e6ab5` |

The archived cubin has SHA256 `d8edf212...`. A clean rebuild generated cubin
`cd0bbd99...`; ptxas output was not byte-identical, but both builds generated
the exact same CUDA source and their 64K backward medians differed by less
than 0.5%.

Authoritative logs:

- archived high-water: `/tmp/k3-three-range2-perf/run.log`;
- archived-cubin replay: `/tmp/k3-three-range-mxfp8-ring-perf.2NvFa2/run.log`;
- archived-cubin numerics: `/tmp/k3-three-range-mxfp8-ring-numeric.hAISxJ/run.log`;
- clean-build performance: `/tmp/k3-three-range-mxfp8-ring-perf.lil4al/run.log`;
- clean-build numerics: `/tmp/k3-three-range-mxfp8-ring-numeric.Iz4QkJ/run.log`.

## EP=8 64K/rank results

All measurements use Kimi-K3's real routed-MoE shape: hidden 3584,
intermediate 3072, 896 global experts, 112 local experts, top-k 16, no CP, and
true unequal rank lengths.

| Phase | Native MXFP4 + DeepEPv2 | Fresh high-water | Speedup |
|---|---:|---:|---:|
| **FWD** | 105.912 ms | 53.728 ms | **1.971x** |
| **BWD** | 190.132 ms | 161.183 ms | **1.180x** |
| **FWD+BWD** | 295.597 ms | 212.989 ms | **1.388x** |

The clean-build useful throughput and phase-specific MFU are:

| Phase | Native TFLOP/s | Candidate TFLOP/s | Native MFU | Candidate MFU |
|---|---:|---:|---:|---:|
| **FWD** | 654.0 | 1289.3 | 14.53% | 28.65% |
| **BWD** | 728.6 | 859.5 | 32.38% | 38.20% |
| **FWD+BWD** | 703.0 | 975.7 | 26.04% | 36.14% |

MFU is logical useful routed-MoE FLOPs divided by rank-max latency and the
phase peak supplied by the benchmark. It is not padded executed tensor-core
throughput.

## Numerical parity

The clean-build cubin produced:

| Tensor | Minimum FP64 cosine | Maximum absolute error |
|---|---:|---:|
| **output** | 0.99999644998 | 0.015625 |
| **dX** | 0.99999313661 | 0.0625 |
| **dRoute** | 0.99999999995 | 0.0919876 |
| **dW1** | 0.99999998178 | 0.25 |
| **dW2** | 0.99999998281 | 0.25 |
| **dW3** | 0.99999998148 | 0.25 |

No candidate or reference tensor contained a nonfinite value.

## Memory accounting

The 64K combined-arm run reported a 30,043,917,312-byte rank-local symmetric
workspace for the candidate. Torch allocator peaks were lower for the
candidate than the native baseline in the same process:

| Scope | Native peak allocated | Candidate peak allocated |
|---|---:|---:|
| **BWD** | 81.03 GB | 65.37 GB |
| **FWD+BWD** | 81.03 GB | 65.37 GB |

These are allocator peaks, not total device NVML peaks; an authoritative
no-regression claim still requires fresh-process single-arm NVML sampling.

## Remaining performance gap

This milestone restores the best reproducible 64K state but does not meet the
final backward target. FWD passes 1.5x; BWD remains 1.18x versus the 1.36x
short-window target, and continuous FWD+BWD remains 1.388x versus 1.75x.

The next optimization must start from this milestone and profile the exposed
ready-BF16 dW2/dW13 and terminal communication overlap. It must not force the
rejected exact-ring selector back on, expand the 28-map descriptor ABI, or
change the two-warp reduction without an isolated A/B that passes full
numerics and 4K--64K non-regression.
