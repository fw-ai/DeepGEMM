# Kimi-K3 EP8 balanced-64K high-water receipt

## Status

This document freezes the strongest balanced 64K/rank Kimi-K3 MegaMoE result
measured before the unified-dW2 experiment.  It is a recovery milestone, not a
claim that the final performance target is complete: backward reaches
`1.1978985x`, just below the relaxed long-sequence requirement of `1.2x`.

The measurement is a direct eight-GPU run on B300, with EP=8, no CP, full
parameter gradients, three packed backward ranges, and true unequal per-rank
token counts:

```text
[43691, 49933, 56174, 62416, 68656, 74898, 81139, 87381]
```

The requested average was 65,536 tokens/rank.  Timings use three warmups and
eight CUDA-event samples, with rank-max medians and identical inputs for the
native/DeepEPv2 and MegaMoE arms.

## Design frozen by this milestone

The forward path is the Kimi-K3 MegaMoE SiTU specialization.  It preserves the
training arithmetic: MXFP4 group-128 weights, route weights after W2, SiTU
`beta=4` and `linear_beta=25`, and the native BF16 rounding points.

Backward remains one persistent MegaMoE parent kernel.  Communication uses
symmetric-memory planes and GPU-scope publication/acquire edges.  The combine
warps execute the true-varlen fixed-top-k dX reduction inside that kernel while
the tensor-core roles execute the gradient GEMMs.  TMA moves operands and
outputs; UMMA performs dgrad and wgrad.  The three-range selector uses one
logical BF16 wgrad product (`mxfp8wgrad=false`, `exact2=false`, `ring=false`),
so this receipt does not silently switch quantization or use the dormant
three-product MXFP8 reconstruction.

The terminal scheduler retains the verified full-rank publication edge for
multi-range inputs.  It uses readiness-driven work selection and the existing
union workspace; it adds no tensor, symmetric-memory plane, process-global
mode, or device allocation.  Sixteen SMs are reserved for the in-kernel
communication roles for this 64K operating point.

## Exact source and runtime provenance

| Artifact | SHA-256 |
|---|---|
| Backward parent header | `2545686139c2ae8e6ed22404717206c4c409c2fd34ecdfee7ba6a5bca7490845` |
| Scheduler header | `ae87b0a68f80a38a7b0ab13451dd9f44547c908ad4e4d0b8aed9c796b421ebf2` |
| Three-term wgrad header | `5eb0cf50b5ef1d7b1f0b794aa90ca70c3a07b34df983391a3893e18dd2a729a8` |
| Rebuilt DeepGEMM extension | `b870dda2e4c0eb779f150ae90228358fd2892850356aa8edf46d348cafc8f887` |
| Frozen benchmark | `fdf65f8982ec3466bd38b6bb79fec32efc2c6f52f9c95c44d1b858b616ece0e2` |
| FireTitan MegaMoE wrapper | `229811d1ddaacece8c40bfdc540a57426497201f927684cd48ce770fe112ce8c` |
| FireTitan native baseline wrapper | `806f27c852760300f6d1320b11ac86f10ba93ca54a934ba03b71a053ed110590` |
| DeepEPv2 baseline | `7d6529e1905021d178ec067e0a34014d7aa4adeaf533f1362f59ff7a83ea1db5` |

The complete installed DeepGEMM header tree in the archived runtime is
byte-identical to this branch.  The archive contains two additional `.orig`
diagnostic backups; neither is compiled or committed.

## Latency, throughput, and MFU

| Phase | Native/DeepEPv2 | MegaMoE | Speedup | Native useful TFLOP/s/GPU | MegaMoE useful TFLOP/s/GPU | Native MFU | MegaMoE MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FWD** | 106.712 ms | 46.682 ms | **2.285952x** | 649.122 | 1483.861 | 14.425% | 32.975% |
| **BWD** | 190.713 ms | 159.206 ms | **1.197899x** | 726.424 | 870.182 | 32.285% | 38.675% |
| **FWD+BWD** | 294.116 ms | 207.201 ms | **1.419470x** | 706.550 | 1002.926 | 26.169% | 37.145% |

Useful TFLOP/s divides identical logical routed-MoE FLOPs by rank-max latency.
It does not credit padding or count implementation-specific extra products as
useful model work.

## Numerical gate

All candidate/reference tensors contain zero nonfinite values.

| Quantity | FP64 cosine |
|---|---:|
| Output | 0.999996449 |
| dX | 0.999993137 |
| dRoute | 0.999999999 |
| dW1 | 0.999999982 |
| dW2 | 0.999999983 |
| dW3 | 0.999999981 |

Every value exceeds the required `0.99999` threshold.

## Peak allocation

Independent phase records report the following rank-max absolute CUDA peak
allocation:

| Phase | Native/DeepEPv2 | MegaMoE | Delta |
|---|---:|---:|---:|
| **FWD** | 71.13 GiB | 39.63 GiB | -31.50 GiB |
| **BWD** | 82.36 GiB | 67.78 GiB | -14.58 GiB |
| **FWD+BWD** | 82.36 GiB | 67.78 GiB | -14.58 GiB |

Thus this milestone has no peak-allocation regression.  Reserved memory is
reported separately by the benchmark because allocator reservation is not a
proxy for live tensor footprint.

## Promotion boundary

This branch is the rollback point for later kernel work.  A candidate may
replace it only after direct EP8 true-varlen numerics, latency, and isolated
memory measurements prove that it preserves the 4K--64K envelope and improves
the target operating point.  The final milestone additionally requires the
128K--512K sweep, MFU tables, and same-input 20-step shrink-Kimi-K3 loss parity.

## Primary references

- [DeepGEMM MegaMoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [Mixture of Kittens](https://github.com/cursor/mixture-of-kittens)
