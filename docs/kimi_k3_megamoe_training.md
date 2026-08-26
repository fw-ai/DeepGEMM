# Kimi K3 MegaMoE training kernel

This document records the design and the verified high-water-mark configuration
for Kimi K3 full-parameter MoE training.  The implementation extends MegaMoE's
fused FP8/FP4 forward dataflow with a fused backward path.  The milestone is
kept immutable so later tuning can be compared with, rather than overwrite, a
known-good kernel.

## Mathematical contract

For each routed token and expert, the forward computes

```text
gate = x W1
up   = x W3
h    = SiTU(gate) * up
y_e  = h W2
y    = sum_e(route_weight_e * y_e)
```

`RouteWeightMode::PostDown` is required for parity with Kimi K3 training: the
route weight is applied after the down projection.  The SiTU parameters are
`beta = 4` and `linear_beta = 25`; the verified configuration does not use the
fast-math approximation.  Activations use group-32 MXFP8 and weights use
MXFP4.  Quantization, route weighting, and accumulation order are fixed by the
kernel configuration rather than inferred by the caller.

Backward produces gradients for the input, route weights, and all three expert
weights.  It retains the exact source activation and prepared gate/up values so
the derivative follows the forward's quantized values and avoids an additional
BF16 reconstruction round trip.

## Kernel dataflow

The backward kernel uses the same distributed execution model as MegaMoE
forward:

- Symmetric-memory workspaces make remotely produced token tiles addressable
  by every expert-parallel rank.
- TMA moves activation, gradient, scale, and weight tiles between global and
  shared memory.
- UMMA performs the W2 data-gradient, W1/W3 data-gradient, and expert
  weight-gradient matrix products.
- Communication and compute advance as a persistent pipeline.  Direct remote
  input-gradient stores remove a separate combine buffer for the verified
  configuration.
- W1/W3 data-gradient, W1/W2/W3 weight-gradient, and route-gradient work are
  compiled into the fused backward path.  The residual MXFP8 weights needed by
  backward are built and consumed in-kernel.
- Variable token counts are represented by per-rank source metadata and expert
  prefix information; tiles outside the actual routed-token ranges are masked.

The verified Kimi K3 specialization is `hidden=3584`,
`intermediate=3072`, `112` local experts, `EP=8`, four pipeline stages, and
`192x128x128` compute tiles.  It uses `SiTU`, post-down route weights, exact
source activations, prepared gate/up tensors, direct remote input-gradient
stores, inline weight gradients, and route gradients.

## Verified milestone

The source header for the verified specialization has SHA-256
`08c8baebfca47b0dc7b7b19a9e70fb7c0b4bd84a46546354468172b71ad469e5`.
The receipt below is EP=8, no context parallelism, true variable-length input,
and 4K tokens per rank on B300 GPUs.

| Phase | Native/DeepEPv2 | MegaMoE | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 8.695 ms | 5.042 ms | 1.724x |
| Backward | 33.732 ms | 20.046 ms | 1.683x |
| Forward + backward | 42.116 ms | 24.014 ms | 1.754x |

| Numeric output | Cosine similarity |
| --- | ---: |
| Forward output | 0.9999964444 |
| Input gradient | 0.9999931863 |
| Route-weight gradient | approximately 1.0 |
| W1 gradient | 0.9999999981 |
| W2 gradient | 0.9999999987 |
| W3 gradient | 0.9999999979 |

Peak allocated memory for the continuous forward-plus-backward measurement was
34,905,634,304 bytes for MegaMoE and 35,113,067,520 bytes for native/DeepEPv2,
so this milestone has no peak-memory regression.  A separate 8K-per-rank
receipt measured 1.828x forward, 1.223x backward, and 1.389x continuous
forward-plus-backward speedup.

## Validation and tuning rules

Generated specializations must be compiled with DeepGEMM's C++20 JIT contract.
GPU validation must compare the continuous end-to-end path as well as isolated
forward and backward phases, check every gradient cosine, exercise nonuniform
per-rank token counts, and record peak allocated and reserved memory.  Any new
optimization is developed on a separate branch and is promoted only after it
beats this milestone without violating numerical or memory requirements.
