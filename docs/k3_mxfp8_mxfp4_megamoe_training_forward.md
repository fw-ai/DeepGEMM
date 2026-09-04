# Kimi-K3 Q128 MegaMoE training Forward

## Scope and status

This document describes the Kimi-K3 Forward specialization used by the
experimental `moe=mxfp8_mxfp4_megamoe` full-parameter training path. It keeps
the training operator contract while targeting the latency of inference's
MegaMoE+SiTU Forward. The implementation remains a single persistent
communication-fused CUDA kernel with symmetric memory, TMA operand movement,
clustered UMMA, and in-kernel dispatch/combine.

The current store64/three-epilogue-warpgroup revision is a source-bound
performance and numerical milestone. Its exact EP8 1K--256K Forward sweep,
true-variable-length numerical gates through 128K, 128K peak-memory check, and
Backward regression suite pass. A direct 256K native Backward reference does
not fit the benchmark process, so 256K Backward remains a labeled segmented
GPU estimate rather than a direct release result.

## Numerical contract

For routed token `r`, expert `e`, BF16 input `x`, and FP32 route weight `p`,
the required Forward is:

```text
g_bf16 = BF16(Q128(x) @ MXFP4(W1_e))
u_bf16 = BF16(Q128(x) @ MXFP4(W3_e))

SiTU(g, u) =
    4 * tanh(g / 4) * sigmoid(g)
    * 25 * tanh(u / 25)

h_bf16 = BF16(SiTU(g_bf16, u_bf16))
z_bf16 = BF16(Q128(h_bf16) @ MXFP4(W2_e))
y       = BF16(sum_topk(FP32(p) * FP32(z_bf16)))
```

`Q128` means one MXFP8 E4M3/UE8M0 activation scale for every 128 contiguous
values. W1, W3, and W2 remain packed MXFP4 with independent group-32 weight
scales. Route weights are applied after W2. Replacing the weights with FP8,
using group-32 activation scaling, or moving route weighting before W2 would
define a different operator and is not permitted.

The optimized SiTU path preserves the mandatory BF16 boundary before W2
quantization. It uses `__tanhf` for K3's two tanh evaluations and derives
`sigmoid(g)` from the already-computed `tanh(g / 4)` with the double-angle
identity. The specialization is compile-time gated by all of the following:

- Q128 L2 activation scaling;
- post-W2 route weighting;
- hidden/intermediate dimensions 3584/3072;
- 896 global experts, top-k 16, and production EP8 (with EP4 enabled only for
  development measurements); and
- SiTU beta/linear-beta 4/25.

All unrelated and Q32 instantiations retain the generic arithmetic path.
Fast intrinsics are accepted only if Forward output, `dX`, `dRoute`, `dW1`,
`dW2`, and `dW3` each have FP64 cosine strictly greater than `0.99999` under
true-variable-length EP8 replay.

## Persistent pipeline

MegaMoE retains the upstream persistent schedule documented by DeepGEMM:

```text
symmetric input registration
        |
        v
dispatch/count/pull -----> clustered W13 TMA + 2-CTA UMMA
        |                              |
        |                              v
        |                     SiTU + BF16 boundary
        |                              |
        |                              v
        +--------------------> Q128 L2 quantization
                                       |
                                       v
                              clustered W2 TMA + UMMA
                                       |
                                       v
                         remote top-k planes + combine
```

Adjacent CTAs own the two halves of each 128-value activation-scale group.
Each CTA computes its local absolute maximum and exchanges the reduction
values through transaction-accounted asynchronous cluster stores. The peer
arms a remote `mbarrier` with the exact byte count before those stores, waits
for their completion, and emits one Q128 scale encoded into the four group-32
descriptor bytes consumed by the existing W2 loader. Repeated descriptor
bytes are a physical encoding; they do not change the logical group-128
quantizer.

Training optionally saves block-padded BF16 gate/up rows and the unweighted
BF16 W2 result for Backward. Inference calls the same Forward with those saves
disabled. This makes the no-save training kernel the replacement candidate
for inference without changing the arithmetic contract.

### Inference adapter reuse

The isolated inference-port branch selects the training contract only for the
exact K3 signature: SiTU 4/25, hidden/intermediate 3584/3072, 896 experts,
top-k 16, and EP8. Every other model retains inference's existing group-32,
pre-W2-route path. K3 bypasses the old standalone group-32 activation
quantizer and copies training's single-grid Q128 quantize-and-stage operation
into the inference workspace. The adapter then invokes the same no-save
DeepGEMM specialization with post-W2 route weighting and Q128 L2 scaling.

Inference workspaces are fixed-capacity and may be reused after a smaller
batch. Therefore, the adapter also clears the routing-only padding suffix to
`(-1, 0)`; it does not quantize padded activations or allocate an invocation-
sized FP8 tensor. The authoritative EP8 benchmark measures this actual adapter
arm separately from the raw no-save trainer call, so staging overhead cannot
be hidden by a kernel-only number.

The adapter's CPU wiring smoke passes in the production serving image. A B300
GPU semantic test also confirms that its FP8 payloads and packed scale words
match a Q128 reference exactly, including active-token masking and padding-row
sentinels. This is staging-only evidence; end-to-end promotion still depends
on the full EP8 A/B and six-cosine gates below.

## Store64 and three epilogue warpgroups

The general prefill configuration uses `BLOCK_M=192` with two epilogue
warpgroups. The K3 Q128 specialization instead uses three 128-thread epilogue
warpgroups and `STORE_BLOCK_M=64`. Together they cover all 192 rows while
increasing parallelism in SiTU, cluster Q128 reduction, L2 quantization, W2
store, and combine. This retains the existing shared-memory footprint class:
the smaller per-warpgroup store tile compensates for the additional
warpgroup.

Dynamic register redistribution is part of the correctness contract. A first
104-register epilogue experiment retained the generic 96/88-register budgets
for dispatch/non-epilogue roles and could wait indefinitely in
`setmaxnreg.inc`. The accepted candidate explicitly assigns:

| Role | Threads | Registers/thread | Register budget |
|---|---:|---:|---:|
| Dispatch | 128 | 88 | 11,264 |
| Non-epilogue TMA/UMMA | 128 | 80 | 10,240 |
| Three epilogue warpgroups | 384 | 104 | 39,936 |
| **Total** | **640** | | **61,440** |

The total is below the 64,512-register launch bound, and both non-epilogue
roles release registers before an epilogue warpgroup requests its allocation.
The direct Forward specialization compiles with 96 static registers, 16
barriers, a 64-byte stack frame, and zero spill stores/loads in the measured
EP4 canary. The training-save instantiation still has separate state-capture
cost and is not used by inference.

## EP8 performance evidence

The table below is the authoritative B300 GPU sweep with the full Kimi-K3
expert shape (`H=3584`, `I=3072`, 896 experts, top-k 16), uniform exact tokens
per rank, no context parallelism, and 16 timed samples. The baseline is the
current inference Q32 MegaMoE+SiTU path; the candidate is the actual inference
adapter port of training's Q128 Forward. This is an efficiency comparison:
the two arms intentionally retain their respective Q32 and Q128 arithmetic.

| Seq/rank | **FWD** inference MegaMoE+SiTU | **FWD** training-Q128 port | Speedup | Inference MFU | Q128 MFU |
|---:|---:|---:|---:|---:|---:|
| 1K | 1.847 ms | 1.769 ms | 1.044x | 13.02% | 13.59% |
| 2K | 3.033 ms | 2.370 ms | 1.280x | 15.86% | 20.30% |
| 4K | 4.063 ms | 4.012 ms | 1.013x | 23.68% | 23.98% |
| 8K | 6.983 ms | 6.766 ms | 1.032x | 27.56% | 28.44% |
| 16K | 10.836 ms | 10.684 ms | 1.014x | 35.52% | 36.02% |
| 32K | 20.275 ms | 19.696 ms | 1.029x | 37.96% | 39.08% |
| 64K | 40.078 ms | 39.093 ms | 1.025x | 38.41% | 39.38% |
| 128K | 79.739 ms | 77.507 ms | 1.029x | 38.61% | 39.72% |
| 256K | 158.646 ms | 154.167 ms | 1.029x | 38.81% | 39.94% |

Every point meets or beats inference. The 1K and 4K deltas remain close to
repeat noise and are treated as parity rather than material speedups. The
receipt is bound to this kernel header's SHA-256
`7bcf1f755f3e881f3ec62ef2bc66de49b9aeb783cfb293e8736fa16148a572e4`
and inference-port source SHA-256
`45d6a8402174fbd8cc090848a33cd28c96fdbe7939ea42f577b2e2f3e39c76b2`.

## Backward and memory invariants

The branch materializes the already-verified prepared-gate physical-layout
fixes from the protected runtime; it introduces no new Backward schedule,
selector, symmetric-memory layout, or saved-state ABI change. It must
nevertheless replay the protected Backward milestone because the number and
order of Forward pool rows can affect the consumer indirectly. Promotion
requires:

1. Backward latency at or above the protected high-water at every covered
   length;
2. no increase in fresh-process peak allocated or reserved memory;
3. identical state-capture shapes and metadata semantics; and
4. all six numerical cosines above `0.99999` for unequal-rank inputs,
   including empty-rank and forced-multi-chunk cases.

## Validation and promotion

1. Bind generated code, host sources, the inference comparison source, and
   benchmark to SHA-256 identities.
2. Compile both no-save and save-enabled SM103 specializations and retain
   ptxas resources; reject hangs or spills in the no-save inference candidate.
3. Measure inference and training-direct Forward at EP8 for exact uniform
   1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, and 256K per rank.
4. Run true-variable-length Forward/Backward parity and peak-memory replay on
   the same revision.
5. Use synchronized multi-rank NCU replay only after numerical parity passes;
   optimize one measured bottleneck at a time.
6. Commit and push only when the complete milestone passes. Slower or invalid
   candidates remain isolated and never replace the protected high-water.

## References

- [DeepGEMM MegaMoE design and NCU workflow](https://github.com/deepseek-ai/DeepGEMM)
- [Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
