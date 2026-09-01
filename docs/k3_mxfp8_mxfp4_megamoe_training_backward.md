# Kimi-K3 MXFP8/MXFP4 MegaMoE training backward

## Status and scope

This report defines the Kimi-K3 full-parameter training specialization built
on DeepGEMM MegaMoE. It is not an inference-only approximation and it is not a
host-orchestrated sequence of collectives. Dispatch, reverse dispatch, expert
math, direct remote gradient publication, top-k reduction, and weight-gradient
work remain inside persistent CUDA kernels using symmetric memory, TMA, and
UMMA.

The protected source baseline is DeepGEMM commit `acef0aa4440cac0867781fbdb852401ec2568031`.
Performance and numerical evidence are recorded separately in
`docs/k3_ep8_performance_highwater.json`. Optimization branches must start from
that source and must not replace it with a slower rollback merely because the
rollback has a newer timestamp or a more recent isolated test.

The exact production shape covered by the specialization is:

| Property | Value |
|---|---:|
| hidden | 3584 |
| intermediate | 3072 |
| global experts | 896 |
| local experts at EP=8 | 112 |
| top-k | 16 |
| activation | SiTU, `beta=4`, `linear_beta=25` |
| route-weight mode | post-down |
| reduction order | fixed top-k slot order |
| weight storage | native group-128 MXFP4 |
| wgrad tensor-core body | BF16 TMA/UMMA with FP32 accumulation |
| target GPUs | SM100/SM103, 148 SMs |

The implementation follows the persistent communication/computation structure
of DeepGEMM's official MegaMoE implementation. Kimi-K3-specific changes are
limited to the native training arithmetic, saved-state contract, backward
pipeline, and exact variable-length scheduling.

## Mathematical contract

The forward path aligns with FireTitan `native_mxfp4`, including its observable
BF16 boundaries:

```text
gate = BF16(Q128(x) @ MXFP4(W1))
up   = BF16(Q128(x) @ MXFP4(W3))
h    = BF16(4*tanh(gate/4)*sigmoid(gate) * 25*tanh(up/25))
down = BF16(Q128(h) @ MXFP4(W2))
y    = BF16(sum_topk(FP32(route_weight) * FP32(down)))
```

Consequently:

- route weights are applied after W2, never before L2 quantization;
- activation quantization uses the native group-128 contract;
- SiTU uses the production constants rather than inference fast-math defaults;
- gate, up, unweighted down, exact source X, and routing metadata are retained
  at the same logical boundaries required by native autograd; and
- the final combine and router-gradient reductions use deterministic top-k
  slot order.

Backward returns five gradients: `dX`, `dRoute`, `dW1`, `dW2`, and `dW3`.
Together with forward output, all six tensors must have FP64 cosine strictly
greater than `0.99999`, and neither the reference nor candidate may contain a
nonfinite value.

## Symmetric-memory ownership

Forward establishes one rank-uniform symmetric buffer. Backward reuses its
live and retired regions rather than allocating a second communication arena:

- saved expert-pool metadata drives reverse dispatch;
- symmetric grad-y and direct remote grad-X planes carry cross-rank payloads;
- retired weight-tile state stores dynamic scheduler cursors and mailboxes;
- consumed saved-down and gate/up arenas provide descriptor-compatible
  scratch or final weight-gradient storage; and
- dead activation-ring regions back compile-dead descriptor operands where
  the common ABI still requires a TensorMap.

All aliases are bounded and phase ordered. No device global, process-global
mode flag, or host-side collective is introduced. Every EP rank launches the
same grid and participates in the same barrier generations, including ranks
whose variable-length suffix is empty.

## Persistent backward pipeline

The selected kernel has these logical phases:

```text
publish grad-y
  -> reverse dispatch through symmetric memory
  -> W2 dgrad UMMA/TMA
  -> exact SiTU derivative and dRoute term
  -> W13 dgrad UMMA/TMA
  -> direct remote dX publication and fixed-order reduction
  -> dynamic dW2 grouped UMMA/TMA
  -> dynamic dW13 grouped UMMA/TMA
```

These are roles inside one persistent launch, not independently synchronized
Python operations. Otherwise-idle warps publish remote-gradient readiness and
perform fixed-top-k combine work while tensor-core roles execute dgrad or
wgrad. Communication completion is therefore overlapped with useful UMMA/TMA
work at kernel level.

### Forward and dgrad geometry

Forward and dgrad use two-CTA clusters and `cta_group::2` TMA/UMMA. A tiles are
multicast within the cluster, B tiles remain CTA-local, FP32 accumulators live
in TMEM, and epilogues publish BF16 values into symmetric-memory rings or
direct remote output planes.

### One-range terminal wgrad

For the protected EP=8 one-range specialization, terminal dW2 and dW13 use a
BF16 grouped wgrad body with:

- tile shape `128 x 256 x 64`;
- six mainloop stages;
- paired-N tasks, giving 84 dW2 and 168 dW13 tasks per expert;
- four-task dynamic claims that never cross an expert boundary; and
- one retained SMEM/TMEM lifecycle across the dW2-to-dW13 descriptor switch.

The producer issues a complete six-stage B1 window before refilling the next
A+B0 window. The consumer uses matching window-major order:

```text
UMMA0(k..k+5) -> UMMA1(k..k+5)
```

This preserves increasing K order for each output while exposing TMA progress
across the resident window. Each cluster may enter dW13 after its claimed dW2
work retires, so late dW2 clusters and early dW13 clusters overlap without a
host-visible phase boundary.

The grouped wgrad body intentionally uses the established single-CTA grouped
ownership contract internally. Changing it to `cta_group::2` would require a
new grouped scheduler, multicast contract, TMEM ownership scheme, and output
partition; it is not a tuning toggle.

## Variable-length scheduling

Sequence lengths are per EP rank and need not match. For a requested average
length `L`, the benchmark uses eight distinct local lengths with a 2/3-to-4/3
spread. At 64K/rank the exact vector is:

```text
[43691, 49933, 56174, 62416, 68656, 74898, 81139, 87381]
```

`K3BackwardRangeSet` records active tokens, symmetric capacity, pool rows,
activation rows, scale rows, and per-expert prefixes. Empty suffix ranges do
not skip the launch or a barrier. The one-range protected replay uses capacity
131072 and proves that active lengths are independent from physical capacity.

## Resource and memory contract

The protected source has no additional symmetric-memory region or full-sized
intermediate tensor. The one-range parent specialization compiles with 64
registers and 16 barriers; exact spill counts are recorded with compiler and
CUTLASS provenance because they can change when those toolchain inputs change.

Promotion requires fresh-process memory measurements for native and candidate
arms. Same-process allocator peaks are diagnostic only. A candidate fails if
its peak memory meaningfully exceeds native or if it introduces a new
full-parameter, token-pool, or symmetric-memory allocation.

Each promotion run therefore uses six separate processes: both arms for
forward, backward, and continuous forward+backward. The harness records both
PyTorch allocator peaks and 50-ms device-wide NVML samples. The latter include
symmetric allocations that may not be attributed clearly by the caching
allocator. A 64-MiB ceiling covers NVML sampling and allocator bookkeeping
noise; it is not permission for a new persistent buffer.

## Performance high-water and promotion

Performance protection is a per-sequence-length, per-phase Pareto frontier.
Historical 4K--32K rows came from numerically qualified GPU runs on different
revisions; they are immutable regression ceilings, not evidence that one
current revision attains the whole table. A faster phase observation is never
discarded because another revision has better provenance or memory evidence:
latency, source identity, and promotion eligibility remain separate fields in
the machine-readable ledger. The 64K row is additionally bound to the
protected source commit and generated CUDA/CUBIN identities.

The historical 4K log predates the surviving source artifacts and does not
contain same-run source hashes. Commit `e21cc60` plus its seeded CUDA/CUBIN is
retained only as a post-hoc replay seed. The 4K latency remains a required
regression ceiling, but it must not be described as source-bound until a fresh
same-revision replay qualifies it.

| Seq/rank | Phase | Native/DeepEPv2 | Candidate | Speedup |
|---:|:---|---:|---:|---:|
| 4K | FWD | 8.415 ms | 4.756 ms | 1.769x |
| 4K | BWD | 33.616 ms | 16.253 ms | 2.068x |
| 4K | FWD+BWD | 41.537 ms | 20.164 ms | 2.060x |
| 8K | FWD | 15.374 ms | 7.570 ms | 2.031x |
| 8K | BWD | 41.729 ms | 25.516 ms | 1.635x |
| 8K | FWD+BWD | 55.907 ms | 32.219 ms | 1.735x |
| 16K | FWD | 26.232 ms | 12.742 ms | 2.059x |
| 16K | BWD | 61.747 ms | 42.662 ms | 1.447x |
| 16K | FWD+BWD | 87.566 ms | 55.040 ms | 1.591x |
| 32K | FWD | 50.694 ms | 24.297 ms | 2.086x |
| 32K | BWD | 101.208 ms | 83.122 ms | 1.218x |
| 32K | FWD+BWD | 152.996 ms | 106.237 ms | 1.440x |

These are composite phase ceilings. In particular, the 16K BWD and FWD+BWD
observations have same-log six-cosine, memory, generated-source, device-header,
and host-binding hashes, but the exact source was not committed at measurement
time. Its device header identity is
`6c61e6647b7acf8ec508e1bdbee919fc2b3db27ac18fef0e1865646d8dbbed08`.
The faster single-sample 32K BWD observation at 66.939 ms is retained only in
`preliminary_phase_observation_frontier`: it has no matching numeric receipt
and is therefore not a verified or promotable ceiling.

The protected EP=8, no-CP, true-variable-length 64K/rank phase records are:

| Phase | Native/DeepEPv2 | Candidate | Speedup |
|---|---:|---:|---:|
| FWD | 106.712 ms | 46.682 ms | 2.286x |
| BWD | 190.443 ms | 139.209 ms | 1.368x |
| FWD+BWD | 294.289 ms | 186.547 ms | 1.578x |

The three phase records are independent high-waters and need not come from one
timing process. A replacement milestone must provide one immutable revision
identity shared by numerics, latency, and memory evidence. It must satisfy:

1. all six FP64 cosines are greater than `0.99999`;
2. true-variable-length EP=8 is proven at the requested length;
3. forward exceeds 1.5x and backward exceeds 1.36x, or the accepted combined
   target is reached;
4. every protected phase remains within its explicit noise ceiling;
5. fresh-process peak memory does not regress; and
6. generated CUDA and cubin hashes are retained with the benchmark receipt.

Failed or slower candidates remain quarantined experiment branches. They do
not become the next optimization base.

The scripts/validate_k3_ep8_receipts.py validator fails closed unless one
receipt proves the protected source hashes, exact EP=8 true-variable-length
vector, one-range selector, six finite cosines, phase targets, per-phase
high-water ceilings, and all six fresh-process memory comparisons. Its
machine-readable output is the promotion decision; file timestamps and
isolated faster-looking samples are not.

## Profiling policy

NCU is run only after a numerically passing replay. The profiler must target
the generated `sm100_fp8_fp4_mega_moe_backward_wave_impl` specialization, not
an unrelated setup or elementwise kernel. Multi-rank kernel replay that ends
with an NCU error is not evidence and must fail closed.

The first useful profile collects phase duration, UMMA activity, TMA issue and
stall behavior, memory throughput, barrier stalls, active warps, and executed
global atomics. SASS is retained beside the generated CUDA and cubin. Tuning
then changes one measured bottleneck at a time while preserving the symmetric
memory, quantization, TMA/UMMA, and fixed-order reduction contracts above.

## Validation sequence

1. Verify source hashes and run CPU scheduler/lifetime contracts.
2. Compile the exact SM103a specialization and retain ptxas resources.
3. Run EP=8 64K numerics, latency, and fresh-process memory from one revision.
4. Sweep 4K, 8K, 16K, 32K, and 64K per rank without CP.
5. Profile the accepted 64K specialization with NCU and inspect SASS.
6. Validate direct 128K, 256K, and 512K latency/MFU using relaxed long-range
   targets only at those lengths.
7. Commit and push a milestone only after all applicable gates pass.
8. Train the shrink Kimi-K3 model for 20 identical-input steps and compare the
   complete loss curves.
