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

### Terminal wgrad

For the protected EP=8 specialization, terminal dW2 and dW13 use a BF16
grouped wgrad body with:

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

### Two-range early-dW2 overlap

For an exact two-range backward whose first rank-uniform range capacity is at
most 65536 tokens, dW2 starts as soon as each two-CTA cluster retires its own
W13 stream. The cluster invalidates only its retired parent barriers, releases
the parent TMEM allocation, restores the BF16 grouped-GEMM register budget,
and joins the existing dynamic two-segment dW2 queue. Other clusters may still
be executing W13, so dW2 UMMA/TMA work overlaps their remaining dgrad and
remote-gradient publication without a grid-wide barrier or a second kernel.

The early dW2 body retains its BF16 SMEM/TMEM resources for terminal dW13.
Consequently, the terminal suffix skips its original dW2 body, reuses the
retained descriptor-compatible allocation for dW13, and does not assign extra
combine threads that would conflict with the early overlap roles. Four-task
claims preserve expert boundaries and the two-range provider publishes the
same rank-wide completion edge as the protected terminal schedule.

This specialization is compile-time separated from the non-early and
three-range paths.
It does not add a tensor, symmetric-memory region, host collective, or Python
launch. Promotion requires the same six-cosine, robust latency, generated-code
identity, and isolated-process peak-memory gates as every other specialization.

The generated two-range control (`e=false`) cubin was also compared offline
with the corresponding cubin built from the protected `6de36dd94839` source
(device-header SHA-256 `957ee39f75614a`). Both disassemblies contain 27,065
lines and differ in only four immediates that encode shifted device-assert
source-line numbers. After those four diagnostic constants are normalized,
the SASS is byte-identical (SHA-256
`fde5df478e9a82d57d1c17a65144318737fce1d337f6a86f50620f7d13629ebe`).
This control audit proves that disabling the new compile-time branch preserves
that two-range body; it does not prove the distinct three-range 64K path and
does not replace its queued source-bound GPU latency, numerical, and
fresh-memory replay.
`cuobjdump --dump-resource-usage` also reports 64 registers and 1,024 bytes of
static shared memory for both variants. The early variant uses a 344-byte stack
versus 392 bytes for the non-early variant, so the specialization adds no
static register or shared-memory pressure. Absolute PyTorch peak allocation is
still checked independently in fresh backend processes.

## Variable-length scheduling

Sequence lengths are per EP rank and need not match. For a requested average
length `L`, the benchmark uses eight distinct local lengths with a 2/3-to-4/3
spread. At 64K/rank the exact vector is:

```text
[43691, 49933, 56174, 62416, 68656, 74898, 81139, 87381]
```

`K3BackwardRangeSet` records active tokens, symmetric capacity, pool rows,
activation rows, scale rows, and per-expert prefixes. Empty suffix ranges do
not skip the launch or a barrier. With the 32K range cap, the current 32K gate
uses two rank-uniform ranges and the current 64K gate uses three because the
largest true-varlen rank has 87,381 tokens. The physical symmetric workspace
is sized to the largest range rather than the full sequence, so active length,
range count, and workspace capacity remain separate receipt fields.

## Resource and memory contract

The protected source has no additional symmetric-memory region or full-sized
intermediate tensor. The non-early parent specialization compiles with 64
registers and 1,024 bytes of static shared memory; exact stack, spill, and
barrier counts are recorded with compiler and CUTLASS provenance because they
can change when those toolchain inputs change.

Promotion requires fresh-process memory measurements for native and candidate
arms. Same-process allocator peaks are diagnostic only. A candidate fails if
its peak memory meaningfully exceeds native or if it introduces a new
full-parameter, token-pool, or symmetric-memory allocation.

Each behavior-changing specialization promotion replay therefore has three
torchrun invocations: one lockstep native/candidate invocation for numerics and
phase latency, then one native-only and one candidate-only invocation. Each
isolated backend invocation measures forward, backward, and continuous
forward+backward using `torch.cuda.max_memory_allocated`. The resulting six
phase-arm peak records form three native/candidate comparisons. Device-wide
NVML sampling is not part of this receipt, so the report makes no claim that it
is. A 64-MiB allocator ceiling covers caching-allocator bookkeeping noise; it
is not permission for a new persistent buffer. The source-level arena/alias
audit separately rejects a new symmetric allocation or full-sized temporary.
The current-source 4K--16K one-range sweep is a compatibility replay: the new
body is compile-disabled there, so its same-process peak fields are diagnostic
and the source contract proves that no new allocation path is selected.

## Numerical and MFU accounting

The source-bound validation matrix uses EP=8, no context parallelism, and
deterministic unequal rank lengths at 4K, 8K, 16K, 32K, 64K, 128K, 256K, and
512K tokens per rank. Every result must bind the benchmark, Python adapter,
DeepGEMM extension, device header, and host generator hashes. It fails closed
unless all of the following hold at every length:

- the rank-length vector has eight unequal entries whose sum is `8 * L`;
- the exact clustered backend and verified wgrad specialization both execute;
- output, `dX`, `dRoute`, `dW1`, `dW2`, and `dW3` contain no nonfinite value;
- every tensor's FP64 cosine is strictly greater than `0.99999`; and
- the phase peak denominators in the completed receipt are unchanged.

Let `L` be source tokens per EP rank. Logical useful routed-expert Forward
FLOPs on one GPU are:

```text
F = L * topk * 6 * hidden * intermediate
```

The factor six is three expert matrix multiplications and two FLOPs per FMA.
The comparison counts identical logical model work for native and candidate;
extra correction UMMAs in the exact candidate are physical implementation
work and are not counted as additional model FLOPs.

On the eight-GPU B300 system, the source-bound receipt uses 4.5 PFLOP/s for
the mixed MXFP8 Forward and dgrad roofs and 2.25 PFLOP/s for logical BF16
wgrad. Phase MFU is derived from additive ideal times:

```text
ideal_forward_ms  = F / 4.5e15 * 1e3
ideal_backward_ms = (F / 4.5e15 + F / 2.25e15) * 1e3
ideal_fwd_bwd_ms  = ideal_forward_ms + ideal_backward_ms
MFU               = ideal_phase_ms / measured_phase_ms
```

Achieved TFLOP/s uses `F`, `2F`, and `3F` for Forward, Backward, and continuous
Forward+Backward respectively. Backward and combined MFU must not be computed
by averaging component percentages or by choosing one peak for the whole
mixed-precision phase.

Kimi-K3 has 93 transformer layers: layer zero is dense and the remaining 92
are MoE layers. The routed-MoE-only full-model projections therefore use:

```text
AC off:  92 * measured_continuous_forward_backward
full AC: 92 * (measured_isolated_forward
                + measured_continuous_forward_backward)
```

The full-AC expression includes the initial Forward and the checkpoint
recompute Forward before Backward. It excludes router, shared experts, latent
projections, attention/KDA, norms, optimizer, and pipeline bubbles. The 512K
point is a kernel scalability test beyond Kimi-K3's configured 256K maximum
sequence length, so it is not a deployable full-model projection.

The FireTitan production selector currently chooses this clustered schedule
through the 64K/rank true-varlen envelope. The source-bound 128K validation
explicitly forces the same kernel with a 64K range capacity, so the largest
174,762-token rank fits the validated three-range schedule. The 256K and 512K
rows compose two and four such GPU segments respectively. These rows test
numerical and utilization scaling; they are not claims about production
selector behavior until that policy is expanded and retested.

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
same-revision replay qualifies it. A current-source one-range replay for 4K,
8K, and 16K is queued for exactly that qualification; pending output is not
treated as evidence.

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

The retained long-sequence GPU observations are:

| Seq/rank | Measurement | Phase | Native/DeepEPv2 | Candidate | Speedup |
|---:|:---|:---|---:|---:|---:|
| 128K | Direct | FWD | 221.598 ms | 99.320 ms | 2.231x |
| 128K | Direct | BWD | 395.732 ms | 325.285 ms | 1.217x |
| 128K | Direct | FWD+BWD | 612.105 ms | 431.342 ms | 1.419x |
| 256K | Segmented GPU estimate | FWD | 445.081 ms | 200.044 ms | 2.225x |
| 256K | Segmented GPU estimate | BWD | 789.877 ms | 654.100 ms | 1.208x |
| 256K | Segmented GPU estimate | FWD+BWD | 1225.944 ms | 864.662 ms | 1.418x |

The 128K values are from the faster of two direct, numerically passing EP=8
GPU replays. The 256K values are retained as a segmented GPU estimate, not a
direct measurement. A current-source direct 128K replay also passed all six
cosines with zero nonfinites and measured 224.573/97.489 ms Forward,
432.577/330.620 ms Backward, and 614.959/429.946 ms continuous
Forward+Backward. The candidate medians remain within 1.7% of the retained
candidate latency high-water. The native Backward median contains early timing
outliers, so that replay does not replace the frozen projection baseline with
its larger apparent 1.308x speedup. There is no qualified direct 256K or 512K
high-water; both remain explicitly labeled as segmented or projected.

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

The replay-specific validators fail closed unless the combined receipt proves
the protected source hashes, exact EP=8 true-variable-length vector, expected
one-, two-, or three-range selector, all six finite cosines, phase targets, and
per-phase high-water ceilings. They then bind the native-only and
candidate-only process receipts and compare the three phase peaks. Their
machine-readable output is the promotion decision; file timestamps and
isolated faster-looking samples are not.

## Profiling policy

NCU is run only after a numerically passing replay. The profiler must target
the generated `sm100_fp8_fp4_mega_moe_backward_wave_impl` specialization, not
an unrelated setup or elementwise kernel. Because the persistent kernel has
mandatory inter-rank progress, the launcher follows DeepGEMM's
[official MegaMoE NCU runner](https://github.com/deepseek-ai/DeepGEMM/blob/main/scripts/run_ncu_mega_moe.sh):
one NCU process per EP rank, TCP communicator lockstep, and application replay.
Multi-rank replay that ends with an NCU error is not evidence and must fail
closed.

The earlier diagnostic PM capture is explicitly excluded from promotion
evidence. It selected the first matching launch, which was the reverse-order
short tail; ranks 3, 4, and 6 had no compute or memory PM samples, and the
reported tensor-pipe activity was zero on the remaining ranks. The promotion
capture skips that tail and profiles the first full 32K range. It runs the
profiling containers with the minimum `SYS_ADMIN` capability required for
performance counters and fails closed unless every EP rank reports positive
SM, tensor-pipe, tensor-memory, TMA, DRAM, and L2 activity.

The `K3PmCompute` section collects duration, SM-active cycles, tensor-pipe and
tensor-memory activity, TMA activity, DRAM throughput, and L2 throughput. It
deliberately does not claim TMA stall, barrier-stall, active-warp, or atomic
metrics that are absent from that section. After the synchronized topology is
proven, targeted sections may add those metrics in separate controlled passes.
SASS is retained beside the generated CUDA and cubin. Tuning then changes one
measured bottleneck at a time while preserving the symmetric-memory,
quantization, TMA/UMMA, and fixed-order reduction contracts above. See also DeepGEMM's
[MegaMoE design overview](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md).

## Validation sequence

1. Verify source hashes and run CPU scheduler/lifetime contracts.
2. Compile the exact SM103a specialization and retain ptxas resources.
3. Run EP=8 64K numerics, latency, and fresh-process memory from one revision.
4. Sweep 4K, 8K, 16K, 32K, and 64K per rank without CP, binding the current
   source revision even for compile-disabled one-range compatibility buckets.
5. Validate direct 128K latency and all six gradients with the three-range
   schedule. Project 256K and 512K only from explicitly labeled two- and
   four-segment GPU composition, while preserving the retained 128K direct and
   256K segmented-GPU speedup high-water values.
6. Check fresh-process phase peaks for no memory regression.
7. Profile the accepted 32K and 64K specializations with NCU and inspect SASS.
8. Commit and push a milestone only after all applicable gates pass.
9. Train the shrink Kimi-K3 model for 20 identical-input steps and compare the
   complete loss curves.
