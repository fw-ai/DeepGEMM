# Mega-MoE prefill: cycle-chunked dispatch pull

## Status: WORKING (mxfp4 + nvfp4, 1-rank + multi-rank EP)

The packed-FP4 mega-MoE kernel now supports prefill (large seqlen) with a token ring
of `ratio × max_seq_len` instead of `num_ranks × max_seq_len` (the `num_min` that forced
a seqlen cliff ~98K/8 ranks, ~49K/16 ranks). The dispatch PULL is split into
`num_cycles = ceil(total_pool_blocks / kNumRingBlocks)` rounds; the GEMM follows via the
ring full/empty counts. Verified correct (`diff < 0.05`) and benchmarked.

Branch: `yingz/deepgemm-mem` in `~/DeepGEMM-worktrees/yingz-deepgemm-mem` (fw-ai/DeepGEMM).

## Design (final)

**The cycle loop lives in the PULL only.** The GEMM roles run the ORIGINAL single-cycle
`for_each_block` (`cap_blocks = 0`) and process all pool blocks continuously, gated by the
ring full/empty counts — exactly the original ring-wrap pipeline. The pull's own cycle loop
(`cycle_end_pool` bound) + a cross-CTA `grid_sync` at each cycle boundary keeps the global
ring counts consistent across CTAs.

```
# host: chunk_ratio (default None = original). When set:
#   num_ring_tokens = align(ceil(chunk_ratio * num_max_tokens_per_rank), 384)
#   clamped to num_max only (NOT num_min) -> permits ring < num_min.
# num_cycles = (!kChunking || total_pool_blocks==0) ? 1 : ceil(total_pool_blocks/kNumRingBlocks)
#   -> big-enough ratio (ring >= num_min) runs exactly 1 cycle (no wasteful empty tail cycles).
for cycle in range(num_cycles):                      # pull only
    for token in this cycle's pool-block range:      # cycle_end_pool bound
        wait ring slot empty (empty_count >= cycle drain target)
        a2a-pull token -> ring slot
    grid_sync (cross-CTA, dispatch warps)            # all CTAs' pulls at the cycle boundary
# GEMM: original for_each_block (all pool blocks), gated by full_count per block.
# Combine reduce: unchanged (post-cycle-loop).
```

**Why not a per-role GEMM cycle loop + shared cycle barrier?** The 5 GEMM roles are
pipelined (A-load → B-load → MMA → 4th → epilogue), each processing the same blocks in
sequence, so A-load is always AHEAD of the epilogue. They reach the cycle boundary at
DIFFERENT cycles. A single mbarrier (counted-arrival) can't sync them — it counts arrivals
across cycles, so a fast role's cycle-N+1 arrival completes a slow role's cycle-N round,
desyncing them (confirmed by printf: the epilogue reached cycle 1 while A-load was still at
cycle 0's mbarrier). The ring full/empty counts already sync the pipelined roles (A-load
waits for `full_count`, the pull waits for `empty_count`), so no per-role cycle barrier is
needed — the GEMM just follows the pull cycle-by-cycle via the counts.

## Key fixes (in order of discovery)

1. **Scheduler `block_idx` carry (first deadlock)** — the persistent stripe counter
   `block_idx` carried a large leftover across the cycle boundary, so
   `m_block_idx = block_idx / kNumL1BlockNs` was a bogus large value for cycle 1's first
   expert, and the cycle-end check `current_pool_block_offset + m_block_idx >= cycle_pool_block_end`
   fired immediately — the GEMM did NO work for cycle >= 1 (the pull filled the ring but
   nothing consumed it). Fixed by resetting `block_idx = blockIdx.x` per cycle AND adding a
   `current_pool_block_offset >= cycle_pool_block_end` guard so the consume-walk stops at the
   cycle boundary instead of overshooting. (This was for the GEMM-cycle-loop approach, which
   was later abandoned for the pull-only approach — but the scheduler fetch logic is still
   cycle-aware for safety.)

2. **Per-rank counts not reloaded across cycles (the REAL root cause)** — `stored_rank_count`
   was only reloaded when the expert changed within a for-loop iteration
   (`old_expert_idx != current_expert_idx`). CTAs that pulled 0 tokens in a cycle (broke at
   the cycle bound before loading) carried stale/zero counts into the next cycle. With
   `num_active_ranks = 0`, the round-robin rank-selection `while (true)` loop spun forever
   (no progress) — a TRUE hang (no grid-sync timeout). Found by printf-isolating which CTAs
   reached the cycle-1 grid_sync: CTAs 132-147 (the highest-numbered, whose token stripes
   land in cycle 1) never reached it. Fixed by tracking `counts_loaded_expert` (the expert
   the cached counts are loaded for) and reloading whenever
   `current_expert_idx != counts_loaded_expert` — even across cycle boundaries.

3. **Cross-CTA desync (grid_sync at the cycle boundary)** — without a cross-CTA sync, CTAs
   desync across cycles (one CTA's cycle-N pull waits for another CTA's cycle-(N-1) GEMM
   drain that never completes because that CTA raced ahead). Fixed with a
   `comm::grid_sync<kNumSMs, kCycleGridSyncIndex=2>` at each cycle boundary, called by the
   dispatch warps (all present at the pull's cycle end, so `sync_aligned(kNumDispatchThreads)`
   is safe). The grid_sync counter (idx 2) alternates 0 ↔ kFinishSumTag mod 2³² across calls,
   so repeated calls work without reset. For multi-rank, the ring counts are PER-RANK, so no
   cross-RANK cycle barrier is needed — only the `num_cycles` all-reduce (cross-rank) ensures
   all ranks run the same cycle count.

## Files (on `yingz/deepgemm-mem`)

- `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_mxfp4_mega_moe.cuh` — `kChunking` constexpr;
  pull's `for (cycle)` loop with `cycle_end_pool` bound + cross-CTA `grid_sync` per cycle;
  `counts_loaded_expert` fix; `num_cycles` compute + cross-rank all-reduce + printf.
- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` — `kNumThreads` template param;
  cycle-aware fetch (no-op when `cap_blocks = 0`). (The cycle_barrier mbarrier machinery is
  unused — the GEMM uses `cap_blocks = 0`.)
- `csrc/jit_kernels/heuristics/mega_moe.hpp` — wave heuristic relaxation gated on
  `ring >= num_min` (force `num_experts_per_wave = 1` when chunking).
- `csrc/jit_kernels/impls/sm100_mxfp4_mxfp4_mega_moe.hpp` — `smem_barriers +1` (cycle_barrier
  mbarrier counted; unused now but kept).
- `csrc/apis/mega.hpp` — relaxed `num_min <= num_ring_tokens` assert; `num_cycles_max`
  workspace slot.
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` — `get_num_cycles_max_ptr` (all-reduce slot).
- `deep_gemm/mega/__init__.py` — `chunk_ratio` param; `num_ring_tokens` sized per-ratio.
- `tests/test_fp4_mega_moe.py` — `PREFILL_1RANK` + `PREFILL_MULTIRANK` shape matrices; `--scope prefill`.
- `benchmarks/bench_packed_fp4.py` — `bench_mega_prefill` (chunk_ratio sweep: 1.3/1.5/2.0/baseline).
- NOT yet applied to `sm100_fp8_fp4_mega_moe.cuh` (mirror is a follow-up).

## Perf analysis (num_cycles=1 / chunking overhead)

The cycle `grid_sync` overhead is **negligible (~0%)**. Measured on the large prefill shape
(EP4, E=512, topk=16, M=16384, h=inter=4096, `total_recv`≈262K tokens/rank):

| ratio | ring | nc | lat us | wraps? |
|-------|------|----|--------|--------|
| 1.3 | 21504 | 13 | 10139 | yes (cycle wrap) |
| 1.5 | 24960 | 11 | 10144 | yes |
| 2.0 | 33024 | 9 | 10141 | yes |
| 4.0 | 66048 | 1 | 10136 | yes (ring < total_recv, 4× wrap) |
| 16 | 264192 | 1 | 9894 | **no** (ring ≥ total_recv) |
| baseline (None) | 786432 | 1 | 9898 | no |

Key observations:
- **ratio 16 (ring=264K ≥ total_recv=262K, no wrap) ≈ baseline** (9894 vs 9898 us). No regression
  when the ring doesn't wrap.
- **ratio 1.3 (13 cycles + 13 grid_syncs) ≈ ratio 4.0 (1 cycle, 0 grid_syncs, 4 wraps)** (10139
  vs 10136 us). So the grid_sync/cycle overhead is ~0%; the 2.4% gap vs baseline is entirely the
  **ring-wrap stall** (the pull waits for `empty_count` at each wrap when the ring is smaller than
  `total_recv`).
- The grid_sync is **required on every cycle** (incl. the last): between cycles it prevents
  cross-CTA desync deadlock (world≥4); on the last cycle it prevents the post-loop cleanup
  (which zeroes the ring counts) from racing with a slow CTA's still-running pull. Skipping it
  on the last cycle was tested and deadlocks EP4.

**Takeaway:** there is no chunking regression. The 2.4% is the ring-wrap stall from choosing a
ring smaller than `total_recv` (= M×topk worst case). To avoid it, use `chunk_ratio ≥ topk`
(ring ≥ M×topk, no wrap) — ratio 16 matches the un-chunked baseline. For `chunk_ratio < topk`,
the ring wraps, which is the memory/perf trade-off (smaller ring = less sym-buffer + wrap stall).
The `grid_sync` itself is free.

## Verification

- `python tests/test_fp4_mega_moe.py --scope baseline` — num_cycles=1, mxfp4 0.00075 / nvfp4 0.00058 (unchanged).
- `python tests/test_fp4_mega_moe.py --scope 1rank` — all 1-rank cases pass (num_cycles=1).
- `python tests/test_fp4_mega_moe.py --scope prefill --worlds 2,4,8` sweeps
  `chunk_ratio` across `[0.5, 1, 1.5, 2, 5]` for 3 1-rank shapes + EP2/EP4/EP8 (60 cases
  total, both formats). Captures the kernel's `num_cycles` printf and asserts:
  `ring >= num_min` → `num_cycles == 1` (big-enough ratio); `ring < num_min && ratio <= 1`
  → `num_cycles >= 2` (chunking happens); otherwise `num_cycles >= 1`. All pass
  (mxfp4 0.00079 / nvfp4 0.00064). Sample `num_cycles`: ratio 0.5 → 3-8, ratio 1.0 → 1-4,
  ratio 5.0 → 1.
- `python benchmarks/bench_packed_fp4.py --scope prefill --world 8 --tokens 1024`:
  ```
  ratio    ring  num_min |   lat us  peak GB
   1.3    1536     8192 |    226.9     7.68
   1.5    1920     8192 |    235.1     7.68
   2.0    2304     8192 |    223.6     7.68
   8.0    9216     8192 |    192.8     7.68   (baseline, ring >= num_min, no chunking)
  ```
  Cycle chunking costs ~15-20% latency at ratio 1.3 vs the un-chunked baseline, for the
  memory savings (ring = ratio×tok instead of num_ranks×tok). At this shape the peak device
  memory is dominated by the sym_buffer's a2a input buffers + combine_token_buffer (which
  don't scale with the ring), so the ring savings are a small fraction of the total; the
  ring size column shows the chunked allocation directly.

## Debugging technique (lesson)

The deadlock was isolated by printf: gate debug prints on `sm_idx` (one CTA) + `lane_idx == 0`
+ the cycle index, and print before/after each mbarrier/pipeline acquire/release. The CUDA
printf fifo is ~1MB by default — bump it via `cudart.cudaDeviceSetLimit(1, 64*1024*1024)`
(`cudaLimitPrintfFifoSize = 1`, NOT 5) before the kernel. Print only the relevant subset
(e.g. `sm_idx >= 130` for the stuck CTAs) to avoid flooding. This pinpointed (a) the GEMM
doing no work in cycle 1 (scheduler `block_idx` carry), (b) the mbarrier mixing arrivals
across cycles (per-role cycle loop is wrong), and (c) CTAs 132-147 stuck in the round-robin
(per-rank counts not reloaded).

## Remaining / follow-up

- Mirror the chunking changes to `sm100_fp8_fp4_mega_moe.cuh` (FP8xFP4 kernel).
- Clean up the now-unused `cycle_barrier` mbarrier machinery in the scheduler + SharedStorage
  + `smem_barriers +1` (the GEMM uses `cap_blocks = 0`, so the cycle_barrier is never called).
- Right-size the SF ring (currently `16×` the data pool for `block_m=8` worst-case; prefill
  uses `block_m=192`, so it's 24× oversized) — a separate memory follow-up.
- Chunking the `combine_token_buffer` (push-direct `y_acc` or pull-mode) for very large
  seqlen — explicitly out of scope here (kept as-is; scales with seqlen, no `num_ranks` factor).
