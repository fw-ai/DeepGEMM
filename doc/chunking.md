# Mega-MoE dispatch cycle chunking — design, progress, and resume guide

## Goal

Enable prefill (large seqlen) for the packed-FP4 mega-MoE kernel by chunking the
dispatch so the per-rank token ring can be `ratio × max_seq_len` instead of
`num_ranks × max_seq_len` (the `num_min` that forces a seqlen cliff ~98K/8 ranks,
~49K/16 ranks). Add a cycle counter logged at kernel end. Verify correctness for
`num_cycles > 1` and benchmark perf under different `chunk_ratio` values.

Branch: `yingz/deepgemm-mem` in `~/DeepGEMM-worktrees/yingz-deepgemm-mem`.
Approved plan: `/home/ipiszy/.claude/plans/lively-noodling-flurry.md`.

## Memory formulas (verified from `csrc/apis/mega.hpp`)

- Ring (L1+L2 token+SF): `num_ring_tokens × (hidden+inter)×0.5` + SFs;
  `num_sf_ring_tokens = 16 × num_ring_tokens` (worst-case `block_m=8` padding),
  so SF pools ≈ data pools (NVFP4 SFs 2×).
- `combine_token_buffer` (BF16): `num_topk × num_max_tokens × hidden × 2` —
  **NO `num_ranks` factor** (per-rank, the token-owner's slots). Kept as-is.
- Dispatch src idx pool: `E × num_ranks × tokens × 4` (linear in `num_ranks`).
- `num_min_ring_tokens = num_ranks × num_max_tokens_per_rank` (one expert's
  worst-case). The seqlen cliff: `num_min ≤ budget` (786K for prefill).

## Commits (all on `yingz/deepgemm-mem`)

- `4bb0397` — cycle-loop scaffolding (host `chunk_ratio` + relaxed assert, scheduler
  cycle-bounds + `cycle_barrier()`, `for_each_block_impl` internal cycle loop, unified
  cycle pull loop, gated wave-heuristic relaxation, `num_cycles` all-reduce + printf).
  `kChunking=false` verified (baseline + 1-rank 16 cases).
- `0eec18b` — two crash fixes found via `compute-sanitizer`: (1) smem OOB
  (`Barrier cycle_barrier` not counted in `smem_barriers` → +1); (2) non-leader MMA
  warp's `for_each_block` was leader-only → added no-op `for_each_block` else branch.
- `f13b122` — mbarrier `cycle_barrier` (counted-arrival); cycle 0 works (all 16 warps
  sync); cycle 1+ hangs on TMEM/TMA pipeline.
- (uncommitted) — `for_each_block_single_cycle` + `get_num_cycles`/`get_total_pool_blocks`
  helpers + A-load lambda-as-variable (in-progress restructure to "caller loops over
  cycles with fresh pipeline state").

## What works

- `kChunking=false` (ring ≥ num_min, the original behavior): baseline + full 1-rank
  matrix (16 cases) pass with diffs unchanged (mxfp4 0.00075 / nvfp4 0.00058).
- `kChunking=true` cycle 0: the mbarrier `cycle_barrier` syncs all 16 warps on SM 0
  (verified via `[cb]` printf). L1/L2 ring wrap counts verified consistent (empty-count
  signals are +1 per N-block, so after cycle 0 `l1_empty_count[j] = kNumL1BlockNs`
  matches the cycle-1 target).
- `num_cycles` computation (pool-blocks-based: `ceil(total_pool_blocks / kNumRingBlocks)`)
  + cross-rank all-reduce (gather-based) + printf — all working.
- `compute-sanitizer` is available at `/usr/local/cuda/bin/compute-sanitizer` — NO torch
  rebuild needed. Use `compute-sanitizer --tool memcheck python <probe>` for fault isolation.

## Current blocker: cycle 1+ TMEM/TMA pipeline hang

### Root cause (identified)

The TMEM/TMA pipeline state — `stage_idx`, `phase` (line ~333), `current_iter_idx`
(MMA line ~923, epilogue line ~1092), `accum_phase` — is declared BEFORE
`for_each_block` in each warp-role branch and captured by the `for_each_block` lambda
(`[&]`). The `for_each_block_impl` internal cycle loop reuses the SAME lambda across
cycles, so the pipeline state **carries across the cycle boundary**. The
`full_barriers`/`empty_barriers`/`tmem_full_barriers`/`tmem_empty_barriers` mbarriers
are in whatever phase cycle 0 left them, and cycle 1's first block waits on a barrier
phase that doesn't match → hang.

The `[cb]` printf shows: cycle 0 — all 16 warps arrive at phase 0 (mbarrier releases).
Cycle 1 — only 3 warps reach phase 1; the other 13 are stuck in the `for_each_block`
while-loop (per-block callback), waiting on a TMEM/TMA mbarrier.

### Fix direction (two options)

**Option A (recommended): "caller loops over cycles with fresh pipeline state"**

Instead of `for_each_block_impl`'s internal cycle loop, the CALLER (each warp-role
branch) loops over cycles, resetting the pipeline state between cycles:

```cpp
// In each of the 5 for_each_block call sites (A-load 778, B-load 853, MMA 924,
// 4th 1046, epilogue 1093):
auto my_lambda = [&](const sched::BlockPhase& block_phase, ...) { ... };  // same body
if constexpr (kChunking) {
    const auto total_pb = scheduler.get_total_pool_blocks();
    const auto nc = scheduler.get_num_cycles();
    for (uint32_t cycle = 0; cycle < nc; ++cycle) {
        scheduler.set_cycle_pool_block_end(cute::min((cycle + 1) * kNumRingBlocks, total_pb));
        stage_idx = 0; phase = 0; current_iter_idx = 0;  // RESET pipeline state per cycle
        scheduler.for_each_block_single_cycle(my_lambda);   // single-cycle, no internal loop
        scheduler.cycle_barrier();
    }
} else {
    scheduler.for_each_block(my_lambda);  // original single-cycle path
}
```

The `for_each_block_single_cycle(func)` method (ALREADY ADDED to the scheduler,
uncommitted) iterates blocks up to `cycle_pool_block_end` with NO internal cycle loop
and NO `cycle_barrier`. The caller sets `cycle_pool_block_end` per cycle + resets the
pipeline state + calls `cycle_barrier`.

**What's done for Option A:**
- `for_each_block_single_cycle` + `get_num_cycles` + `get_total_pool_blocks` added to
  the scheduler (uncommitted).
- A-load (778) partially changed: `scheduler.for_each_block([&]` → `auto load_a = [&]`
  (the lambda is now a variable). The closing `});` → `};` + the `if constexpr` cycle
  loop is NOT yet done.

**What's remaining for Option A:**
1. Finish the A-load (778): change the closing `});` (line ~847) to `};` + the
   `if constexpr (kChunking) { cycle loop } else { scheduler.for_each_block(load_a); }`.
2. Do the same for the other 4 call sites (B-load 853, MMA 924, 4th 1046, epilogue 1093).
   - The MMA (924) + epilogue (1093) also need `current_iter_idx = 0` reset.
   - The MMA (924) is leader-only (`if (is_leader_cta)`) — the non-leader's no-op
     `for_each_block` (added in `0eec18b`) also needs the cycle loop.
3. The dispatch pull already loops over cycles (the `for cycle` loop in the dispatch
   warps) — but it should also reset its `pull_mbarrier_phase` per cycle (currently
   carries). Check if needed.
4. Test 1-rank `chunk_ratio=0.5` → expect `num_cycles=4`, `diff < 0.05`.
5. Add cross-rank NVLink barrier to `cycle_barrier()` for multi-rank (currently
   mbarrier + cluster_sync only; the `nvlink_barrier` with grid_sync index 2 timed out
   when tried — needs investigation).
6. Test 2-rank/8-rank → seqlen-cliff shape → perf-vs-`chunk_ratio` benchmark.

**Option B: re-init the mbarriers per cycle**

Re-init `full_barriers`/`empty_barriers`/`tmem_full_barriers`/`tmem_empty_barriers` at
each cycle boundary (after the `cycle_barrier`) + reset `stage_idx`/`phase`/
`current_iter_idx`. This requires a "cycle-start" callback in `for_each_block_impl`
(since the pipeline state is in the lambda closure, not the scheduler). More complex
than Option A. Not recommended.

## Key files (all in `~/DeepGEMM-worktrees/yingz-deepgemm-mem`)

- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` — scheduler: `cycle_pool_block_end`,
  `cap_blocks`, `cycle_ended`, `cycle_barrier()` (mbarrier + cluster_sync),
  `for_each_block_impl` (internal cycle loop), `for_each_block` (no-arg, uses chunking
  path when `cap_blocks > 0`), `for_each_block_single_cycle` (NEW, no cycle loop),
  `get_num_cycles`/`get_total_pool_blocks` (NEW), `fetch_next_l1/l2_block` (cycle-end
  stop + `cycle_ended` flag), `get_next_block` (cycle-end break via `cycle_ended`).
- `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_mxfp4_mega_moe.cuh` — kernel:
  `kChunking` constexpr, scheduler construction with `kNumThreads` + sym_buffer/sm_idx/
  thread_idx, `set_cap_blocks` + `set_cycle_barrier`, `num_cycles` computation + all-reduce
  + printf, unified cycle pull loop, `shared_storage.cycle_barrier` (mbarrier),
  `smem_barriers +1`, non-leader MMA no-op `for_each_block`, A-load lambda-as-variable
  (in-progress).
- `csrc/jit_kernels/heuristics/mega_moe.hpp` — wave heuristic: gated relaxation
  (ring < num_min → force `num_experts_per_wave = 1`).
- `csrc/jit_kernels/impls/sm100_mxfp4_mxfp4_mega_moe.hpp` — `smem_barriers +1` (cycle
  barrier counted).
- `csrc/apis/mega.hpp` — relaxed `num_min ≤ num_ring_tokens` assert; `num_cycles_max`
  workspace slot.
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` — `get_num_cycles_max_ptr` (all-reduce
  slot).
- `deep_gemm/mega/__init__.py` — `chunk_ratio` param (default None = original).
- NOT yet applied to `sm100_fp8_fp4_mega_moe.cuh` (mirror later).

## Probe scripts (in `$CLAUDE_JOB_DIR/tmp/`, may be cleaned up — recreate if needed)

- `chunk_1r.py` — 1-rank, `chunk_ratio=0.5`, tok=1024, (8,2,512,512):
  `ring=768 < num_min=1024` → `kChunking=true`, `num_cycles=4`. The primary test.
- `chunk_ep.py` — 8-rank, `chunk_ratio=1.3`, (8,32,4,1024,4096,1536).
- `chunk_ep2.py` — 2-rank, `chunk_ratio=1.3`, (2,8,2,1024,512,512).

## Resume steps (for a new session)

1. `cd ~/DeepGEMM-worktrees/yingz-deepgemm-mem && git log --oneline -5` — confirm
   the 3 commits + uncommitted changes.
2. `bash develop.sh && python tests/test_fp4_mega_moe.py --scope baseline` — confirm
   `kChunking=false` still works.
3. Finish Option A: change the 5 `for_each_block` call sites to the caller-loops-over-
   cycles pattern (A-load partially done; B-load, MMA, 4th, epilogue remaining).
   - Use `auto my_lambda = [&](...) { ... };` + `if constexpr (kChunking) { cycle loop
     with for_each_block_single_cycle + reset stage_idx/phase/current_iter_idx +
     cycle_barrier } else { for_each_block(my_lambda) }`.
   - The MMA (924) is leader-only — the non-leader's no-op `for_each_block` also needs
     the cycle loop.
4. `bash develop.sh && python $CLAUDE_JOB_DIR/tmp/chunk_1r.py` (or recreate) — test
   1-rank `chunk_ratio=0.5`, expect `num_cycles=4`, `diff < 0.05`.
5. If it passes: add cross-rank NVLink barrier to `cycle_barrier()` → test 2-rank/8-rank.
6. Run `python tests/test_fp4_mega_moe.py --scope 1rank` — confirm no regression.
7. Add prefill shapes to `tests/test_fp4_mega_moe.py` + perf-vs-`chunk_ratio` benchmark
   to `benchmarks/bench_packed_fp4.py`.
8. Mirror all changes to `sm100_fp8_fp4_mega_moe.cuh`.
9. Commit + push to `fw-ai` (`yingz/deepgemm-mem`).

## Key subtleties (lessons learned)

- `compute-sanitizer --tool memcheck` is the debugging tool (available, no torch rebuild).
  It gives the faulting PC offset (e.g. `+0x900`, `+0xc470`) — cross-ref with SASS/PTX.
- The smem allocation (`smem_barriers` in `sm100_mxfp4_mxfp4_mega_moe.hpp:75`) must
  count ALL barriers in `SharedStorage` (adding `cycle_barrier` required `+1`).
- The 2-CTA cluster: the MMA issue warp (`warp_idx == kNumDispatchWarps+2`) runs
  leader-only (`if (is_leader_cta)`). Its `for_each_block` must be called by BOTH CTAs
  (else the all-thread `cycle_barrier` faults). The non-leader needs a no-op
  `for_each_block`.
- `bar.sync` (sync_aligned) and `barrier.sync` (sync_unaligned) with `kNumThreads` both
  fault if not all threads arrive. The counted-arrival mbarrier
  (`ClusterTransactionBarrier` with `init(kNumThreads)` + `mbarrier_arrive` +
  `mbarrier_wait_and_flip_phase`) tolerates staggered arrivals and works for cycle 0.
- The `for_each_block_impl` internal cycle loop must NOT break early
  (`if (current >= kNumExpertsPerRank) break`) — all threads must call `cycle_barrier`
  every cycle.
- `num_cycles` must be `ceil(total_pool_blocks / kNumRingBlocks)` (pool BLOCKS, not
  tokens) — per-expert rounding means pool blocks ≠ `total_recv / BLOCK_M`.
- The dispatch pull's cycle bound must be in POOL-BLOCK space (`pool_block_idx >=
  (cycle+1)*kNumRingBlocks`), not token space, to stay block-aligned with the scheduler.
- The `cycle_ended` flag (set by `fetch_next_l1/l2_block` when the cycle end is hit)
  is used by `get_next_block` to distinguish cycle-end from wave-end (a stale
  `m_block_idx` check was wrong — it false-positived when the wave's blocks were done
  but the expert advanced).
