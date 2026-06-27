# Handoff: Packed MXFP4 × MXFP4 for SM100 (Mega MoE)

Status: **DONE. Packed MXFP4 × MXFP4 mega-MoE kernel PASSES end-to-end on B200
(`tests/test_mxfp4_mega_moe.py`, `diff = 0.00075`).** Standalone GEMM also validated.

## Update (final) — mega-MoE FIXED end-to-end

Two fixes closed it out:
1. **Hang** (see below): per-CTA `SM90_TMA_LOAD_2D` + cross-CTA loads-done barrier.
2. **L1 packed-FP4 epilogue numerics**: the swap-AB epilogue does NOT need a
   transposing `stmatrix`. Using the verified `SM100_TMEM_LOAD_16dp256b1x` fragment
   map, each lane writes its e2m1 nibbles DIRECTLY to the `[token][inter]` packed
   smem position (plain layout, swizzle-0 TMA store). The map (lane `a=lane%4`,
   `b=lane/4`, warp `w`): `swiglu[i*2+0].{x,y} -> (tok 2a/2a+1, inter w*16+b)`,
   `swiglu[i*2+1].{x,y} -> (tok 2a/2a+1, inter w*16+8+b)`. K-major packing of 2
   consecutive inter per byte is done by an even-`b` lane fetching its `b+1`
   partner's nibble via `__shfl_sync(lane+4)`. Verified `warp_reduce<4,true>`
   reduces over `b` (inter) via `shfl_xor 4/8/16`, and the SF write is guarded by
   `lane_idx < 4` (b==0), so the existing UE8M0 SF code is correct as-is.
   Isolation (`/tmp/iso2.py`, identity L2 weights): `diff 0.98 -> 0.00078`.

Run: `python tests/test_mxfp4_mega_moe.py` -> `diff = 0.00075  MXFP4 mega MoE passed.`

### Update — TRUE 2-CTA `cta_group::2` multicast TMA (standalone + mega)

Both kernels now use genuine `cute::SM100_TMA_2SM_LOAD` (`cp.async.bulk.tensor.cta_group::2`)
instead of per-CTA `SM90_TMA_LOAD`. Key facts were obtained by **instrumenting the real
CUTLASS kernel** (example 72b forced to a 2x1 cluster, `-DDG_TMA_LOG` printf in
`cute/arch/copy_sm100_tma.hpp` + `cutlass/pipeline/sm100_pipeline.hpp`):

- Each CTA issues its OWN per-CTA box at its OWN coord (acts: leader `m=0`, peer `m=128`;
  i.e. keep `load_block_m` box + the per-CTA M offset). It is NOT full-box + shared-coord.
- Peer→leader tx routing is AUTOMATIC under cluster launch: the peer's smem/barrier address
  has bit 24 set (cluster addressing), and the atom masks it (`Sm100MmaPeerBitMask=0xFEFFFFFF`)
  → all tx lands on the LEADER's `full` barrier. No manual cluster-mapping needed.
- For a 2x1 cluster, B is NOT multicast (cluster N=1); both A and B use `SM100_TMA_2SM_LOAD`
  (weights replicated by using the same n coord on both CTAs).
- SF must stay a PER-CTA SM90 load into each CTA's own smem (the `..._2cta` UTCCP reads both
  CTAs' SF). So the leader's `full` expects `2x data + own SF` (both CTAs' data routed in),
  while the non-leader's `full` expects only its own SF; both CTAs' transposer/sync warps wait
  their own `full` and arrive on the leader's `with_sf`.

Standalone `tests/test_mxfp4_gemm.py`: all sizes `diff=0.0`. Mega `diff=0.00075`. Perf
unchanged vs the per-CTA path (2x1 has no multicast bandwidth saving) — still ~1.12-1.23x
over FP8xFP4. Bug that caused the earlier hang: full-box+shared-coord + under-counted
`expect_tx`; fixed by per-CTA box/coord + correct `2x data + own SF` accounting.

### Performance vs FP8×FP4 (single B200, kernel-only via kineto, BF16 out)

MXFP4×MXFP4 is consistently faster than `fp8_fp4_mega_moe` (same shapes/routing):

| tokens | topk/experts | hidden×inter | mxfp4 | fp8fp4 | speedup |
|---|---|---|---|---|---|
| 512  | 4/32 | 4096×4096 | 183µs/1126TF | 192µs/1076TF | 1.05x |
| 1024 | 2/8  | 4096×4096 | 109µs/1886TF | 122µs/1689TF | 1.12x |
| 4096 | 2/8  | 4096×4096 | 306µs/2697TF | 342µs/2409TF | 1.12x |
| 8192 | 2/8  | 4096×4096 | 490µs/3365TF | 600µs/2748TF | 1.22x |
| 4096 | 4/32 | 7168×2048 | 499µs/2893TF | 641µs/2253TF | 1.28x |

Both share the FP4 tensor-core path + FP4 L2 input; MXFP4 wins because L1 activations
are FP4 (0.5 B/elem) vs FP8 (1 B/elem), halving L1 activation traffic. Gap widens with
size (more bandwidth/compute-bound). Trade-off: lower L1-input precision than FP8.
Bench: `/tmp/bench_compare.py [tokens experts topk hidden inter]`.

---
(earlier-session notes below)

## Update (latest session) — mega-MoE hang FIXED, GEMM path working

The long-standing 2-CTA TMA **deadlock is resolved**. Root cause + fix:

- The mega used `cute::SM100_TMA_2SM_LOAD_2D` (`cta_group::2` multicast) for the
  packed-FP4 A/B loads, but with DeepGEMM's hand-rolled half-box descriptor +
  per-CTA arrive scheme. cute's true 2-SM multicast needs a different (full-box,
  cluster-layout) descriptor, so the leader's `full_barrier` `expect_tx` never
  matched what the atom delivered → MMA never fired → hang.
- Fix = adopt the **validated standalone's proven load mechanism** (per-CTA
  `cute::SM90_TMA_LOAD_2D` + a cross-CTA "loads-done" barrier funneled to the
  leader). The host descriptors were already correct for this (acts half-box
  `load_block_m`, weights full `load_block_n`). Changes in
  `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_mxfp4_mega_moe.cuh`:
  - A/B load warps: `SM100_TMA_2SM_LOAD_2D` → `SM90_TMA_LOAD_2D`; SF `tma::copy`
    multicast `2`→`1`; both CTAs `arrive_and_expect_tx(own bytes)` (dropped the
    leader/non-leader split).
  - New `with_sf_full_barriers[kNumStages]` in `SharedStorage`; init `full=2`
    (A+B per CTA), `with_sf = 2*32`.
  - Idle warp `kNumDispatchWarps+3` repurposed as the **cross-CTA sync warp**:
    waits its CTA's `full_barriers[stage]`, then all 32 lanes `arrive(0u)` on the
    leader's `with_sf_full_barriers[stage]`. The leader-only MMA now waits
    `with_sf_full_barriers` instead of `full_barriers`.
  - Host smem sizing updated for the 3rd per-stage barrier (`2*8`→`3*8`) in BOTH
    `csrc/jit_kernels/impls/sm100_mxfp4_mxfp4_mega_moe.hpp` and
    `csrc/jit_kernels/heuristics/mega_moe.hpp` (this was the illegal-access cause
    after enlarging `SharedStorage`).

Result: `python tests/test_mxfp4_mega_moe.py` runs the kernel cleanly; output
magnitude is sane (`y~0.23` vs `ref~0.19`) but **`diff≈0.99`** (values scrambled).

### Remaining task: L1 packed-FP4 swap-AB epilogue transpose (numerics)

The first-GEMM (L1) epilogue writes packed E2M1 + UE8M0 SF to the `l2_acts` pool
buffer that the L2 GEMM reads. The working FP8 sibling
(`sm100_fp8_fp4_mega_moe.cuh`) uses a **transposing** store-matrix
(`SM100_U8x4_STSM_T`, `.b8`) for the swap-AB layout; the MXFP4 port currently does
a **plain, non-transposed** `uint16` write (lines ~1091-1101 of the mega cuh,
marked `// VALIDATE` / stage-1 placeholder) → scrambled L1 output → `diff≈0.99`.
Isolation (`/tmp/iso_l1.py`, 1 expert / top-1) confirms L1 packed bytes are
scrambled (~11% match, confounded by gate/up gran-8 interleave + pool order).

Direction: transpose at **b16 granularity** (each `e2m1x4` = 4 packed FP4 = the
K-major packing unit must stay together). A `SM90_U32x1_STSM_T`
(`stmatrix.x1.m8n8.b16.trans`) primitive was added to `ptx/ld_st.cuh`. Still need
to: gather `kNumAtomsPerStore` `e2m1x4` per lane into `uint32`s, pick x1/x2/x4 by
`kNumAtomsPerStore/2`, and get the transposed smem addressing + the L1-output TMA
descriptor swizzle (`swizzle_acts_mode/2`) consistent. Validate with an L1-output
isolation that accounts for the gate/up interleave and pool token order.

---
(Previous session below)

Status: **Standalone packed `mxf4` 2-CTA GEMM is COMPILED + VALIDATED on B200
(numerically exact, `calc_diff == 0.0`).** Remaining work is the port into the
mega-MoE kernel.

## Update (this session) — standalone de-risk GEMM DONE

Environment: this box has CUDA 13.0 + 8× B200 (unlike the original box), so the
standalone path was fully built and validated on hardware.

What now exists and passes:
- `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_gemm.cuh` — **rewritten** to mirror
  the proven `sm100_fp8_fp4_gemm_1d1d` structure (its scheduler, warp roles, barrier
  scheme incl. the dedicated **SF warp-transpose warp + `with_sf` barrier**, and the
  swap-AB BF16 epilogue), specialized to packed `mxf4`: both operands packed E2M1
  (2/byte, byte-addressed smem), `UMMA_K = 64`, `SM100_MMA_MXF4_2x1SM_SS`, UE8M0 SF
  gran-32 (`sf_id = k*2`), K-major packed swizzle = `BLOCK_K/2` bytes, descriptors via
  `make_smem_desc` (mirrors `sm100_fp4_mqa_logits`).
- `csrc/jit_kernels/impls/sm100_mxfp4_gemm.hpp` — host/JIT `LaunchRuntime` with packed
  A/B tensormaps (`make_tma_2d_desc(..., fp4_unpacked_smem=false)` → `16U4_ALIGN8B`),
  UE8M0 SF descriptors, BF16 D, smem sizing, cluster-2 launch. Registered in
  `csrc/apis/gemm.hpp` as `mxfp4_gemm_nt` (+ pybind), exported in `deep_gemm/__init__.py`.
- `tests/test_mxfp4_gemm.py` — quantizes A/B to packed E2M1 + UE8M0 SF (gran-32, via
  `per_token_cast_to_fp4` + `get_mn_major_tma_aligned_packed_ue8m0_tensor`), runs the
  kernel, compares to a dequant reference. All cases `diff == 0.0`.

The three original `// VALIDATE` spots are now CONFIRMED correct on SM100. Two bugs were
fixed during bring-up:
1. The original draft did UTCCP **without the mandatory SF warp-transpose**; the proven
   kernel runs a separate transposer warp on **all** CTAs + a `with_sf` barrier — now done.
2. Packed-FP4 data must be loaded with a **single raw `cute::SM90_TMA_LOAD_2D::copy`**
   (the descriptor's smem box already spans the full packed `BLOCK_K`); the `tma::copy`
   atom-splitter assumes byte-sized elems and overruns the smem stage for sub-byte FP4.

Run it: `PYTHONPATH=$PWD python tests/test_mxfp4_gemm.py` (after `./develop.sh`).

### Current standalone limitations (intentional for the de-risk)
- Hardcoded config: `BLOCK_M=BLOCK_N=BLOCK_K=128`, `kNumStages=4`, swap-AB,
  **2-CTA (`cluster_n=2`)**, 1 epilogue warpgroup. So it requires **N % 256 == 0**
  (even N-blocks), `M % 128 == 0`, `K % 128 == 0` (host-asserted; smaller N deadlocks
  by construction, exactly like the production heuristics would reject it).
- No autotuning / `get_best_config` — explicit template params only.

Original starting commit: `641d7a3` ("Add packed MXFP4 2-CTA foundations + standalone GEMM (WIP)").

## Goal

Add a **pure, packed MXFP4 × MXFP4** path to DeepGEMM on SM100 (Blackwell / B200):
both operands E2M1 (FP4), UE8M0 scale factors at gran-K 32, **packed 2 elements per
byte**, using the dedicated `tcgen05.mma.kind::mxf4` tensor-core path in **2-CTA**
(`cta_group::2`) mode.

The end target is the **Mega MoE** kernel (`sm100_fp8_fp4_mega_moe.cuh`), whose two
GEMMs currently run FP8×FP4 via `kind::mxf8f6f4`. The plan is to first de-risk the
packed `mxf4` 2-CTA path in a **standalone GEMM**, validate it on B200, then port the
proven pieces into the mega kernel.

## Why packed `mxf4` (not `mxf8f6f4`)

Key facts established during design (see also the chat that produced this work):

- `kind::mxf8f6f4` is the flexible mixed kind (fp8/fp6/fp4, A and B may differ). It
  stores every element in an **8-bit container** (FP4 = 1 byte, low nibble) and
  contracts **K=32 per instruction**. So FP4 under `mxf8f6f4` gets **neither**
  bandwidth/footprint savings **nor** compute speedup — it runs at FP8 storage *and*
  FP8 TFLOPS.
- `kind::mxf4` is FP4-only, **packed 2 elements/byte** (`float_e2m1_t`,
  `sizeof_bits == 4`), and contracts **K=64 per instruction** → ~2× MACs/instruction →
  ~2× FP8 TFLOPS, plus half the smem/gmem/dispatch bandwidth. This is the real FP4 win
  and the reason we must move to `mxf4`.
- Hardware constraint: a single block-scaled MMA has **one** `scale_format_` (E4M3 vs
  E8M0) and **one** scale-vector size for *both* operands (see
  `cute::UMMA::InstrDescriptorBlockScaled` in `cute/arch/mma_sm100_desc.hpp`). Element
  data formats (`a_format_`/`b_format_`) can differ, but scale formats cannot. So you
  cannot mix e.g. MXFP4 (E8M0/32) and NVFP4 (E4M3/16) **within one GEMM**; the two mega
  GEMMs are independent instructions and may differ from each other.

## Design decisions (carried from the chat)

- Scope is **MXFP4 only** for now (UE8M0 SF, gran-32, no global scale). NVFP4 (E4M3 SF,
  gran-16, per-tensor global scale) was discussed and deferred — it adds an on-device
  global-scale problem that MXFP4 avoids entirely.
- The only existing **packed** FP4 precedent in the repo is **1-CTA** in
  `sm100_mqa_logits.cuh` (`SM100_MMA_MXF4_SS`, `float_e2m1_t`, `make_smem_desc` with
  `/2` addressing, `sf_id = k*2`). The production GEMM (`sm100_fp8_fp4_gemm_1d1d.cuh`)
  and the mega kernel both use **unpacked** `mxf8f6f4`. So packed-2CTA is new territory
  and needs hardware iteration.

## What's done (committed in `641d7a3`)

1. `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`
   - Added `SM100_MMA_MXF4_2x1SM_SS`: `tcgen05.mma.cta_group::2.kind::mxf4.block_scale`
     with `.block32` (CUDA ≥ 12.9) / `.scale_vec::2X` fallback.
2. `deep_gemm/include/deep_gemm/common/types.cuh`
   - Added `MmaKind::MXFP4` and `get_element_bits()` (4-bit) so byte math can handle
     sub-byte FP4. NOTE: `get_element_size()` still returns 1 for MXFP4; callers must
     use bit math / divide element counts by 2 for packed storage.
3. `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_gemm.cuh` (new)
   - First-draft **standalone** packed MXFP4 2-CTA GEMM. Conventions mirror the mega L1
     GEMM so the validated pieces port back cleanly:
     - swap-AB, K-major, `UMMA_M = 256` (2-CTA), `UMMA_N = BLOCK_M`, `UMMA_K = 64`.
     - `smem_a` = activations (M axis, multicast, `LOAD_BLOCK_M = BLOCK_M/2`),
       `smem_b` = weights (N axis). SFA on M, SFB on N. UTCCP-2cta SF → TMEM.
     - packed-FP4 byte addressing: smem tiles are `uint8_t[... * BLOCK_K/2]`, swizzle =
       `BLOCK_K/2` bytes, MMA descriptors via `make_smem_desc` with `/2` offsets,
       `sf_id = (umma_k_block * (UMMA_BLOCK_K/UMMA_K) + k) * 2`.
     - BF16 epilogue (TMEM → STSM → `SM90_TMA_STORE_2D`).

## `// VALIDATE` spots (need SM100 compile/run to confirm)

Grep `VALIDATE` in `sm100_mxfp4_gemm.cuh`:

1. **Packed-FP4 TMA load + swizzle**: that `tma::copy<BLOCK_K, LOAD_BLOCK_*,
   kSwizzleABMode=BLOCK_K/2, float_e2m1_t>` produces the correct K-major packed layout,
   and the `arrive_and_expect_tx` byte counts match (A multicast counts ×2).
2. **`make_smem_desc` layout/stride for E2M1**: `to_umma_layout_type<K, BLOCK_K/2,
   false, float_e2m1_t>()` and `stride_byte_offset = 8 * BLOCK_K/2` are right for packed
   FP4 (cross-check against the 1-CTA MQA usage).
3. **`sf_id` / K64 mapping**: that 2 gran-32 SFs per K64 instruction with base
   `sf_id = k*2` and the UTCCP TMEM column placement is correct for 2-CTA `mxf4`.

Other likely tuning points: epilogue STSM swizzle/`STORE_BLOCK_M`, register split
(`kNumEpilogueRegisters`/`kNumNonEpilogueRegisters`), `__grid_constant__` arg order.

## Next steps

1. ~~Toolchain~~ — DONE (CUDA 13.0 + B200; submodules initialized).
2. ~~Host runtime + JIT wiring~~ — DONE (`sm100_mxfp4_gemm.hpp`).
3. ~~Python entry + test~~ — DONE (`mxfp4_gemm_nt`, `tests/test_mxfp4_gemm.py`).
4. ~~Iterate the `// VALIDATE` spots on B200~~ — DONE (all confirmed, `diff == 0.0`).
## Mega-MoE port — STAGE 1 done (compiles + launches on B200), STAGE 2 = numerics

New files / edits (all compile into `_C` and JIT-compile on B200):
- `deep_gemm/include/deep_gemm/impls/sm100_mxfp4_mxfp4_mega_moe.cuh` (fork of the fp8 mega
  kernel). Converted to **uniform packed `mxf4`**: both operands packed E2M1, token byte
  layouts halved, dispatch pulls `hidden/2` bytes, both L1/L2 GEMMs use the validated
  packed path (raw 2-CTA TMA loads, `make_smem_desc`, `SM100_MMA_MXF4_2x1SM_SS`,
  `UMMA_K=64`, `sf_id=k*2`, swizzle `BLOCK_K/2`), L1 SwiGLU epilogue emits packed E2M1
  (`cvt.rn.satfinite.e2m1x2.f32`) + UE8M0 SF (`amax/6`). SF UTCCP path kept (SF is
  pre-transposed in dispatch / weight transform).
- `csrc/jit_kernels/impls/sm100_mxfp4_mxfp4_mega_moe.hpp` — host runtime + `get_mxfp4_mega_moe_config`
  (packed byte math: swizzle `block_k/2`, A/B/output smem halved, `num_bytes_per_pull = hidden/2`).
- `csrc/jit_kernels/heuristics/sm100.hpp` — `get_sf_uttcp_aligned_block_sizes` handles `MmaKind::MXFP4`.
- `csrc/apis/mega.hpp` — `get_symm_buffer_size_for_mxfp4_mega_moe` (packed `hidden/2` token
  buffers, int8 views) + `mxfp4_mxfp4_mega_moe` entry (registered in pybind).
- `deep_gemm/mega/__init__.py` — `SymmBuffer(mma_type='mxfp4xmxfp4')` + `mxfp4_mxfp4_mega_moe`;
  reuses `transform_weights_for_mega_moe` (interleave is on N, works on packed weights).
- Exported `mxfp4_mxfp4_mega_moe` from `deep_gemm/__init__.py`.

STAGE 1 result: the kernel **compiles** (168 regs, 16 barriers, smem ~225KB < 232KB cap) and
**launches** on B200 (single-rank harness `/tmp/mega_compile.py`, shapes hidden=512,
inter=512, E8/top2). It currently **hangs / wrong numerics** — expected; the FP4 L1 epilogue
store layout (the only `// VALIDATE` piece) is a plain non-transposed store that almost
certainly doesn't match the L2 `mxf4` TMA read-back, and there may be dispatch/barrier
interplay to debug.

STAGE 2 (in progress):
- DONE: single-rank **torch MoE reference** at `tests/test_mxfp4_mega_moe.py`
  (FP4 quant → grouped L1×W1 → clamp+SwiGLU×weight → per-32 UE8M0 FP4 requant →
  grouped L2×W2 → top-k combine). Single-rank harness: `/tmp/mega_compile.py`.
- DONE: confirmed the **FP8 mega kernel runs in the same single-rank harness** (so the
  harness/dispatch setup is correct; the hang is in the mxfp4 changes).
- DIAGNOSED the hang (via `DG_JIT_WITH_LINEINFO=1` + `cuda-gdb --batch -ex run -ex 'info cuda
  threads'`, interrupting the hung kernel): the **L2-GEMM acts-load warp spins at
  `sm100_mxfp4_mxfp4_mega_moe.cuh:695`** — `while (ld_acq_gpu(l2_arrival_mask) != kExpectedMask)` —
  waiting for the L2 arrival mask (`0xFF`, 8 bits) that the **L1 SwiGLU epilogue** must set via
  `red_or_rel_gpu(l2_arrival_mask, 1 << n_block_idx)`. All other warps are downstream waiters
  (dispatch clean-workspace `:605`, epilogue combine grid-sync `barrier.cuh:32/51`). GEMM
  warps 4–7 have exited on idle SMs. So the culprit is the **packed-FP4 L1 epilogue** (the only
  `// VALIDATE` piece): its placeholder non-transposed store is numerically wrong AND something
  in that path prevents the L2 arrival mask from reaching `0xFF` (epilogues appear to complete,
  so suspect: the SF/`red_or` interaction, or a per-`n_block` epilogue that never runs because
  its L1 MMA output layout is wrong). The OOB that originally corrupted barriers was fixed by
  over-sizing `smem_d.l1`.

UPDATE (debugging done — root cause is NOT the epilogue): instrumented the kernel with
`printf`s (now removed) and traced across all SMs. Findings:
- `L2 stuck ... mask=0` — the L2 arrival mask is **0** (never set), not partial.
- `LA_WAIT/LA_PASS` fire (load warp clears the L1-arrival wait) and `LA_ISSUE` fires (leader
  issues the TMA + `arrive_and_expect_tx`), **but `MMAFULL` never fires on any SM** — the
  GEMM `full_barriers` never complete, so the MMA never runs, no `tmem_full`, L1 epilogue
  never runs (`EPIFULL`=0), mask stays 0, L2 load spins at `:695`.
- => ROOT CAUSE: the GEMM **`full_barrier` `arrive_and_expect_tx` byte count does not match
  what the raw `cute::SM100_TMA_2SM_LOAD_2D` delivers for packed FP4**, so the barrier waits
  forever for tx that never arrives. The `expect_tx` formulas (`sizeof(smem_a[0])*2 + sizeof(smem_sfa[0])*2`
  for the A/SFA warp; `sizeof(smem_b[0]) + sizeof(smem_sfb[0])*2` for the B/SFB warp) were
  carried over from the FP8 kernel and are wrong for the packed-FP4 2-CTA load tx accounting.

FURTHER PROBING (done — rules out the easy fix): tried `expect_tx` for the A/B data at
`*2`, `*1`, and even **SF-only** (data dropped). `MMAFULL` stays 0 in ALL cases — i.e. the
barrier never completes even when almost nothing is expected. Since the 4 arrivals do happen,
this means the packed A/B **2-CTA TMA loads deliver ~0 tx**: `cute::SM100_TMA_2SM_LOAD_2D`
does not work with the packed `16U4_ALIGN8B` descriptor (it's silently rejected, and the
failed load poisons the `full_barrier` so no tx is counted). This is consistent with the fact
that BOTH the validated standalone (`sm100_mxfp4_gemm.cuh`, diff=0) and `sm100_fp4_mqa_logits.cuh`
load packed FP4 with **per-CTA `cute::SM90_TMA_LOAD_2D` (1-CTA), never the 2-CTA multicast load**.

ROOT CAUSE (from CUTLASS dig, 2026-06): CUTLASS's SM100 packed-FP4 (mxf4/nvf4) GEMMs use
**`MmaTileShape_MNK = Shape<_128,_128,_256>`** — i.e. **K-tile = 256**. For packed FP4
(0.5 B/elem) that is a **128-byte** K extent → `cute::detail::sm100_smem_selector` picks
`Layout_K_SW128_Atom` (**128B swizzle**). The cute TMA atom is built with the packed
`float_e2m1_t` element type and `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B` (identical to DeepGEMM's
`make_tma_2d_desc(..., fp4_unpacked_smem=false)`), and the 2-SM-ness comes purely from the copy
atom (`SM100_TMA_2SM_LOAD[_MULTICAST]` = `cp.async.bulk.tensor.2d.cta_group::2...`) + multicast
masks. So packed-FP4 + 2-CTA multicast IS supported.

The DeepGEMM mega uses `BLOCK_K=128` packed → only **64B** (`SW64`) swizzle. Every working 2-SM
kernel (fp8) uses **128B** swizzle (fp8 gets it free: `128 elems × 1B = 128B`). Strong hypothesis:
the `cta_group::2` 2-SM TMA path requires (or is only exercised with) a 128B-swizzle smem layout,
so the 64B packed layout silently fails to deliver. **FIX: use `BLOCK_K = 256` for the mxfp4
mega+standalone** → `kSwizzleAMode = BLOCK_K/2 = 128` → `SW128`, matching CUTLASS and fp8. This
keeps the desired **2-CTA multicast** (`SM100_TMA_2SM_LOAD_2D`). Mechanical follow-through:
`BLOCK_K/UMMA_BLOCK_K = 2` UMMA-K sub-blocks (already handled), and SF load/indexing for 2 packed
SF-ints per K-block (vs 1 today; `sf_smem_outer_dim = BLOCK_K/(32*4) = 2`).

UPDATE (BLOCK_K=256 tested, swizzle hypothesis REFUTED): set `BLOCK_K=256` (confirmed in the
instantiation: template arg `256`, TMA desc `swizzle:128`, smem box `[256,32]`) — the 2-SM load
STILL delivers 0 (`MMAFULL`=0, hang). So swizzle (64B vs 128B) is NOT the cause. Empirical matrix:
  - packed `16U4` + `SM90` 1-CTA + DeepGEMM `make_tma_2d_desc`  => WORKS (standalone, diff=0)
  - 8-bit + `cta_group::2` (2-SM) + DeepGEMM `make_tma_2d_desc` => WORKS (fp8 mega)
  - packed `16U4` + `cta_group::2` (2-SM) + DeepGEMM `make_tma_2d_desc` => DELIVERS 0 (hang), any swizzle
So the blocker is specifically the **(`cta_group::2` 2-SM load) × (`16U4` packed descriptor built by
DeepGEMM's hand-rolled `make_tma_2d_desc`)** combination. CUTLASS makes the same combination work,
but it builds the descriptor via cute's `make_tma_atom_A_sm100<float_e2m1_t>(... ClusterLayout_VMNK ...)`,
which derives the 2-SM box partitioning / coords from the cluster layout — something DeepGEMM's
`make_tma_2d_desc` does not replicate for the 2-SM packed case.

*** ROOT CAUSE FOUND (descriptor field-diff, option A) ***
Built a cute harness (`/tmp/tma_dump.cu`, compiles with `nvcc -arch=sm_100a -I third-party/cutlass/include
-I third-party/cutlass/tools/util/include`) that instantiates a CUTLASS 2-SM mxf4 mainloop
(`mx_float4_t<float_e2m1_t>`, MmaTileShape `<128,128,256>`, ClusterShape `<2,1,1>`), extracts its
A-operand `CUtensorMap` via `params.tma_load_a.get_tma_descriptor()`, and diffs it against
DeepGEMM's `make_tma_2d_desc` (replicated) for the same A. Only 2 of 16 words differ:
  word[7]: cute `0x7f` (=127 → box-M **128**) vs deepgemm `0x3f` (=63 → box-M **64**)
  word[1]: the dependent box-stride field (differs for the same reason)
=> **cute uses the FULL tile-M (128) as the TMA box and lets `cta_group::2` split it 64/64 across
the 2 CTAs; DeepGEMM uses the pre-split `load_block_m` (64).** For 8-bit (fp8) the half-box works,
but for packed `16U4` the `cta_group::2` load needs the FULL-tile box and silently delivers 0 with
the half-box — the exact hang observed.

THE FIX: build the mxfp4 mega's A/B packed-FP4 TMA descriptors with the **full block-M / block-N**
box (not `load_block_m`/`load_block_n`), and issue the **base coord** (drop the per-CTA
`m_idx += get_valid_m/2` offset) so `cta_group::2` does the split — matching cute. Keep the smem
tile sized per-CTA (`load_block_m`). NOTE: the mega is swap-AB (acts=UMMA-N, weights=UMMA-M=256), so
apply the full-box to the operand actually split across the 2 SMs and re-derive the descriptor via a
packed-specific `make_tma_2d_desc` variant that takes the full tile extent for the box.

UPDATE (full-box fix tested in mega — insufficient): applied cute's convention to the ACTS
(box = full `block_m` via host, dropped per-CTA `m_idx` offset; confirmed in DG_JIT_DEBUG: acts
smem box outer 32 -> 64). Still hangs (`MMAFULL`=0). Harness also dumped cute's smem: `SmemLayoutA`
per-CTA M = 64 = MmaTile_M/2 (half), box = 128 (full) — confirms the "full-box descriptor +
half-smem, cta_group::2 splits" model. But the mega is **swap-AB** (weights=UMMA-A spanning
UMMA_M=256 across the 2 CTAs; acts=UMMA-B/UMMA_N), so the 2-SM-split operand mapping differs from
cute's standard GEMM (where A=M is the split operand). Fixing only the acts box doesn't line up,
and A+B share one `full_barrier`. Cracking cta_group::2 here ≈ adopting cute's full non-swap 2-SM
mainloop structure (effectively a GEMM rewrite), not a descriptor tweak.

DECISION POINT (both are real work; pick one):
  (A) Keep 2-CTA multicast: rebuild the mega GEMM around cute's exact 2-SM model (full-tile boxes
      for the correct split operand, base coords, matching smem). Largest; verify each descriptor
      byte-matches cute via `/tmp/tma_dump.cu`.
  (B) Solve now via the PROVEN path: use the validated standalone's load mechanism — per-CTA
      `SM90_TMA_LOAD_2D` (each CTA loads its own split half) + a cross-CTA "loads-done" barrier
      (repurpose the idle GEMM warp, like the standalone's SF-transpose warp arrives `with_sf`).
      This is exactly what `sm100_fp4_mqa_logits` and the validated standalone (`diff=0`) do for
      packed FP4. Reworks the mega's `full_barrier` arrive scheme (each CTA expects its own tx).

(superseded options below)
REMAINING OPTIONS to get 2-CTA multicast + packed FP4:
  (A) Match cute exactly: dump the `CUtensorMap` cute produces for a 2-SM packed-FP4 atom
      (`make_tma_atom_A_sm100<float_e2m1_t>` with a 2-SM `ClusterLayout`) and diff its fields vs
      `make_tma_2d_desc`; fix the box/coords in `make_tma_2d_desc` (or call cute's builder in the host).
      This is the path that keeps true 2-CTA multicast.
  (B) Fall back to the repo's own packed-FP4 precedent — per-CTA `SM90_TMA_LOAD_2D` (what
      `sm100_fp4_mqa_logits` and the validated standalone use) + a cross-CTA "loads-done" barrier.

(Superseded) earlier candidate — per-CTA SM90 loads:
- Replace the GEMM A/B loads with per-CTA `cute::SM90_TMA_LOAD_2D` (`num_tma_multicast=1`):
  each CTA loads its own A half (split `m_idx`) and its own B into its own smem, and
  `arrive_and_expect_tx`-es ITS OWN bytes on ITS OWN `full_barrier` (init per-CTA, not the
  `2*2` leader-combined scheme).
- Add the cross-CTA "loads done" sync the 2-CTA TMA used to provide implicitly. The standalone
  gets this for free via its SF-transpose warp + `with_sf` barrier (both CTAs arrive at the
  leader after their own `full` completes, and the leader MMA waits on `with_sf`). The mega
  removed the transpose warp (SF pre-transposed), so add an equivalent: after each CTA's
  `full_barrier` is satisfied, have it `arrive` at a shared (leader) barrier that the MMA warp
  waits on, so the leader doesn't issue the 2-CTA UMMA until BOTH CTAs' smem is loaded.
  (SFA/SFB still load fine via the existing `tma::copy<...,2>`; only the packed A/B data loads
  must move off the 2-CTA TMA.)

After the GEMM loads complete, the next blocker is still the **packed-FP4 L1 epilogue** layout
(currently a placeholder, numerically wrong) — see below.

REMAINING (focused): implement the **correct packed-FP4 L1 epilogue** — derive the TMEM→register
mapping (`SM100_TMEM_LOAD_16dp256b1x`) and a packed-E2M1 smem store (no 4-bit `stmatrix`
exists; needs a manual transpose/pack) whose layout matches the L2 `mxf4` TMA read
(`swizzle = BLOCK_K/2`), and verify the L2 arrival mask reaches `0xFF`. Then validate against
the torch reference and add a `mxfp4xmxfp4` case to `tests/test_mega_moe.py`. Debug loop:
`DG_JIT_WITH_LINEINFO=1 cuda-gdb --batch -ex run -ex 'info cuda threads' --args python -u
tests/test_mxfp4_mega_moe.py` (interrupt with `timeout -s INT`); inspect the workspace
`l2_arrival_mask` value for a stuck pool block.

### (superseded) original step 5

### (Original) Port to mega notes
- create `sm100_mxfp4_mxfp4_mega_moe.cuh` (fork of the fp8 mega impl)
   reusing the proven packed paths — dispatch pulls `hidden/2` bytes, the L1-output
   epilogue emits packed E2M1 + UE8M0 SF (SF math is unchanged from today, only the
   cast/STSM packing differs), and both GEMMs use the new `mxf4` 2-CTA MMA. Then add
   `MmaKind::MXFP4` handling in `heuristics/mega_moe.hpp` + `heuristics/sm100.hpp`
   (FP4 0.5-byte arithmetic, per-stage smem, SF aligned sizes), the host
   `sm100_mxfp4_mxfp4_mega_moe.hpp`, the API in `csrc/apis/mega.hpp`
   (`mma_type="mxfp4xmxfp4"` + buffer sizing), Python + weight transform, and a
   `tests/test_mega_moe.py` case.

## Useful references in-repo

- `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh` — the only packed `mxf4`
  (1-CTA) usage; authoritative for descriptor `/2` addressing and `sf_id` mapping.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` — the 2-CTA structure,
  UTCCP-2cta SF, TMEM pipeline, and epilogue this draft mirrors.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh` +
  `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp` — host/JIT/tensormap wiring to
  copy for the standalone host runtime.
- `deep_gemm/include/deep_gemm/mma/sm100.cuh` — `make_smem_desc`,
  `to_umma_layout_type`, `make_runtime_instr_desc_with_sf_id`.
