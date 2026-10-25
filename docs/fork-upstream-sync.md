# Upstream sync through PR #478

This fork incorporates upstream commit
`5081f8513af57e61696de4f4f04b0e707ffb503e` (mega mHC split barriers under CUDA
graph replay), including Public Release 26/09/30 (#462) and the FP8 epilogue and
paged sparse MQA alignment update (#471). The previous sync stopped at PR #441.

The fork's GeGLU and SiTU activations, native MXFP4/BF16 training backward
kernels, activation saves across ring wraps, route-weight and combine modes,
strided PSUM scales, side-LoRA kernels, `DG_MEGA_MOE_SM_HEADROOM`, the BF16-logit
sigmoid gate and the additive NVFP4 MegaMoE (`deep_gemm.mega_nvfp4`) remain
available.

## Compatibility

- Mega gate: upstream's rewritten kernel only scores with `sqrtsoftplus`. The
  fork keeps `sigmoid`, `sigmoid_bf16` and `identity` as a `kScoringType`
  template parameter of the new kernel, with the same ranking semantics as before
  (scores are ranked raw without a bias and activated after selection).
  `get_bf16_mega_gate_config` is kept and reports the new heuristics. The default
  `scoring_func` is now upstream's `sqrtsoftplus`.
- Dense and paged MQA logits: upstream's SM100 rewrite requires MXFP8 Q scales,
  BF16 weights and MX-scaled index caches, returns BF16 logits without cleaning,
  and restricts paged SM100 to varlen `next_n == 1`. The fork keeps the previous
  kernels and API (`clean_logits`, `logits_dtype`, non-varlen paged, and the
  `fp8_mqa_logits` / `fp8_paged_mqa_logits` wrappers), so FP8 indexer caches with
  per-token float scales keep working. Sparse MQA logits follow upstream.
  `fp8_gemm_nt_skip_head_mid` follows upstream's removal.
- GEMM epilogues: upstream's `EpilogueClass` objects replace `EpilogueInput`.
  Fork launchers (BF16 wgrad, side-LoRA rank GEMMs) pass `IdentityEpilogue`, and
  fork kernels construct `epilogue::operators::Identity` from
  `EpilogueOperatorArgs`.
- Mega MoE store blocks: the FP8xFP4 kernel uses upstream's asymmetric L1/L2
  store blocks (24/8 rows at `BLOCK_M = 240`). Training saves use the store block
  of their layer. Side-LoRA kernels, which have a single store block, request the
  previous symmetric sizes through `single_store_block`.
- Mega MoE weights may be localized (`deep_gemm.localize`) when MLOPart is
  available. The scheduler is locality-domain aware either way; without MLOPart
  it uses an even SM-to-domain mapping.
- SM headroom: upstream requires Mega MoE to launch on the whole GPU so both
  locality domains have CTAs. The fork keeps `DG_MEGA_MOE_SM_HEADROOM`; the
  domain queues are dynamic with work stealing, and a headroom below half the SMs
  leaves CTAs in both domains.
- Mega mHC: upstream tags split barriers with an epoch, which supersedes the
  fork's per-call barrier reset (one fewer launch per call).
  `mega_mhc_supports_graph_replay` stays `True`.
- The BF16 Mega MoE combine reduces the shared-expert slot first, then routed
  slots in top-k order (upstream change). The fork rejects shared experts with
  the DeepEP combine orders, so only `FixedTopK` is affected.

## Validation

Build with initialized CUTLASS/DeepJIT submodules, CUDA, PyTorch, a C++20
compiler, and libdw development headers. CPU API regressions:

```bash
python -m pytest -q tests/test_mega_moe_compat.py tests/test_mega_moe_headroom.py \
  tests/test_mega_moe_side_lora_contracts.py tests/test_mega_moe_side_lora_reference.py \
  tests/test_mega_moe_side_lora_block_m.py tests/test_mega_moe_wgrad_layout.py
```

Compile the representative SM100 variants without a GPU:

```bash
python tests/compile_mega_moe_compat.py --output-dir /tmp/mega-moe-compile
```

The compilation suite covers BF16, FP8/FP4 and FP8/FP8 inference, shared
experts, training saves, 32/240-row tiles, backward and fused weight-gradient
kernels, side-LoRA, and both strided PSUM scale layouts. Compilation does not
validate numerical results, GPU synchronization, or performance.

GPU suites: `tests/test_bf16.py`, `tests/test_fp8_fp4.py`, `tests/test_einsum.py`,
`tests/test_attention.py`, `tests/test_mega_gate.py`, `tests/test_mega_mhc.py`,
`tests/test_layout.py`, `tests/test_mega_moe.py` (gated activations),
`tests/test_mega_moe_nvfp4.py --num-processes 2`, and the training matrix in
`tests/run_mega_moe_post_down_tests.sh`. `tests/test_locality_domain.py`
requires MLOPart.
