# Upstream sync through PR #441

This fork incorporates upstream commit
`78b69000794d0937b47ae3387eff7663410264d1` (Mega MoE task info slot release
ordering), including Public Release 26/09, the preceding July release, and
the tensormap race fix.

The scheduler completes task metadata reads before signaling that its slot
can be reused. This ordering fix applies to both BF16 and FP8/FP4 Mega MoE.

The fork's GeGLU activation, native MXFP4/BF16 training backward kernels,
activation saves across ring wraps, route-weight and combine modes, strided
PSUM scales, and `DG_MEGA_MOE_SM_HEADROOM` remain available. Their launchers
now use DeepJIT. SM headroom is applied to the configured `set_num_sms()` limit.

## Compatibility

- Upstream's shared-expert inference and FP8 expert weights are available.
  Pass `num_shared_experts`, `shared_l1_weights`, and `shared_l2_weights` by
  keyword. The fork's existing positional arguments remain in their original
  positions.
- Training saves and backward pools cover routed experts. Combining shared
  experts with training saves or nondefault route/combine modes is rejected.
- Buffer storage follows the new upstream layout. Allocate fresh buffers with
  this version; do not reuse raw storage allocated by an older build.
- Ring and maximum-token alignment is now 1920, reflecting upstream's new
  240-row tile. Omitting `num_ring_tokens` selects upstream's capacity estimate.
  Explicit capacities must satisfy that alignment and be at least the estimate.
- The legacy native buffer slicers retain their 10- and 11-view tuple formats.
  The Python wrapper uses a new v3 slicer that also exposes shared-expert views.
- `DG_BF16_MEGA_MOE_EXPERTS_PER_WAVE` no longer applies: upstream replaced the
  expert-wave scheduler. The block, stage, and epilogue override settings remain.
- Upstream now defaults to nondeterministic algorithms. Use
  `deep_gemm.use_deterministic_algorithms(True)` when deterministic algorithms
  are required.

## Validation

Build with initialized CUTLASS/DeepJIT submodules, CUDA, PyTorch, a C++20
compiler, and libdw development headers. CPU API regressions:

```bash
python -m pytest -q tests/test_mega_moe_compat.py \
  tests/test_mega_moe_training.py::test_fp8_backward_canonicalizes_block_m_and_clears_padding
```

Compile 26 representative SM100 variants without a GPU:

```bash
python tests/compile_mega_moe_compat.py --output-dir /tmp/mega-moe-compile
```

The compilation suite covers BF16, FP8/FP4 and FP8/FP8 inference, shared
experts, training saves, 32/240-row tiles, backward and fused weight-gradient
kernels, and both strided PSUM scale layouts. Compilation does not validate
numerical results, GPU synchronization, or performance.

`tests/test_mega_moe.py` retains upstream's inference suite.
`tests/test_mega_moe_training.py` retains the fork's training suite, and
`tests/run_mega_moe_post_down_tests.sh` runs its route/combine regression matrix.
These GPU tests, together with `tests/test_layout.py`, require working SM100
GPUs and the requested distributed topology.
