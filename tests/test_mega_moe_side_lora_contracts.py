"""Source-level contracts for MegaMoE side-LoRA kernels and wrappers.

These tests intentionally avoid importing :mod:`deep_gemm`, so they can run
without a compiled extension or a GPU.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKWARD_PY = ROOT / "deep_gemm/mega/backward.py"
BACKWARD_HOST = (
    ROOT / "csrc/jit_kernels/impls/sm100_bf16_mega_moe_side_lora_backward.hpp"
)
BASE_BACKWARD_HOST = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
BACKWARD_KERNEL = (
    ROOT / "deep_gemm/include/deep_gemm/impls/"
    "sm100_bf16_mega_moe_side_lora_backward.cuh"
)
FORWARD_KERNEL = (
    ROOT / "deep_gemm/include/deep_gemm/impls/"
    "sm100_bf16_mega_moe_side_lora_forward.cuh"
)
SCHEDULER = ROOT / "deep_gemm/include/deep_gemm/scheduler/mega_moe_side_lora.cuh"
MEGA_MOE_API = ROOT / "csrc/apis/mega_moe.hpp"


def _python_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _python_assignment(tree: ast.Module, name: str) -> ast.Assign:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _default(function: ast.FunctionDef, argument: str) -> object:
    positional = function.args.posonlyargs + function.args.args
    positional_defaults = dict(
        zip(positional[-len(function.args.defaults) :], function.args.defaults)
    )
    keyword_defaults = dict(zip(function.args.kwonlyargs, function.args.kw_defaults))
    defaults = {arg.arg: value for arg, value in positional_defaults.items()}
    defaults.update(
        {arg.arg: value for arg, value in keyword_defaults.items() if value is not None}
    )
    return ast.literal_eval(defaults[argument])


def _assigns_name(function: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(function):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return True
    return False


def _cpp_block(source: str, start: str, end: str) -> str:
    start_offset = source.index(start)
    end_offset = source.index(end, start_offset + len(start))
    return source[start_offset:end_offset]


def _compact(source: str) -> str:
    return " ".join(source.split())


def test_backward_trace_schema_matches_kernel_barriers() -> None:
    source = BACKWARD_PY.read_text()
    tree = ast.parse(source)
    base_assignment = _python_assignment(tree, "_BF16_BACKWARD_KERNEL_TRACE_SITES")
    base_trace_sites = ast.literal_eval(base_assignment.value)
    assert len(base_trace_sites) == 22

    side_assignment = _python_assignment(
        tree, "_BF16_SIDE_LORA_BACKWARD_KERNEL_TRACE_SITES"
    )
    assert isinstance(side_assignment.value, ast.BinOp)
    assert isinstance(side_assignment.value.op, ast.Add)
    assert isinstance(side_assignment.value.left, ast.Name)
    assert side_assignment.value.left.id == "_BF16_BACKWARD_KERNEL_TRACE_SITES"
    side_trace_sites = base_trace_sites + ast.literal_eval(side_assignment.value.right)
    assert len(side_trace_sites) == 23
    assert side_trace_sites[-1] == ("canonical_gate_up_grad_deinterleave_handoff")

    base_function = _python_function(tree, "bf16_mega_moe_backward_dgrad")
    base_source = ast.get_source_segment(source, base_function)
    assert base_source is not None
    assert "_BF16_BACKWARD_KERNEL_TRACE_SITES" in base_source
    assert "_BF16_SIDE_LORA_BACKWARD_KERNEL_TRACE_SITES" not in base_source
    side_function = _python_function(tree, "bf16_mega_moe_side_lora_backward")
    side_source = ast.get_source_segment(source, side_function)
    assert side_source is not None
    assert "_BF16_SIDE_LORA_BACKWARD_KERNEL_TRACE_SITES" in side_source
    assert "float(side_lora_scale), kernel_trace)" in side_source

    host = BACKWARD_HOST.read_text()
    kernel = BACKWARD_KERNEL.read_text()
    assert "constexpr int num_trace_sites = 22;" in BASE_BACKWARD_HOST.read_text()
    assert "constexpr int num_trace_sites = 23;" in host
    assert "kTraceSiteCount = 23" in kernel
    assert "full_grid_phase_barrier(22)" in kernel


def test_side_lora_backward_always_materializes_grad_x_pool() -> None:
    source = BACKWARD_PY.read_text()
    tree = ast.parse(source)
    function_names = (
        "bf16_mega_moe_side_lora_backward",
        "fp8_fp4_mega_moe_side_lora_backward",
    )
    for name in function_names:
        function = _python_function(tree, name)
        assert _default(function, "write_grad_x_pool") is True
        assert not _assigns_name(function, "write_grad_x_pool")
        function_source = ast.get_source_segment(source, function)
        assert function_source is not None
        assert "if not write_grad_x_pool:" in function_source
        assert (
            'ValueError("side-LoRA backward requires ' 'write_grad_x_pool=True")'
        ) in function_source

    allocator = _python_function(tree, "_allocate_side_lora_backward_outputs")
    assert _default(allocator, "write_grad_x_pool") is True
    allocator_source = ast.get_source_segment(source, allocator)
    assert allocator_source is not None
    assert "if not write_grad_x_pool:" in allocator_source
    assert "torch.empty((0, hidden)" not in allocator_source

    host = BACKWARD_HOST.read_text()
    bf16_host = _cpp_block(
        host,
        "static void sm100_bf16_mega_moe_side_lora_backward(",
        "static void sm100_fp8_fp4_mega_moe_side_lora_backward(",
    )
    fp8_host = host[
        host.index("static void sm100_fp8_fp4_mega_moe_side_lora_backward(") :
    ]
    for block in (bf16_host, fp8_host):
        assert "DG_HOST_ASSERT(write_grad_x_pool);" in block
        assert ".write_grad_x_pool = write_grad_x_pool" in block
        assert "write_grad_x_pool || direct_remote_grad_x" not in block
    assert "grad_x_pool_output.sizes() == x_pool_output.sizes()" in _compact(bf16_host)
    assert (
        "grad_x_pool_output.sizes() == "
        "torch::IntArrayRef({num_pool_rows, hidden})" in _compact(fp8_host)
    )


def test_mxfp4_backward_requires_prepared_gate_up() -> None:
    host = _compact(BACKWARD_HOST.read_text())
    kernel = _compact(BACKWARD_KERNEL.read_text())
    assert "DG_HOST_ASSERT(args.bf16_mode || args.gate_up_prepared);" in host
    assert "kBF16Mode || kGateUpPrepared" in kernel
    assert "MXFP4 side-LoRA backward requires prepared gate/up inputs" in kernel


def test_scheduler_and_forward_prefix_cover_all_experts() -> None:
    scheduler = SCHEDULER.read_text()
    forward = _compact(FORWARD_KERNEL.read_text())
    assert "uint32_t valid_value = 0;" in scheduler
    assert "expert_idx += kNumGlobalWarps" in forward
    assert "source_warp * kNumExperts + expert_idx" in forward
    assert "global_warp_idx < kNumExperts && lane_idx == 0" not in forward


def test_bf16_side_lora_requires_full_expanded_buffer() -> None:
    source = MEGA_MOE_API.read_text()
    function = _cpp_block(
        source,
        "static void bf16_mega_moe_side_lora(",
        "static void register_apis(",
    )
    compact = _compact(function)
    assert (
        "sym_buffer.nbytes() >= " "static_cast<size_t>(expanded_num_required_bytes)"
    ) in compact
    assert "expanded_num_required_bytes -" not in function
    assert "const auto num_required_bytes" not in function


def test_side_lora_remote_grad_x_clear_is_kernel_owned() -> None:
    source = BACKWARD_PY.read_text()
    tree = ast.parse(source)
    for name in (
        "bf16_mega_moe_side_lora_backward",
        "fp8_fp4_mega_moe_side_lora_backward",
    ):
        function = _python_function(tree, name)
        function_source = ast.get_source_segment(source, function)
        assert function_source is not None
        assert "_direct_grad_x_planes(sym_buffer).zero_()" not in function_source

    kernel = BACKWARD_KERNEL.read_text()
    assert "reinterpret_cast<uint4*>(combine_buffer)[vector_idx]" in kernel
    assert "make_uint4(0, 0, 0, 0);" in kernel
    assert "comm::nvlink_barrier<kNumRanks, kNumSMs, 256, 1, 73>" in kernel
