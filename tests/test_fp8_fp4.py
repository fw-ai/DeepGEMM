import numpy as np
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    ignore_env, get_arch_major
)

from generators import (
    KernelType, get_ue8m0_usage, layout_masked_to_psum, align,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
    generate_k_grouped_contiguous_psum,
    get_mk_alignment_for_contiguous_layout, set_mk_alignment_for_contiguous_layout
)


def check_fp8_fp4_psum_zero_padding(a: tuple, d: torch.Tensor, grouped_layout: torch.Tensor) -> None:
    for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
        aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
        if current_m < aligned_m:
            data_padding = a[0][current_m: aligned_m]
            d_padding = d[current_m: aligned_m]
            assert torch.equal(data_padding, torch.zeros_like(data_padding)), f'{group_idx=}, nonzero FP8/FP4 input padding'
            assert torch.equal(d_padding, torch.zeros_like(d_padding)), f'{group_idx=}, nonzero FP8/FP4 output padding'


def _ceil_to_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    bits = scales.abs().float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().to(torch.int32)
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def _ue8m0_per_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    grouped = x.float().view(rows, columns // 128, 128)
    scales = _ceil_to_ue8m0(grouped.abs().amax(dim=-1).clamp_min(1e-4) / 448.0)
    quantized = (grouped / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).view_as(x)
    return quantized.contiguous(), scales.contiguous()


def _ue8m0_blockwise_weight_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rows, columns = weight.shape
    blocks = (
        weight.float()
        .view(groups, rows // 128, 128, columns // 128, 128)
        .permute(0, 1, 3, 2, 4)
    )
    scales = _ceil_to_ue8m0(
        blocks.abs().amax(dim=(-1, -2)).clamp_min(1e-4) / 448.0
    )
    quantized = (
        (blocks / scales[..., None, None])
        .to(torch.float8_e4m3fn)
        .permute(0, 1, 3, 2, 4)
        .reshape_as(weight)
        .contiguous()
    )
    return quantized, scales.contiguous()


def test_psum_grouped_gemm_accepts_deepep_column_major_scales() -> None:
    """Keep GLM's expanded DeepEP scale layout valid in the K3 wheel."""
    assert deep_gemm._C.supports_noncontiguous_psum_sfa is True
    torch.manual_seed(5750)
    alignment = 128
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    groups, rows, n, k = 2, 2 * alignment, 256, 256
    psum = torch.tensor([5, alignment + 5], device='cuda', dtype=torch.int32)

    activations_bf16 = torch.zeros(rows, k, device='cuda', dtype=torch.bfloat16)
    activations_bf16[:5].normal_()
    activations_bf16[alignment:alignment + 5].normal_()
    activations, contiguous_scales = _ue8m0_per_token_quantize(activations_bf16)
    column_major_scales = torch.empty_strided(
        contiguous_scales.shape,
        (1, rows),
        dtype=contiguous_scales.dtype,
        device=contiguous_scales.device,
    )
    column_major_scales.copy_(contiguous_scales)
    assert not column_major_scales.is_contiguous()
    assert column_major_scales.stride() == (1, rows)

    weights_bf16 = torch.randn(groups, n, k, device='cuda', dtype=torch.bfloat16)
    weights, weight_scales = _ue8m0_blockwise_weight_quantize(weights_bf16)
    expected = torch.empty(rows, n, device='cuda', dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (activations, contiguous_scales),
        (weights, weight_scales),
        expected,
        psum,
        use_psum_layout=True,
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (activations, column_major_scales),
        (weights, weight_scales),
        actual,
        psum,
        use_psum_layout=True,
        allow_strided_psum_scales=True,
    )
    assert torch.equal(actual, expected)


def test_gemm() -> None:
    print('Testing GEMM:')
    scores = []
    for kernel_type, quant_config, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=(kernel_type.is_1d1d() and accumulate))

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0, quant_config=quant_config)
            func_name = f'fp8_fp4_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            diff = calc_diff(d, ref_d)
            assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                                    f'{diff:.5f}, alias={test_alias}')

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0, quant_config=quant_config)
        t = bench_kineto(lambda: deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b),
                         'gemm_', suppress_kineto_output=True)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a[0], b[0], d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True) \
                              if not quant_config.is_fp4_a and not quant_config.is_fp4_b else (0, 0)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
        if cublas_t > 0:
            scores.append((cublas_t + split_k_t) / t)
    print(f"Average FP8xFP8 GEMM speedup over cuBLASLt: {float(np.prod(scores)) ** (1.0 / len(scores)):.3f}x\n")


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, quant_config, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for test_alias in (False, True):
            m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                              use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                              quant_config=quant_config)
            func_name = f"m_grouped_fp8_fp4_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast,
                                          use_psum_layout=use_psum_layout, ensure_zero_padding=ensure_zero_padding,
                                          recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            if use_psum_layout:
                for j in range(num_groups):
                    start = 0 if j == 0 else align(grouped_layout[j - 1], get_mk_alignment_for_contiguous_layout())
                    end = grouped_layout[j]
                    diff = calc_diff(d[start : end], ref_d[start : end])
                    assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, '
                                                            f'{diff:.5f}, alias={test_alias}, {ensure_zero_padding=}')
                if ensure_zero_padding:
                    check_fp8_fp4_psum_zero_padding(a, d, grouped_layout)
            else:
                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                          quant_config=quant_config)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast, use_psum_layout=use_psum_layout,
                                                           ensure_zero_padding=ensure_zero_padding,
                                                           recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}, '
              f'psum={use_psum_layout}, zero_pad={ensure_zero_padding}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m_per_group * 1.2))
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for i in range(num_tests):
            a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k,
                                                                         use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                         quant_config=quant_config)
            if use_psum_layout:
                a_psum = (layout_masked_to_psum(a[0], psum_m), layout_masked_to_psum(a[1], psum_m))
                d_psum = layout_masked_to_psum(d, psum_m)

            # noinspection PyShadowingNames
            def test_func():
                if use_psum_layout:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a_psum, b, d_psum, psum_m, disable_ue8m0_cast=disable_ue8m0_cast,
                                                                   use_psum_layout=True, expected_m_for_psum_layout=int(expected_m_per_group * 1.2),
                                                                   recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                else:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, int(expected_m_per_group * 1.2), disable_ue8m0_cast=disable_ue8m0_cast,
                                                               recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

            test_func()
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                if use_psum_layout:
                    d_slice = d_psum[: psum_m[j]] if j == 0 else d_psum[align(psum_m[j - 1], get_mk_alignment_for_contiguous_layout()): psum_m[j]]
                else:
                    d_slice = d[j, :masked_m[j].item()]
                diff = calc_diff(d_slice, ref_d[j, :masked_m[j].item()])
                assert diff < quant_config.max_diff(), f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

            # Test performance with fixed shapes
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'{kernel_opt}, psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    k_grouped_fp8_gemm_contiguous = deep_gemm.k_grouped_fp8_gemm_nt_contiguous if get_arch_major() == 9 \
                                    else deep_gemm.k_grouped_fp8_gemm_tn_contiguous
    for num_groups, m, n, major_a, major_b, real_ks_cpu, aligned_ks_cpu, _, gran_k, k_alignment, use_psum_layout in enumerate_k_grouped_contiguous(torch.float8_e4m3fn):
        recipe = (1, 1, gran_k)
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_real_ks_cpu, test_aligned_ks_cpu, _, _ in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu, k_alignment, use_psum_layout):
            if use_psum_layout:
                total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous_psum(num_groups, m, n, major_a, major_b, test_real_ks_cpu, k_alignment=k_alignment, use_ue8m0=use_ue8m0, gran_k=gran_k)
            else:
                total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, test_aligned_ks_cpu, use_ue8m0=use_ue8m0, gran_k=gran_k)
            c_orig = c.clone() if use_psum_layout else None
            k_grouped_fp8_gemm_contiguous(a, b, d, test_aligned_ks_cpu, grouped_layout, c, recipe=recipe, use_psum_layout=use_psum_layout)

            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, {test_aligned_ks_cpu=}, {use_psum_layout=}, {diff:.5f}'

            # Unsynced psum paths
            if use_psum_layout:
                c.copy_(c_orig)
                k_grouped_fp8_gemm_contiguous(a, b, d, None, grouped_layout, c, recipe=recipe,
                                              use_psum_layout=True)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f'None ks_cpu path: {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, {diff:.5f}'

                c.copy_(c_orig)
                k_grouped_fp8_gemm_contiguous(a, b, d, [], grouped_layout, c, recipe=recipe,
                                              use_psum_layout=True)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f'empty ks_cpu path: {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, {diff:.5f}'

        # Test performance
        if use_psum_layout:
            total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous_psum(num_groups, m, n, major_a, major_b, real_ks_cpu, k_alignment=k_alignment, use_ue8m0=use_ue8m0, gran_k=gran_k)
        else:
            total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, aligned_ks_cpu, use_ue8m0=use_ue8m0, gran_k=gran_k)

        # noinspection PyShadowingNames
        def test_func():
            k_grouped_fp8_gemm_contiguous(a, b, d, aligned_ks_cpu, grouped_layout, c, recipe=recipe, use_psum_layout=use_psum_layout)

        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={total_k:5}, gran_k={gran_k:3}, k_alignment={k_alignment:3}, psum={int(use_psum_layout)}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * total_k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    test_k_grouped_gemm_contiguous()
