#pragma once

// Derived from sgl-project/DeepGEMM@ec226bcf400c11d62755137b916fae9ee359c684:
//   csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp
//   (and `aten_dtype_to_tensor_map_dtype` / `make_tma_2d_desc` / `make_tma_sf_desc` from
//   csrc/jit_kernels/impls/runtime_utils.hpp)
// Copyright (c) 2025 DeepSeek, with modifications by the sgl-project/DeepGEMM contributors.
// Licensed under the MIT License (see LICENSE at the repository root).
// Fireworks port, all host-side; the generated kernel instantiation is unchanged apart from the
// header path and namespace:
//   - namespace `deep_gemm::mega_nvfp4`, host functions renamed, JIT kernel name `sm100_nvfp4_mega_moe`;
//   - the TMA descriptor helpers are copied because sgl's version maps `torch::kUInt8` (the packed
//     FP4 activation views) to `CU_TENSOR_MAP_DATA_TYPE_UINT8`, which the fw-ai helper rejects;
//   - removed the TRT-LLM weight layout (`make_trtllm_fp4_*_tma_desc`; `kUseTrtllmWeights` is
//     always instantiated `false`), which Fireworks does not use;
//   - compiled with `get_nvfp4_mega_moe_compiler_options` (sgl's `sm_100f` target), launched with
//     `get_nvfp4_mega_moe_num_sms` SMs and with PDL off (sgl's runtime default; fw-ai's global
//     `set_pdl` must not change these kernels).

#include <format>
#include <torch/python.h>

#include <deep_gemm/mega_nvfp4/layout/mega_moe.cuh>
#include <deep_gemm/mega_nvfp4/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/nvfp4_mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm::mega_nvfp4 {

static CUtensorMapDataType nvfp4_aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype,
                                                                const bool& allow_tf32,
                                                                const bool& fp4_unpacked_smem) {
    if (allow_tf32 and dtype == torch::kFloat)
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;

    switch (dtype) {
        case torch::kInt:           return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16:      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case torch::kUInt8:         return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case kPackedFP4:            return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                                                             : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMap nvfp4_make_tma_2d_desc(const torch::Tensor& t,
                                          int gmem_inner_dim, int gmem_outer_dim,
                                          int smem_inner_dim, int smem_outer_dim,
                                          const int& gmem_outer_stride,
                                          const int& swizzle_mode, const int& swizzle_base = 0,
                                          const bool& allow_tf32 = false,
                                          const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    const auto num_gmem_outer_stride_bytes = static_cast<uint32_t>(gmem_outer_stride) * static_cast<uint32_t>(elem_size);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
    DG_HOST_ASSERT(num_gmem_outer_stride_bytes % 16u == 0u);
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_inner_dim % 128 == 0);

        // Fix FP4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_inner_dim = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, swizzle: %d (base: %d), elem size: %d, pointer: %llu\n",
               gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
               gmem_outer_stride, swizzle_mode, swizzle_base, elem_size,
               reinterpret_cast<unsigned long long>(t.data_ptr()));
    }
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
        &tensor_map, nvfp4_aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap nvfp4_make_tma_sf_desc(const cute::UMMA::Major& major,
                                          const torch::Tensor& t,
                                          int shape_mn, int shape_k,
                                          const int& block_mn, const int& gran_k,
                                          const int& num_groups,
                                          const int& swizzle_mode, const int& swizzle_base = 0,
                                          const bool& allow_tf32 = false,
                                          const int& smem_outer_dim = 1) {
    DG_HOST_ASSERT(major == cute::UMMA::Major::MN);

    // TODO: maybe swizzle SF as well
    DG_HOST_ASSERT(swizzle_mode == 0);

    shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
    return nvfp4_make_tma_2d_desc(t,
                                  shape_mn, ceil_div(shape_k, gran_k * (t.scalar_type() == torch::kFloat ? 1 : 4)) * num_groups,
                                  block_mn, smem_outer_dim,
                                  shape_mn,
                                  swizzle_mode, swizzle_base,
                                  allow_tf32);
}

static void sm100_nvfp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const float& swiglu_alpha,
    const bool& use_situ,
    const bool& fast_math,
    const bool& use_x_scales,
    const float* l1_alphas,
    const float* l2_alphas,
    const float* l2_act_scales,
    const MmaKind& mma_kind,
    const bool& use_fp8_combine
) {
    // NOTES: sgl's TRT-LLM weight layout is not ported; the kernel is always instantiated without it
    constexpr bool use_trtllm_weights = false;

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_nvfp4_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        mma_kind);

    // Make tensormap
    const bool is_packed_fp4 = mma_kind == MmaKind::NVFP4 or mma_kind == MmaKind::MXFP4;
    const int kGranK = config.gran_k;
    const int elem_bits = get_element_bits(mma_kind);
    const auto to_inner = [=](const int& num_elems) { return num_elems * elem_bits / 8; };
    const auto weight_tensor = [=](const torch::Tensor& t) { return is_packed_fp4 ? t.view(torch::kUInt8) : t; };
    const auto weight_inner = [=](const int& num_elems) { return is_packed_fp4 ? num_elems / 2 : num_elems; };
    const int block_k_inner = to_inner(config.block_k);
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = nvfp4_make_tma_2d_desc(l1_acts,
                                                           to_inner(hidden), config.num_ring_tokens,
                                                           block_k_inner, config.load_block_m,
                                                           static_cast<int>(l1_acts.stride(-2)),
                                                           config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = nvfp4_make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                              config.num_sf_ring_tokens, hidden,
                                                              config.sf_block_m, kGranK,
                                                              1, 0, 0, false,
                                                              sf_smem_outer_dim);
    const auto tensor_map_l1_weights = nvfp4_make_tma_2d_desc(weight_tensor(l1_weights),
                                                              weight_inner(hidden), num_experts_per_rank * intermediate_hidden * 2,
                                                              block_k_inner, config.load_block_n,
                                                              static_cast<int>(l1_weights.stride(-2)),
                                                              config.swizzle_weights_mode, 0, false, not is_packed_fp4);
    const auto tensor_map_l1_weights_sf = nvfp4_make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                                 intermediate_hidden * 2, hidden,
                                                                 config.block_n, kGranK,
                                                                 num_experts_per_rank, 0, 0, false,
                                                                 sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const int l1_out_block_n_bytes = to_inner(config.block_n / 2);
    const auto tensor_map_l1_output = nvfp4_make_tma_2d_desc(l2_acts,
                                                             to_inner(intermediate_hidden), config.num_ring_tokens,
                                                             l1_out_block_n_bytes, config.store_block_m,
                                                             static_cast<int>(l2_acts.stride(-2)),
                                                             l1_out_block_n_bytes);
    const auto tensor_map_l2_acts = nvfp4_make_tma_2d_desc(l2_acts,
                                                           to_inner(intermediate_hidden), config.num_ring_tokens,
                                                           block_k_inner, config.load_block_m,
                                                           static_cast<int>(l2_acts.stride(-2)),
                                                           config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = nvfp4_make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                              config.num_sf_ring_tokens, intermediate_hidden,
                                                              config.sf_block_m, kGranK,
                                                              1, 0, 0, false,
                                                              sf_smem_outer_dim);
    const auto tensor_map_l2_weights = nvfp4_make_tma_2d_desc(weight_tensor(l2_weights),
                                                              weight_inner(intermediate_hidden), num_experts_per_rank * hidden,
                                                              block_k_inner, config.load_block_n,
                                                              static_cast<int>(l2_weights.stride(-2)),
                                                              config.swizzle_weights_mode, 0, false, not is_packed_fp4);
    const auto tensor_map_l2_weights_sf = nvfp4_make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                                 hidden, intermediate_hidden,
                                                                 config.block_n, kGranK,
                                                                 num_experts_per_rank, 0, 0, false,
                                                                 sf_smem_outer_dim);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? nvfp4_make_tma_2d_desc(
        shared_l1_acts,
        to_inner(hidden), num_max_tokens_per_rank,
        block_k_inner, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? nvfp4_make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? nvfp4_make_tma_2d_desc(
        weight_tensor(shared_l1_weights),
        to_inner(hidden), shared_intermediate_hidden * 2,
        block_k_inner, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? nvfp4_make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? nvfp4_make_tma_2d_desc(
        shared_l2_acts,
        to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
        l1_out_block_n_bytes, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        l1_out_block_n_bytes) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? nvfp4_make_tma_2d_desc(
        shared_l2_acts,
        to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
        block_k_inner, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? nvfp4_make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? nvfp4_make_tma_2d_desc(
        weight_tensor(shared_l2_weights),
        to_inner(shared_intermediate_hidden), hidden,
        block_k_inner, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? nvfp4_make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = get_nvfp4_mega_moe_num_sms();

    // Compile
    const auto kernel = jit->compile("sm100_nvfp4_mega_moe", std::format(R"(
#include <deep_gemm/mega_nvfp4/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm::mega_nvfp4;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}, {}, {}, {},
        {}
    >);
}};
)", num_max_tokens_per_rank,
    hidden, intermediate_hidden,
    num_experts, num_shared_experts,
    num_topk,
    config.block_m, config.block_n, config.block_k,
    config.store_block_m,
    config.sf_block_m, config.sf_block_n,
    nvfp4_to_mma_kind_name(mma_kind),
    config.num_ring_tokens,
    config.num_sf_ring_tokens,
    config.num_stages,
    config.num_bytes_per_pull,
    config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
    num_sms, num_ranks,
    to_string(activation_clamp), to_string(swiglu_alpha),
    use_situ ? "true" : "false",
    fast_math ? "true" : "false",
    use_x_scales ? "true" : "false",
    (l1_alphas != nullptr) ? "true" : "false",
    (l2_alphas != nullptr) ? "true" : "false",
    (l2_act_scales != nullptr) ? "true" : "false",
    use_fp8_combine ? "true" : "false",
    to_string(l1_weights.scalar_type()),
    use_trtllm_weights ? "true" : "false"), get_nvfp4_mega_moe_compiler_options());
    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
            .enable_pdl = false,
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        l1_alphas, l2_alphas, l2_act_scales,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_acts_sf,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_weights_sf,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_acts_sf,
        tensor_map_shared_l2_weights,
        tensor_map_shared_l2_weights_sf);
}

} // namespace deep_gemm::mega_nvfp4
