#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "runtime_utils.hpp"
#include "sm100_bf16_mega_moe_wgrad.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/impls/mega_moe_side_lora_params.cuh>

namespace deep_gemm {

// MegaMoE's physical expert boundaries are padded by the forward BLOCK_M.
// Select a grouped-GEMM tile that divides that exact boundary alignment; this
// keeps every tensor-core tile inside one expert without repacking the pool or
// launching a GEMM per expert.
static GemmConfig sm100_bf16_mega_moe_side_lora_rank_config(
    const GemmDesc& desc,
    const int pool_block_m) {
    DG_HOST_ASSERT(pool_block_m >= 16 && pool_block_m <= 256 &&
                   pool_block_m % 16 == 0);
    // The swapped M-grouped kernel maps logical M onto UMMA N, which accepts
    // every 16-row step through 256. Matching the forward tile exactly avoids
    // both cross-expert tiles and any extra pool padding.
    const Layout layout{
        .swap_ab = true,
        .block_m = pool_block_m,
        .block_n = 128,
        .block_k = 64,
        .cluster_m = 1,
        .cluster_n = 1,
    };
    const auto storage = SM100ArchSpec::get_storage_config(desc, layout);
    return GemmConfig{
        .layout = layout,
        .storage_config = storage,
        .pipeline_config =
            SM100ArchSpec::get_pipeline_config(desc, layout, storage),
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout),
    };
}

static void sm100_bf16_mega_moe_side_lora_rank_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& expert_psum_rows,
    const torch::Tensor&,
    const int num_groups,
    const int num_pool_rows,
    const int n,
    const int k,
    const int pool_block_m,
    const cute::UMMA::Major major_a,
    const cute::UMMA::Major major_b) {
    const GemmDesc desc{
        .gemm_type = GemmType::MGroupedContiguousWithPsumLayout,
        .kernel_type = KernelType::KernelNoSF,
        .m = num_pool_rows,
        .n = n,
        .k = k,
        .num_groups = num_groups,
        .a_dtype = a.scalar_type(),
        .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a,
        .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = "nk",
        .ensure_zero_padding = false,
        .expected_m = num_pool_rows / num_groups,
        .expected_n = n,
        .expected_k = k,
        .expected_num_groups = num_groups,
    };
    const auto config =
        sm100_bf16_mega_moe_side_lora_rank_config(desc, pool_block_m);
    const auto tensor_map_a = make_tma_a_desc(
        major_a, a, num_pool_rows, k,
        config.storage_config.load_block_m, config.layout.block_k,
        static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
        config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(
        major_b, b, n, k,
        config.storage_config.load_block_n, config.layout.block_k,
        static_cast<int>(b.stride(get_non_contiguous_dim(major_b))),
        num_groups, config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(
        d, num_pool_rows, n,
        config.storage_config.store_block_m,
        config.storage_config.store_block_n,
        static_cast<int>(d.stride(-2)), 1,
        config.storage_config.swizzle_cd_mode);
    const SM100BF16GemmRuntime::Args args{
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(
            config.launch_config.num_sms,
            config.launch_config.num_threads,
            config.pipeline_config.smem_size,
            config.layout.get_cluster_size()),
        .grouped_layout = expert_psum_rows.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_d,
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build(
        "sm100_bf16_mega_moe_side_lora_rank_gemm", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

// Shared A1/A3/B2 factors are one matrix for the whole EP-local route pool.
// Keep their contractions in the native SM100 tensor-core path, but use one
// dense GEMM instead of repeating the same weight through every expert group.
static void sm100_bf16_mega_moe_side_lora_shared_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const int m,
    const int n,
    const int k,
    const std::string& compiled_dims = "mnk") {
    sm100_bf16_gemm(
        a, b, std::nullopt, d, m, n, k,
        get_major_type_ab(a), get_major_type_ab(b), compiled_dims);
}

static std::string get_side_lora_backward_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

static std::string get_side_lora_backward_combine_order_mode_name(
    const std::string& combine_order_mode) {
    if (combine_order_mode == "fixed_topk")
        return "CombineOrderMode::FixedTopK";
    if (combine_order_mode == "deepep")
        return "CombineOrderMode::DeepEP";
    if (combine_order_mode == "deepep_v1")
        return "CombineOrderMode::DeepEPV1";
    DG_HOST_UNREACHABLE("Unsupported combine order mode");
}

class SM100BF16MegaMoESideLoraBackwardWaveRuntime final
    : public LaunchRuntime<SM100BF16MegaMoESideLoraBackwardWaveRuntime> {
public:
    struct Args {
        int hidden;
        int intermediate_hidden;
        int num_experts;
        int num_pool_rows;
        int num_acts_rows;
        int num_sf_pool_rows;
        int block_m;
        int block_n;
        int block_k;
        int sf_block_m;
        int sf_block_n;
        int num_stages;
        int num_sms;
        int num_ranks;
        bool bf16_mode = false;
        std::string activation = "swiglu";
        bool fast_math = false;
        std::string route_weight_mode = "pre_down";
        std::string combine_order_mode = "fixed_topk";

        const int* expert_counts;
        layout::SymBuffer<> backward_sym_buffer;
        layout::Workspace backward_workspace;
        const cutlass::bfloat16_t* backward_grad_y;
        const cutlass::bfloat16_t* backward_x;
        const float* backward_topk_weights;
        float* backward_grad_route;
        const layout::TokenSrcMetadata* token_src_metadata;
        uint32_t num_topk;
        uint32_t acts_sf_stride;
        CUtensorMap tensor_map_acts;
        CUtensorMap tensor_map_acts_sf;
        CUtensorMap tensor_map_weights;
        CUtensorMap tensor_map_weights_sf;
        CUtensorMap tensor_map_output;
        CUtensorMap tensor_map_grad_ye;
        CUtensorMap tensor_map_w2_dequant;
        CUtensorMap tensor_map_w2_weights;
        CUtensorMap tensor_map_w2_scales;
        CUtensorMap tensor_map_w13_dequant;
        CUtensorMap tensor_map_w13_weights;
        CUtensorMap tensor_map_w13_scales;
        CUtensorMap tensor_map_grad_gate_up;
        const cutlass::float_e4m3_t* acts_ptr;
        const uint32_t* acts_sf_ptr;
        const int8_t* w2_weights;
        const float* w2_scales;
        cutlass::bfloat16_t* w2_dequant_scratch;
        const int8_t* w13_weights;
        const float* w13_scales;
        cutlass::bfloat16_t* w13_dequant_scratch;
        const cutlass::bfloat16_t* gate_up_output;
        cutlass::bfloat16_t* grad_ye_output;
        cutlass::bfloat16_t* grad_y_unweighted_output;
        cutlass::bfloat16_t* route_weights;
        float* route_weights_fp32;
        cutlass::bfloat16_t* grad_h_output;
        cutlass::bfloat16_t* grad_gate_up_output;
        cutlass::bfloat16_t* h_act_output;
        cutlass::bfloat16_t* h_weighted_output;
        cutlass::bfloat16_t* x_pool_output;
        cutlass::bfloat16_t* grad_x_pool_output;
        const cutlass::bfloat16_t* down_unweighted_output;
        float* grad_route_output;
        uint32_t* grid_sync_counter;
        uint32_t launch_epoch;
        float activation_limit;
        MegaMoESideLoraBackwardParams side_lora{};
        bool compute_w13_dgrad;
        bool direct_remote_grad_x;
        bool write_grad_x_pool;
        bool clear_wgrad_padding;
        bool compute_route_grad = false;
        bool trace_kernel = false;
        bool vectorized_grad_x_store = false;
        bool wide_grad_x_store = false;
        bool gate_up_prepared = false;
        uint64_t* kernel_trace = nullptr;
        bool inputs_prepared = false;
        bool dispatch_inputs_prepared = false;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_side_lora_backward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_side_lora_backward_wave_impl<
            {}, {},
            {},
            {}, {}, {},
            {}, {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {}
        >);
}};
)",
            args.hidden, args.intermediate_hidden,
            args.num_experts,
            args.block_m, args.block_n, args.block_k,
            args.sf_block_m, args.sf_block_n,
            args.num_stages,
            args.num_sms,
            args.num_ranks,
            args.compute_w13_dgrad ? "true" : "false",
            args.bf16_mode ? "true" : "false",
            args.activation == "geglu"
                ? "ActivationType::GeGLU"
                : "ActivationType::SwiGLU",
            args.fast_math ? "true" : "false",
            get_side_lora_backward_route_weight_mode_name(
                args.route_weight_mode),
            get_side_lora_backward_combine_order_mode_name(
                args.combine_order_mode),
            args.inputs_prepared ? "true" : "false",
            args.dispatch_inputs_prepared ? "true" : "false",
            args.direct_remote_grad_x ? "true" : "false",
            args.write_grad_x_pool ? "true" : "false",
            args.clear_wgrad_padding ? "true" : "false",
            args.compute_route_grad ? "true" : "false",
            args.trace_kernel ? "true" : "false",
            args.vectorized_grad_x_store ? "true" : "false",
            args.wide_grad_x_store ? "true" : "false",
            args.gate_up_prepared ? "true" : "false");
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.expert_counts,
            args.backward_sym_buffer,
            args.backward_workspace,
            args.backward_grad_y,
            args.backward_x,
            args.backward_topk_weights,
            args.backward_grad_route,
            args.token_src_metadata,
            args.num_topk,
            args.num_pool_rows,
            args.num_acts_rows,
            args.acts_sf_stride,
            args.tensor_map_acts,
            args.tensor_map_acts_sf,
            args.tensor_map_weights,
            args.tensor_map_weights_sf,
            args.tensor_map_output,
            args.tensor_map_grad_ye,
            args.tensor_map_w2_dequant,
            args.tensor_map_w2_weights,
            args.tensor_map_w2_scales,
            args.tensor_map_w13_dequant,
            args.tensor_map_w13_weights,
            args.tensor_map_w13_scales,
            args.tensor_map_grad_gate_up,
            args.acts_ptr,
            args.acts_sf_ptr,
            args.w2_weights,
            args.w2_scales,
            args.w2_dequant_scratch,
            args.w13_weights,
            args.w13_scales,
            args.w13_dequant_scratch,
            args.gate_up_output,
            args.grad_ye_output,
            args.grad_y_unweighted_output,
            args.route_weights,
            args.route_weights_fp32,
            args.grad_h_output,
            args.grad_gate_up_output,
            args.h_act_output,
            args.h_weighted_output,
            args.x_pool_output,
            args.grad_x_pool_output,
            args.down_unweighted_output,
            args.grad_route_output,
            args.grid_sync_counter,
            args.launch_epoch,
            args.activation_limit,
            args.side_lora,
            args.kernel_trace));
    }
};

class SM100BF16MegaMoESideLoraGradXRuntime final
    : public LaunchRuntime<SM100BF16MegaMoESideLoraGradXRuntime> {
public:
    struct Args {
        int hidden;
        int num_experts;
        int block_m;
        int num_ranks;
        int num_sms;
        bool write_grad_x_pool;
        bool direct_remote_grad_x;
        const int* expert_counts;
        cutlass::bfloat16_t* grad_x_pool;
        const layout::TokenSrcMetadata* token_src_metadata;
        cutlass::bfloat16_t* combine_buffer;
        layout::SymBuffer<> sym_buffer;
        layout::Workspace workspace;
        uint32_t num_pool_rows;
        uint32_t num_topk;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_side_lora_backward.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_side_lora_grad_x_impl<
            {}, {}, {}, {}, {}, {}, {}>);
}}
)", args.hidden, args.num_experts, args.block_m, args.num_ranks,
            args.num_sms,
            args.write_grad_x_pool ? "true" : "false",
            args.direct_remote_grad_x ? "true" : "false");
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.expert_counts, args.grad_x_pool,
            args.token_src_metadata, args.combine_buffer,
            args.sym_buffer, args.workspace, args.num_pool_rows,
            args.num_topk));
    }
};

class SM100BF16MegaMoESideLoraScaleGradsRuntime final
    : public LaunchRuntime<SM100BF16MegaMoESideLoraScaleGradsRuntime> {
public:
    struct Args {
        int hidden;
        int intermediate_hidden;
        int num_experts;
        MegaMoESideLoraBackwardParams side_lora;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_side_lora_backward.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_side_lora_scale_grads_impl<{}, {}, {}>);
}}
)", args.hidden, args.intermediate_hidden, args.num_experts);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.side_lora));
    }
};

class SM100BF16MegaMoESideLoraAxpy2Runtime final
    : public LaunchRuntime<SM100BF16MegaMoESideLoraAxpy2Runtime> {
public:
    struct Args {
        int num_sms;
        cutlass::bfloat16_t* dst;
        const cutlass::bfloat16_t* src1;
        const cutlass::bfloat16_t* src3;
        uint64_t num_elements;
        float scale;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_side_lora_backward.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_side_lora_axpy2_impl<{}>);
}}
)", args.num_sms);
    }
    static void launch_impl(
        const KernelHandle& kernel, const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.dst, args.src1, args.src3,
            args.num_elements, args.scale));
    }
};

class SM100BF16MegaMoESideLoraClearPaddingRuntime final
    : public LaunchRuntime<SM100BF16MegaMoESideLoraClearPaddingRuntime> {
public:
    struct Args {
        int hidden;
        int intermediate_hidden;
        int num_experts;
        int block_m;
        int num_sms;
        const int* expert_counts;
        uint32_t num_pool_rows;
        cutlass::bfloat16_t* saved_h;
        cutlass::bfloat16_t* q13;
        cutlass::bfloat16_t* q2;
        cutlass::bfloat16_t* t13;
        cutlass::bfloat16_t* t2;
        cutlass::bfloat16_t* x_pool;
        cutlass::bfloat16_t* grad_ye;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_side_lora_backward.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_side_lora_clear_padding_impl<{}, {}, {}, {}, {}>);
}}
)", args.hidden, args.intermediate_hidden, args.num_experts, args.block_m,
            args.num_sms);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.expert_counts, args.num_pool_rows, args.saved_h,
            args.q13, args.q2, args.t13, args.t2,
            args.x_pool, args.grad_ye));
    }
};


static void sm100_bf16_mega_moe_side_lora_backward(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& grad_ye,
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w13_weights,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const std::string& activation,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const std::string& combine_order_mode,
    const torch::Tensor& down_unweighted_output,
    const int& block_m,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::string& memory_mode,
    const torch::Tensor& side_lora_a1,
    const torch::Tensor& side_lora_b1,
    const torch::Tensor& side_lora_a3,
    const torch::Tensor& side_lora_b3,
    const torch::Tensor& side_lora_a2,
    const torch::Tensor& side_lora_b2,
    const torch::Tensor& side_lora_q13,
    const torch::Tensor& side_lora_q2,
    const torch::Tensor& side_lora_saved_h,
    const torch::Tensor& side_lora_t13,
    const torch::Tensor& side_lora_t2,
    const torch::Tensor& grad_side_lora_a1,
    const torch::Tensor& grad_side_lora_b1,
    const torch::Tensor& grad_side_lora_a3,
    const torch::Tensor& grad_side_lora_b3,
    const torch::Tensor& grad_side_lora_a2,
    const torch::Tensor& grad_side_lora_b2,
    const torch::Tensor& expert_psum_rows,
    const torch::Tensor& padded_expert_counts,
    const float& side_lora_scale,
    const std::optional<torch::Tensor>& kernel_trace =
        std::nullopt) {
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    constexpr int dgrad_block_k = 64;
    constexpr int store_block_m = 16;
    constexpr int smem_capacity = 232448;
    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;

    const auto [num_experts, intermediate_hidden_2, hidden] =
        get_shape<3>(w13_weights);
    const auto [num_experts_w2, hidden_w2, intermediate_hidden] =
        get_shape<3>(w2_weights);
    const int num_pool_rows = static_cast<int>(grad_ye.size(0));
    const int num_ranks =
        static_cast<int>(backward_sym_buffer_ptrs.size());
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = block_n;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int num_dispatch_warps = num_ranks > 1 ? 4 : 0;

    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(activation == "swiglu" || activation == "geglu");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    // The BF16 side-LoRA specialization reuses the now-dead grad-h and
    // activation planes for canonical gate/up derivatives after its W13
    // dgrad consumes the forward-format interleave. Current training uses
    // PRE_DOWN, whose phase plan keeps those planes distinct.
    DG_HOST_ASSERT(route_weight_mode == "pre_down");
    DG_HOST_ASSERT(
        memory_mode == "legacy" ||
        memory_mode == "phase_ordered" ||
        memory_mode == "dispatch_prepared");
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        backward_rank >= 0 && backward_rank < num_ranks);
    DG_HOST_ASSERT(num_experts == num_experts_w2);
    DG_HOST_ASSERT(hidden == hidden_w2);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(hidden % 256 == 0);
    DG_HOST_ASSERT(intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(num_pool_rows > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_topk > 0);
    constexpr int side_lora_rank = 128;

    const auto check_bf16_contiguous = [](const torch::Tensor& tensor) {
        DG_HOST_ASSERT(tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
    };
    check_bf16_contiguous(gate_up_output);
    check_bf16_contiguous(grad_h_output);
    check_bf16_contiguous(grad_gate_up_output);
    check_bf16_contiguous(h_act_output);
    check_bf16_contiguous(h_weighted_output);
    check_bf16_contiguous(x_pool_output);
    check_bf16_contiguous(grad_x_pool_output);
    check_bf16_contiguous(grad_ye);
    check_bf16_contiguous(grad_y_unweighted_output);
    check_bf16_contiguous(down_unweighted_output);
    check_bf16_contiguous(w2_weights);
    check_bf16_contiguous(w13_weights);
    check_bf16_contiguous(backward_grad_y);
    check_bf16_contiguous(backward_x);
    for (const auto* tensor : {
             &side_lora_a1, &side_lora_b1,
             &side_lora_a3, &side_lora_b3,
             &side_lora_a2, &side_lora_b2,
             &side_lora_q13, &side_lora_q2,
             &side_lora_saved_h,
             &side_lora_t13, &side_lora_t2,
             &grad_side_lora_a1, &grad_side_lora_b1,
             &grad_side_lora_a3, &grad_side_lora_b3,
             &grad_side_lora_a2, &grad_side_lora_b2})
        check_bf16_contiguous(*tensor);
    DG_HOST_ASSERT(backward_topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(backward_topk_weights.is_contiguous());
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(backward_grad_route->is_contiguous());
    }
    DG_HOST_ASSERT(route_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(route_weights.is_contiguous());
    DG_HOST_ASSERT(grad_route_output.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(grad_route_output.is_contiguous());
    DG_HOST_ASSERT(expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(expert_counts.is_contiguous());
    DG_HOST_ASSERT(expert_counts.numel() == num_experts);
    DG_HOST_ASSERT(grid_sync_counter.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(grid_sync_counter.is_contiguous());
    DG_HOST_ASSERT(token_src_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(token_src_metadata.is_contiguous());
    DG_HOST_ASSERT(expert_psum_rows.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(expert_psum_rows.is_contiguous());
    DG_HOST_ASSERT(padded_expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(padded_expert_counts.is_contiguous());

    DG_HOST_ASSERT(
        gate_up_output.sizes() ==
        torch::IntArrayRef(
            {num_pool_rows, intermediate_hidden_2}));
    DG_HOST_ASSERT(grad_gate_up_output.sizes() ==
                   gate_up_output.sizes());
    DG_HOST_ASSERT(
        grad_h_output.sizes() ==
        torch::IntArrayRef(
            {num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(h_act_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        h_weighted_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        x_pool_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(
        write_grad_x_pool
            ? grad_x_pool_output.sizes() == x_pool_output.sizes()
            : grad_x_pool_output.sizes() ==
                  torch::IntArrayRef({0, hidden}));
    DG_HOST_ASSERT(grad_ye.sizes() == x_pool_output.sizes());
    DG_HOST_ASSERT(
        grad_y_unweighted_output.sizes() ==
        x_pool_output.sizes());
    DG_HOST_ASSERT(
        down_unweighted_output.sizes() ==
        x_pool_output.sizes());
    DG_HOST_ASSERT(route_weights.numel() == num_pool_rows);
    DG_HOST_ASSERT(grad_route_output.numel() == num_pool_rows);
    DG_HOST_ASSERT(
        backward_grad_y.size(0) >= num_max_tokens_per_rank &&
        backward_grad_y.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_x.size(0) >= num_max_tokens_per_rank &&
        backward_x.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_topk_weights.size(0) >=
            num_max_tokens_per_rank &&
        backward_topk_weights.size(1) == num_topk);
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->size(0) >=
                num_max_tokens_per_rank &&
            backward_grad_route->size(1) == num_topk);
    }
    DG_HOST_ASSERT(
        token_src_metadata.size(0) >= num_pool_rows &&
        token_src_metadata.size(1) == 3);
    DG_HOST_ASSERT(write_grad_x_pool || direct_remote_grad_x);
    DG_HOST_ASSERT(side_lora_a1.sizes() == torch::IntArrayRef(
        {side_lora_rank, hidden}));
    DG_HOST_ASSERT(side_lora_a3.sizes() == side_lora_a1.sizes());
    DG_HOST_ASSERT(side_lora_b1.sizes() == torch::IntArrayRef(
        {num_experts, intermediate_hidden, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_b3.sizes() == side_lora_b1.sizes());
    DG_HOST_ASSERT(side_lora_a2.sizes() == torch::IntArrayRef(
        {num_experts, side_lora_rank, intermediate_hidden}));
    DG_HOST_ASSERT(side_lora_b2.sizes() == torch::IntArrayRef(
        {hidden, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_q13.sizes() == torch::IntArrayRef(
        {num_pool_rows, 2, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_q2.sizes() == torch::IntArrayRef(
        {num_pool_rows, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_saved_h.sizes() == torch::IntArrayRef(
        {num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(side_lora_t13.sizes() == side_lora_q13.sizes());
    DG_HOST_ASSERT(side_lora_t2.sizes() == side_lora_q2.sizes());
    DG_HOST_ASSERT(grad_side_lora_a1.sizes() == torch::IntArrayRef(
        {hidden, side_lora_rank}));
    DG_HOST_ASSERT(grad_side_lora_a3.sizes() ==
                   grad_side_lora_a1.sizes());
    DG_HOST_ASSERT(grad_side_lora_b1.sizes() == torch::IntArrayRef(
        {num_experts, side_lora_rank, intermediate_hidden}));
    DG_HOST_ASSERT(grad_side_lora_b3.sizes() ==
                   grad_side_lora_b1.sizes());
    DG_HOST_ASSERT(grad_side_lora_a2.sizes() == torch::IntArrayRef(
        {num_experts, intermediate_hidden, side_lora_rank}));
    DG_HOST_ASSERT(grad_side_lora_b2.sizes() == torch::IntArrayRef(
        {side_lora_rank, hidden}));
    DG_HOST_ASSERT(expert_psum_rows.numel() == num_experts);
    DG_HOST_ASSERT(padded_expert_counts.numel() == num_experts);
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(kernel_trace->is_cuda());
        DG_HOST_ASSERT(
            kernel_trace->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(kernel_trace->is_contiguous());
        DG_HOST_ASSERT(kernel_trace->device() == grad_ye.device());
    }

    const auto exact_alias = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        return lhs.data_ptr() == rhs.data_ptr() &&
               lhs.numel() == rhs.numel();
    };
    const auto overlaps = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        if (lhs.numel() == 0 || rhs.numel() == 0)
            return false;
        const auto lhs_begin = reinterpret_cast<uintptr_t>(
            lhs.data_ptr());
        const auto rhs_begin = reinterpret_cast<uintptr_t>(
            rhs.data_ptr());
        const auto lhs_end =
            lhs_begin + lhs.numel() * lhs.element_size();
        const auto rhs_end =
            rhs_begin + rhs.numel() * rhs.element_size();
        return std::max(lhs_begin, rhs_begin) <
               std::min(lhs_end, rhs_end);
    };
    const std::array<std::reference_wrapper<const torch::Tensor>, 10>
        alias_tensors = {
            gate_up_output, grad_gate_up_output,
            grad_h_output, h_act_output, h_weighted_output,
            x_pool_output, grad_x_pool_output, grad_ye,
            grad_y_unweighted_output, down_unweighted_output};
    const auto allowed_overlap = [&](const int lhs, const int rhs) {
        const auto pair_is = [=](
            const int first, const int second) {
            return (lhs == first && rhs == second) ||
                   (lhs == second && rhs == first);
        };
        if (route_weight_mode == "pre_down" &&
            pair_is(7, 8))
            return exact_alias(
                grad_ye, grad_y_unweighted_output);
        if (route_weight_mode == "pre_down" &&
            pair_is(7, 9))
            return exact_alias(
                grad_ye, down_unweighted_output);
        if (route_weight_mode == "pre_down" &&
            pair_is(8, 9))
            return exact_alias(
                grad_y_unweighted_output,
                down_unweighted_output);
        if (route_weight_mode == "post_down" &&
            pair_is(3, 4))
            return exact_alias(
                h_act_output, h_weighted_output);
        if (memory_mode != "phase_ordered")
            return false;
        if (pair_is(0, 1))
            return exact_alias(
                gate_up_output, grad_gate_up_output);
        if (route_weight_mode == "post_down") {
            if (pair_is(7, 9))
                return exact_alias(
                    grad_ye, down_unweighted_output);
            if (pair_is(2, 8))
                return
                    grad_h_output.data_ptr() ==
                        grad_y_unweighted_output.data_ptr() &&
                    grad_h_output.numel() <=
                        grad_y_unweighted_output.numel();
            if (pair_is(2, 3) || pair_is(2, 4))
                return exact_alias(
                    grad_h_output,
                    pair_is(2, 3)
                        ? h_act_output
                        : h_weighted_output);
            if (pair_is(3, 8) || pair_is(4, 8))
                return
                    grad_y_unweighted_output.data_ptr() ==
                    (pair_is(3, 8)
                         ? h_act_output.data_ptr()
                         : h_weighted_output.data_ptr());
        } else if (pair_is(3, 4)) {
            return exact_alias(
                h_act_output, h_weighted_output);
        }
        return false;
    };
    for (int lhs = 0; lhs < alias_tensors.size(); ++lhs) {
        for (int rhs = lhs + 1;
             rhs < alias_tensors.size(); ++rhs) {
            DG_HOST_ASSERT(
                !overlaps(
                    alias_tensors[lhs].get(),
                    alias_tensors[rhs].get()) ||
                allowed_overlap(lhs, rhs));
        }
    }
    if (memory_mode == "phase_ordered") {
        DG_HOST_ASSERT(exact_alias(
            gate_up_output, grad_gate_up_output));
        if (route_weight_mode == "post_down") {
            DG_HOST_ASSERT(exact_alias(
                grad_ye, down_unweighted_output));
            DG_HOST_ASSERT(
                grad_h_output.data_ptr() ==
                    grad_y_unweighted_output.data_ptr() &&
                grad_h_output.numel() <=
                    grad_y_unweighted_output.numel());
            DG_HOST_ASSERT(exact_alias(
                grad_h_output, h_act_output));
            DG_HOST_ASSERT(exact_alias(
                grad_h_output, h_weighted_output));
        } else {
            DG_HOST_ASSERT(exact_alias(
                grad_ye, grad_y_unweighted_output));
            DG_HOST_ASSERT(exact_alias(
                h_act_output, h_weighted_output));
        }
    }

    const int num_w2_states =
        num_experts * (hidden / dgrad_block_k) *
        (intermediate_hidden / block_n);
    const int num_w13_states =
        num_experts *
        (intermediate_hidden_2 / dgrad_block_k) *
        (hidden / block_n);
    DG_HOST_ASSERT(
        grid_sync_counter.numel() >=
        num_w2_states + num_w13_states + 2);

    const int smem_cd =
        store_block_m * block_n *
        static_cast<int>(sizeof(cutlass::bfloat16_t)) *
        num_tma_store_stages;
    const int smem_per_stage =
        load_block_m * block_k +
        load_block_n * block_k +
        sf_block_m * static_cast<int>(sizeof(uint32_t)) +
        sf_block_n * static_cast<int>(sizeof(uint32_t)) +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        num_dispatch_warps * hidden *
            static_cast<int>(sizeof(cutlass::bfloat16_t)) +
        smem_cd +
        2 * num_epilogue_stages *
            static_cast<int>(sizeof(uint64_t)) +
        num_dispatch_warps *
            static_cast<int>(sizeof(uint64_t)) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages =
        std::min(
            32,
            (smem_capacity - smem_fixed) / smem_per_stage);
    const int smem_size =
        align(
            smem_fixed + num_stages * smem_per_stage,
            1024);
    DG_HOST_ASSERT(num_stages >= 2);

    // Recompute descriptors are compile-time dead in BF16 mode. Keep valid
    // descriptors in every slot so descriptor prefetch remains well-defined.
    const auto tensor_map_acts = make_tma_2d_desc(
        x_pool_output, hidden, num_pool_rows,
        block_k, load_block_m,
        static_cast<int>(x_pool_output.stride(-2)), 128);
    const auto tensor_map_w13 = make_tma_2d_desc(
        w13_weights, hidden,
        num_experts * intermediate_hidden_2,
        block_k, load_block_n,
        static_cast<int>(w13_weights.stride(-2)), 128);
    const auto tensor_map_w13_dgrad = make_tma_2d_desc(
        w13_weights, hidden,
        num_experts * intermediate_hidden_2,
        load_block_n, dgrad_block_k,
        static_cast<int>(w13_weights.stride(-2)), 128);
    const auto tensor_map_gate_up = make_tma_2d_desc(
        gate_up_output, intermediate_hidden_2, num_pool_rows,
        block_n, store_block_m,
        static_cast<int>(gate_up_output.stride(-2)), 128);
    const auto tensor_map_grad_ye = make_tma_2d_desc(
        grad_ye, hidden, num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_ye.stride(-2)), 128);
    const auto tensor_map_w2 = make_tma_2d_desc(
        w2_weights, intermediate_hidden,
        num_experts * hidden,
        load_block_n, dgrad_block_k,
        intermediate_hidden, 128);
    const auto tensor_map_grad_gate_up = make_tma_2d_desc(
        grad_gate_up_output, intermediate_hidden_2,
        num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(
            grad_gate_up_output.stride(-2)), 128);

    const int num_sms = device_runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0);
    constexpr int num_trace_sites = 22;
    constexpr int num_trace_values = 5;
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(
            kernel_trace->dim() == 3 &&
            kernel_trace->size(0) == num_trace_sites &&
            kernel_trace->size(1) == num_sms &&
            kernel_trace->size(2) == num_trace_values);
    }
    static std::atomic<uint32_t> next_launch_epoch{1};
    uint32_t launch_epoch =
        next_launch_epoch.fetch_add(
            1, std::memory_order_relaxed);
    if (launch_epoch == 0) {
        launch_epoch =
            next_launch_epoch.fetch_add(
                1, std::memory_order_relaxed);
    }

    const auto backward_sym_buffer =
        layout::SymBuffer<>(
            backward_sym_buffer_ptrs, backward_rank);
    const auto backward_workspace = layout::Workspace(
        reinterpret_cast<void*>(
            backward_sym_buffer_ptrs[backward_rank]),
        num_ranks, num_experts * num_ranks,
        num_max_tokens_per_rank, num_topk,
        layout::kMinCandidateBlockM);

    // The only pre-wave result is rank-width t2. Its expansion is consumed
    // directly by the W2 dgrad epilogue below, so no [pool, I] side gradient
    // is ever allocated or written. q2 was saved by the forward specialization.
    const auto side_lora_b2_nt = side_lora_b2.transpose(0, 1);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        grad_ye, side_lora_b2_nt, side_lora_t2,
        num_pool_rows, side_lora_rank, hidden, "nk");
    // Expand t2 through expert-local A2 on tensor cores before entering the
    // persistent base dgrad wave. Reuse its required h_weighted output plane
    // before the wave recomputes it; the wave consumes and overwrites each value
    // with the final combined gradient. This replaces 128 scalar FMAs in the
    // hot W2 epilogue without allocating a side-width gradient buffer.
    const auto side_lora_a2_nt = side_lora_a2.transpose(1, 2);
    sm100_bf16_mega_moe_side_lora_rank_gemm(
        side_lora_t2, side_lora_a2_nt, h_weighted_output,
        expert_psum_rows, padded_expert_counts,
        num_experts, num_pool_rows, intermediate_hidden, side_lora_rank,
        block_m, cute::UMMA::Major::K,
        get_major_type_ab(side_lora_a2_nt));

    MegaMoESideLoraBackwardParams side_lora_params{
        .saved_x = reinterpret_cast<const cutlass::bfloat16_t*>(
            x_pool_output.data_ptr<at::BFloat16>()),
        .saved_h = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_saved_h.data_ptr<at::BFloat16>()),
        .a1 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_a1.data_ptr<at::BFloat16>()),
        .b1 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_b1.data_ptr<at::BFloat16>()),
        .a3 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_a3.data_ptr<at::BFloat16>()),
        .b3 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_b3.data_ptr<at::BFloat16>()),
        .a2 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_a2.data_ptr<at::BFloat16>()),
        .b2 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_b2.data_ptr<at::BFloat16>()),
        .q1 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q13.data_ptr<at::BFloat16>()),
        .q3 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q13.data_ptr<at::BFloat16>()) + side_lora_rank,
        .q2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q2.data_ptr<at::BFloat16>()),
        .t1 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t13.data_ptr<at::BFloat16>()),
        .t3 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t13.data_ptr<at::BFloat16>()) + side_lora_rank,
        .t2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t2.data_ptr<at::BFloat16>()),
        .grad_a1 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_a1.data_ptr<at::BFloat16>()),
        .grad_b1 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_b1.data_ptr<at::BFloat16>()),
        .grad_a3 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_a3.data_ptr<at::BFloat16>()),
        .grad_b3 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_b3.data_ptr<at::BFloat16>()),
        .grad_a2 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_a2.data_ptr<at::BFloat16>()),
        .grad_b2 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_b2.data_ptr<at::BFloat16>()),
        .scale = side_lora_scale,
    };

    const SM100BF16MegaMoESideLoraBackwardWaveRuntime::Args args = {
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_pool_rows = num_pool_rows,
        .num_acts_rows = 0,
        .num_sf_pool_rows = 1,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_ranks = num_ranks,
        .bf16_mode = true,
        .activation = activation,
        .fast_math = fast_math,
        .route_weight_mode = route_weight_mode,
        .combine_order_mode = combine_order_mode,
        .expert_counts = expert_counts.data_ptr<int>(),
        .backward_sym_buffer = backward_sym_buffer,
        .backward_workspace = backward_workspace,
        .backward_grad_y =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                backward_grad_y.data_ptr<at::BFloat16>()),
        .backward_x =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                backward_x.data_ptr<at::BFloat16>()),
        .backward_topk_weights =
            backward_topk_weights.data_ptr<float>(),
        .backward_grad_route =
            backward_grad_route.has_value()
            ? backward_grad_route->data_ptr<float>()
            : nullptr,
        .token_src_metadata =
            reinterpret_cast<
                const layout::TokenSrcMetadata*>(
                token_src_metadata.data_ptr<int>()),
        .num_topk = static_cast<uint32_t>(num_topk),
        .acts_sf_stride = 0,
        .tensor_map_acts = tensor_map_acts,
        .tensor_map_acts_sf = tensor_map_acts,
        .tensor_map_weights = tensor_map_w13,
        .tensor_map_weights_sf = tensor_map_w13,
        .tensor_map_output = tensor_map_gate_up,
        .tensor_map_grad_ye = tensor_map_grad_ye,
        .tensor_map_w2_dequant = tensor_map_w2,
        .tensor_map_w2_weights = tensor_map_w2,
        .tensor_map_w2_scales = tensor_map_w2,
        .tensor_map_w13_dequant = tensor_map_w13_dgrad,
        .tensor_map_w13_weights = tensor_map_w13,
        .tensor_map_w13_scales = tensor_map_w13,
        .tensor_map_grad_gate_up =
            tensor_map_grad_gate_up,
        .acts_ptr = nullptr,
        .acts_sf_ptr = nullptr,
        .w2_weights = nullptr,
        .w2_scales = nullptr,
        .w2_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w2_weights.data_ptr<at::BFloat16>()),
        .w13_weights = nullptr,
        .w13_scales = nullptr,
        .w13_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w13_weights.data_ptr<at::BFloat16>()),
        .gate_up_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                gate_up_output.data_ptr<at::BFloat16>()),
        .grad_ye_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_ye.data_ptr<at::BFloat16>()),
        .grad_y_unweighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_y_unweighted_output
                    .data_ptr<at::BFloat16>()),
        .route_weights =
            nullptr,
        .route_weights_fp32 = route_weights.data_ptr<float>(),
        .grad_h_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_h_output.data_ptr<at::BFloat16>()),
        .grad_gate_up_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_gate_up_output.data_ptr<at::BFloat16>()),
        .h_act_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_act_output.data_ptr<at::BFloat16>()),
        .h_weighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_weighted_output.data_ptr<at::BFloat16>()),
        .x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                x_pool_output.data_ptr<at::BFloat16>()),
        .grad_x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_pool_output.data_ptr<at::BFloat16>()),
        .down_unweighted_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                down_unweighted_output
                    .data_ptr<at::BFloat16>()),
        .grad_route_output =
            grad_route_output.data_ptr<float>(),
        .grid_sync_counter =
            reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
        .launch_epoch = launch_epoch,
        .activation_limit = activation_limit,
        .side_lora = side_lora_params,
        .compute_w13_dgrad = true,
        // The side publisher sends the final base+side value once. Emitting
        // the base value remotely here would only be overwritten later.
        .direct_remote_grad_x = false,
        .write_grad_x_pool = true,
        .clear_wgrad_padding = clear_wgrad_padding,
        .compute_route_grad = true,
        .trace_kernel = kernel_trace.has_value(),
        .vectorized_grad_x_store = get_env<int>(
            "DG_BF16_MEGA_MOE_VECTORIZED_GRAD_X_STORE",
            1) == 1,
        .wide_grad_x_store = get_env<int>(
            "DG_BF16_MEGA_MOE_WIDE_GRAD_X_STORE",
            0) == 1,
        .kernel_trace =
            kernel_trace.has_value()
            ? reinterpret_cast<uint64_t*>(
                  kernel_trace->data_ptr<int64_t>())
            : nullptr,
        .inputs_prepared =
            memory_mode == "phase_ordered" &&
            route_weight_mode == "post_down",
        .dispatch_inputs_prepared =
            memory_mode == "phase_ordered" ||
            memory_mode == "dispatch_prepared",
        .launch_args =
            LaunchArgs(num_sms, 1024, smem_size, 2),
    };
    const auto code =
        SM100BF16MegaMoESideLoraBackwardWaveRuntime::generate(args);
    const auto runtime = compiler->build(
        fmt::format(
            "sm100_bf16_mega_moe_side_lora_backward_trace{}_vec{}_wide{}",
            kernel_trace.has_value(),
            args.vectorized_grad_x_store,
            args.wide_grad_x_store),
        code);
    SM100BF16MegaMoESideLoraBackwardWaveRuntime::launch(
        runtime, args);

    // The activation gradients now exist. Contract each one to rank 128,
    // then let the dedicated publisher consume both contractions directly
    // into the base grad-x destinations. There is no full-width side grad-x.
    const auto grad_gate = h_act_output;
    const auto grad_up = grad_h_output;
    const auto t1 = side_lora_t13.select(1, 0);
    const auto t3 = side_lora_t13.select(1, 1);
    const auto b1_nt = side_lora_b1.transpose(1, 2);
    const auto b3_nt = side_lora_b3.transpose(1, 2);
    sm100_bf16_mega_moe_side_lora_rank_gemm(
        grad_gate, b1_nt, t1, expert_psum_rows,
        padded_expert_counts, num_experts, num_pool_rows, side_lora_rank,
        intermediate_hidden, block_m, cute::UMMA::Major::K,
        get_major_type_ab(b1_nt));
    sm100_bf16_mega_moe_side_lora_rank_gemm(
        grad_up, b3_nt, t3, expert_psum_rows,
        padded_expert_counts, num_experts, num_pool_rows, side_lora_rank,
        intermediate_hidden, block_m, cute::UMMA::Major::K,
        get_major_type_ab(b3_nt));

    // Clear forward padding before the adapter wgrads. B2 is formed now so
    // the dead grad-ye plane can hold the second L1 dgrad expansion.
    const SM100BF16MegaMoESideLoraClearPaddingRuntime::Args clear_args{
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .block_m = block_m,
        .num_sms = num_sms,
        .expert_counts = expert_counts.data_ptr<int>(),
        .num_pool_rows = static_cast<uint32_t>(num_pool_rows),
        .saved_h = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_saved_h.data_ptr<at::BFloat16>()),
        .q13 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q13.data_ptr<at::BFloat16>()),
        .q2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q2.data_ptr<at::BFloat16>()),
        .t13 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t13.data_ptr<at::BFloat16>()),
        .t2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t2.data_ptr<at::BFloat16>()),
        .x_pool = reinterpret_cast<cutlass::bfloat16_t*>(
            x_pool_output.data_ptr<at::BFloat16>()),
        .grad_ye = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_ye.data_ptr<at::BFloat16>()),
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto clear_code =
        SM100BF16MegaMoESideLoraClearPaddingRuntime::generate(clear_args);
    const auto clear_runtime = compiler->build(
        "sm100_bf16_mega_moe_side_lora_clear_padding", clear_code);
    SM100BF16MegaMoESideLoraClearPaddingRuntime::launch(
        clear_runtime, clear_args);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        side_lora_q2.transpose(0, 1), grad_ye.transpose(0, 1),
        grad_side_lora_b2, side_lora_rank, hidden, num_pool_rows, "mn");

    // Reuse the now-dead saved-down and grad-ye planes for the two tensor-core
    // L1 dgrad expansions. One vectorized native add consumes both, so no
    // additional hidden-width side buffer or framework tensor op is needed.
    const auto side_grad_x_scratch1 = down_unweighted_output;
    const auto side_grad_x_scratch3 = grad_ye;
    const auto a1_nt = side_lora_a1.transpose(0, 1);
    const auto a3_nt = side_lora_a3.transpose(0, 1);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        t1, a1_nt, side_grad_x_scratch1,
        num_pool_rows, hidden, side_lora_rank, "nk");
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        t3, a3_nt, side_grad_x_scratch3,
        num_pool_rows, hidden, side_lora_rank, "nk");
    const SM100BF16MegaMoESideLoraAxpy2Runtime::Args axpy2_args{
        .num_sms = num_sms,
        .dst = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_x_pool_output.data_ptr<at::BFloat16>()),
        .src1 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_grad_x_scratch1.data_ptr<at::BFloat16>()),
        .src3 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_grad_x_scratch3.data_ptr<at::BFloat16>()),
        .num_elements = static_cast<uint64_t>(num_pool_rows) * hidden,
        .scale = side_lora_scale,
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto axpy2_code =
        SM100BF16MegaMoESideLoraAxpy2Runtime::generate(axpy2_args);
    const auto axpy2_runtime = compiler->build(
        "sm100_bf16_mega_moe_side_lora_axpy2_grad_x", axpy2_code);
    SM100BF16MegaMoESideLoraAxpy2Runtime::launch(
        axpy2_runtime, axpy2_args);

    const SM100BF16MegaMoESideLoraGradXRuntime::Args grad_x_args{
        .hidden = hidden,
        .num_experts = num_experts,
        .block_m = block_m,
        .num_ranks = num_ranks,
        .num_sms = num_sms,
        .write_grad_x_pool = write_grad_x_pool,
        .direct_remote_grad_x = direct_remote_grad_x,
        .expert_counts = expert_counts.data_ptr<int>(),
        .grad_x_pool = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_x_pool_output.data_ptr<at::BFloat16>()),
        .token_src_metadata = reinterpret_cast<
            const layout::TokenSrcMetadata*>(
            token_src_metadata.data_ptr<int>()),
        .combine_buffer = reinterpret_cast<cutlass::bfloat16_t*>(
            backward_grad_y.data_ptr<at::BFloat16>()),
        .sym_buffer = backward_sym_buffer,
        .workspace = backward_workspace,
        .num_pool_rows = static_cast<uint32_t>(num_pool_rows),
        .num_topk = static_cast<uint32_t>(num_topk),
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto grad_x_code =
        SM100BF16MegaMoESideLoraGradXRuntime::generate(grad_x_args);
    const auto grad_x_runtime = compiler->build(
        "sm100_bf16_mega_moe_side_lora_grad_x", grad_x_code);
    SM100BF16MegaMoESideLoraGradXRuntime::launch(
        grad_x_runtime, grad_x_args);

    // Six rank-128 adapter wgrads replace the three frozen full-width base
    // wgrads. A1/A3/B2 are shared, so each is one reduction across the route
    // pool rather than one output matrix per local expert.
    const auto q1 = side_lora_q13.select(1, 0);
    const auto q3 = side_lora_q13.select(1, 1);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        x_pool_output.transpose(0, 1), t1.transpose(0, 1),
        grad_side_lora_a1, hidden, side_lora_rank, num_pool_rows, "mn");
    sm100_bf16_mega_moe_wgrad_1sm(
        q1, grad_gate, grad_side_lora_b1,
        padded_expert_counts, block_m, {}, true,
        "sm100_bf16_mega_moe_side_lora_wgrad_1sm");
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        x_pool_output.transpose(0, 1), t3.transpose(0, 1),
        grad_side_lora_a3, hidden, side_lora_rank, num_pool_rows, "mn");
    sm100_bf16_mega_moe_wgrad_1sm(
        q3, grad_up, grad_side_lora_b3,
        padded_expert_counts, block_m, {}, true,
        "sm100_bf16_mega_moe_side_lora_wgrad_1sm");
    sm100_bf16_mega_moe_wgrad_1sm(
        side_lora_saved_h, side_lora_t2, grad_side_lora_a2,
        padded_expert_counts, block_m, {}, true,
        "sm100_bf16_mega_moe_side_lora_wgrad_1sm");

    const SM100BF16MegaMoESideLoraScaleGradsRuntime::Args scale_args{
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .side_lora = side_lora_params,
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto scale_code =
        SM100BF16MegaMoESideLoraScaleGradsRuntime::generate(scale_args);
    const auto scale_runtime = compiler->build(
        "sm100_bf16_mega_moe_side_lora_scale_grads", scale_code);
    SM100BF16MegaMoESideLoraScaleGradsRuntime::launch(
        scale_runtime, scale_args);
}

static void sm100_fp8_fp4_mega_moe_side_lora_backward(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& acts,
    const torch::Tensor& acts_sf,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l1_weights_sf,
    const torch::Tensor& grad_ye,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w2_scales,
    const torch::Tensor& w2_dequant_scratch,
    const torch::Tensor& w13_weights,
    const torch::Tensor& w13_scales,
    const torch::Tensor& w13_dequant_scratch,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const bool& compute_w13_dgrad,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const int& block_m,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::optional<torch::Tensor>& backward_grad_y,
    const std::optional<torch::Tensor>& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const std::optional<torch::Tensor>& token_src_metadata,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& grad_y_unweighted_output,
    const std::optional<torch::Tensor>& down_unweighted_output,
    const std::optional<torch::Tensor>& grad_route_output,
    const torch::Tensor& side_lora_a1,
    const torch::Tensor& side_lora_b1,
    const torch::Tensor& side_lora_a3,
    const torch::Tensor& side_lora_b3,
    const torch::Tensor& side_lora_a2,
    const torch::Tensor& side_lora_b2,
    const torch::Tensor& side_lora_q13,
    const torch::Tensor& side_lora_q2,
    const torch::Tensor& side_lora_saved_h,
    const torch::Tensor& side_lora_t13,
    const torch::Tensor& side_lora_t2,
    const torch::Tensor& grad_side_lora_a1,
    const torch::Tensor& grad_side_lora_b1,
    const torch::Tensor& grad_side_lora_a3,
    const torch::Tensor& grad_side_lora_b3,
    const torch::Tensor& grad_side_lora_a2,
    const torch::Tensor& grad_side_lora_b2,
    const torch::Tensor& expert_psum_rows,
    const torch::Tensor& padded_expert_counts,
    const float& side_lora_scale) {
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    constexpr int dgrad_block_k = 64;
    constexpr int store_block_m = 16;
    constexpr int gran_k = 32;
    constexpr int smem_capacity = 232448;
    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;

    const auto [num_experts, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(
            l1_weights, cute::UMMA::Major::K,
            device_runtime->get_arch_major());
    const int intermediate_hidden = intermediate_hidden_2 / 2;
    const int num_pool_rows = static_cast<int>(grad_ye.size(0));
    const int num_acts_rows = static_cast<int>(acts.size(0));
    const int num_sf_pool_rows = static_cast<int>(acts_sf.size(0));
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = block_n;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int num_ranks = backward_sym_buffer_ptrs.empty()
        ? 1
        : static_cast<int>(backward_sym_buffer_ptrs.size());
    const int num_dispatch_warps = num_ranks > 1 ? 4 : 0;

    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(num_ranks == 1 ||
                   (backward_rank >= 0 && backward_rank < num_ranks));
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(hidden % block_k == 0);
    DG_HOST_ASSERT(hidden % dgrad_block_k == 0);
    DG_HOST_ASSERT(intermediate_hidden_2 % block_n == 0);
    DG_HOST_ASSERT(acts.dim() == 2 and acts.size(1) == hidden);
    DG_HOST_ASSERT(num_acts_rows > 0 && num_acts_rows <= num_pool_rows);
    DG_HOST_ASSERT(num_sf_pool_rows > 0);
    DG_HOST_ASSERT(grad_ye.dim() == 2);
    DG_HOST_ASSERT(grad_ye.size(0) == num_pool_rows);
    DG_HOST_ASSERT(grad_ye.size(1) == hidden);
    DG_HOST_ASSERT(grad_ye.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_ye.is_contiguous());
    DG_HOST_ASSERT(
        gate_up_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, intermediate_hidden_2}));
    DG_HOST_ASSERT(grad_gate_up_output.sizes() ==
                   gate_up_output.sizes());
    DG_HOST_ASSERT(grad_h_output.sizes() ==
                   torch::IntArrayRef({num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(h_act_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(h_weighted_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        x_pool_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(grad_x_pool_output.sizes() ==
                   torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(gate_up_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_h_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_gate_up_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(h_act_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(
        h_weighted_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(x_pool_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_x_pool_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(gate_up_output.is_contiguous());
    DG_HOST_ASSERT(grad_h_output.is_contiguous());
    DG_HOST_ASSERT(grad_gate_up_output.is_contiguous());
    DG_HOST_ASSERT(h_act_output.is_contiguous());
    DG_HOST_ASSERT(h_weighted_output.is_contiguous());
    DG_HOST_ASSERT(x_pool_output.is_contiguous());
    DG_HOST_ASSERT(grad_x_pool_output.is_contiguous());
    DG_HOST_ASSERT(
        route_weights.scalar_type() == torch::kFloat ||
        route_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(route_weights.numel() == num_pool_rows);
    DG_HOST_ASSERT(route_weights.is_contiguous());
    const auto check_bf16_hidden_pool =
        [&](const torch::Tensor& tensor) {
            DG_HOST_ASSERT(
                tensor.scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(tensor.is_contiguous());
            DG_HOST_ASSERT(
                tensor.sizes() ==
                torch::IntArrayRef(
                    {num_pool_rows, hidden}));
        };
    if (grad_y_unweighted_output.has_value())
        check_bf16_hidden_pool(
            *grad_y_unweighted_output);
    if (down_unweighted_output.has_value()) {
        DG_HOST_ASSERT(
            down_unweighted_output->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(down_unweighted_output->is_contiguous());
        DG_HOST_ASSERT(down_unweighted_output->dim() == 2);
        DG_HOST_ASSERT(
            down_unweighted_output->size(1) == hidden);
        DG_HOST_ASSERT(
            down_unweighted_output->size(0) > 0 &&
            down_unweighted_output->size(0) <= num_pool_rows);
    }
    if (grad_route_output.has_value()) {
        DG_HOST_ASSERT(
            grad_route_output->scalar_type() ==
            torch::kFloat);
        DG_HOST_ASSERT(grad_route_output->is_contiguous());
        DG_HOST_ASSERT(
            grad_route_output->numel() ==
            num_pool_rows);
    }
    if (route_weight_mode == "post_down") {
        DG_HOST_ASSERT(
            route_weights.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(
            grad_y_unweighted_output.has_value());
        DG_HOST_ASSERT(
            down_unweighted_output.has_value());
        DG_HOST_ASSERT(grad_route_output.has_value());
    }
    DG_HOST_ASSERT(
        w2_weights.scalar_type() ==
        torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(w2_weights.dim() == 3);
    DG_HOST_ASSERT(w2_weights.size(0) == num_experts);
    DG_HOST_ASSERT(w2_weights.size(1) == hidden);
    DG_HOST_ASSERT(w2_weights.size(2) == intermediate_hidden / 2);
    DG_HOST_ASSERT(w2_weights.is_contiguous());
    DG_HOST_ASSERT(w2_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(w2_scales.dim() == 3);
    DG_HOST_ASSERT(w2_scales.size(0) == num_experts);
    DG_HOST_ASSERT(w2_scales.size(1) == hidden);
    DG_HOST_ASSERT(w2_scales.size(2) == intermediate_hidden / gran_k);
    DG_HOST_ASSERT(w2_scales.is_contiguous());
    DG_HOST_ASSERT(
        w2_dequant_scratch.sizes() ==
        torch::IntArrayRef(
            {num_experts, hidden, intermediate_hidden}));
    DG_HOST_ASSERT(
        w2_dequant_scratch.scalar_type() ==
        torch::kBFloat16);
    DG_HOST_ASSERT(w2_dequant_scratch.is_contiguous());
    DG_HOST_ASSERT(
        w13_weights.scalar_type() ==
        torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(w13_weights.dim() == 3);
    DG_HOST_ASSERT(w13_weights.size(0) == 2 * num_experts);
    DG_HOST_ASSERT(w13_weights.size(1) == intermediate_hidden);
    DG_HOST_ASSERT(w13_weights.size(2) == hidden / 2);
    DG_HOST_ASSERT(w13_weights.is_contiguous());
    DG_HOST_ASSERT(w13_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(w13_scales.dim() == 3);
    DG_HOST_ASSERT(w13_scales.size(0) == 2 * num_experts);
    DG_HOST_ASSERT(w13_scales.size(1) == intermediate_hidden);
    DG_HOST_ASSERT(w13_scales.size(2) == hidden / gran_k);
    DG_HOST_ASSERT(w13_scales.is_contiguous());
    DG_HOST_ASSERT(
        w13_dequant_scratch.sizes() ==
        torch::IntArrayRef(
            {num_experts, intermediate_hidden_2, hidden}));
    DG_HOST_ASSERT(
        w13_dequant_scratch.scalar_type() ==
        torch::kBFloat16);
    DG_HOST_ASSERT(w13_dequant_scratch.is_contiguous());
    DG_HOST_ASSERT(expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(expert_counts.numel() == num_experts);
    DG_HOST_ASSERT(expert_counts.is_contiguous());
    DG_HOST_ASSERT(grid_sync_counter.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(
        grid_sync_counter.numel() >=
        num_experts *
            ((hidden / dgrad_block_k) *
                 (intermediate_hidden / block_n) +
             (intermediate_hidden_2 / dgrad_block_k) *
                 (hidden / block_n)) +
            2);
    DG_HOST_ASSERT(grid_sync_counter.is_contiguous());
    if (compute_w13_dgrad)
        DG_HOST_ASSERT(write_grad_x_pool || direct_remote_grad_x);
    else
        DG_HOST_ASSERT(!direct_remote_grad_x);
    if (direct_remote_grad_x) {
        DG_HOST_ASSERT(compute_w13_dgrad);
        DG_HOST_ASSERT(num_ranks > 1);
    }
    if (!backward_sym_buffer_ptrs.empty()) {
        DG_HOST_ASSERT(backward_grad_y.has_value());
        DG_HOST_ASSERT(backward_topk_weights.has_value());
        DG_HOST_ASSERT(token_src_metadata.has_value());
        DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
        DG_HOST_ASSERT(num_topk > 0);
        DG_HOST_ASSERT(
            backward_grad_y->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(backward_grad_y->dim() == 2);
        DG_HOST_ASSERT(backward_grad_y->size(0) >= num_max_tokens_per_rank);
        DG_HOST_ASSERT(backward_grad_y->size(1) == hidden);
        DG_HOST_ASSERT(backward_grad_y->is_contiguous());
        DG_HOST_ASSERT(
            backward_topk_weights->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(backward_topk_weights->dim() == 2);
        DG_HOST_ASSERT(backward_topk_weights->size(0) >=
                       num_max_tokens_per_rank);
        DG_HOST_ASSERT(backward_topk_weights->size(1) == num_topk);
        DG_HOST_ASSERT(backward_topk_weights->is_contiguous());
        if (grad_route_output.has_value()) {
            DG_HOST_ASSERT(backward_grad_route.has_value());
            DG_HOST_ASSERT(
                backward_grad_route->scalar_type() == torch::kFloat);
            DG_HOST_ASSERT(backward_grad_route->dim() == 2);
            DG_HOST_ASSERT(backward_grad_route->size(0) >=
                           num_max_tokens_per_rank);
            DG_HOST_ASSERT(backward_grad_route->size(1) == num_topk);
            DG_HOST_ASSERT(backward_grad_route->is_contiguous());
        }
        DG_HOST_ASSERT(
            token_src_metadata->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(token_src_metadata->dim() == 2);
        // Metadata is an immutable active-layout snapshot. Bucket-only output
        // rows are never visited because expert_counts guards every access.
        DG_HOST_ASSERT(token_src_metadata->size(0) >= num_acts_rows);
        DG_HOST_ASSERT(token_src_metadata->size(1) == 3);
        DG_HOST_ASSERT(token_src_metadata->is_contiguous());
    }

    check_sf_layout(
        l1_weights_sf, intermediate_hidden_2, hidden,
        1, gran_k, num_experts, true, false, torch::kInt);

    const int smem_cd =
        store_block_m * block_n *
        static_cast<int>(sizeof(cutlass::bfloat16_t)) *
        num_tma_store_stages;
    const int smem_per_stage =
        load_block_m * block_k +
        load_block_n * block_k +
        sf_block_m * static_cast<int>(sizeof(uint32_t)) +
        sf_block_n * static_cast<int>(sizeof(uint32_t)) +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        num_dispatch_warps * hidden *
            static_cast<int>(sizeof(cutlass::bfloat16_t)) +
        smem_cd +
        2 * num_epilogue_stages *
            static_cast<int>(sizeof(uint64_t)) +
        num_dispatch_warps *
            static_cast<int>(sizeof(uint64_t)) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages =
        std::min(32, (smem_capacity - smem_fixed) / smem_per_stage);
    const int smem_size =
        align(smem_fixed + num_stages * smem_per_stage, 1024);
    DG_HOST_ASSERT(num_stages >= 2);

    const auto tensor_map_acts = make_tma_2d_desc(
        acts, hidden, num_acts_rows,
        block_k, load_block_m,
        static_cast<int>(acts.stride(-2)), 128);
    const auto tensor_map_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, acts_sf,
        num_sf_pool_rows, hidden,
        sf_block_m, gran_k, 1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, hidden, num_experts * intermediate_hidden_2,
        block_k, load_block_n,
        static_cast<int>(l1_weights.stride(-2)), 128);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_weights_sf,
        intermediate_hidden_2, hidden,
        block_n, gran_k, num_experts, 0);
    const auto tensor_map_gate_up = make_tma_2d_desc(
        gate_up_output, intermediate_hidden_2, num_pool_rows,
        block_n, store_block_m,
        static_cast<int>(gate_up_output.stride(-2)), 128);
    const auto tensor_map_grad_ye = make_tma_2d_desc(
        grad_ye, hidden, num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_ye.stride(-2)), 128);
    const auto tensor_map_w2_dequant = make_tma_2d_desc(
        w2_dequant_scratch, intermediate_hidden,
        num_experts * hidden,
        load_block_n, dgrad_block_k,
        intermediate_hidden, 128);
    const auto tensor_map_w2_weights = make_tma_2d_desc(
        w2_weights, intermediate_hidden / 2,
        num_experts * hidden,
        load_block_n / 2, 256,
        intermediate_hidden / 2, 0);
    const auto tensor_map_w2_scales = make_tma_2d_desc(
        w2_scales, intermediate_hidden / gran_k,
        num_experts * hidden,
        load_block_n / gran_k, 256,
        intermediate_hidden / gran_k, 0);
    const auto tensor_map_w13_dequant = make_tma_2d_desc(
        w13_dequant_scratch, hidden,
        num_experts * intermediate_hidden_2,
        load_block_n, dgrad_block_k,
        hidden, 128);
    const auto tensor_map_w13_weights = make_tma_2d_desc(
        w13_weights, hidden / 2,
        num_experts * intermediate_hidden_2,
        load_block_n / 2, 256,
        hidden / 2, 0);
    const auto tensor_map_w13_scales = make_tma_2d_desc(
        w13_scales, hidden / gran_k,
        num_experts * intermediate_hidden_2,
        load_block_n / gran_k, 256,
        hidden / gran_k, 0);
    const auto tensor_map_grad_gate_up = make_tma_2d_desc(
        grad_gate_up_output, intermediate_hidden_2,
        num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_gate_up_output.stride(-2)), 128);
    // Each launch gets a unique readiness epoch; no host memset is required.
    const int num_sms = device_runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0);
    static std::atomic<uint32_t> next_launch_epoch{1};
    uint32_t launch_epoch =
        next_launch_epoch.fetch_add(1, std::memory_order_relaxed);
    if (launch_epoch == 0)
        launch_epoch =
            next_launch_epoch.fetch_add(1, std::memory_order_relaxed);
    layout::SymBuffer<> backward_sym_buffer{};
    void* backward_workspace_base = nullptr;
    if (!backward_sym_buffer_ptrs.empty()) {
        backward_sym_buffer =
            layout::SymBuffer<>(
                backward_sym_buffer_ptrs, backward_rank);
        backward_workspace_base =
            reinterpret_cast<void*>(
                backward_sym_buffer_ptrs[backward_rank]);
    }
    const auto backward_workspace = layout::Workspace(
        backward_workspace_base, num_ranks,
        num_experts * num_ranks,
        std::max(num_max_tokens_per_rank, 1),
        std::max(num_topk, 1),
        layout::kMinCandidateBlockM);
    constexpr int side_lora_rank = 128;
    DG_HOST_ASSERT(compute_w13_dgrad);
    const auto check_side_bf16 = [](const torch::Tensor& tensor) {
        DG_HOST_ASSERT(tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
    };
    for (const auto* tensor : {
             &side_lora_a1, &side_lora_b1, &side_lora_a3,
             &side_lora_b3, &side_lora_a2, &side_lora_b2,
             &side_lora_q13, &side_lora_q2, &side_lora_saved_h,
             &side_lora_t13, &side_lora_t2,
             &grad_side_lora_a1, &grad_side_lora_b1,
             &grad_side_lora_a3, &grad_side_lora_b3,
             &grad_side_lora_a2, &grad_side_lora_b2})
        check_side_bf16(*tensor);
    DG_HOST_ASSERT(side_lora_a1.sizes() == torch::IntArrayRef(
        {side_lora_rank, hidden}));
    DG_HOST_ASSERT(side_lora_a3.sizes() == side_lora_a1.sizes());
    DG_HOST_ASSERT(side_lora_b1.sizes() == torch::IntArrayRef(
        {num_experts, intermediate_hidden, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_b3.sizes() == side_lora_b1.sizes());
    DG_HOST_ASSERT(side_lora_a2.sizes() == torch::IntArrayRef(
        {num_experts, side_lora_rank, intermediate_hidden}));
    DG_HOST_ASSERT(side_lora_b2.sizes() == torch::IntArrayRef(
        {hidden, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_q13.sizes() == torch::IntArrayRef(
        {num_pool_rows, 2, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_q2.sizes() == torch::IntArrayRef(
        {num_pool_rows, side_lora_rank}));
    DG_HOST_ASSERT(side_lora_saved_h.sizes() == torch::IntArrayRef(
        {num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(side_lora_t13.sizes() == side_lora_q13.sizes());
    DG_HOST_ASSERT(side_lora_t2.sizes() == side_lora_q2.sizes());
    DG_HOST_ASSERT(expert_psum_rows.scalar_type() == torch::kInt &&
                   expert_psum_rows.is_contiguous() &&
                   expert_psum_rows.numel() == num_experts);
    DG_HOST_ASSERT(padded_expert_counts.scalar_type() == torch::kInt &&
                   padded_expert_counts.is_contiguous() &&
                   padded_expert_counts.numel() == num_experts);

    DG_HOST_ASSERT(grad_side_lora_a1.sizes() == torch::IntArrayRef(
        {hidden, side_lora_rank}));
    DG_HOST_ASSERT(grad_side_lora_a3.sizes() == grad_side_lora_a1.sizes());
    DG_HOST_ASSERT(grad_side_lora_b2.sizes() == torch::IntArrayRef(
        {side_lora_rank, hidden}));

    const auto side_lora_b2_nt = side_lora_b2.transpose(0, 1);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        grad_ye, side_lora_b2_nt, side_lora_t2,
        num_pool_rows, side_lora_rank, hidden, "nk");
    const auto side_lora_a2_nt = side_lora_a2.transpose(1, 2);
    sm100_bf16_mega_moe_side_lora_rank_gemm(
        side_lora_t2, side_lora_a2_nt, h_weighted_output,
        expert_psum_rows, padded_expert_counts,
        num_experts, num_pool_rows, intermediate_hidden, side_lora_rank,
        block_m, cute::UMMA::Major::K,
        get_major_type_ab(side_lora_a2_nt));

    MegaMoESideLoraBackwardParams side_lora_params{
        .saved_x = reinterpret_cast<const cutlass::bfloat16_t*>(
            x_pool_output.data_ptr<at::BFloat16>()),
        .saved_h = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_saved_h.data_ptr<at::BFloat16>()),
        .a1 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_a1.data_ptr<at::BFloat16>()),
        .b1 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_b1.data_ptr<at::BFloat16>()),
        .a3 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_a3.data_ptr<at::BFloat16>()),
        .b3 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_b3.data_ptr<at::BFloat16>()),
        .a2 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_a2.data_ptr<at::BFloat16>()),
        .b2 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_lora_b2.data_ptr<at::BFloat16>()),
        .q1 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q13.data_ptr<at::BFloat16>()),
        .q3 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q13.data_ptr<at::BFloat16>()) + side_lora_rank,
        .q2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q2.data_ptr<at::BFloat16>()),
        .t1 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t13.data_ptr<at::BFloat16>()),
        .t3 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t13.data_ptr<at::BFloat16>()) + side_lora_rank,
        .t2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t2.data_ptr<at::BFloat16>()),
        .grad_a1 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_a1.data_ptr<at::BFloat16>()),
        .grad_b1 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_b1.data_ptr<at::BFloat16>()),
        .grad_a3 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_a3.data_ptr<at::BFloat16>()),
        .grad_b3 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_b3.data_ptr<at::BFloat16>()),
        .grad_a2 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_a2.data_ptr<at::BFloat16>()),
        .grad_b2 = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_side_lora_b2.data_ptr<at::BFloat16>()),
        .scale = side_lora_scale,
    };
    const SM100BF16MegaMoESideLoraBackwardWaveRuntime::Args args = {
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_pool_rows = num_pool_rows,
        .num_acts_rows = num_acts_rows,
        .num_sf_pool_rows = num_sf_pool_rows,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_ranks = num_ranks,
        .route_weight_mode = route_weight_mode,
        .expert_counts = expert_counts.data_ptr<int>(),
        .backward_sym_buffer = backward_sym_buffer,
        .backward_workspace = backward_workspace,
        .backward_grad_y = num_ranks > 1
            ? reinterpret_cast<const cutlass::bfloat16_t*>(
                  backward_grad_y->data_ptr<at::BFloat16>())
            : nullptr,
        .backward_x = nullptr,
        .backward_topk_weights = num_ranks > 1
            ? backward_topk_weights->data_ptr<float>()
            : nullptr,
        .backward_grad_route =
            backward_grad_route.has_value()
            ? backward_grad_route->data_ptr<float>()
            : nullptr,
        .token_src_metadata = !backward_sym_buffer_ptrs.empty()
            ? reinterpret_cast<const layout::TokenSrcMetadata*>(
                  token_src_metadata->data_ptr<int>())
            : nullptr,
        .num_topk = static_cast<uint32_t>(num_topk),
        .acts_sf_stride =
            static_cast<uint32_t>(acts_sf.stride(1)),
        .tensor_map_acts = tensor_map_acts,
        .tensor_map_acts_sf = tensor_map_acts_sf,
        .tensor_map_weights = tensor_map_l1_weights,
        .tensor_map_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_output = tensor_map_gate_up,
        .tensor_map_grad_ye = tensor_map_grad_ye,
        .tensor_map_w2_dequant = tensor_map_w2_dequant,
        .tensor_map_w2_weights = tensor_map_w2_weights,
        .tensor_map_w2_scales = tensor_map_w2_scales,
        .tensor_map_w13_dequant = tensor_map_w13_dequant,
        .tensor_map_w13_weights = tensor_map_w13_weights,
        .tensor_map_w13_scales = tensor_map_w13_scales,
        .tensor_map_grad_gate_up = tensor_map_grad_gate_up,
        .acts_ptr =
            reinterpret_cast<const cutlass::float_e4m3_t*>(
                acts.data_ptr()),
        .acts_sf_ptr =
            reinterpret_cast<const uint32_t*>(
                acts_sf.data_ptr<int>()),
        .w2_weights =
            reinterpret_cast<const int8_t*>(
                w2_weights.data_ptr()),
        .w2_scales = w2_scales.data_ptr<float>(),
        .w2_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w2_dequant_scratch.data_ptr<at::BFloat16>()),
        .w13_weights =
            reinterpret_cast<const int8_t*>(
                w13_weights.data_ptr()),
        .w13_scales = w13_scales.data_ptr<float>(),
        .w13_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w13_dequant_scratch.data_ptr<at::BFloat16>()),
        .gate_up_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                gate_up_output.data_ptr<at::BFloat16>()),
        .grad_ye_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_ye.data_ptr<at::BFloat16>()),
        .grad_y_unweighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_y_unweighted_output
                    .value_or(grad_ye)
                    .data_ptr<at::BFloat16>()),
        .route_weights =
            route_weights.scalar_type() == torch::kBFloat16
            ? reinterpret_cast<cutlass::bfloat16_t*>(
                  route_weights.data_ptr<at::BFloat16>())
            : nullptr,
        .route_weights_fp32 =
            route_weights.scalar_type() == torch::kFloat
            ? route_weights.data_ptr<float>()
            : nullptr,
        .grad_h_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_h_output.data_ptr<at::BFloat16>()),
        .grad_gate_up_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_gate_up_output.data_ptr<at::BFloat16>()),
        .h_act_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_act_output.data_ptr<at::BFloat16>()),
        .h_weighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_weighted_output.data_ptr<at::BFloat16>()),
        .x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                x_pool_output.data_ptr<at::BFloat16>()),
        .grad_x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_pool_output.data_ptr<at::BFloat16>()),
        .down_unweighted_output =
            down_unweighted_output.has_value()
            ? reinterpret_cast<
                  const cutlass::bfloat16_t*>(
                  down_unweighted_output
                      ->data_ptr<at::BFloat16>())
            : nullptr,
        .grad_route_output =
            grad_route_output.has_value()
            ? grad_route_output->data_ptr<float>()
            : nullptr,
        .grid_sync_counter =
            reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
        .launch_epoch = launch_epoch,
        .activation_limit = activation_limit,
        .side_lora = side_lora_params,
        .compute_w13_dgrad = compute_w13_dgrad,
        .direct_remote_grad_x = false,
        .write_grad_x_pool = true,
        .clear_wgrad_padding = clear_wgrad_padding,
        .compute_route_grad =
            grad_route_output.has_value(),
        // The native side-LoRA forward already saved the full base+LoRA
        // gate/up preactivation. Replaying the base-only MXFP4 W13 GEMM here
        // would both waste work and erase the side delta before dSwiGLU.
        .gate_up_prepared = true,
        .inputs_prepared = route_weight_mode == "post_down",
        .dispatch_inputs_prepared = true,
        .launch_args = LaunchArgs(
            num_sms, 1024, smem_size, 2),
    };
    const auto code =
        SM100BF16MegaMoESideLoraBackwardWaveRuntime::generate(args);
    const auto runtime = compiler->build(fmt::format(
        "sm100_fp8_fp4_mega_moe_backward_dgrad_swiglu_{}_r{}",
        route_weight_mode,
        grad_route_output.has_value()), code);
    SM100BF16MegaMoESideLoraBackwardWaveRuntime::launch(runtime, args);

    const auto grad_gate =
        grad_gate_up_output.slice(1, 0, intermediate_hidden);
    const auto grad_up = grad_gate_up_output.slice(
        1, intermediate_hidden, intermediate_hidden_2);
    const auto t1 = side_lora_t13.select(1, 0);
    const auto t3 = side_lora_t13.select(1, 1);
    const auto b1_nt = side_lora_b1.transpose(1, 2);
    const auto b3_nt = side_lora_b3.transpose(1, 2);
    sm100_bf16_mega_moe_side_lora_rank_gemm(
        grad_gate, b1_nt, t1, expert_psum_rows,
        padded_expert_counts, num_experts, num_pool_rows,
        side_lora_rank, intermediate_hidden, block_m,
        cute::UMMA::Major::K,
        get_major_type_ab(b1_nt));
    sm100_bf16_mega_moe_side_lora_rank_gemm(
        grad_up, b3_nt, t3, expert_psum_rows,
        padded_expert_counts, num_experts, num_pool_rows,
        side_lora_rank, intermediate_hidden, block_m,
        cute::UMMA::Major::K,
        get_major_type_ab(b3_nt));

    const SM100BF16MegaMoESideLoraClearPaddingRuntime::Args clear_args{
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .block_m = block_m,
        .num_sms = num_sms,
        .expert_counts = expert_counts.data_ptr<int>(),
        .num_pool_rows = static_cast<uint32_t>(num_pool_rows),
        .saved_h = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_saved_h.data_ptr<at::BFloat16>()),
        .q13 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q13.data_ptr<at::BFloat16>()),
        .q2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_q2.data_ptr<at::BFloat16>()),
        .t13 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t13.data_ptr<at::BFloat16>()),
        .t2 = reinterpret_cast<cutlass::bfloat16_t*>(
            side_lora_t2.data_ptr<at::BFloat16>()),
        .x_pool = reinterpret_cast<cutlass::bfloat16_t*>(
            x_pool_output.data_ptr<at::BFloat16>()),
        .grad_ye = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_ye.data_ptr<at::BFloat16>()),
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto clear_code =
        SM100BF16MegaMoESideLoraClearPaddingRuntime::generate(clear_args);
    const auto clear_runtime = compiler->build(
        "sm100_fp8_fp4_mega_moe_side_lora_clear_padding", clear_code);
    SM100BF16MegaMoESideLoraClearPaddingRuntime::launch(
        clear_runtime, clear_args);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        side_lora_q2.transpose(0, 1), grad_ye.transpose(0, 1),
        grad_side_lora_b2, side_lora_rank, hidden, num_pool_rows, "mn");

    const auto side_grad_x_scratch1 = *down_unweighted_output;
    const auto side_grad_x_scratch3 = grad_ye;
    const auto a1_nt = side_lora_a1.transpose(0, 1);
    const auto a3_nt = side_lora_a3.transpose(0, 1);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        t1, a1_nt, side_grad_x_scratch1,
        num_pool_rows, hidden, side_lora_rank, "nk");
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        t3, a3_nt, side_grad_x_scratch3,
        num_pool_rows, hidden, side_lora_rank, "nk");
    const SM100BF16MegaMoESideLoraAxpy2Runtime::Args axpy2_args{
        .num_sms = num_sms,
        .dst = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_x_pool_output.data_ptr<at::BFloat16>()),
        .src1 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_grad_x_scratch1.data_ptr<at::BFloat16>()),
        .src3 = reinterpret_cast<const cutlass::bfloat16_t*>(
            side_grad_x_scratch3.data_ptr<at::BFloat16>()),
        .num_elements = static_cast<uint64_t>(num_pool_rows) * hidden,
        .scale = side_lora_scale,
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto axpy2_code =
        SM100BF16MegaMoESideLoraAxpy2Runtime::generate(axpy2_args);
    const auto axpy2_runtime = compiler->build(
        "sm100_fp8_fp4_mega_moe_side_lora_axpy2_grad_x", axpy2_code);
    SM100BF16MegaMoESideLoraAxpy2Runtime::launch(
        axpy2_runtime, axpy2_args);

    const SM100BF16MegaMoESideLoraGradXRuntime::Args grad_x_args{
        .hidden = hidden,
        .num_experts = num_experts,
        .block_m = block_m,
        .num_ranks = num_ranks,
        .num_sms = num_sms,
        .write_grad_x_pool = write_grad_x_pool,
        .direct_remote_grad_x = direct_remote_grad_x,
        .expert_counts = expert_counts.data_ptr<int>(),
        .grad_x_pool = reinterpret_cast<cutlass::bfloat16_t*>(
            grad_x_pool_output.data_ptr<at::BFloat16>()),
        .token_src_metadata = !backward_sym_buffer_ptrs.empty()
            ? reinterpret_cast<const layout::TokenSrcMetadata*>(
                  token_src_metadata->data_ptr<int>())
            : nullptr,
        .combine_buffer = backward_grad_y.has_value()
            ? reinterpret_cast<cutlass::bfloat16_t*>(
                  backward_grad_y->data_ptr<at::BFloat16>())
            : nullptr,
        .sym_buffer = backward_sym_buffer,
        .workspace = backward_workspace,
        .num_pool_rows = static_cast<uint32_t>(num_pool_rows),
        .num_topk = static_cast<uint32_t>(num_topk),
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto grad_x_code =
        SM100BF16MegaMoESideLoraGradXRuntime::generate(grad_x_args);
    const auto grad_x_runtime = compiler->build(
        "sm100_fp8_fp4_mega_moe_side_lora_grad_x", grad_x_code);
    SM100BF16MegaMoESideLoraGradXRuntime::launch(
        grad_x_runtime, grad_x_args);

    const auto q1 = side_lora_q13.select(1, 0);
    const auto q3 = side_lora_q13.select(1, 1);
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        x_pool_output.transpose(0, 1), t1.transpose(0, 1),
        grad_side_lora_a1, hidden, side_lora_rank, num_pool_rows, "mn");
    sm100_bf16_mega_moe_wgrad_1sm(
        q1, grad_gate, grad_side_lora_b1,
        padded_expert_counts, block_m, {}, true,
        "sm100_fp8_fp4_mega_moe_side_lora_wgrad_1sm");
    sm100_bf16_mega_moe_side_lora_shared_gemm(
        x_pool_output.transpose(0, 1), t3.transpose(0, 1),
        grad_side_lora_a3, hidden, side_lora_rank, num_pool_rows, "mn");
    sm100_bf16_mega_moe_wgrad_1sm(
        q3, grad_up, grad_side_lora_b3,
        padded_expert_counts, block_m, {}, true,
        "sm100_fp8_fp4_mega_moe_side_lora_wgrad_1sm");
    sm100_bf16_mega_moe_wgrad_1sm(
        side_lora_saved_h, side_lora_t2, grad_side_lora_a2,
        padded_expert_counts, block_m, {}, true,
        "sm100_fp8_fp4_mega_moe_side_lora_wgrad_1sm");

    const SM100BF16MegaMoESideLoraScaleGradsRuntime::Args scale_args{
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .side_lora = side_lora_params,
        .launch_args = LaunchArgs(num_sms, 256, 0, 1),
    };
    const auto scale_code =
        SM100BF16MegaMoESideLoraScaleGradsRuntime::generate(scale_args);
    const auto scale_runtime = compiler->build(
        "sm100_fp8_fp4_mega_moe_side_lora_scale_grads", scale_code);
    SM100BF16MegaMoESideLoraScaleGradsRuntime::launch(
        scale_runtime, scale_args);
}


}  // namespace deep_gemm
