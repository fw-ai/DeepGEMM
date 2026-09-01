#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// Map an activation name to its `deep_gemm::ActivationType` enumerator token
// (resolved inside the JIT-generated translation unit via `using namespace deep_gemm`).
static std::string get_activation_type_name(const std::string& activation) {
    if (activation == "swiglu")
        return "ActivationType::SwiGLU";
    if (activation == "geglu")
        return "ActivationType::GeGLU";
    if (activation == "situ")
        return "ActivationType::SiTU";
    DG_HOST_UNREACHABLE("Unsupported activation");
}

static std::string get_fp8_fp4_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        float situ_beta;
        float situ_linear_beta;
        bool fast_math;
        std::string activation;
        bool save_l1_preact;
        std::string route_weight_mode;
        bool save_down_unweighted;
        int l2_activation_group_size;
        int source_token_offset;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        void* saved_l1_preact;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        int num_saved_pool_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        CUtensorMap tensor_map_down_unweighted;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define DEEP_GEMM_MEGA_MOE_SOURCE_TOKEN_OFFSET {}
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
#undef DEEP_GEMM_MEGA_MOE_SOURCE_TOKEN_OFFSET

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {}, {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)", args.source_token_offset,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_ring_tokens,
    args.config.num_sf_ring_tokens,
    args.config.num_stages,
    args.config.num_bytes_per_pull,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    get_activation_type_name(args.activation),
    to_string(args.situ_beta),
    to_string(args.situ_linear_beta),
    args.save_l1_preact ? "true" : "false",
    get_fp8_fp4_route_weight_mode_name(
        args.route_weight_mode),
    args.save_down_unweighted ? "true" : "false",
    args.l2_activation_group_size == 128 ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.saved_l1_preact,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.num_saved_pool_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.tensor_map_down_unweighted
        ));
    }
};

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_config_tokens,
    const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& activation,
    const float& activation_clamp,
    const float& situ_beta,
    const float& situ_linear_beta,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const int& l2_activation_group_size,
    const int& source_token_offset
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_config_tokens, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4,
        l2_activation_group_size);
    const auto num_max_pool_tokens =
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts_per_rank);
    if (saved_l1_preact.has_value()) {
        DG_HOST_ASSERT(saved_l1_preact->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_l1_preact->is_contiguous());
        DG_HOST_ASSERT(saved_l1_preact->dim() == 2);
        DG_HOST_ASSERT(saved_l1_preact->size(1) ==
                       2 * intermediate_hidden);
        DG_HOST_ASSERT(saved_l1_preact->size(0) > 0);
        DG_HOST_ASSERT(saved_l1_preact->size(0) %
                       config.block_m == 0);
        DG_HOST_ASSERT(saved_l1_preact->size(0) <=
                       num_max_pool_tokens);
    }
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        l2_activation_group_size == 32 ||
        l2_activation_group_size == 128);
    DG_HOST_ASSERT(source_token_offset >= 0);
    DG_HOST_ASSERT(
        source_token_offset + num_tokens <=
        num_max_tokens_per_rank);
    if (saved_l1_preact.has_value() &&
        saved_down_unweighted.has_value()) {
        DG_HOST_ASSERT(saved_l1_preact->size(0) ==
                       saved_down_unweighted->size(0));
    }
    if (saved_down_unweighted.has_value()) {
        DG_HOST_ASSERT(
            saved_down_unweighted->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(saved_down_unweighted->is_contiguous());
        DG_HOST_ASSERT(saved_down_unweighted->dim() == 2);
        DG_HOST_ASSERT(saved_down_unweighted->size(1) == hidden);
        DG_HOST_ASSERT(saved_down_unweighted->size(0) > 0);
        DG_HOST_ASSERT(
            saved_down_unweighted->size(0) %
                config.block_m == 0);
        DG_HOST_ASSERT(
            saved_down_unweighted->size(0) <=
                       num_max_pool_tokens);
    }
    // Make tensormap
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_down_unweighted =
        saved_down_unweighted.has_value()
        ? make_tma_2d_desc(
              *saved_down_unweighted,
              hidden, saved_down_unweighted->size(0),
              config.block_n, config.store_block_m,
              static_cast<int>(
                  saved_down_unweighted->stride(-2)),
              config.swizzle_acts_mode)
        : tensor_map_l2_acts;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .situ_beta = situ_beta,
        .situ_linear_beta = situ_linear_beta,
        .fast_math = fast_math,
        .activation = activation,
        .save_l1_preact = saved_l1_preact.has_value(),
        .route_weight_mode = route_weight_mode,
        .save_down_unweighted =
            saved_down_unweighted.has_value(),
        .l2_activation_group_size = l2_activation_group_size,
        .source_token_offset = source_token_offset,
        .config = config,
        .y = y.data_ptr(),
        .saved_l1_preact = saved_l1_preact.has_value()
            ? saved_l1_preact->data_ptr()
            : nullptr,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .num_saved_pool_tokens =
            saved_down_unweighted.has_value()
            ? static_cast<int>(saved_down_unweighted->size(0))
            : saved_l1_preact.has_value()
            ? static_cast<int>(saved_l1_preact->size(0))
            : num_max_pool_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .tensor_map_down_unweighted =
            tensor_map_down_unweighted,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
