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

static std::string get_bf16_side_lora_activation_type_name(
    const std::string& activation) {
    if (activation == "swiglu")
        return "ActivationType::SwiGLU";
    if (activation == "geglu")
        return "ActivationType::GeGLU";
    DG_HOST_UNREACHABLE("Unsupported activation");
}

static std::string get_bf16_side_lora_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

static std::string get_bf16_side_lora_combine_order_mode_name(
    const std::string& combine_order_mode) {
    if (combine_order_mode == "fixed_topk")
        return "CombineOrderMode::FixedTopK";
    if (combine_order_mode == "deepep")
        return "CombineOrderMode::DeepEP";
    if (combine_order_mode == "deepep_v1")
        return "CombineOrderMode::DeepEPV1";
    DG_HOST_UNREACHABLE("Unsupported combine order mode");
}

class SM100BF16MegaMoESideLoraForwardRuntime final : public LaunchRuntime<SM100BF16MegaMoESideLoraForwardRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        std::string activation;
        bool save_l1_preact;
        bool save_stage_activations;
        std::string route_weight_mode;
        std::string combine_order_mode;
        bool save_down_unweighted;
        bool save_x;
        int side_lora_rank;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        void* saved_l1_preact;
        void* saved_h_unweighted;
        void* saved_h_weighted;
        void* saved_x;
        void* saved_down_unweighted;
        void* side_lora_a1;
        void* side_lora_b1;
        void* side_lora_a3;
        void* side_lora_b3;
        void* side_lora_a2;
        void* side_lora_b2;
        void* side_lora_l1_scratch;
        void* side_lora_l2_scratch;
        int* side_lora_ready;
        float side_lora_scale;
        int* cumulative_local_expert_recv_stats;
        const int* precomputed_route_counts;
        int* route_count_mismatch;
        int num_tokens;
        int num_saved_pool_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_lora_a1;
        CUtensorMap tensor_map_lora_a3;
        CUtensorMap tensor_map_lora_b1;
        CUtensorMap tensor_map_lora_b3;
        CUtensorMap tensor_map_lora_a2;
        CUtensorMap tensor_map_lora_b2;
        CUtensorMap tensor_map_lora_l1_scratch;
        CUtensorMap tensor_map_lora_l2_scratch;
        CUtensorMap tensor_map_lora_l1_scratch_store;
        CUtensorMap tensor_map_lora_l2_scratch_store;
        CUtensorMap tensor_map_down_unweighted;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe_side_lora_forward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_side_lora_forward_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {},
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
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.num_ring_tokens,
    args.config.num_stages,
    args.config.num_bytes_per_pull,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    get_bf16_side_lora_activation_type_name(args.activation),
    args.save_l1_preact ? "true" : "false",
    args.save_stage_activations ? "true" : "false",
    get_bf16_side_lora_route_weight_mode_name(args.route_weight_mode),
    get_bf16_side_lora_combine_order_mode_name(args.combine_order_mode),
    args.save_down_unweighted ? "true" : "false",
    args.save_x ? "true" : "false",
    args.side_lora_rank);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.saved_l1_preact,
            args.saved_h_unweighted,
            args.saved_h_weighted,
            args.saved_x,
            args.saved_down_unweighted,
            args.side_lora_a1,
            args.side_lora_b1,
            args.side_lora_a3,
            args.side_lora_b3,
            args.side_lora_a2,
            args.side_lora_b2,
            args.side_lora_l1_scratch,
            args.side_lora_l2_scratch,
            args.side_lora_ready,
            args.side_lora_scale,
            args.cumulative_local_expert_recv_stats,
            args.precomputed_route_counts,
            args.route_count_mismatch,
            args.num_tokens,
            args.num_saved_pool_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_weights,
            args.tensor_map_lora_a1,
            args.tensor_map_lora_a3,
            args.tensor_map_lora_b1,
            args.tensor_map_lora_b3,
            args.tensor_map_lora_a2,
            args.tensor_map_lora_b2,
            args.tensor_map_lora_l1_scratch,
            args.tensor_map_lora_l2_scratch,
            args.tensor_map_lora_l1_scratch_store,
            args.tensor_map_lora_l2_scratch_store,
            args.tensor_map_down_unweighted
        ));
    }
};

static void sm100_bf16_mega_moe_side_lora_forward(
    const torch::Tensor& y,
    const std::optional<torch::Tensor>& saved_l1_preact,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_config_tokens,
    const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& activation,
    const float& activation_clamp,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_h_unweighted,
    const std::optional<torch::Tensor>& saved_h_weighted,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const std::string& combine_order_mode,
    const std::optional<torch::Tensor>& precomputed_route_counts,
    const std::optional<int>& active_pool_rows,
    const std::optional<torch::Tensor>& route_count_mismatch,
    const std::optional<torch::Tensor>& saved_x,
    const std::optional<torch::Tensor>& side_lora_a1,
    const std::optional<torch::Tensor>& side_lora_b1,
    const std::optional<torch::Tensor>& side_lora_a3,
    const std::optional<torch::Tensor>& side_lora_b3,
    const std::optional<torch::Tensor>& side_lora_a2,
    const std::optional<torch::Tensor>& side_lora_b2,
    const std::optional<torch::Tensor>& side_lora_l1_scratch,
    const std::optional<torch::Tensor>& side_lora_l2_scratch,
    const std::optional<torch::Tensor>& side_lora_ready,
    const float& side_lora_scale
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_config_tokens, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);
    const auto num_max_pool_tokens =
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts_per_rank);
    const auto num_saved_pool_tokens =
        active_pool_rows.value_or(num_max_pool_tokens);
    DG_HOST_ASSERT(
        num_saved_pool_tokens > 0 &&
        num_saved_pool_tokens <= num_max_pool_tokens);
    if (saved_l1_preact.has_value()) {
        DG_HOST_ASSERT(saved_l1_preact->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_l1_preact->is_contiguous());
        DG_HOST_ASSERT(
            saved_l1_preact->sizes() ==
            torch::IntArrayRef(
                {num_saved_pool_tokens, 2 * intermediate_hidden}));
    }
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    DG_HOST_ASSERT(
        saved_h_unweighted.has_value() ==
        saved_h_weighted.has_value());
    if (saved_h_unweighted.has_value()) {
        for (const auto* saved :
             {&*saved_h_unweighted, &*saved_h_weighted}) {
            DG_HOST_ASSERT(
                saved->scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(saved->is_contiguous());
            DG_HOST_ASSERT(
                saved->sizes() == torch::IntArrayRef(
                    {num_saved_pool_tokens, intermediate_hidden}));
        }
    }
    if (saved_down_unweighted.has_value()) {
        DG_HOST_ASSERT(
            saved_down_unweighted->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_down_unweighted->is_contiguous());
        DG_HOST_ASSERT(
            saved_down_unweighted->sizes() ==
            torch::IntArrayRef({num_saved_pool_tokens, hidden}));
    }
    if (saved_x.has_value()) {
        DG_HOST_ASSERT(saved_x->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_x->is_contiguous());
        DG_HOST_ASSERT(
            saved_x->sizes() ==
            torch::IntArrayRef({num_saved_pool_tokens, hidden}));
    }

    const bool has_side_lora = side_lora_a1.has_value();
    DG_HOST_ASSERT(has_side_lora);
    DG_HOST_ASSERT(
        has_side_lora == side_lora_b1.has_value() &&
        has_side_lora == side_lora_a3.has_value() &&
        has_side_lora == side_lora_b3.has_value() &&
        has_side_lora == side_lora_a2.has_value() &&
        has_side_lora == side_lora_b2.has_value() &&
        has_side_lora == side_lora_l1_scratch.has_value() &&
        has_side_lora == side_lora_l2_scratch.has_value() &&
        has_side_lora == side_lora_ready.has_value());
    int side_lora_rank = 0;
    if (has_side_lora) {
        side_lora_rank = static_cast<int>(side_lora_a1->size(0));
        DG_HOST_ASSERT(side_lora_rank == 128);
        const auto check_bf16_contiguous = [&y](
                                                const torch::Tensor& tensor) {
            DG_HOST_ASSERT(tensor.scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(tensor.is_contiguous());
            DG_HOST_ASSERT(tensor.device() == y.device());
        };
        for (const auto* tensor : {
                 &*side_lora_a1, &*side_lora_b1,
                 &*side_lora_a3, &*side_lora_b3,
                 &*side_lora_a2, &*side_lora_b2,
                 &*side_lora_l1_scratch,
                 &*side_lora_l2_scratch})
            check_bf16_contiguous(*tensor);
        DG_HOST_ASSERT(side_lora_a1->sizes() == torch::IntArrayRef(
            {side_lora_rank, hidden}));
        DG_HOST_ASSERT(side_lora_b1->sizes() == torch::IntArrayRef(
            {num_experts_per_rank, intermediate_hidden, side_lora_rank}));
        DG_HOST_ASSERT(side_lora_a3->sizes() == side_lora_a1->sizes());
        DG_HOST_ASSERT(side_lora_b3->sizes() == side_lora_b1->sizes());
        DG_HOST_ASSERT(side_lora_a2->sizes() == torch::IntArrayRef(
            {num_experts_per_rank, side_lora_rank, intermediate_hidden}));
        DG_HOST_ASSERT(side_lora_b2->sizes() == torch::IntArrayRef(
            {hidden, side_lora_rank}));
        DG_HOST_ASSERT(side_lora_l1_scratch->sizes() == torch::IntArrayRef(
            {num_saved_pool_tokens, 2, side_lora_rank}));
        DG_HOST_ASSERT(side_lora_l2_scratch->sizes() == torch::IntArrayRef(
            {num_saved_pool_tokens, side_lora_rank}));
        DG_HOST_ASSERT(side_lora_ready->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(side_lora_ready->is_contiguous());
        DG_HOST_ASSERT(side_lora_ready->device() == y.device());
        DG_HOST_ASSERT(
            side_lora_ready->numel() >=
            4 * num_ring_tokens / config.block_m);
        side_lora_ready->zero_();
    }

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_lora_a1 = has_side_lora
        ? make_tma_2d_desc(*side_lora_a1, hidden,
              side_lora_rank,
              config.block_k, config.load_block_n,
              static_cast<int>(side_lora_a1->stride(-2)),
              config.swizzle_weights_mode)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_a3 = has_side_lora
        ? make_tma_2d_desc(*side_lora_a3, hidden,
              side_lora_rank,
              config.block_k, config.load_block_n,
              static_cast<int>(side_lora_a3->stride(-2)),
              config.swizzle_weights_mode)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_b1 = has_side_lora
        ? make_tma_2d_desc(*side_lora_b1, side_lora_rank,
              num_experts_per_rank * intermediate_hidden,
              config.block_k, 8,
              static_cast<int>(side_lora_b1->stride(-2)),
              config.swizzle_weights_mode)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_b3 = has_side_lora
        ? make_tma_2d_desc(*side_lora_b3, side_lora_rank,
              num_experts_per_rank * intermediate_hidden,
              config.block_k, 8,
              static_cast<int>(side_lora_b3->stride(-2)),
              config.swizzle_weights_mode)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_a2 = has_side_lora
        ? make_tma_2d_desc(*side_lora_a2, intermediate_hidden,
              num_experts_per_rank * side_lora_rank,
              config.block_k, config.load_block_n,
              static_cast<int>(side_lora_a2->stride(-2)),
              config.swizzle_weights_mode)
        : tensor_map_l2_weights;
    const auto tensor_map_lora_b2 = has_side_lora
        ? make_tma_2d_desc(*side_lora_b2, side_lora_rank,
              hidden,
              config.block_k, config.load_block_n,
              static_cast<int>(side_lora_b2->stride(-2)),
              config.swizzle_weights_mode)
        : tensor_map_l2_weights;
    const auto tensor_map_lora_l1_scratch = has_side_lora
        ? make_tma_2d_desc(*side_lora_l1_scratch,
              2 * side_lora_rank, num_saved_pool_tokens,
              config.block_k, config.load_block_m,
              static_cast<int>(side_lora_l1_scratch->stride(0)),
              config.swizzle_acts_mode)
        : tensor_map_l1_acts;
    const auto tensor_map_lora_l2_scratch = has_side_lora
        ? make_tma_2d_desc(*side_lora_l2_scratch,
              side_lora_rank, num_saved_pool_tokens,
              config.block_k, config.load_block_m,
              static_cast<int>(side_lora_l2_scratch->stride(0)),
              config.swizzle_acts_mode)
        : tensor_map_l2_acts;
    const auto tensor_map_lora_l1_scratch_store = has_side_lora
        ? make_tma_2d_desc(*side_lora_l1_scratch,
              2 * side_lora_rank, num_saved_pool_tokens,
              config.block_k, config.store_block_m,
              static_cast<int>(side_lora_l1_scratch->stride(0)),
              config.swizzle_acts_mode)
        : tensor_map_l1_acts;
    const auto tensor_map_lora_l2_scratch_store = has_side_lora
        ? make_tma_2d_desc(*side_lora_l2_scratch,
              side_lora_rank, num_saved_pool_tokens,
              config.block_k, config.store_block_m,
              static_cast<int>(side_lora_l2_scratch->stride(0)),
              config.swizzle_acts_mode)
        : tensor_map_l2_acts;
    const auto tensor_map_down_unweighted =
        saved_down_unweighted.has_value()
        ? make_tma_2d_desc(
              *saved_down_unweighted,
              hidden, saved_down_unweighted->size(0),
              config.block_n, config.store_block_m,
              static_cast<int>(saved_down_unweighted->stride(-2)),
              config.swizzle_acts_mode)
        : tensor_map_l2_acts;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto physical_num_sms = device_runtime->get_num_sms();
    const auto num_sms = get_env<int>(
        "DG_BF16_MEGA_MOE_NUM_SMS",
        physical_num_sms);
    DG_HOST_ASSERT(num_sms > 0 && num_sms <= physical_num_sms);
    const SM100BF16MegaMoESideLoraForwardRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .activation = activation,
        .save_l1_preact = saved_l1_preact.has_value(),
        .save_stage_activations =
            saved_h_unweighted.has_value(),
        .route_weight_mode = route_weight_mode,
        .combine_order_mode = combine_order_mode,
        .save_down_unweighted =
            saved_down_unweighted.has_value(),
        .save_x = saved_x.has_value(),
        .side_lora_rank = side_lora_rank,
        .config = config,
        .y = y.data_ptr(),
        .saved_l1_preact = saved_l1_preact.has_value()
            ? saved_l1_preact->data_ptr()
            : nullptr,
        .saved_h_unweighted =
            saved_h_unweighted.has_value()
            ? saved_h_unweighted->data_ptr()
            : nullptr,
        .saved_h_weighted =
            saved_h_weighted.has_value()
            ? saved_h_weighted->data_ptr()
            : nullptr,
        .saved_x = saved_x.has_value()
            ? saved_x->data_ptr()
            : nullptr,
        .saved_down_unweighted =
            saved_down_unweighted.has_value()
            ? saved_down_unweighted->data_ptr()
            : nullptr,
        .side_lora_a1 = has_side_lora
            ? side_lora_a1->data_ptr() : nullptr,
        .side_lora_b1 = has_side_lora
            ? side_lora_b1->data_ptr() : nullptr,
        .side_lora_a3 = has_side_lora
            ? side_lora_a3->data_ptr() : nullptr,
        .side_lora_b3 = has_side_lora
            ? side_lora_b3->data_ptr() : nullptr,
        .side_lora_a2 = has_side_lora
            ? side_lora_a2->data_ptr() : nullptr,
        .side_lora_b2 = has_side_lora
            ? side_lora_b2->data_ptr() : nullptr,
        .side_lora_l1_scratch = has_side_lora
            ? side_lora_l1_scratch->data_ptr() : nullptr,
        .side_lora_l2_scratch = has_side_lora
            ? side_lora_l2_scratch->data_ptr() : nullptr,
        .side_lora_ready = has_side_lora
            ? side_lora_ready->data_ptr<int>() : nullptr,
        .side_lora_scale = side_lora_scale,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .precomputed_route_counts =
            precomputed_route_counts.has_value()
            ? precomputed_route_counts->data_ptr<int>()
            : nullptr,
        .route_count_mismatch =
            route_count_mismatch.has_value()
            ? route_count_mismatch->data_ptr<int>()
            : nullptr,
        .num_tokens = num_tokens,
        .num_saved_pool_tokens = num_saved_pool_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_lora_a1 = tensor_map_lora_a1,
        .tensor_map_lora_a3 = tensor_map_lora_a3,
        .tensor_map_lora_b1 = tensor_map_lora_b1,
        .tensor_map_lora_b3 = tensor_map_lora_b3,
        .tensor_map_lora_a2 = tensor_map_lora_a2,
        .tensor_map_lora_b2 = tensor_map_lora_b2,
        .tensor_map_lora_l1_scratch =
            tensor_map_lora_l1_scratch,
        .tensor_map_lora_l2_scratch =
            tensor_map_lora_l2_scratch,
        .tensor_map_lora_l1_scratch_store =
            tensor_map_lora_l1_scratch_store,
        .tensor_map_lora_l2_scratch_store =
            tensor_map_lora_l2_scratch_store,
        .tensor_map_down_unweighted =
            tensor_map_down_unweighted,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100BF16MegaMoESideLoraForwardRuntime::generate(args);
    const auto runtime = compiler->build(
        "sm100_bf16_mega_moe_side_lora_forward", code);
    SM100BF16MegaMoESideLoraForwardRuntime::launch(runtime, args);
}

} // namespace deep_gemm
