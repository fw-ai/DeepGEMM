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
static std::string get_mxfp4_side_lora_activation_type_name(
    const std::string& activation) {
    if (activation == "swiglu")
        return "ActivationType::SwiGLU";
    if (activation == "geglu")
        return "ActivationType::GeGLU";
    DG_HOST_UNREACHABLE("Unsupported activation");
}

static std::string get_mxfp4_side_lora_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

class SM100FP8FP4MegaMoESideLoraForwardRuntime final : public LaunchRuntime<SM100FP8FP4MegaMoESideLoraForwardRuntime> {
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
        std::string route_weight_mode;
        bool save_down_unweighted;
        int side_lora_rank;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        void* saved_l1_preact;
        void* saved_x;
        void* saved_h;
        void* saved_down_unweighted;
        int* side_lora_ready;
        float side_lora_scale;
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
        CUtensorMap tensor_map_saved_x;
        CUtensorMap tensor_map_saved_h;
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

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_side_lora_forward.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_side_lora_forward_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
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
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
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
    get_mxfp4_side_lora_activation_type_name(args.activation),
    args.save_l1_preact ? "true" : "false",
    get_mxfp4_side_lora_route_weight_mode_name(
        args.route_weight_mode),
    args.save_down_unweighted ? "true" : "false",
    args.side_lora_rank);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.saved_l1_preact,
            args.saved_x,
            args.saved_h,
            args.saved_down_unweighted,
            args.side_lora_ready,
            args.side_lora_scale,
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
            args.tensor_map_down_unweighted,
            args.tensor_map_saved_x,
            args.tensor_map_saved_h,
            args.tensor_map_lora_a1,
            args.tensor_map_lora_a3,
            args.tensor_map_lora_b1,
            args.tensor_map_lora_b3,
            args.tensor_map_lora_a2,
            args.tensor_map_lora_b2,
            args.tensor_map_lora_l1_scratch,
            args.tensor_map_lora_l2_scratch,
            args.tensor_map_lora_l1_scratch_store,
            args.tensor_map_lora_l2_scratch_store
        ));
    }
};

static void sm100_fp8_fp4_mega_moe_side_lora_forward(
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
    const bool& fast_math,
    const std::string& route_weight_mode,
    const std::optional<torch::Tensor>& saved_down_unweighted,
    const torch::Tensor& side_lora_source,
    const std::optional<torch::Tensor>& saved_x,
    const std::optional<torch::Tensor>& saved_h_unweighted,
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
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const bool has_side_lora = side_lora_a1.has_value();
    const int side_lora_rank = has_side_lora ? 128 : 0;
    DG_HOST_ASSERT(has_side_lora);
    DG_HOST_ASSERT(
        saved_x.has_value() && saved_h_unweighted.has_value() &&
        side_lora_b1.has_value() && side_lora_a3.has_value() &&
        side_lora_b3.has_value() && side_lora_a2.has_value() &&
        side_lora_b2.has_value() && side_lora_l1_scratch.has_value() &&
        side_lora_l2_scratch.has_value() && side_lora_ready.has_value());

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_config_tokens, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4);
    const auto num_max_pool_tokens =
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts_per_rank);
    const int num_saved_pool_tokens =
        static_cast<int>(saved_x->size(0));
    const auto check_bf16_contiguous = [](const torch::Tensor& tensor) {
        DG_HOST_ASSERT(tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
    };
    for (const auto* tensor : {
             &side_lora_source, &*saved_x, &*saved_h_unweighted,
             &*side_lora_a1, &*side_lora_b1, &*side_lora_a3,
             &*side_lora_b3, &*side_lora_a2, &*side_lora_b2,
             &*side_lora_l1_scratch, &*side_lora_l2_scratch})
        check_bf16_contiguous(*tensor);
    DG_HOST_ASSERT(saved_x->sizes() == torch::IntArrayRef(
        {num_saved_pool_tokens, hidden}));
    DG_HOST_ASSERT(saved_h_unweighted->sizes() == torch::IntArrayRef(
        {num_saved_pool_tokens, intermediate_hidden}));
    DG_HOST_ASSERT(side_lora_a1->sizes() == torch::IntArrayRef(
        {side_lora_rank, hidden}));
    DG_HOST_ASSERT(side_lora_a3->sizes() == side_lora_a1->sizes());
    DG_HOST_ASSERT(side_lora_b1->sizes() == torch::IntArrayRef(
        {num_experts_per_rank, intermediate_hidden, side_lora_rank}));
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
    DG_HOST_ASSERT(side_lora_ready->numel() >=
                   4 * config.num_ring_tokens / config.block_m);
    side_lora_ready->zero_();
    if (saved_l1_preact.has_value()) {
        DG_HOST_ASSERT(saved_l1_preact->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(saved_l1_preact->is_contiguous());
        DG_HOST_ASSERT(saved_l1_preact->dim() == 2);
        DG_HOST_ASSERT(saved_l1_preact->size(0) > 0 &&
                       saved_l1_preact->size(0) <= num_max_pool_tokens);
        DG_HOST_ASSERT(saved_l1_preact->size(0) % config.block_m == 0);
        DG_HOST_ASSERT(saved_l1_preact->size(1) ==
                       2 * intermediate_hidden);
    }
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
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
    constexpr int kSideBlockK = 64;
    const auto tensor_map_saved_x = has_side_lora
        ? make_tma_2d_desc(*saved_x, hidden, saved_x->size(0),
              kSideBlockK, config.load_block_m,
              static_cast<int>(saved_x->stride(0)), 128)
        : tensor_map_l1_acts;
    const auto tensor_map_saved_h = has_side_lora
        ? make_tma_2d_desc(*saved_h_unweighted, intermediate_hidden,
              saved_h_unweighted->size(0), kSideBlockK,
              config.load_block_m,
              static_cast<int>(saved_h_unweighted->stride(0)), 128)
        : tensor_map_l2_acts;
    const auto make_side_weight_desc = [&](const torch::Tensor& tensor,
                                            int k, int n) {
        return make_tma_2d_desc(tensor, k, n, kSideBlockK,
                                config.load_block_n,
                                static_cast<int>(tensor.stride(-2)), 128);
    };
    const auto make_l1_expand_weight_desc = [&](const torch::Tensor& tensor) {
        // B1/B3 are compact [E, I, R].  The kernel scatters eight-row TMA
        // strips into the gate/up slots of the interleaved 128-column tile.
        return make_tma_2d_desc(tensor, side_lora_rank,
                                num_experts_per_rank * intermediate_hidden,
                                kSideBlockK, 8,
                                static_cast<int>(tensor.stride(-2)), 128);
    };
    const auto tensor_map_lora_a1 = has_side_lora
        ? make_side_weight_desc(*side_lora_a1, hidden,
              side_lora_rank)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_a3 = has_side_lora
        ? make_side_weight_desc(*side_lora_a3, hidden,
              side_lora_rank)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_b1 = has_side_lora
        ? make_l1_expand_weight_desc(*side_lora_b1)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_b3 = has_side_lora
        ? make_l1_expand_weight_desc(*side_lora_b3)
        : tensor_map_l1_weights;
    const auto tensor_map_lora_a2 = has_side_lora
        ? make_side_weight_desc(*side_lora_a2, intermediate_hidden,
              num_experts_per_rank * side_lora_rank)
        : tensor_map_l2_weights;
    const auto tensor_map_lora_b2 = has_side_lora
        ? make_side_weight_desc(*side_lora_b2, side_lora_rank,
              hidden)
        : tensor_map_l2_weights;
    const auto tensor_map_lora_l1_scratch = has_side_lora
        ? make_tma_2d_desc(*side_lora_l1_scratch, 2 * side_lora_rank,
              num_saved_pool_tokens, kSideBlockK, config.load_block_m,
              static_cast<int>(side_lora_l1_scratch->stride(0)), 128)
        : tensor_map_l1_acts;
    const auto tensor_map_lora_l2_scratch = has_side_lora
        ? make_tma_2d_desc(*side_lora_l2_scratch, side_lora_rank,
              num_saved_pool_tokens, kSideBlockK, config.load_block_m,
              static_cast<int>(side_lora_l2_scratch->stride(0)), 128)
        : tensor_map_l2_acts;
    // Shrink epilogues stage STORE_BLOCK_M rows, while the following expand
    // GEMM loads LOAD_BLOCK_M rows. A TMA descriptor encodes that box height,
    // so using the load descriptor for the store reads beyond the staged
    // shared-memory tile whenever a full M block is present.
    const auto tensor_map_lora_l1_scratch_store = has_side_lora
        ? make_tma_2d_desc(*side_lora_l1_scratch, 2 * side_lora_rank,
              num_saved_pool_tokens, kSideBlockK, config.store_block_m,
              static_cast<int>(side_lora_l1_scratch->stride(0)), 128)
        : tensor_map_l1_acts;
    const auto tensor_map_lora_l2_scratch_store = has_side_lora
        ? make_tma_2d_desc(*side_lora_l2_scratch, side_lora_rank,
              num_saved_pool_tokens, kSideBlockK, config.store_block_m,
              static_cast<int>(side_lora_l2_scratch->stride(0)), 128)
        : tensor_map_l2_acts;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoESideLoraForwardRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .activation = activation,
        .save_l1_preact = saved_l1_preact.has_value(),
        .route_weight_mode = route_weight_mode,
        .save_down_unweighted =
            saved_down_unweighted.has_value(),
        .side_lora_rank = side_lora_rank,
        .config = config,
        .y = y.data_ptr(),
        .saved_l1_preact = saved_l1_preact.has_value()
            ? saved_l1_preact->data_ptr()
            : nullptr,
        .saved_x = has_side_lora ? saved_x->data_ptr() : nullptr,
        .saved_h = has_side_lora
            ? saved_h_unweighted->data_ptr() : nullptr,
        .saved_down_unweighted = saved_down_unweighted.has_value()
            ? saved_down_unweighted->data_ptr() : nullptr,
        .side_lora_ready = has_side_lora
            ? side_lora_ready->data_ptr<int>() : nullptr,
        .side_lora_scale = side_lora_scale,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .num_saved_pool_tokens = num_saved_pool_tokens,
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
        .tensor_map_saved_x = tensor_map_saved_x,
        .tensor_map_saved_h = tensor_map_saved_h,
        .tensor_map_lora_a1 = tensor_map_lora_a1,
        .tensor_map_lora_a3 = tensor_map_lora_a3,
        .tensor_map_lora_b1 = tensor_map_lora_b1,
        .tensor_map_lora_b3 = tensor_map_lora_b3,
        .tensor_map_lora_a2 = tensor_map_lora_a2,
        .tensor_map_lora_b2 = tensor_map_lora_b2,
        .tensor_map_lora_l1_scratch = tensor_map_lora_l1_scratch,
        .tensor_map_lora_l2_scratch = tensor_map_lora_l2_scratch,
        .tensor_map_lora_l1_scratch_store =
            tensor_map_lora_l1_scratch_store,
        .tensor_map_lora_l2_scratch_store =
            tensor_map_lora_l2_scratch_store,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoESideLoraForwardRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe_side_lora_forward", code);
    SM100FP8FP4MegaMoESideLoraForwardRuntime::launch(runtime, args);
}

} // namespace deep_gemm
