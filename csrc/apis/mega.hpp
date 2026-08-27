#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"

namespace deep_gemm::mega {

using MegaMoEBufferViews = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor>;
using MegaMoEBufferSlicer = std::function<MegaMoEBufferViews(const torch::Tensor&)>;

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static std::pair<int, int> get_ring_limit_for_mega_moe(
    const int& num_max_tokens_per_rank, const int& num_experts_per_rank, const int& num_topk, const int& num_ranks) {
    return {
        get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, 1, layout::kLCMCandidateBlockM),
        get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, num_experts_per_rank, layout::kLCMCandidateBlockM)
    };
}

static std::tuple<int64_t, MegaMoEBufferSlicer> get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens,
    const int& lora_rank,
    const int& num_lora_slots,
    const bool& enable_lora_down) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu" or activation == "situ");
    DG_HOST_ASSERT(lora_rank == 0 or lora_rank == 128);
    DG_HOST_ASSERT(lora_rank == 0 or mma_type == "fp8xfp4");
    DG_HOST_ASSERT(num_lora_slots >= 0);
    DG_HOST_ASSERT(not enable_lora_down or
                   (lora_rank == 128 and 2 <= num_lora_slots and num_lora_slots <= 32));

    // Pool capacity must fit at least one full wave (one expert per wave) and aligned to block size
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto [num_min_ring_tokens, num_max_ring_tokens] =
        get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts_per_rank, num_topk, num_ranks);
    DG_HOST_ASSERT(num_ring_tokens % layout::kLCMCandidateBlockM == 0);
    DG_HOST_ASSERT(num_min_ring_tokens <= num_ring_tokens and num_ring_tokens <= num_max_ring_tokens);

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);
    const auto with_sf = is_mma_with_sf(mma_kind);

    // Workspace
    const auto workspace = layout::Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_ring_tokens);

    // Layouts
    const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
    const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0);
    const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        input_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        input_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Padded SF pool tokens
    int num_sf_ring_tokens = 0;
    for (int block_m: layout::kCandidateBlockM) {
        num_sf_ring_tokens = std::max(
            num_sf_ring_tokens,
            layout::get_num_sf_ring_tokens(num_ring_tokens, block_m)
        );
    }

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        input_token_layout, 1, num_ring_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        input_sf_layout, 1, num_sf_ring_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_ring_tokens,
        with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        intermediate_token_layout, 1, num_ring_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        intermediate_sf_layout, 1, num_sf_ring_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());

    // LoRA payload is deliberately appended after every baseline buffer.
    // Consequently, a LoRA-sized allocation remains ABI-compatible with the
    // no-LoRA kernel specialization and its original offsets.
    const auto lora_payload_layout = layout::MegaMoELoraPayload(lora_rank);
    const auto input_lora_gate_up_acts_buffer = layout::Buffer(
        lora_payload_layout.get_gate_up_acts_layout(), 1, num_max_tokens_per_rank,
        combine_token_buffer.get_end_ptr());
    const auto ring_lora_gate_up_acts_buffer = layout::Buffer(
        lora_payload_layout.get_gate_up_acts_layout(), 1, num_ring_tokens,
        input_lora_gate_up_acts_buffer.get_end_ptr());
    const auto input_lora_adapter_slot_buffer = layout::Buffer(
        lora_payload_layout.get_adapter_slot_layout(), 1, num_max_tokens_per_rank,
        ring_lora_gate_up_acts_buffer.get_end_ptr());
    const auto ring_lora_adapter_slot_buffer = layout::Buffer(
        lora_payload_layout.get_adapter_slot_layout(), 1, num_ring_tokens,
        input_lora_adapter_slot_buffer.get_end_ptr());
    // FC1 supports at most 32 slots.  Larger slot spaces are PayloadOnly and
    // intentionally retain the original expert-only dispatch.
    const auto num_lora_subgroup_slots =
        lora_rank > 0 and num_lora_slots <= 32 ? num_lora_slots : 0;
    const auto lora_subgroups = layout::MegaMoELoraSubgroups(
        ring_lora_adapter_slot_buffer.get_end_ptr(),
        num_ranks, num_experts, num_lora_subgroup_slots);
    const auto lora_readiness = layout::MegaMoELoraReadiness(
        lora_subgroups.get_end_ptr(),
        num_lora_subgroup_slots > 0
            ? num_ring_tokens / layout::kMinCandidateBlockM
            : 0);
    // Down-projection data is an append-only symmetric suffix.  Keeping both
    // route and source rows separate avoids widening the established [D]
    // combine payload and preserves every Disabled/PayloadOnly/FC1 offset.
    const auto down_sideband = layout::MegaMoEDownLoraSideband(
        enable_lora_down ? lora_rank : 0);
    const auto routed_lora_rank_buffer = layout::Buffer(
        down_sideband.get_rank_acts_layout(), num_topk,
        num_max_tokens_per_rank, lora_readiness.get_end_ptr());
    const auto combined_lora_rank_buffer = layout::Buffer(
        down_sideband.get_rank_acts_layout(), 1,
        num_max_tokens_per_rank, routed_lora_rank_buffer.get_end_ptr());

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates baseline input/ring views followed by the
    // optional source and dispatched LoRA payload views.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    MegaMoEBufferSlicer slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto x_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_ring_tokens, hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l1_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto input_lora_gate_up_acts = lora_rank > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_lora_gate_up_acts_buffer.base)),
            {num_max_tokens_per_rank, 2, lora_rank},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto input_lora_adapter_slots = lora_rank > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_lora_adapter_slot_buffer.base)),
            {num_max_tokens_per_rank},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto ring_lora_gate_up_acts = lora_rank > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(ring_lora_gate_up_acts_buffer.base)),
            {num_ring_tokens, 2, lora_rank},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto ring_lora_adapter_slots = lora_rank > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(ring_lora_adapter_slot_buffer.base)),
            {num_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto lora_subgroup_offsets = num_lora_subgroup_slots > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(lora_subgroups.get_offset_ptr())),
            {num_experts_per_rank, num_lora_subgroup_slots + 1},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto routed_lora_rank_acts = enable_lora_down ? torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(routed_lora_rank_buffer.base)),
            {num_topk, num_max_tokens_per_rank, lora_rank},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto combined_lora_rank_acts = enable_lora_down ? torch::from_blob(
            math::advance_ptr(
                buffer.data_ptr(),
                reinterpret_cast<int64_t>(combined_lora_rank_buffer.base)),
            {num_max_tokens_per_rank, lora_rank},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        return std::make_tuple(
            x, x_sf, topk_idx, topk_weights,
            l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
            input_lora_gate_up_acts, input_lora_adapter_slots,
            ring_lora_gate_up_acts, ring_lora_adapter_slots,
            lora_subgroup_offsets, routed_lora_rank_acts,
            combined_lora_rank_acts);
    };
    return std::make_tuple(
        reinterpret_cast<int64_t>(combined_lora_rank_buffer.get_end_ptr()),
        slice_input_buffers);
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const std::optional<float>& situ_beta_opt,
    const std::optional<float>& situ_linear_beta_opt,
    const bool& fast_math,
    const int& num_ring_tokens,
    const std::string& lora_mode,
    const int& lora_rank,
    const int& num_lora_slots,
    const std::optional<torch::Tensor>& lora_gate_b,
    const std::optional<torch::Tensor>& lora_up_b,
    const std::optional<torch::Tensor>& lora_down_a,
    const float& lora_scaling
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "geglu" or activation == "situ");
    DG_HOST_ASSERT(lora_mode == "disabled" or lora_mode == "payload_only" or
                   lora_mode == "fc1" or lora_mode == "fc1_down");
    DG_HOST_ASSERT(lora_rank == 0 or lora_rank == 128);
    DG_HOST_ASSERT(num_lora_slots >= 0);
    if (lora_mode == "disabled") {
        DG_HOST_ASSERT(lora_rank == 0 or lora_rank == 128);
    } else {
        DG_HOST_ASSERT(lora_rank == 128);
        DG_HOST_ASSERT(num_lora_slots > 0);
    }
    DG_HOST_ASSERT(std::isfinite(lora_scaling));
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.dim() == 2 and y.is_contiguous() and y.is_cuda());
    DG_HOST_ASSERT(sym_buffer.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(sym_buffer.dim() == 1 and sym_buffer.is_contiguous() and sym_buffer.is_cuda());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    const auto situ_beta = situ_beta_opt.value_or(1.0f);
    const auto situ_linear_beta =
        situ_linear_beta_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    DG_HOST_ASSERT(situ_beta > 0);
    DG_HOST_ASSERT(situ_linear_beta > 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(y.size(1) == hidden);

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_ring_tokens, lora_rank, num_lora_slots,
        lora_mode == "fc1_down");
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                input_lora_gate_up_acts, input_lora_adapter_slots,
                ring_lora_gate_up_acts, ring_lora_adapter_slots,
                lora_subgroup_offsets, routed_lora_rank_acts,
                combined_lora_rank_acts] = slice(sym_buffer);
    if (lora_rank > 0) {
        DG_HOST_ASSERT(input_lora_gate_up_acts.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(input_lora_gate_up_acts.sizes() ==
                       torch::IntArrayRef({num_max_tokens_per_rank, 2, lora_rank}));
        DG_HOST_ASSERT(input_lora_gate_up_acts.is_contiguous());
        DG_HOST_ASSERT(input_lora_adapter_slots.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(input_lora_adapter_slots.sizes() ==
                       torch::IntArrayRef({num_max_tokens_per_rank}));
        DG_HOST_ASSERT(input_lora_adapter_slots.is_contiguous());
        DG_HOST_ASSERT(ring_lora_gate_up_acts.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(ring_lora_gate_up_acts.sizes() ==
                       torch::IntArrayRef({num_ring_tokens, 2, lora_rank}));
        DG_HOST_ASSERT(ring_lora_gate_up_acts.is_contiguous());
        DG_HOST_ASSERT(ring_lora_adapter_slots.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(ring_lora_adapter_slots.sizes() ==
                       torch::IntArrayRef({num_ring_tokens}));
        DG_HOST_ASSERT(ring_lora_adapter_slots.is_contiguous());
        if (num_lora_slots <= 32) {
            DG_HOST_ASSERT(lora_subgroup_offsets.scalar_type() == torch::kInt);
            DG_HOST_ASSERT(lora_subgroup_offsets.sizes() == torch::IntArrayRef(
                {num_experts_per_rank, num_lora_slots + 1}));
            DG_HOST_ASSERT(lora_subgroup_offsets.is_contiguous());
        }
    }
    if (lora_mode == "fc1" or lora_mode == "fc1_down") {
        DG_HOST_ASSERT(2 <= num_lora_slots and num_lora_slots <= 32);
        DG_HOST_ASSERT(lora_gate_b.has_value() and lora_up_b.has_value());
        for (const auto* tensor: {&lora_gate_b.value(), &lora_up_b.value()}) {
            DG_HOST_ASSERT(tensor->scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(tensor->sizes() == torch::IntArrayRef(
                {num_lora_slots, num_experts_per_rank, intermediate_hidden, lora_rank}));
            DG_HOST_ASSERT(tensor->is_contiguous() and tensor->is_cuda());
            DG_HOST_ASSERT(tensor->device() == y.device());
        }
    } else {
        DG_HOST_ASSERT(not lora_gate_b.has_value() and not lora_up_b.has_value());
    }
    if (lora_mode == "fc1_down") {
        DG_HOST_ASSERT(lora_down_a.has_value());
        DG_HOST_ASSERT(lora_down_a->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(lora_down_a->sizes() == torch::IntArrayRef(
            {num_lora_slots, num_experts_per_rank, lora_rank,
             intermediate_hidden}));
        DG_HOST_ASSERT(lora_down_a->is_contiguous() and lora_down_a->is_cuda());
        DG_HOST_ASSERT(lora_down_a->device() == y.device());
        DG_HOST_ASSERT(routed_lora_rank_acts.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(routed_lora_rank_acts.sizes() == torch::IntArrayRef(
            {num_topk, num_max_tokens_per_rank, lora_rank}));
        DG_HOST_ASSERT(routed_lora_rank_acts.is_contiguous());
        DG_HOST_ASSERT(combined_lora_rank_acts.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(combined_lora_rank_acts.sizes() == torch::IntArrayRef(
            {num_max_tokens_per_rank, lora_rank}));
        DG_HOST_ASSERT(combined_lora_rank_acts.is_contiguous());
    } else {
        DG_HOST_ASSERT(not lora_down_a.has_value());
    }

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation, activation_clamp,
                               situ_beta, situ_linear_beta,
                               fast_math,
                               lora_mode, lora_rank, num_lora_slots,
                               ring_lora_gate_up_acts,
                               lora_gate_b, lora_up_b,
                               lora_down_a, lora_scaling);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_ring_tokens, 0, 0, false);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights,
                l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf,
                _input_lora_gate_up_acts, _input_lora_adapter_slots,
                _ring_lora_gate_up_acts, _ring_lora_adapter_slots,
                _lora_subgroup_offsets, _routed_lora_rank_acts,
                _combined_lora_rank_acts] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts, 
                            l1_weights, l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_ring_limit_for_mega_moe", &get_ring_limit_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("bf16_mega_moe", &bf16_mega_moe);
#endif
}

} // namespace deep_gemm::mega
