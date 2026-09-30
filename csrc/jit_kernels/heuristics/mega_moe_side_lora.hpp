#pragma once

#include "mega_moe.hpp"
#include "../impls/runtime_utils.hpp"

namespace deep_gemm {

// The side kernel adds shrink/expand phases around both expert GEMMs and
// therefore retains a distinct scheduler. Keep its wave-capacity policy out
// of the ordinary upstream scheduler/configuration.
static int get_num_wave_pool_tokens(
    const int& num_ranks, const int& num_topk, const int& num_max_tokens_per_rank, const int& num_experts_per_wave, const int& block_m) {
    DG_HOST_ASSERT(num_max_tokens_per_rank % block_m == 0);
    const auto num_tokens_from_all_ranks = num_max_tokens_per_rank * num_ranks;
    if (num_experts_per_wave == 1)
        return num_tokens_from_all_ranks;

    return std::min(
        // All tokens come to all local experts in the wave
        num_tokens_from_all_ranks * num_experts_per_wave,
        // All routed tokens come to this local wave, and each expert needs a padding
        math::align(num_tokens_from_all_ranks * num_topk + num_experts_per_wave * (block_m - 1), block_m)
    );
};

static int get_num_experts_per_wave_for_mega_moe(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    
    // Get max experts per wave limitation
    int num_max_experts_per_wave = num_experts_per_rank;
    while (num_max_experts_per_wave > 0 and
           get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, num_max_experts_per_wave, block_m) > num_ring_tokens)
        num_max_experts_per_wave --;
    DG_HOST_ASSERT(num_max_experts_per_wave > 0 and "Buffer size is too small");

    // Reduce per-expert block count by this factor since uneven routing leaves some experts with fewer tokens
    constexpr int kImbalanceFactor = 2;

    // Count L1 blocks per expert assuming tokens are evenly spread across experts
    const float num_expected_tokens_per_expert = static_cast<float>(num_tokens * num_topk) / num_experts_per_rank;
    const int num_expected_m_blocks = std::max(ceil_div(static_cast<int>(std::ceil(num_expected_tokens_per_expert)), block_m), 1);
    const int num_l1_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_expected_l1_blocks_per_expert = num_expected_m_blocks * num_l1_n_blocks;

    // Pick the smallest value whose total blocks (after imbalance reduction) can keep all SMs busy
    int num_min_expected_experts_to_fill_sms = ceil_div(kImbalanceFactor * num_sms, num_expected_l1_blocks_per_expert);

    // Most experts don't have tokens, calculate all experts at once
    if (num_expected_tokens_per_expert < 1)
        num_min_expected_experts_to_fill_sms = num_experts_per_rank;

    // Ring capacity is the bottleneck
    if (num_min_expected_experts_to_fill_sms >= num_max_experts_per_wave)
        return num_max_experts_per_wave;

    // When each expert nearly fills all SMs, use the smallest wave to maximize L2 cache reuse
    if (num_expected_l1_blocks_per_expert >= num_sms) 
        return num_min_expected_experts_to_fill_sms;

    // Search to 2 * num_min_expected_experts_to_fill_sms for a value where the last partial
    // wave has as many experts as possible relative to a full wave
    const int num_sweep_max_experts_per_wave = std::min(num_max_experts_per_wave, num_min_expected_experts_to_fill_sms * 2);
    int best_num_experts_per_wave = num_min_expected_experts_to_fill_sms;
    float best_tail_ratio = -1.0f;
    for (int num_experts_per_wave = num_min_expected_experts_to_fill_sms; 
             num_experts_per_wave <= num_sweep_max_experts_per_wave; ++ num_experts_per_wave) {
        int remainder = num_experts_per_rank % num_experts_per_wave;
        float tail_ratio = (remainder == 0) ? 1.0f : static_cast<float>(remainder) / num_experts_per_wave;
        if (tail_ratio > best_tail_ratio) {
            best_tail_ratio = tail_ratio;
            best_num_experts_per_wave = num_experts_per_wave;
        }
    }
    return best_num_experts_per_wave;
}


struct SideLoraMegaMoEConfig : MegaMoEConfig {
    int num_experts_per_wave;
};

static SideLoraMegaMoEConfig get_side_lora_mega_moe_config(
    const int num_ranks, const int num_experts, const int num_experts_per_rank,
    const int num_max_tokens_per_rank, const int num_tokens, const int num_topk,
    const int hidden, const int intermediate_hidden,
    const int num_ring_tokens, const int num_sf_ring_tokens, const MmaKind mma_kind) {
    const auto base = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank, num_max_tokens_per_rank,
        num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens, mma_kind);
    const int experts_per_wave = get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk, intermediate_hidden,
        base.block_m, base.block_n, get_mega_moe_num_sms(),
        num_ring_tokens, num_max_tokens_per_rank, num_ranks);
    return {base, experts_per_wave};
}

}  // namespace deep_gemm
