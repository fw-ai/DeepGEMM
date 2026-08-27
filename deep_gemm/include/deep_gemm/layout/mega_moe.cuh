#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Ring-buffer capacity used by reusable token/data buffers
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    // For both grid barrier and NVLink barrier
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 full token count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L1 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 full block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

// Optional per-source-token LoRA payload transported by MegaMoE dispatch.
//
// Keep these buffers as a suffix of the normal MegaMoE allocation.  That
// invariant lets a buffer provisioned for LoRA still run the compile-time
// no-LoRA kernel: every pre-existing workspace/ring offset remains unchanged.
struct MegaMoELoraPayload {
    uint32_t rank;

    CUTLASS_HOST_DEVICE
    constexpr explicit MegaMoELoraPayload(const uint32_t& rank): rank(rank) {}

    CUTLASS_HOST_DEVICE
    constexpr bool enabled() const {
        return rank != 0;
    }

    // Two BF16 shrink results, in gate/up order, for each source token.
    CUTLASS_HOST_DEVICE
    constexpr Data get_gate_up_acts_layout() const {
        return Data(enabled() ? 2 * rank * sizeof(nv_bfloat16) : 0);
    }

    // Adapter pool slot.  The final pool slot is reserved as the zero sentinel.
    CUTLASS_HOST_DEVICE
    constexpr Data get_adapter_slot_layout() const {
        return Data(enabled() ? sizeof(int32_t) : 0, false);
    }
};

// Rank-128 down-projection sideband.  Both the per-route transport buffer and
// the source-local combined output use this row layout.  Callers append those
// buffers after `MegaMoELoraSubgroups`, so every pre-existing LoRA address is
// stable when the down sidecar is disabled.
struct MegaMoEDownLoraSideband {
    uint32_t rank;

    CUTLASS_HOST_DEVICE
    constexpr explicit MegaMoEDownLoraSideband(const uint32_t& rank):
        rank(rank) {}

    CUTLASS_HOST_DEVICE
    constexpr Data get_rank_acts_layout() const {
        return Data(rank * sizeof(nv_bfloat16));
    }
};

// Fixed-address symmetric metadata used to dispatch FC1 LoRA rows in
// (local expert, adapter slot) order.  This is appended after all baseline and
// payload buffers, so Disabled and PayloadOnly retain their original layouts.
struct MegaMoELoraSubgroups {
    void* base;
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_slots;

    CUTLASS_HOST_DEVICE
    constexpr MegaMoELoraSubgroups(
        void* base,
        const uint32_t& num_ranks,
        const uint32_t& num_experts,
        const uint32_t& num_slots):
        base(base),
        num_ranks(num_ranks),
        num_experts(num_experts),
        num_experts_per_rank(num_experts / num_ranks),
        num_slots(num_slots) {}

    CUTLASS_HOST_DEVICE
    constexpr bool enabled() const {
        return num_slots != 0;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        if (not enabled())
            return 0;

        uint64_t num_bytes = 0;
        // Per-source-rank count and write cursor.  Low 32 bits are the count;
        // high 32 bits are reused as the device-side write cursor.
        num_bytes +=
            static_cast<uint64_t>(num_experts) * num_slots * sizeof(uint64_t);
        // Counts received from every source rank.
        num_bytes +=
            static_cast<uint64_t>(num_experts_per_rank) * num_ranks *
            num_slots * sizeof(uint32_t);
        // Exclusive subgroup offsets within each expert.  The final entry is
        // the expert's total row count.
        num_bytes +=
            static_cast<uint64_t>(num_experts_per_rank) *
            (num_slots + 1) * sizeof(uint32_t);
        return math::align<uint64_t>(num_bytes, 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_DEVICE
    uint64_t* get_send_count_ptr(
        const uint32_t& expert_idx = 0,
        const uint32_t& slot_idx = 0) const {
        return static_cast<uint64_t*>(base) +
            expert_idx * num_slots + slot_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_recv_count_ptr(
        const uint32_t& source_rank_idx = 0,
        const uint32_t& local_expert_idx = 0,
        const uint32_t& slot_idx = 0) const {
        const auto recv_base = reinterpret_cast<uint32_t*>(
            get_send_count_ptr(num_experts));
        return recv_base +
            (local_expert_idx * num_ranks + source_rank_idx) * num_slots +
            slot_idx;
    }

    CUTLASS_HOST_DEVICE
    uint32_t* get_offset_ptr(
        const uint32_t& local_expert_idx = 0,
        const uint32_t& slot_idx = 0) const {
        const auto offset_base = reinterpret_cast<uint32_t*>(
            math::advance_ptr(
                base,
                static_cast<uint64_t>(num_experts) * num_slots *
                sizeof(uint64_t) +
                static_cast<uint64_t>(num_experts_per_rank) * num_ranks *
                num_slots * sizeof(uint32_t)));
        return offset_base +
            local_expert_idx * (num_slots + 1) + slot_idx;
    }
};

// Ring generation counters for routed rank payload arrival.  This is a LoRA
// suffix so the baseline workspace layout and counters remain unchanged.
struct MegaMoELoraReadiness {
    void* base;
    uint32_t num_ring_blocks;

    CUTLASS_HOST_DEVICE
    constexpr MegaMoELoraReadiness(
        void* base,
        const uint32_t& num_ring_blocks):
        base(base), num_ring_blocks(num_ring_blocks) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return math::align<uint64_t>(
            static_cast<uint64_t>(num_ring_blocks) * sizeof(uint32_t), 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_DEVICE
    uint32_t* get_full_count_ptr(
        const uint32_t& ring_block_idx = 0) const {
        return static_cast<uint32_t*>(base) + ring_block_idx;
    }
};

// Kernel launch payload is empty for disabled/payload-only specializations.
// FC1 descriptors are grouped into one typed argument so their lifetime and
// register impact are compile-time absent from those specializations.
template <MegaMoELoraMode kMode>
struct MegaMoELoraTensorMaps {};

struct alignas(64) TmaDescriptorStorage {
    uint8_t bytes[128];
};

template <>
struct MegaMoELoraTensorMaps<MegaMoELoraMode::FC1> {
    TmaDescriptorStorage gate_a;
    TmaDescriptorStorage up_a;
    TmaDescriptorStorage gate_b;
    TmaDescriptorStorage up_b;
};

template <>
struct MegaMoELoraTensorMaps<MegaMoELoraMode::FC1Down> {
    TmaDescriptorStorage gate_a;
    TmaDescriptorStorage up_a;
    TmaDescriptorStorage gate_b;
    TmaDescriptorStorage up_b;
    TmaDescriptorStorage down_a;
    float scaling;
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

} // namespace deep_gemm::layout
