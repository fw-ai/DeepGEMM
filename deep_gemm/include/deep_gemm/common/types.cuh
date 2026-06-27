#pragma once

#include <cute/arch/mma_sm100_desc.hpp>

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
    // Pure packed MXFP4 x MXFP4 (E2M1 data, UE8M0 SF gran-32, 2 elements per byte)
    MXFP4       = 2,
    // Pure packed NVFP4 x NVFP4 (E2M1 data, E4M3 SF gran-16, per-tensor global scale)
    NVFP4       = 3,
};

// NOTES: element size in *bits*, since packed FP4 is sub-byte (4 bits)
constexpr CUTLASS_HOST_DEVICE int get_element_bits(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 16;
        case MmaKind::MXFP8FP4: return 8;
        case MmaKind::MXFP4:    return 4;
        case MmaKind::NVFP4:    return 4;
        default: return 0;
    }
}

// SF granularity (elements per scale factor): MXFP4 -> 32, NVFP4 -> 16
constexpr CUTLASS_HOST_DEVICE int get_sf_gran_k(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::NVFP4 ? 16 : 32;
}

constexpr CUTLASS_HOST_DEVICE int get_element_size(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 2;
        case MmaKind::MXFP8FP4: return 1;
        // NOTES: packed FP4 is 0.5 byte/elem; callers must use byte math that
        // divides element counts by 2 (see `get_element_bits`)
        case MmaKind::MXFP4:    return 1;
        case MmaKind::NVFP4:    return 1;
        default: return 0;
    }
}

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

constexpr CUTLASS_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

} // namespace deep_gemm
