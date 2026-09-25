#pragma once

#include <cutlass/bfloat16.h>

namespace deep_gemm {

// Rank-128 native MegaMoE side-LoRA backward ABI. All pool tensors are
// expert-major with the same BLOCK_M padding as the base backward wave.
// Only rank-width intermediates cross phases; full-width side dgrad tensors
// are deliberately absent from this contract.
struct alignas(16) MegaMoESideLoraBackwardParams {
    using bf16 = cutlass::bfloat16_t;

    const bf16* saved_x;
    const bf16* saved_h;

    const bf16* a1;
    const bf16* b1;
    const bf16* a3;
    const bf16* b3;
    const bf16* a2;
    const bf16* b2;

    bf16* q1;
    bf16* q3;
    bf16* q2;
    bf16* t1;
    bf16* t3;
    bf16* t2;

    bf16* grad_a1;
    bf16* grad_b1;
    bf16* grad_a3;
    bf16* grad_b3;
    bf16* grad_a2;
    bf16* grad_b2;

    float scale;
};

}  // namespace deep_gemm
