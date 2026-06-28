"""FlashInfer TRT-LLM-gen NVFP4 fused MoE driver (`trtllm_fp4_block_scale_moe`).

Reuses flashinfer's own (version-matched, vendored) offline weight-shuffle +
routing-reference harness from `_fi_vendor/test_trtllm_gen_fused_moe.py`, then
exposes a thin timeable closure that calls the kernel directly (mirroring the
harness's `_run_moe_computation`, but without the per-call CUDA-graph wrapper).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _fi_vendor.test_trtllm_gen_fused_moe import (
    FP4Moe, moe_args, routing_reference_renormalize,
)
from _fi_vendor.utils import QuantMode
from flashinfer import RoutingMethodType, ActivationType
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe


class _Dequant:
    """Minimal stand-in for moe_args_dequant: prepare_static_weights only reads
    `c_global_sf`. Scale values don't affect kernel timing, so we stub it (avoids
    the slow global reference MoE and lets each rank build only its expert slice)."""
    def __init__(self, c_global_sf):
        self.c_global_sf = c_global_sf


def build_flashinfer_trtllm(x_bf16, w1_bf16, w2_bf16, num_experts, top_k, hidden, inter,
                            local_expert_offset=0, activation_type=None):
    """TRT-LLM-gen NVFP4 MoE. `num_experts` is the GLOBAL expert count (routing
    space); `w1_bf16`/`w2_bf16` hold only this rank's local expert slice. The
    kernel computes only the local experts via local_expert_offset/local_num_experts."""
    dev = x_bf16.device
    act = activation_type if activation_type is not None else ActivationType.Swiglu
    moe = FP4Moe(QuantMode.FP4_NVFP4_NVFP4)
    moe._cache_permute_indices = {}
    num_tokens = x_bf16.shape[0]
    num_local = w1_bf16.shape[0]
    padding = 8

    weights_data = moe.quantize_weights(w1_bf16, w2_bf16, x_bf16)
    inputs_data = moe.quantize_inputs(x_bf16, weights_data['hidden_states_scale_global'])
    q = {**weights_data, **inputs_data}

    args = moe_args(
        num_tokens, num_local, hidden, inter, top_k, padding,
        q['hidden_states'], q['hidden_states_scale'], q['hidden_states_scale_global'],
        None, q['gemm1_weights'], q['gemm1_scales'], q['gemm1_scales_global'],
        q['gemm2_weights'], q['gemm2_scales'], q['gemm2_scales_global'],
        None, False, act)

    args_dequant = _Dequant(torch.tensor(1.0, device=dev, dtype=torch.float32))
    static = moe.prepare_static_weights_for_kernel(
        args_dequant, args, w1_bf16, w2_bf16, hidden, inter, num_local, None)

    # Global routing space (top-k over all `num_experts`); each rank computes only its slice.
    expert_logits = torch.randn(num_tokens, num_experts, device=dev).to(torch.bfloat16)
    inp = moe.quantize_inputs(x_bf16, weights_data['hidden_states_scale_global'], is_swizzling=False)
    hs, hs_sf = inp['hidden_states'], inp['hidden_states_scale']

    def run():
        return trtllm_fp4_block_scale_moe(
            routing_logits=expert_logits, routing_bias=None,
            hidden_states=hs, hidden_states_scale=hs_sf,
            gemm1_weights=static['gemm1_weights_fp4_shuffled'],
            gemm1_weights_scale=static['gemm1_scales_fp4_shuffled'],
            gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=static['gemm2_weights_fp4_shuffled'],
            gemm2_weights_scale=static['gemm2_scales_fp4_shuffled'],
            gemm2_bias=None,
            output1_scale_scalar=static['scale_c_fc1'],
            output1_scale_gate_scalar=static['scale_gate_fc1'],
            output2_scale_scalar=static['scale_c_fc2'],
            num_experts=num_experts, top_k=top_k, n_group=None, topk_group=None,
            intermediate_size=inter, local_expert_offset=local_expert_offset,
            local_num_experts=num_local,
            routed_scaling_factor=None, routing_method_type=RoutingMethodType.Renormalize,
            activation_type=act, do_finalize=True,
            tune_max_num_tokens=8192, norm_topk_prob=True)
    return run


if __name__ == '__main__':
    torch.manual_seed(0)
    he, it, ne, tk, nt = 4608, 2560, 128, 16, 32
    x = torch.randn(nt, he, dtype=torch.bfloat16, device='cuda') / 10
    w1 = torch.randn(ne, it * 2, he, dtype=torch.bfloat16, device='cuda') / 10
    w2 = torch.randn(ne, he, it, dtype=torch.bfloat16, device='cuda') / 10
    run = build_flashinfer_trtllm(x, w1, w2, ne, tk, he, it)
    out = run()
    torch.cuda.synchronize()
    o = out[0] if isinstance(out, (list, tuple)) else out
    print('trtllm output:', type(out), o.shape, o.dtype, 'mean', o.float().abs().mean().item())
    print('OK')
