"""B300 EP benchmark for native MegaMoE side-LoRA specializations."""

import argparse
import json

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import init_dist


def _block_m(tokens: int, ranks: int, topk: int, experts: int) -> int:
    expected = tokens * ranks * topk / experts
    if expected <= 8.5:
        return 16
    if expected <= 16.5:
        return 32
    if expected <= 32.5:
        return 64
    if expected <= 64.5:
        return 96
    if expected <= 96.5:
        return 128
    return 192


def _adapters(experts: int, hidden: int, intermediate: int):
    rank = 128
    return (
        torch.randn(hidden, rank, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(experts, rank, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(hidden, rank, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(experts, rank, intermediate, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(experts, intermediate, rank, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(rank, hidden, device="cuda", dtype=torch.bfloat16) * 0.02,
    )


def _time(fn, group, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    elapsed = 0.0
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        sample = torch.tensor(
            start.elapsed_time(end), dtype=torch.float64, device="cuda")
        dist.all_reduce(sample, op=dist.ReduceOp.MAX, group=group)
        elapsed += float(sample.item())
    return elapsed / iterations


def run_bf16(local_rank: int, world: int, args) -> None:
    rank, ranks, group = init_dist(local_rank, world)
    if ranks != args.num_processes:
        raise RuntimeError("unexpected EP world size")
    torch.manual_seed(9000 + rank)
    tokens = args.tokens
    hidden = args.hidden
    intermediate = args.intermediate
    experts = args.experts
    topk = args.topk
    local_experts = experts // ranks
    block_m = _block_m(tokens, ranks, topk, experts)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, experts, tokens, topk, hidden, intermediate,
        mma_type="bf16xbf16", activation="swiglu")

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(
        local_experts, 2 * intermediate, hidden,
        device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(
        local_experts, hidden, intermediate,
        device="cuda", dtype=torch.bfloat16) * 0.02
    transformed_w13, transformed_w2 = deep_gemm.transform_weights_for_mega_moe(
        w13, w2)
    adapters = _adapters(local_experts, hidden, intermediate)
    side = deep_gemm.transform_side_lora_for_mega_moe(adapters)

    route_id = (
        rank * tokens * topk +
        torch.arange(tokens, device="cuda")[:, None] * topk +
        torch.arange(topk, device="cuda")[None, :])
    topk_idx = (route_id % experts).to(torch.int64)
    topk_weights = torch.sigmoid(
        torch.randn(tokens, topk, device="cuda", dtype=torch.float32))
    buffer.x[:tokens].copy_(x)
    buffer.topk_idx[:tokens].copy_(topk_idx)
    buffer.topk_weights[:tokens].copy_(topk_weights)

    local_start = rank * local_experts
    mask = (topk_idx >= local_start) & (topk_idx < local_start + local_experts)
    counts = torch.bincount(
        topk_idx[mask] - local_start, minlength=local_experts).to(torch.int32)
    dist.all_reduce(counts, group=group)
    padded = ((counts + block_m - 1) // block_m * block_m).to(torch.int32)
    pool_rows = int(padded.sum().item())
    route_counts = torch.bincount(
        topk_idx.flatten(), minlength=experts).to(torch.int32)

    options = dict(device="cuda", dtype=torch.bfloat16)
    y = torch.empty((tokens, hidden), **options)
    base_gate = torch.empty((pool_rows, 2 * intermediate), **options)
    base_h = torch.empty((pool_rows, intermediate), **options)
    base_hw = torch.empty_like(base_h)
    base_down = torch.empty((pool_rows, hidden), **options)
    mismatch = torch.zeros(1, device="cuda", dtype=torch.int32)

    def base_forward():
        deep_gemm.bf16_mega_moe(
            y, transformed_w13, transformed_w2, buffer,
            saved_l1_preact=base_gate,
            saved_h_unweighted=base_h,
            saved_h_weighted=base_hw,
            saved_down_unweighted=base_down,
            precomputed_route_counts=route_counts,
            active_pool_rows=pool_rows,
            route_count_mismatch=mismatch,
            num_config_tokens=tokens,
            fast_math=True)

    base_forward_ms = _time(
        base_forward, group, args.warmup, args.iterations)
    if mismatch.item() != 0:
        raise RuntimeError("forward route histogram mismatch")

    grad_y = torch.randn_like(y)
    grad_ye = torch.empty((pool_rows, hidden), **options)
    grad_h = torch.empty((pool_rows, intermediate), **options)
    grad_gate = torch.empty_like(base_gate)
    h_act = torch.empty_like(base_h)
    h_weighted = torch.empty_like(base_h)
    x_pool = torch.empty((pool_rows, hidden), **options)
    grad_x_pool = torch.empty((0, hidden), **options)
    grad_route = torch.empty(pool_rows, device="cuda", dtype=torch.float32)
    route_weights = torch.empty_like(grad_route)
    grid_states = local_experts * (
        (hidden // 64) * (intermediate // 128) +
        ((2 * intermediate) // 64) * (hidden // 128)) + 2
    grid = torch.zeros(grid_states, device="cuda", dtype=torch.int32)
    grad_w13 = torch.empty_like(w13)
    grad_w2 = torch.empty_like(w2)
    grad_x = torch.empty_like(y)

    def original_full_backward():
        deep_gemm.bf16_mega_moe_backward_dgrad(
            base_gate, grad_h, grad_gate, h_act, h_weighted,
            x_pool, grad_x_pool, grad_route, grad_ye, route_weights,
            w2, w13, counts, grid, grad_y, buffer,
            activation_limit=float("inf"), block_m=block_m,
            fast_math=True, direct_remote_grad_x=True,
            write_grad_x_pool=False, clear_wgrad_padding=True,
            route_weight_mode=deep_gemm.RouteWeightMode.PRE_DOWN,
            grad_y_unweighted_output=grad_ye,
            down_unweighted_output=base_down,
            combine_order_mode=deep_gemm.CombineOrderMode.FIXED_TOPK,
            memory_mode="legacy", rank_uniform_block_m=True)
        deep_gemm.bf16_mega_moe_backward_w2_combine(
            grad_w2, grad_ye, h_weighted, padded, block_m,
            grad_x, buffer)
        deep_gemm.bf16_mega_moe_backward_w13_combine(
            grad_w13, grad_gate, x_pool, padded, block_m,
            grad_x, buffer)

    original_backward_ms = _time(
        original_full_backward, group, args.warmup, args.iterations)

    side_gate = torch.empty_like(base_gate)
    side_h = torch.empty_like(base_h)
    side_hw = torch.empty_like(base_h)
    side_down = torch.empty_like(base_down)
    saved_x = torch.empty_like(base_down)
    q13 = torch.empty((pool_rows, 2, 128), **options)
    q2 = torch.empty((pool_rows, 128), **options)
    ready = torch.empty(
        (4, buffer.num_ring_tokens // 8), device="cuda", dtype=torch.int32)

    def side_forward():
        deep_gemm.bf16_mega_moe_side_lora(
            y, transformed_w13, transformed_w2, buffer,
            saved_l1_preact=side_gate,
            saved_h_unweighted=side_h,
            saved_h_weighted=side_hw,
            saved_down_unweighted=side_down,
            precomputed_route_counts=route_counts,
            active_pool_rows=pool_rows,
            route_count_mismatch=mismatch,
            num_config_tokens=tokens,
            saved_x=saved_x, side_lora=side,
            side_lora_scale=args.scale,
            side_lora_scratch=(q13, q2, ready), fast_math=True)

    side_forward_ms = _time(
        side_forward, group, args.warmup, args.iterations)

    # First call allocates reusable outputs; timed calls reuse every tensor and
    # the immutable padded-prefix plan.
    side_result = deep_gemm.bf16_mega_moe_side_lora_backward(
        side_gate, side_hw, side_down, q13, q2, side,
        w2, transformed_w13, counts, padded, grad_y, buffer, block_m,
        fast_math=True, side_lora_scale=args.scale,
        direct_remote_grad_x=True, write_grad_x_pool=True)
    side_grid = torch.zeros_like(grid)
    expert_psum = padded.cumsum(0).to(torch.int32)

    def side_backward():
        nonlocal side_result
        side_result = deep_gemm.bf16_mega_moe_side_lora_backward(
            side_gate, side_hw, side_down, q13, q2, side,
            w2, transformed_w13, counts, padded, grad_y, buffer, block_m,
            fast_math=True, side_lora_scale=args.scale,
            direct_remote_grad_x=True, write_grad_x_pool=True,
            out=side_result, grid_sync_counter=side_grid,
            expert_psum_rows=expert_psum)
        # The caller represents the replicated 2-D A1/A3/B2
        # factors as DTensors with Partial gradients. Include those reductions
        # so this is an end-to-end EP training comparison, not only kernel time.
        for shared_grad in (
                side_result.grad_side_lora[0],
                side_result.grad_side_lora[2],
                side_result.grad_side_lora[5]):
            dist.all_reduce(shared_grad, group=group)

    side_backward_ms = _time(
        side_backward, group, args.warmup, args.iterations)
    profile_table = None
    if args.profile_side:
        dist.barrier(group=group)
        if rank == 0:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as profile:
                side_backward()
            profile_table = profile.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40)
        else:
            side_backward()
        dist.barrier(group=group)
    result = {
        "mode": "bf16",
        "device": torch.cuda.get_device_name(),
        "ep_ranks": ranks,
        "tokens_per_rank": tokens,
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "local_experts": local_experts,
        "topk": topk,
        "block_m": block_m,
        "pool_rows_per_rank": pool_rows,
        "routes_per_local_expert": counts.tolist(),
        "forward_ms": {
            "original": base_forward_ms,
            "side_lora": side_forward_ms,
            "ratio": side_forward_ms / base_forward_ms,
        },
        "backward_ms": {
            "original_full_wgrad": original_backward_ms,
            "side_lora_no_base_wgrad": side_backward_ms,
            "speedup": original_backward_ms / side_backward_ms,
        },
        "forward_backward_ms": {
            "original_full_wgrad": base_forward_ms + original_backward_ms,
            "side_lora_no_base_wgrad": side_forward_ms + side_backward_ms,
            "speedup": ((base_forward_ms + original_backward_ms) /
                        (side_forward_ms + side_backward_ms)),
        },
        "frozen_base_wgrads_emitted": False,
        "full_width_side_scratch_emitted": False,
        "shared_factor_ep_reductions_included": True,
    }
    if rank == 0:
        print(json.dumps(result, indent=2), flush=True)
        if profile_table is not None:
            print(profile_table, flush=True)
    dist.barrier(group=group)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--profile-side", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        run_bf16, args=(args.num_processes, args),
        nprocs=args.num_processes)
