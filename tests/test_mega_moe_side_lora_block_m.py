"""Two-rank NCCL tile guard and uniform-metadata CUDA capture regression.

Run with torchrun --standalone --nproc-per-node=2. No model or kernel timing.
"""
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.mega.backward import _side_backward_block_m


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    capacity, experts, topk = 4096, 8, 2
    expected = deep_gemm._C.get_block_m_for_mega_moe(world, experts, capacity, capacity, topk, "bf16xbf16")
    assert expected > 16, expected
    for local_tokens in (0, 7, capacity):
        tokens = local_tokens if rank == 0 else capacity
        local_tile = deep_gemm._C.get_block_m_for_mega_moe(world, experts, capacity, tokens, topk, "bf16xbf16")
        actual = _side_backward_block_m(local_tile, SimpleNamespace(group=group), torch.device("cuda", local_rank), False)
        assert actual == expected, (tokens, local_tile, actual, expected)
        if rank == 0:
            print(f"NCCL tile guard passed: source_tokens=({tokens},{capacity}), local_tile={local_tile}, uniform_tile={actual}", flush=True)
    value = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _side_backward_block_m(expected, SimpleNamespace(group=group), value.device, True)
        value.add_(actual)
    graph.replay()
    torch.cuda.synchronize()
    assert value.item() == expected
    dist.barrier()
    if rank == 0:
        print("Uniform side-backward tile fast path CUDA capture/replay passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
