import torch
import types
import warnings
import math
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_ring_tokens: int,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 lora_rank: int = 0,
                 num_lora_slots: int = 0,
                 enable_lora_down: bool = False):
        assert activation in ('swiglu', 'geglu', 'situ'), f'Unsupported activation: `{activation}`'
        if lora_rank not in (0, 128):
            raise ValueError(f'MegaMoE LoRA rank must be 0 or 128, got {lora_rank}')
        if lora_rank and mma_type != 'fp8xfp4':
            raise ValueError('MegaMoE LoRA payloads are only supported by fp8xfp4')
        if num_lora_slots < 0 or (lora_rank == 0 and num_lora_slots != 0):
            raise ValueError('num_lora_slots must be zero when LoRA is disabled')
        if lora_rank and num_lora_slots < 1:
            raise ValueError('LoRA payloads require at least one slot (the zero sentinel)')
        if enable_lora_down and not (
                lora_rank == 128 and 2 <= num_lora_slots <= 32):
            raise ValueError(
                'down LoRA requires rank 128 and 1..31 active slots plus one sentinel')
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_ring_tokens = num_ring_tokens
        self.lora_rank = lora_rank
        self.num_lora_slots = num_lora_slots
        self.lora_sentinel_slot = num_lora_slots - 1 if lora_rank else None
        self.enable_lora_down = enable_lora_down

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_ring_tokens,
            lora_rank, num_lora_slots, enable_lora_down
        )
        allocator = torch if group.size() == 1 else symm_mem
        device = torch.device('cuda', torch.cuda.current_device())
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device=device)
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf,
         self.lora_gate_up_acts, self.lora_adapter_slots,
         self.dispatched_lora_gate_up_acts,
         self.dispatched_lora_adapter_slots,
         self.lora_subgroup_offsets,
         self.routed_lora_rank_acts,
         self.combined_lora_rank_acts) = slice_input_buffers(self.buffer)
        if lora_rank:
            # Unwritten/inactive source rows are sentinel rows by default.  The
            # corresponding expert B slot is required to be all zero.
            self.lora_adapter_slots.fill_(self.lora_sentinel_slot)
            self.dispatched_lora_adapter_slots.fill_(self.lora_sentinel_slot)
        if enable_lora_down:
            self.routed_lora_rank_acts.zero_()
            self.combined_lora_rank_acts.zero_()

    def set_lora_payload(self,
                         gate_up_acts: torch.Tensor,
                         adapter_slots: torch.Tensor,
                         num_tokens: Optional[int] = None):
        """Validate and copy source-token LoRA payload into symmetric memory."""
        if not self.lora_rank:
            raise ValueError('this symmetric buffer was allocated without LoRA payload space')
        if num_tokens is None:
            num_tokens = gate_up_acts.size(0)
        if not 0 <= num_tokens <= self.num_max_tokens_per_rank:
            raise ValueError(f'invalid LoRA payload token count: {num_tokens}')
        expected_acts_shape = (num_tokens, 2, self.lora_rank)
        if tuple(gate_up_acts.shape) != expected_acts_shape:
            raise ValueError(
                f'gate_up_acts must have shape {expected_acts_shape}, got {tuple(gate_up_acts.shape)}')
        if gate_up_acts.dtype != torch.bfloat16:
            raise TypeError(f'gate_up_acts must be BF16, got {gate_up_acts.dtype}')
        if not gate_up_acts.is_cuda or gate_up_acts.device != self.buffer.device:
            raise ValueError(f'gate_up_acts must be on {self.buffer.device}')
        if not gate_up_acts.is_contiguous():
            raise ValueError('gate_up_acts must be contiguous')
        if tuple(adapter_slots.shape) != (num_tokens,):
            raise ValueError(
                f'adapter_slots must have shape {(num_tokens,)}, got {tuple(adapter_slots.shape)}')
        if adapter_slots.dtype != torch.int32:
            raise TypeError(f'adapter_slots must be int32, got {adapter_slots.dtype}')
        if not adapter_slots.is_cuda or adapter_slots.device != self.buffer.device:
            raise ValueError(f'adapter_slots must be on {self.buffer.device}')
        if not adapter_slots.is_contiguous():
            raise ValueError('adapter_slots must be contiguous')

        self.lora_gate_up_acts[:num_tokens].copy_(gate_up_acts)
        self.lora_adapter_slots[:num_tokens].copy_(adapter_slots)
        if num_tokens < self.num_max_tokens_per_rank:
            self.lora_adapter_slots[num_tokens:].fill_(self.lora_sentinel_slot)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.lora_gate_up_acts = None
        self.lora_adapter_slots = None
        self.dispatched_lora_gate_up_acts = None
        self.dispatched_lora_adapter_slots = None
        self.lora_subgroup_offsets = None
        self.routed_lora_rank_acts = None
        self.combined_lora_rank_acts = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu',
                                 lora_rank: int = 0,
                                 num_lora_slots: int = 0,
                                 enable_lora_down: bool = False) -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # To save buffer size, we enable ring buffer
    # TODO: move the wave concept into kernel and dynamically schedule
    # TODO: currently decoding may consume more memory than prefill
    # TODO: finer-grained wave
    num_min_ring_tokens, num_max_ring_tokens = \
        _C.get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts // group.size(), num_topk, group.size())
    if num_max_tokens_per_rank >= 6144:
        # We assume must be prefill (decode cannot have such size)
        # We try to give ~8 GB budget (within V4 Pro config)
        # And batch size is mostly stable, to save buffer size, we use 1 expert per wave
        num_ring_tokens = align(768 * 1024, _C.get_token_alignment_for_mega_moe())
    else:
        # Otherwise, we must ensure, like for EP64, 4K decoding batch size,
        # the wave heuristics can select the best number of experts per wave
        # In this case, the budget is roughly ~18 GB
        num_ring_tokens = _C.get_ring_limit_for_mega_moe(
            align(4096, _C.get_token_alignment_for_mega_moe()), 432 // 72, 6, 72)[1]
    num_ring_tokens = max(num_ring_tokens, num_min_ring_tokens)
    num_ring_tokens = min(num_ring_tokens, num_max_ring_tokens)

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_ring_tokens,
        mma_type=mma_type, activation=activation,
        lora_rank=lora_rank, num_lora_slots=num_lora_slots,
        enable_lora_down=enable_lora_down
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    # Gate/up interleaving is independent of the gated-activation variant
    assert activation in ('swiglu', 'geglu', 'situ'), f'Unsupported activation: `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     situ_beta: Optional[float] = None,
                     situ_linear_beta: Optional[float] = None,
                     lora_mode: str = 'disabled',
                     lora_gate_b: Optional[torch.Tensor] = None,
                     lora_up_b: Optional[torch.Tensor] = None,
                     lora_down_a: Optional[torch.Tensor] = None,
                     lora_scaling: float = 1.0):
    if lora_mode not in ('disabled', 'payload_only', 'fc1', 'fc1_down'):
        raise ValueError(f'Unsupported MegaMoE LoRA mode: `{lora_mode}`')
    if lora_mode != 'disabled' and sym_buffer.lora_rank != 128:
        raise ValueError(f'{lora_mode} mode requires a symmetric buffer with lora_rank=128')
    if lora_mode in ('fc1', 'fc1_down'):
        if not 2 <= sym_buffer.num_lora_slots <= 32:
            raise ValueError(
                'fc1 requires 1..31 active adapter slots plus one sentinel slot')
        expected_shape = (
            sym_buffer.num_lora_slots,
            sym_buffer.num_experts // sym_buffer.group.size(),
            sym_buffer.intermediate_hidden, sym_buffer.lora_rank)
        for name, tensor in (('lora_gate_b', lora_gate_b), ('lora_up_b', lora_up_b)):
            if tensor is None:
                raise ValueError(f'{name} is required in {lora_mode} mode')
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(tensor.shape)}')
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f'{name} must be BF16, got {tensor.dtype}')
            if not tensor.is_cuda or tensor.device != sym_buffer.buffer.device:
                raise ValueError(f'{name} must be on {sym_buffer.buffer.device}')
            if not tensor.is_contiguous():
                raise ValueError(f'{name} must be contiguous')
    elif lora_gate_b is not None or lora_up_b is not None:
        raise ValueError('LoRA B pools are only accepted in fc1/fc1_down mode')
    if lora_mode == 'fc1_down':
        if not sym_buffer.enable_lora_down:
            raise ValueError(
                'fc1_down requires a symmetric buffer allocated with '
                'enable_lora_down=True')
        expected_down_shape = (
            sym_buffer.num_lora_slots,
            sym_buffer.num_experts // sym_buffer.group.size(),
            sym_buffer.lora_rank,
            sym_buffer.intermediate_hidden)
        if lora_down_a is None:
            raise ValueError('lora_down_a is required in fc1_down mode')
        if tuple(lora_down_a.shape) != expected_down_shape:
            raise ValueError(
                f'lora_down_a must have shape {expected_down_shape}, '
                f'got {tuple(lora_down_a.shape)}')
        if lora_down_a.dtype != torch.bfloat16:
            raise TypeError(f'lora_down_a must be BF16, got {lora_down_a.dtype}')
        if not lora_down_a.is_cuda or lora_down_a.device != sym_buffer.buffer.device:
            raise ValueError(f'lora_down_a must be on {sym_buffer.buffer.device}')
        if not lora_down_a.is_contiguous():
            raise ValueError('lora_down_a must be contiguous')
        if not isinstance(lora_scaling, (float, int)):
            raise TypeError('lora_scaling must be a finite float')
        if not math.isfinite(float(lora_scaling)):
            raise ValueError('lora_scaling must be finite')
    elif lora_down_a is not None:
        raise ValueError('lora_down_a is only accepted in fc1_down mode')
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        situ_beta, situ_linear_beta,
        fast_math,
        sym_buffer.num_ring_tokens,
        lora_mode,
        sym_buffer.lora_rank,
        sym_buffer.num_lora_slots,
        lora_gate_b,
        lora_up_b,
        lora_down_a,
        float(lora_scaling)
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        sym_buffer.num_ring_tokens
    )
