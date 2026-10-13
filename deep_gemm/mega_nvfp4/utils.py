# Derived from sgl-project/DeepGEMM@ec226bcf400c11d62755137b916fae9ee359c684:
#   deep_gemm/utils/math.py (`cast_to_ue4m3`, `pack_ue4m3_to_int`, `unpack_ue4m3_from_int`,
#   `per_token_cast_to_nvfp4`, `transform_ue4m3_sf_into_required_layout`)
# Copyright (c) 2025 DeepSeek, with modifications by the sgl-project/DeepGEMM contributors.
# Licensed under the MIT License (see LICENSE at the repository root).
# Fireworks port: unchanged, kept out of `deep_gemm.utils` so that module stays as it is.
from typing import Tuple

import torch

from ..utils.math import _quantize_to_fp4_e2m1


def cast_to_ue4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).float()


def pack_ue4m3_to_int(x: torch.Tensor) -> torch.Tensor:
    """Pack 4 UE4M3 scaling factor bytes into one int32, matching the kernel's SF word."""
    assert x.size(-1) % 4 == 0
    return x.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).view(torch.int)


def unpack_ue4m3_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
    return packed_sf.view(torch.uint8).view(torch.float8_e4m3fn).float()


def per_token_cast_to_nvfp4(x: torch.Tensor, gran_k: int = 16,
                            use_packed_ue4m3: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4: E2M1 values with one UE4M3 (not pow2 UE8M0) scaling factor per `gran_k` elements."""
    m, n = x.shape
    assert n % gran_k == 0
    x_view = x.view(m, -1, gran_k)
    sf = cast_to_ue4m3(x_view.abs().float().amax(dim=2) / 6.0)
    sf_inv = torch.where(sf > 0, 1.0 / sf, torch.zeros_like(sf))
    codes = _quantize_to_fp4_e2m1(x_view.float() * sf_inv.unsqueeze(2)).view(m, n)
    codes2 = codes.view(m, n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed.contiguous(), pack_ue4m3_to_int(sf) if use_packed_ue4m3 else sf


def transform_ue4m3_sf_into_required_layout(sf: torch.Tensor, mn: int) -> torch.Tensor:
    """MN-major, TMA-aligned, int32-packed UE4M3 SFs — the weight-side layout the kernel reads.

    `transform_sf_into_required_layout` only knows how to pack UE8M0, so NVFP4 weight SFs are
    packed here and handed to its already-packed `(INT, 1, gran_k)` branch.
    """
    assert sf.dim() in (2, 3) and sf.size(-2) == mn
    assert mn % 4 == 0, f'MN must be TMA-aligned for int32 SFs, got {mn}'
    return pack_ue4m3_to_int(sf).transpose(-1, -2).contiguous().transpose(-1, -2)
