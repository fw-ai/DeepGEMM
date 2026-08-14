"""Run the native MegaMoE side-LoRA forward/backward edge matrix.

The numerical worker lives in ``test_mega_moe_native_side_lora.py``. Keeping
the matrix here makes every production contract explicit while each case gets
a fresh distributed process group and symmetric buffer.
"""

import argparse
import subprocess
import sys
from pathlib import Path


WORKER = Path(__file__).with_name("test_mega_moe_native_side_lora.py")
COMMON = (
    "--hidden", "1024",
    "--intermediate", "512",
    "--experts", "8",
)


def _case(name: str, *arguments: str) -> tuple[str, tuple[str, ...]]:
    return name, (*COMMON, *arguments)


def _production_case(
    mode: str, tokens: int,
) -> tuple[str, tuple[str, ...]]:
    return (
        f"{mode}_production_boundary_{tokens}_ep4",
        (
            "--hidden", "4096",
            "--intermediate", "2048",
            "--experts", "256",
            "--topk", "6",
            "--mode", mode,
            "--tokens", str(tokens),
            "--routing", "remote",
            "--masked-ratio", "0.2",
        ),
    )


EP1_CASES = (
    _case(
        "bf16_single_route_single_token",
        "--mode", "bf16", "--tokens", "1", "--topk", "1"),
    _case(
        "bf16_empty_experts_masked_bm_minus_one",
        "--mode", "bf16", "--tokens", "15", "--topk", "2",
        "--routing", "empty_experts", "--masked-ratio", "0.2",
        "--activation-limit", "2.0"),
    _case(
        "bf16_geglu_bm_boundary",
        "--mode", "bf16", "--tokens", "16", "--topk", "1",
        "--routing", "skewed", "--activation", "geglu",
        "--activation-limit", "1.5"),
    _case(
        "bf16_swiglu_bm_plus_one",
        "--mode", "bf16", "--tokens", "17", "--topk", "2"),
    _case(
        "bf16_zero_scale_base_preservation",
        "--mode", "bf16", "--tokens", "17", "--topk", "2",
        "--scale", "0"),
    _case(
        "mxfp4_balanced_bm_plus_one",
        "--mode", "mxfp4", "--tokens", "17", "--topk", "2",
        "--default-side-lora-scratch", "--check-short-saved-down"),
    _case(
        "mxfp4_zero_scale_base_preservation",
        "--mode", "mxfp4", "--tokens", "17", "--topk", "2",
        "--scale", "0"),
    _case(
        "mxfp4_empty_experts_masked_clamped",
        "--mode", "mxfp4", "--tokens", "15", "--topk", "2",
        "--routing", "empty_experts", "--masked-ratio", "0.2",
        "--activation-limit", "2.0"),
)


EP_CASES = (
    _case(
        "bf16_remote_masked_ep",
        "--mode", "bf16", "--tokens", "17", "--topk", "2",
        "--routing", "remote", "--masked-ratio", "0.2"),
    _case(
        "mxfp4_remote_masked_ep",
        "--mode", "mxfp4", "--tokens", "17", "--topk", "2",
        "--routing", "remote", "--masked-ratio", "0.2"),
)


# Every point immediately below and above a production scheduler transition,
# plus a zero-active-rank case and a larger steady-state case. These exact
# DSV4 Flash widths reproduce padding and rank-uniform route-pool suffixes that
# the compact default cases cannot exercise.
PRODUCTION_BOUNDARY_TOKENS = (
    1, 15, 16, 17, 90, 91, 176, 177, 346, 347, 688, 689, 1029, 1030, 2048,
)
PRODUCTION_EP4_CASES = tuple(
    _production_case(mode, tokens)
    for mode in ("bf16", "mxfp4")
    for tokens in PRODUCTION_BOUNDARY_TOKENS
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ep-processes", type=int, default=2,
        help="number of GPUs for remote-route cases; use 1 to skip them")
    parser.add_argument(
        "--production-boundaries", action="store_true",
        help=(
            "also run the exact 4096x2048, 256-expert EP4 boundary sweep "
            "on four GPUs"
        ),
    )
    args = parser.parse_args()
    if args.ep_processes < 1:
        parser.error("ep-processes must be positive")

    cases = list(EP1_CASES)
    if args.ep_processes > 1:
        cases.extend(EP_CASES)
    if args.production_boundaries:
        cases.extend(PRODUCTION_EP4_CASES)
    for name, arguments in cases:
        if name.endswith("_ep4"):
            processes = 4
        else:
            processes = args.ep_processes if name.endswith("_ep") else 1
        command = (
            sys.executable, str(WORKER),
            "--num-processes", str(processes), *arguments)
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
