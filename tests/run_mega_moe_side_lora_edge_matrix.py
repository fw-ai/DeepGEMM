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
        "--mode", "mxfp4", "--tokens", "17", "--topk", "2"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ep-processes", type=int, default=2,
        help="number of GPUs for remote-route cases; use 1 to skip them")
    args = parser.parse_args()
    if args.ep_processes < 1:
        parser.error("ep-processes must be positive")

    cases = list(EP1_CASES)
    if args.ep_processes > 1:
        cases.extend(EP_CASES)
    for name, arguments in cases:
        processes = (
            args.ep_processes if name.endswith("_ep") else 1)
        command = (
            sys.executable, str(WORKER),
            "--num-processes", str(processes), *arguments)
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
