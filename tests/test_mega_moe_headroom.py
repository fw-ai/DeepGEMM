"""CPU-only checks for headroom policy and side-LoRA launch wiring.

Compile the actual host helper with a stub device runtime; no CUDA or torch
installation is needed. GPU numerical tests remain a separate requirement.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


IMPLS = Path(__file__).resolve().parents[1] / "csrc/jit_kernels/impls"


class TestMegaMoEHeadroom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        source = (IMPLS / "runtime_utils.hpp").read_text()
        start = source.index("static int get_mega_moe_num_sms() {")
        end = source.index("\n}", start) + 2
        helper = source[start:end]
        harness = r'''
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#define DG_HOST_ASSERT(condition) \
    do { if (!(condition)) throw std::runtime_error("invalid headroom"); } while (0)
struct Runtime { int sms; int get_num_sms() { return sms; } };
Runtime runtime;
Runtime* device_runtime = &runtime;
template <typename T> T get_env(const char* name, T fallback) {
    const char* value = std::getenv(name);
    return value ? static_cast<T>(std::atoi(value)) : fallback;
}
int align(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}
''' + helper + r'''
int main(int argc, char** argv) {
    runtime.sms = std::atoi(argv[1]);
    try { std::cout << get_mega_moe_num_sms(); }
    catch (const std::runtime_error&) { return 2; }
}
'''
        cls.directory = tempfile.TemporaryDirectory(prefix="megamoe-headroom-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = str(Path(cls.directory.name) / "headroom")
        subprocess.run(
            [compiler, "-std=c++17", "-x", "c++", "-", "-o", cls.binary],
            input=harness, text=True, check=True, capture_output=True,
        )

    def test_host_policy(self):
        # Device runtime may already be limited by set_num_sms().
        for sms, headroom, expected in (
            (148, None, 148), (148, "0", 148), (148, "8", 140),
            (148, "7", 140), (148, "1", 146), (132, "8", 124),
            (148, "146", 2), (148, "-1", None),
            (148, "147", None), (148, "148", None), (148, "150", None),
        ):
            with self.subTest(sms=sms, headroom=headroom):
                env = dict(os.environ)
                env.pop("DG_MEGA_MOE_SM_HEADROOM", None)
                if headroom is not None:
                    env["DG_MEGA_MOE_SM_HEADROOM"] = headroom
                result = subprocess.run(
                    [self.binary, str(sms)], env=env,
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 2 if expected is None else 0)
                if expected is not None:
                    self.assertEqual(int(result.stdout), expected)

    def test_bf16_forward_preserves_absolute_override(self):
        source = (IMPLS / "sm100_bf16_mega_moe_side_lora_forward.hpp").read_text()
        self.assertRegex(
            source,
            r'get_env<int>\(\s*"DG_BF16_MEGA_MOE_NUM_SMS",\s*get_mega_moe_num_sms\(\)\)',
        )

    def test_mxfp4_forward_uses_headroom(self):
        source = (IMPLS / "sm100_fp8_fp4_mega_moe_side_lora_forward.hpp").read_text()
        self.assertIn("const auto num_sms = get_mega_moe_num_sms();", source)
        self.assertNotIn("device_runtime->get_num_sms()", source)

    def test_both_backward_persistent_grids_use_headroom(self):
        source = (IMPLS / "sm100_bf16_mega_moe_side_lora_backward.hpp").read_text()
        self.assertEqual(source.count("const int num_sms = get_mega_moe_num_sms();"), 2)
        self.assertNotIn("const int num_sms = device_runtime->get_num_sms();", source)
        # The ordinary dense GEMM helper has no whole-grid barrier and keeps
        # its own occupancy heuristic; do not indiscriminately cap every GEMM.
        self.assertIn(".num_sms = device_runtime->get_num_sms(),", source)


if __name__ == "__main__":
    unittest.main()
