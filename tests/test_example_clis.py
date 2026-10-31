"""Example CLIs build, parse and print help: argparse conflicts must fail here, not at launch.

Covers the example scripts whose CLIs changed for the default-off HEVC inter prediction
(``--hevc-inter-prediction``, one BooleanOptionalAction flag defaulting to False).
"""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("causvid_rank_worker", "runtime_demo", "causvid_runtime_controls", "native_causvid")
NATIVE = ["run", "--model-code-root", "src", "--base-model", "wan", "--checkpoint", "model.pt", "--config", "c.yaml",
          "--case-manifest", "cases.json", "--run-id", "smoke"]


def flags(parser, option):
    return [action for action in parser._actions if option in action.option_strings]


class ExampleCLITests(unittest.TestCase):
    def test_parsers_build_with_one_default_off_inter_flag(self):
        native = importlib.import_module("examples.native_causvid")
        controls = importlib.import_module("examples.causvid_runtime_controls")
        demo = importlib.import_module("examples.runtime_demo")
        cases = (
            (native.parser, NATIVE),
            (controls.parser, NATIVE + ["--cpu-receipt", "r.json", "--gpu-uuid", "GPU-0", "--store-root", "s"]),
            (demo.parser, ["--output", "out"]),
        )
        for build, argv in cases:
            with self.subTest(parser=build.__module__):
                parser = build()
                self.assertEqual(len(flags(parser, "--hevc-inter-prediction")), 1)
                self.assertEqual(len(flags(parser, "--no-hevc-inter-prediction")), 1)
                self.assertIs(parser.parse_args(argv).hevc_inter_prediction, False)
                self.assertIs(build().parse_args(argv + ["--hevc-inter-prediction"]).hevc_inter_prediction, True)
                self.assertIs(build().parse_args(argv + ["--no-hevc-inter-prediction"]).hevc_inter_prediction, False)

    def test_rank_worker_parser_builds(self):
        worker = importlib.import_module("examples.causvid_rank_worker")
        args = worker.parser().parse_args([
            "--protocol", "p.json", "--assets", "assets.json", "--asset-map", "map.json",
            "--launch-receipt", "r.json", "--requests", "q.json", "--output-root", "out",
        ])
        self.assertEqual((args.operation, args.threads, args.allow_cuda), ("generate", 1, False))

    def test_entry_points_print_help(self):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        commands = {f"-m examples.{name}": ["-m", f"examples.{name}", "--help"] for name in EXAMPLES}
        commands["examples/causvid_runtime_controls.py"] = [str(ROOT / "examples/causvid_runtime_controls.py"), "--help"]
        commands["examples/native_causvid.py worker"] = [str(ROOT / "examples/native_causvid.py"), "worker", "--help"]
        for label, arguments in commands.items():
            with self.subTest(command=label):
                result = subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, env=environment,
                                        capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
