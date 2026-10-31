import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from examples import causvid_runtime_controls as driver


class RuntimeControlDriverTests(unittest.TestCase):
    def test_gpu_gate_checks_uuid_type_free_memory_and_existing_jobs(self):
        uuid = "GPU-test"
        device = "GPU-test, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97000\n"
        with mock.patch.object(driver.subprocess, "check_output", side_effect=[device, ""]), \
                mock.patch.object(Path, "read_text", return_value="MemFree: 67108864 kB\n"):
            self.assertEqual(driver.admit_device(uuid, minimum_free_gib=48)["uuid"], uuid)
        for inventory, processes in (
            ("GPU-other, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97000\n", ""),
            ("GPU-test, NVIDIA A100, 97000\n", ""),
            ("GPU-test, NVIDIA RTX PRO 6000 Blackwell Server Edition, 2000\n", ""),
            (device, "GPU-test, 123\n"),
        ):
            with mock.patch.object(driver.subprocess, "check_output", side_effect=[inventory, processes]):
                with self.assertRaises(ValueError):
                    driver.admit_device(uuid, minimum_free_gib=48)

    def test_cpu_receipt_requires_exact_inventory_source_and_no_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("reelcache", "tests", "examples", "scripts"):
                (root / name).mkdir()
                (root / name / "fixture.py").write_text("pass\n")
            (root / "pyproject.toml").write_text("[project]\n")
            files = {str(p.relative_to(root)): driver.native.sha256(p) for p in root.rglob("*") if p.is_file()}
            report = {
                "status": "passed", "source_unchanged_during_tests": True,
                "failures": 0, "errors": 0, "skipped": 0, "require_no_skips": True,
                "discovery": {"full_suite": True}, "source_files": files, "source_tree_sha256": "fixture",
            }
            path = root / "receipt.json"
            with mock.patch.object(driver, "ROOT", root):
                path.write_text(json.dumps(report))
                self.assertEqual(driver.verify_cpu_receipt(path), "fixture")
                for changes in ({"skipped": 1}, {"require_no_skips": False},
                                {"discovery": {"full_suite": False}}, {"source_files": {}}):
                    path.write_text(json.dumps({**report, **changes}))
                    with self.assertRaises(ValueError):
                        driver.verify_cpu_receipt(path)
                path.write_text(json.dumps(report))
                (root / "reelcache/fixture.py").write_text("changed\n")
                with self.assertRaisesRegex(ValueError, "source differs"):
                    driver.verify_cpu_receipt(path)


if __name__ == "__main__":
    unittest.main()
