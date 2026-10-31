"""Import isolation of the out-of-tree SGLang model package ``reelcache_sglang_models``.

* Core ``reelcache`` never imports the package (static scan of import statements + fresh interpreter).
* Importing the package and its SGLang-free modules registers nothing and imports no SGLang.
* When SGLang's hook names the package (``SGLANG_EXTERNAL_MODEL_PACKAGE``) but SGLang is not importable,
  the import fails loudly instead of silently skipping registration.
* The package is picked up by the project's setuptools discovery pattern.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PURE_MODULES = ["spec", "weights", "rope", "schedule", "conditioning_cache", "lora", "dit_core", "video_io", "text",
                "segment_resume", "reelcache_backend", "context_parallel"]
SGLANG_MODULES = ["sglang_configs", "sglang_dit", "sglang_loader", "sglang_stages", "sglang_pipeline"]


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "SGLANG_EXTERNAL_MODEL_PACKAGE"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True,
                          timeout=900)


class ImportIsolationTests(unittest.TestCase):
    def test_core_reelcache_never_imports_the_package(self):
        # Imports only (statements and importlib strings); a docstring cross-reference is not a dependency.
        pattern = re.compile(r"(^|\n)\s*(from|import)\s+reelcache_sglang_models\b"
                             r"|import_module\(\s*[\"']reelcache_sglang_models|__import__\(\s*[\"']reelcache_sglang_models")
        offenders = []
        for path in (ROOT / "reelcache").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_pure_modules_import_without_sglang_or_registration(self):
        mods = ", ".join(f"reelcache_sglang_models.longcat_video.{m}" for m in PURE_MODULES)
        code = (
            "import json, sys\n"
            "import reelcache_sglang_models as pkg\n"
            f"import {mods}\n"
            "import reelcache_sglang_models.longcat_video.registration as reg\n"
            "print(json.dumps({'sglang': sorted(m for m in sys.modules if m == 'sglang' or m.startswith('sglang.')),"
            " 'sglang_side': sorted(m for m in sys.modules if m.startswith('reelcache_sglang_models.longcat_video.sglang_')),"
            " 'registered': pkg.registered(), 'loaded_by_sglang': pkg.loaded_by_sglang()}))\n")
        result = _run(code)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        info = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(info["sglang"], [])
        self.assertEqual(info["sglang_side"], [])
        self.assertEqual(info["registered"], {"longcat_video": None})
        self.assertFalse(info["loaded_by_sglang"])

    def test_core_reelcache_import_does_not_load_the_package(self):
        code = ("import json, sys, reelcache\n"
                "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('reelcache_sglang_models', 'sglang')))))\n")
        result = _run(code)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), [])

    def test_requested_registration_without_sglang_fails_loudly(self):
        probe = _run("import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('sglang') is None else 3)")
        self.assertIn(probe.returncode, (0, 3))
        if probe.returncode == 3:
            # SGLang is importable in this interpreter: registration must then succeed instead.
            result = _run("import reelcache_sglang_models as p; print(p.registered())",
                          {"SGLANG_EXTERNAL_MODEL_PACKAGE": "reelcache_sglang_models"})
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            self.assertIn("LongCatVideoContinuationPipeline", result.stdout)
            return
        result = _run("import reelcache_sglang_models", {"SGLANG_EXTERNAL_MODEL_PACKAGE": "reelcache_sglang_models"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No module named 'sglang'", result.stderr)

    def test_sglang_side_modules_are_only_reached_through_register(self):
        for name in SGLANG_MODULES:
            source = (ROOT / "reelcache_sglang_models" / "longcat_video" / f"{name}.py").read_text()
            self.assertRegex(source, r"\nfrom sglang\.multimodal_gen", name)
        for name in PURE_MODULES + ["__init__"]:
            source = (ROOT / "reelcache_sglang_models" / "longcat_video" / f"{name}.py").read_text()
            self.assertNotRegex(source, r"(^|\n)\s*(from|import) sglang", name)
            self.assertNotRegex(source, r"from \.sglang_|import \.sglang_", name)
        registration = (ROOT / "reelcache_sglang_models" / "longcat_video" / "registration.py").read_text()
        top = registration.split("def register_longcat_video", 1)[0]
        self.assertNotRegex(top, r"(from|import) sglang")

    def test_setuptools_discovery_includes_the_package(self):
        text = (ROOT / "pyproject.toml").read_text()
        patterns = re.search(r"\[tool\.setuptools\.packages\.find\]\s*include\s*=\s*\[([^\]]*)\]", text)
        self.assertIsNotNone(patterns)
        includes = [p.strip().strip("\"'") for p in patterns.group(1).split(",") if p.strip()]
        for pkg in ("reelcache_sglang_models", "reelcache_sglang_models.longcat_video"):
            self.assertTrue(any(fnmatch.fnmatch(pkg, pat) for pat in includes), pkg)


if __name__ == "__main__":
    unittest.main()
