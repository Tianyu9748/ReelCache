"""Run CPU regressions and persist the result bound to the tested source tree."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    files = sorted({
        *(path for directory in ("reelcache", "reelcache_sglang_models", "tests", "examples", "scripts")
          for path in (ROOT / directory).rglob("*.py")),
        ROOT / "pyproject.toml",
    })
    identities = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    digest = hashlib.sha256(json.dumps(
        identities, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return digest, identities


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.started = {}

    def startTest(self, test):
        self.started[test.id()] = time.monotonic()
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        if self.testsRun % 25 == 0:
            print(f"Completed {self.testsRun} CPU tests", flush=True)

    def _record(self, test, status, detail=None):
        started = self.started.get(test.id())
        self.records.append({
            "test": test.id(), "status": status, "detail": detail,
            "seconds": None if started is None else time.monotonic() - started,
        })

    def addSuccess(self, test):
        self._record(test, "passed")
        super().addSuccess(test)

    def addError(self, test, error):
        self._record(test, "error", self._exc_info_to_string(error, test))
        super().addError(test, error)

    def addFailure(self, test, error):
        self._record(test, "failed", self._exc_info_to_string(error, test))
        super().addFailure(test, error)

    def addSkip(self, test, reason):
        self._record(test, "skipped", reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, error):
        self._record(test, "expected_failure", self._exc_info_to_string(error, test))
        super().addExpectedFailure(test, error)

    def addUnexpectedSuccess(self, test):
        self._record(test, "unexpected_success")
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, error):
        if error is not None:
            self._record(subtest, "subtest_failed", self._exc_info_to_string(error, test))
        super().addSubTest(test, subtest, error)


def test_inventory(suite):
    result = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            result.extend(test_inventory(test))
        else:
            result.append(test.id())
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch-closure", type=Path)
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from reelcache.errors import IntegrityError
    from reelcache.study import read_json, verify_source_closure

    closure = None if args.launch_closure is None else read_json(args.launch_closure)
    if closure is not None:
        verify_source_closure(closure)
    before, files = source_identity()
    mandatory = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT),
    )
    mandatory_inventory = test_inventory(mandatory)
    suite = mandatory if args.pattern == "test_*.py" else unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern=args.pattern, top_level_dir=str(ROOT),
    )
    selected_inventory = test_inventory(suite)
    output = io.StringIO()
    started = time.monotonic()
    result = unittest.TextTestRunner(
        stream=output, verbosity=2, resultclass=RecordingResult
    ).run(suite)
    duration = time.monotonic() - started
    after, _ = source_identity()
    closure_error = None
    if closure is not None:
        try:
            verify_source_closure(closure)
        except (IntegrityError, OSError) as exc:
            closure_error = str(exc)
    require_no_skips = args.require_no_skips or closure is not None
    passed = (
        result.wasSuccessful() and not result.expectedFailures
        and result.testsRun > len(result.skipped)
        and before == after
        and (not require_no_skips or not result.skipped)
        and closure_error is None
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    destination = args.output or ROOT / "artifacts/verification" / (
        now.strftime("cpu-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "reelcache_cpu_verification_v1",
        "status": "passed" if passed else "failed",
        "utc": now.isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "source_tree_sha256": before,
        "source_files": files,
        "source_unchanged_during_tests": before == after,
        "launch_closure_id": None if closure is None else closure["closure_id"],
        "launch_closure_error": closure_error,
        "require_no_skips": require_no_skips,
        "mandatory_inventory": mandatory_inventory,
        "selected_inventory": selected_inventory,
        "discovery": {
            "root": str(ROOT / "tests"), "pattern": args.pattern,
            "full_suite": args.pattern == "test_*.py" and selected_inventory == mandatory_inventory,
        },
        "tests_run": result.testsRun, "skipped": len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "seconds": duration, "records": result.records,
        "log": output.getvalue(),
        "native_model_validation": False,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    with destination.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(output.getvalue(), end="")
    print(f"Result: {destination}")
    if before != after:
        print("FAILED: source changed during the run; rerun against a stable tree.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
