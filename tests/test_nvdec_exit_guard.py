"""NVDEC exit guard (``reelcache.nvdec.install_exit_guard``): a process that used NVDEC never hangs in C ``exit()``.

libnvcuvid's native exit handler can wait forever after the Python exit completed (RTX PRO 6000: Step 18 job
4429779, Step 11 job 4419034). The CPU tests stand a never-returning native exit handler, registered before the
guard as the library's is, in for it, in child processes. CUDA tests (``REELCACHE_STEP7_GPU_TESTS=1``, GPU jobs only)
exit a child with real NVDEC sessions left open; they are collected only there, so the no-skip CPU suite never
skips them. The RTX before/after reproduction is
``artifacts/sglang_integration/benchmarks/restore_e2e/scripts/nvdec_exit_bridge.py``.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import reelcache
from reelcache import nvdec

# argv: [actions...]; "hang" registers a native exit handler that never returns (pause()) before anything else,
# "late" a Python exit hook registered before reelcache's import (so it runs after the guard's hook) that writes Python
# and C stdio output, "session" an open canvas session that an exit hook closes (as a decoder pool's does), "leak"
# one left open, "nvdec" uses NVDEC (arms the guard), "fork" exits a forked child of this process first, and the
# last argument is the exit: an integer status, "raise" or "return".
CHILD = r'''
import atexit
import ctypes
import os
import sys

*actions, ending = sys.argv[1:]
libc = ctypes.CDLL(None)
libc.__cxa_atexit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]  # what atexit() and C++ use
libc.fputs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")


def c_print(text):
    libc.fputs(text.encode(), c_stdout)  # buffered C stdio (stdout is a pipe), no newline


if "hang" in actions:
    libc.__cxa_atexit(ctypes.cast(libc.pause, ctypes.c_void_p).value, None, None)
if "late" in actions:
    atexit.register(lambda: (print("PY-LATE", flush=True), c_print(" C-LATE")))

from reelcache import nvdec  # after the "late" hook

if "session" in actions or "leak" in actions:
    session = object.__new__(nvdec.CanvasSession)  # as CanvasSession.__init__ leaves it after CreateDecoder
    session._decoder = object()
    nvdec._SESSION_OWNERS.add(session)
    if "session" in actions:
        atexit.register(setattr, session, "_decoder", None)
if "nvdec" in actions:
    nvdec.note_nvdec_use()
print("WORK", nvdec.exit_guard_state()["armed"], flush=True)
c_print("C-EARLY")
if "fork" in actions:
    child = os.fork()
    if child == 0:
        sys.exit(3)  # the child's exit runs the inherited exit hooks and native handlers
    _, status = os.waitpid(child, 0)
    print("CHILD", os.waitstatus_to_exitcode(status), flush=True)
if ending == "raise":
    raise RuntimeError("the child failed")
if ending != "return":
    sys.exit(int(ending))
'''


def run_child(*args, guard=None, ready="WORK", exit_timeout=30.0, startup=300.0, script=CHILD, report=False):
    """Run a child with this tree first on its path; its exit is timed from its ``ready`` line on.

    Returns ``(returncode, stdout, stderr, exit_seconds)``; ``returncode`` is None when the exit took longer than
    ``exit_timeout`` and the child was killed.
    """
    tree = Path(reelcache.__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        part for part in (str(tree), os.environ.get("PYTHONPATH", "")) if part))
    env.pop(nvdec.EXIT_GUARD_ENV, None)
    env.pop(nvdec.EXIT_REPORT_ENV, None)
    if guard is not None:
        env[nvdec.EXIT_GUARD_ENV] = guard
    if report:
        env[nvdec.EXIT_REPORT_ENV] = "1"
    with tempfile.TemporaryDirectory() as scratch, subprocess.Popen(  # neutral working directory: imports from PYTHONPATH
            [sys.executable, "-c", script, *args], cwd=scratch, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE) as child:
        stdout, stderr, seen = [], [], threading.Event()

        def pump(stream, sink, marker):
            for line in stream:
                sink.append(line)
                if marker is not None and line.startswith(marker):
                    seen.set()

        pumps = [threading.Thread(target=pump, args=(child.stdout, stdout, ready), daemon=True),
                 threading.Thread(target=pump, args=(child.stderr, stderr, None), daemon=True)]
        for thread in pumps:
            thread.start()
        deadline = time.monotonic() + startup
        while not seen.wait(0.05) and child.poll() is None:
            if time.monotonic() > deadline:
                child.kill()
                child.wait()
                raise AssertionError(f"the child never printed {ready!r}: {''.join(stderr)[-3000:]}")
        started = time.monotonic()
        try:
            returncode = child.wait(exit_timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            returncode = None
        seconds = time.monotonic() - started
        for thread in pumps:
            thread.join(30)
    return returncode, "".join(stdout), "".join(stderr), seconds


class ExitGuardChildTests(unittest.TestCase):
    """End to end in child processes: status, output and a bounded exit despite a never-returning native handler."""

    def test_guard_skips_a_hanging_native_handler_and_keeps_status_and_output(self):
        """Same status and output, Python and C stdio (also written after the guard's hook), as a regular exit."""
        expected = run_child("late", "nvdec", "7", guard="0")
        self.assertEqual(expected[0], 7, expected[2][-3000:])
        returncode, stdout, stderr, seconds = run_child("hang", "late", "nvdec", "7")
        self.assertEqual(returncode, 7, stderr[-3000:])
        self.assertLess(seconds, 20.0)
        self.assertEqual((stdout, stderr), expected[1:3])
        for text in ("WORK True", "PY-LATE", "C-EARLY", " C-LATE"):
            self.assertIn(text, stdout)

    def test_without_the_guard_the_native_handler_hangs_the_exit(self):
        """Negative control: the stand-in deadlocks C ``exit()`` like libnvcuvid's handler did."""
        for args, guard in ((("hang", "nvdec", "0"), "0"), (("hang", "0"), None)):  # disabled; never armed
            with self.subTest(args=args, guard=guard):
                returncode, stdout, _, seconds = run_child(*args, guard=guard, exit_timeout=3.0)
                self.assertIsNone(returncode)
                self.assertIn("WORK", stdout)
                self.assertGreaterEqual(seconds, 3.0)

    def test_an_uncaught_exception_keeps_its_status_and_traceback(self):
        returncode, stdout, stderr, _ = run_child("hang", "nvdec", "raise")
        self.assertEqual(returncode, 1)
        self.assertIn("RuntimeError: the child failed", stderr)
        self.assertIn("C-EARLY", stdout)

    def test_returning_from_main_exits_zero(self):
        returncode, stdout, stderr, _ = run_child("hang", "nvdec", "return")
        self.assertEqual(returncode, 0, stderr[-3000:])
        self.assertIn("C-EARLY", stdout)

    def test_the_exit_report_shows_the_guard_and_the_open_sessions(self):
        for guard, installed, error in ((None, True, None), ("0", False, f"disabled by {nvdec.EXIT_GUARD_ENV}=0")):
            with self.subTest(guard=guard):
                returncode, _, stderr, _ = run_child("nvdec", "0", guard=guard, report=True)
                self.assertEqual(returncode, 0, stderr[-3000:])
                state = {"armed": True, "installed": installed, "error": error,
                         "open_at_exit": {"canvas_sessions": 0, "decoder_sessions": 0}}
                self.assertEqual(stderr, f"reelcache.nvdec exit guard {state}\n")

    def test_the_guard_runs_after_the_exit_hooks_that_close_sessions(self):
        """Registered at import, the guard's hook runs after those of pools and bridges and sees what they left."""
        for action, left in (("session", 0), ("leak", 1)):
            with self.subTest(action=action):
                returncode, _, stderr, _ = run_child("hang", action, "nvdec", "0", report=True)
                self.assertEqual(returncode, 0, stderr[-3000:])
                state = {"armed": True, "installed": True, "error": None,
                         "open_at_exit": {"canvas_sessions": left, "decoder_sessions": 0}}
                self.assertEqual(stderr, f"reelcache.nvdec exit guard {state}\n")

    def test_a_forked_child_inherits_the_guard(self):
        """A fork copies the library's exit handler but not its threads: the child is guarded as well."""
        returncode, stdout, stderr, seconds = run_child("hang", "nvdec", "fork", "5")
        self.assertEqual(returncode, 5, stderr[-3000:])
        self.assertIn("CHILD 3", stdout)
        self.assertLess(seconds, 20.0)


class ExitGuardStateTests(unittest.TestCase):
    """In-process: arming and installation decisions (the real installation is only exercised in children)."""

    def setUp(self):
        state = mock.patch.dict(nvdec._EXIT_GUARD, {"armed": False, "installed": False, "error": None,
                                                    "open_at_exit": None})
        state.start()
        self.addCleanup(state.stop)

    def test_use_arms_the_guard(self):
        self.assertFalse(nvdec.exit_guard_state()["armed"])
        nvdec.note_nvdec_use()
        nvdec.note_nvdec_use()
        self.assertEqual(nvdec.exit_guard_state(), {"armed": True, "installed": False, "error": None,
                                                    "open_at_exit": None})

    def test_an_unarmed_or_disabled_guard_installs_nothing(self):
        with mock.patch.object(nvdec.ctypes, "CDLL") as cdll:
            self.assertFalse(nvdec.install_exit_guard())  # NVDEC never used
            nvdec.note_nvdec_use()
            with mock.patch.dict(os.environ, {nvdec.EXIT_GUARD_ENV: "0"}):
                self.assertFalse(nvdec.install_exit_guard())
        cdll.assert_not_called()
        self.assertEqual(nvdec.exit_guard_state()["error"], f"disabled by {nvdec.EXIT_GUARD_ENV}=0")
        self.assertFalse(nvdec.exit_guard_state()["installed"])

    def test_without_glibc_exit_handlers_the_exit_is_unchanged(self):
        nvdec.note_nvdec_use()
        with mock.patch.object(nvdec.ctypes, "CDLL", return_value=object()):  # e.g. musl: no on_exit
            self.assertFalse(nvdec.install_exit_guard())
        self.assertIn("no glibc exit handlers", nvdec.exit_guard_state()["error"])
        self.assertFalse(nvdec.exit_guard_state()["installed"])

    def test_open_sessions_are_counted_until_closed(self):
        with mock.patch.object(nvdec, "_SESSION_OWNERS", weakref.WeakSet()):
            decoder = nvdec.NVDECDecoder(gpu_id=0, gpu_uuid="GPU-00000000")  # no CUDA work at construction
            session = object.__new__(nvdec.CanvasSession)  # as CanvasSession.__init__ leaves it after CreateDecoder
            session._decoder = object()
            nvdec._SESSION_OWNERS.add(session)
            self.assertEqual(nvdec.open_nvdec_sessions(), {"canvas_sessions": 1, "decoder_sessions": 0})
            decoder._sessions[(64, 32, "yuv444p", 0)] = object()
            self.assertEqual(nvdec.open_nvdec_sessions(), {"canvas_sessions": 1, "decoder_sessions": 1})
            session._decoder = None  # closed
            decoder._sessions.clear()
            self.assertEqual(nvdec.open_nvdec_sessions(), {"canvas_sessions": 0, "decoder_sessions": 0})
            del session, decoder
            self.assertEqual(len(nvdec._SESSION_OWNERS), 0)  # held weakly

    def test_failing_diagnostics_never_stop_the_guard(self):
        nvdec.note_nvdec_use()
        libc = mock.Mock(on_exit=mock.Mock(return_value=0))
        with mock.patch.object(nvdec.ctypes, "CDLL", return_value=libc), \
                mock.patch.object(nvdec.ctypes, "cast", return_value=mock.Mock(value=1)), \
                mock.patch.object(nvdec, "open_nvdec_sessions", side_effect=RuntimeError("diagnostics failed")), \
                mock.patch.dict(os.environ, {nvdec.EXIT_REPORT_ENV: "1"}):
            self.assertTrue(nvdec.install_exit_guard())
        self.assertEqual(libc.on_exit.call_count, 2)  # _exit, then fcloseall
        self.assertEqual(nvdec.exit_guard_state(), {"armed": True, "installed": True, "error": None,
                                                    "open_at_exit": "RuntimeError('diagnostics failed')"})

    def test_counting_retries_a_concurrent_change(self):
        class Changing(weakref.WeakSet):
            failures = 2

            def __iter__(self):
                if Changing.failures:
                    Changing.failures -= 1
                    raise RuntimeError("Set changed size during iteration")
                return super().__iter__()

        with mock.patch.object(nvdec, "_SESSION_OWNERS", Changing()):
            self.assertEqual(nvdec.open_nvdec_sessions(), {"canvas_sessions": 0, "decoder_sessions": 0})
        self.assertEqual(Changing.failures, 0)

    def test_decoder_caps_arms_before_calling_the_library(self):
        calls = []

        class FakeCaps(dict):
            pass

        class FakeAPI:
            cudaVideoChromaFormat = SimpleNamespace(__members__={"444": "444", "420": "420"})
            cudaVideoCodec = SimpleNamespace(HEVC="hevc")

            @staticmethod
            def GetDecoderCaps(*args):
                calls.append(nvdec.exit_guard_state()["armed"])
                return FakeCaps(supported=True)

        with mock.patch.dict(nvdec._CAPS, clear=True):
            self.assertEqual(nvdec.decoder_caps(FakeAPI, 0, "yuv444p"), {"supported": True})
            self.assertEqual(nvdec.decoder_caps(FakeAPI, 0, "yuv444p"), {"supported": True})  # cached
        self.assertEqual(calls, [True])

    def test_nvdec_decoder_arms_before_its_first_library_call(self):
        decoder = nvdec.NVDECDecoder(gpu_id=0, gpu_uuid="GPU-00000000")

        class Stop(Exception):
            pass

        def caps(*args):
            self.assertTrue(nvdec.exit_guard_state()["armed"])
            raise Stop

        api = SimpleNamespace(GetDecoderCaps=caps, cudaVideoChromaFormat=SimpleNamespace(__members__={"444": 1, "420": 2}),
                              cudaVideoCodec=SimpleNamespace(HEVC="hevc"))
        with mock.patch.object(decoder, "_initialize"), self.assertRaises(Stop):
            decoder._api = api
            decoder._session({"image_width": 64, "image_height": 32, "pix_fmt": "yuv444p"})


# A process that decodes on real NVDEC sessions and exits with them still open (no close), like a harness ending
# with sys.exit and only the exit hooks: argv [mode] "pool" (DecoderPool, A8 closes its sessions at exit) or
# "session" (a bare CanvasSession never closed).
GPU_CHILD = r'''
import sys

import torch

from reelcache import nvdec
from reelcache.decoder_pool import DecoderPool, synthetic_unit

mode = sys.argv[1]
uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
unit = synthetic_unit(256, 160, "yuv444p")  # NVDEC HEVC decodes 144..8192 per side
if mode == "pool":
    pool = DecoderPool(gpu_id=0, gpu_uuid=uuid, workers=2)
    pool.prewarm_shapes([(256, 160, "yuv444p"), (320, 160, "yuv444p")], timeout=120)
    canvas = torch.zeros((1, 3, 160, 256), dtype=torch.uint8, device="cuda")
    pool.submit([unit], canvas).synchronize(120)
else:
    session = nvdec.CanvasSession(gpu_id=0, width=256, height=160, pix_fmt="yuv444p")
    session.warm(unit)
torch.cuda.synchronize()
print("DONE", nvdec.exit_guard_state()["armed"], flush=True)
sys.exit(0)
'''


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDANVDECExitTests(unittest.TestCase):
        """Real NVDEC (inside GPU jobs only; collected only with REELCACHE_STEP7_GPU_TESTS=1): a bounded exit."""

        EXIT_SECONDS = 60.0  # the decoder-pool exit hook waits at most EXIT_CLOSE_SECONDS=30 s per pool

        @classmethod
        def setUpClass(cls):
            import torch

            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            try:
                import PyNvVideoCodec  # noqa: F401
            except ImportError as exc:
                raise AssertionError("the NVDEC exit tests require PyNvVideoCodec") from exc

        def test_open_sessions_exit_within_the_bound(self):
            for mode in ("pool", "session"):
                with self.subTest(mode=mode):
                    returncode, stdout, stderr, _ = run_child(mode, script=GPU_CHILD, ready="DONE",
                                                              exit_timeout=self.EXIT_SECONDS)
                    self.assertEqual(returncode, 0, stderr[-3000:])
                    self.assertIn("DONE True", stdout)


if __name__ == "__main__":
    unittest.main()
