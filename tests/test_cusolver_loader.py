import ctypes
import os
import tempfile
import unittest

import torch

from reelcache import cusolver_eigh
from reelcache.errors import UnsupportedError


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass
    return path


class _FakeLib:
    def __init__(self, symbols):
        for symbol in symbols:
            setattr(self, symbol, object())


class CusolverCandidateTests(unittest.TestCase):
    def test_mapped_libraries_parse_proc_maps_once_and_skip_mg(self):
        maps = "\n".join([
            "7f00-7f01 r-xp 00000000 00:2a 11 /env/nvidia/cu13/lib/libcusolver.so.12",
            "7f01-7f02 r--p 00010000 00:2a 11 /env/nvidia/cu13/lib/libcusolver.so.12",
            "7f02-7f03 r-xp 00000000 00:2a 12 /env/nvidia/cu13/lib/libcusolverMg.so.12",
            "7f03-7f04 r-xp 00000000 00:2a 13 /env/nvidia/cu13/lib/libcublas.so.13",
            "7f04-7f05 rw-p 00000000 00:00 0 ",
            "7f05-7f06 rw-p 00000000 00:00 0 [heap]",
        ])
        self.assertEqual(cusolver_eigh._mapped_libraries(maps), ["/env/nvidia/cu13/lib/libcusolver.so.12"])

    def test_cuda13_order_is_mapped_package_cuda_home_then_sonames(self):
        with tempfile.TemporaryDirectory() as root:
            mapped = _touch(os.path.join(root, "torch_loaded", "libcusolver.so.12"))
            versioned = _touch(os.path.join(root, "pkg", "cu13", "lib", "libcusolver.so.12.0.4.66"))
            soname = _touch(os.path.join(root, "pkg", "cu13", "lib", "libcusolver.so.12"))
            home12 = _touch(os.path.join(root, "cuda", "lib64", "libcusolver.so.12"))
            home11 = _touch(os.path.join(root, "cuda", "lib64", "libcusolver.so.11"))
            _touch(os.path.join(root, "cuda", "lib64", "libcusolverMg.so.12"))
            candidates = cusolver_eigh._search_candidates(
                mapped=[mapped], package_dirs=[os.path.join(root, "pkg", "cu13", "lib")],
                environ={"CUDA_HOME": os.path.join(root, "cuda")}, cuda_version="13.0",
            )
        self.assertEqual(candidates, [
            (mapped, "mapped_by_process"), (soname, "python_package"), (versioned, "python_package"),
            (home12, "cuda_home"), (home11, "cuda_home"),
            ("libcusolver.so.12", "loader_soname"), ("libcusolver.so.11", "loader_soname"),
            ("libcusolver.so", "loader_soname"),
        ])

    def test_cuda12_prefers_libcusolver_so_11(self):
        with tempfile.TemporaryDirectory() as root:
            old = _touch(os.path.join(root, "lib", "libcusolver.so.11"))
            new = _touch(os.path.join(root, "lib", "libcusolver.so.12"))
            candidates = cusolver_eigh._search_candidates(
                mapped=[], package_dirs=[os.path.join(root, "lib")], environ={}, cuda_version="12.8",
            )
        self.assertEqual([path for path, _ in candidates][:3], [old, new, "libcusolver.so.11"])

    def test_same_file_reached_twice_is_listed_once(self):
        with tempfile.TemporaryDirectory() as root:
            real = _touch(os.path.join(root, "pkg", "libcusolver.so.12"))
            os.makedirs(os.path.join(root, "cuda", "lib64"))
            os.symlink(real, os.path.join(root, "cuda", "lib64", "libcusolver.so.12"))
            candidates = cusolver_eigh._search_candidates(
                mapped=[real], package_dirs=[os.path.join(root, "pkg")],
                environ={"CUDA_HOME": os.path.join(root, "cuda")}, cuda_version="13.2",
            )
        paths = [path for path, source in candidates if source != "loader_soname"]
        self.assertEqual(paths, [real])


class CusolverBindingTests(unittest.TestCase):
    def test_bind_skips_unloadable_and_symbolless_libraries(self):
        def loader(candidate):
            if candidate == "missing":
                raise OSError("cannot open shared object file")
            if candidate == "old":
                return _FakeLib(["cusolverDnCreate"])
            return _FakeLib(cusolver_eigh._SYMBOLS)

        with tempfile.TemporaryDirectory() as root:
            good = _touch(os.path.join(root, "libcusolver.so.12"))
            expected = os.path.realpath(good)
            lib, info = cusolver_eigh._bind(
                [("missing", "mapped_by_process"), ("old", "python_package"), (good, "cuda_home")], loader=loader,
            )
        self.assertTrue(hasattr(lib, "cusolverDnXsyevBatched"))
        self.assertEqual(info["candidate"], good)
        self.assertEqual(info["source"], "cuda_home")
        self.assertEqual(info["path"], expected)
        self.assertIn("load failed", info["attempts"][0]["rejected"])
        self.assertIn("missing symbols", info["attempts"][1]["rejected"])
        self.assertEqual(info["attempts"][2]["bound"], expected)

    def test_failure_names_every_candidate_and_reason(self):
        def loader(candidate):
            if candidate == "libcusolver.so.12":
                raise OSError("not found")
            return _FakeLib([])

        with self.assertRaises(UnsupportedError) as caught:
            cusolver_eigh._bind([("/opt/old/libcusolver.so.11", "cuda_home"),
                                 ("libcusolver.so.12", "loader_soname")], loader=loader)
        message = str(caught.exception)
        self.assertIn("/opt/old/libcusolver.so.11 [cuda_home]: missing symbols", message)
        self.assertIn("libcusolver.so.12 [loader_soname]: load failed: not found", message)

    def test_installed_torch_cuda_wheel_binds_the_matching_soname(self):
        saved = (cusolver_eigh._lib, cusolver_eigh._lib_info)
        cusolver_eigh._lib, cusolver_eigh._lib_info = None, None
        try:
            expected = {12: "libcusolver.so.11", 13: "libcusolver.so.12"}
            cuda = torch.version.cuda
            if cuda is None:
                self.assertIsInstance(cusolver_eigh.available(), bool)
                return
            info = cusolver_eigh.binding_info()
            soname = expected[int(cuda.split(".")[0])]
            self.assertTrue(os.path.basename(info["path"]).startswith(soname), info)
            self.assertIn(info["source"], {"mapped_by_process", "python_package", "cuda_home", "loader_soname"})
            loaded = ctypes.CDLL(info["path"])
            self.assertTrue(all(hasattr(loaded, symbol) for symbol in cusolver_eigh._SYMBOLS))
            # A handle injected directly is still described by the file that defines its symbols.
            cusolver_eigh._lib, cusolver_eigh._lib_info = loaded, None
            preset = cusolver_eigh.binding_info()
            self.assertEqual((preset["source"], preset["path"]), ("preset_handle", info["path"]))
        finally:
            cusolver_eigh._lib, cusolver_eigh._lib_info = saved


if __name__ == "__main__":
    unittest.main()
