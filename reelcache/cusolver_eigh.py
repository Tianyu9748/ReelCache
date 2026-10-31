"""Batched symmetric eigensolver through cuSOLVER ``cusolverDnXsyevBatched`` (CUDA 12.6+).

Torch up to 2.8 loops ``syevd`` over a batch of large matrices, which is
launch/latency bound for 1536x1536 Grams; Torch 2.13 sends batched
``torch.linalg.eigh`` to ``XsyevBatched`` itself, so this binding is the
explicit, verifiable path rather than the only fast one. ``XsyevBatched``
solves the whole batch in one call.

The library is loaded lazily with ``ctypes``. Candidates, in order:

1. a libcusolver already mapped into this process (the one Torch loaded),
2. Python CUDA wheels: the CUDA 12 layout ``nvidia/cusolver/lib``
   (``libcusolver.so.11``), the CUDA 13 layout ``nvidia/cu13/lib``
   (``libcusolver.so.12``) and Torch's own ``lib`` directory,
3. ``$CUDA_HOME`` / ``$CUDA_PATH`` (``lib64``, ``lib``, ``targets/*/lib``),
4. the dynamic loader's default search by soname.

Only a library exporting ``cusolverDnXsyevBatched`` and its buffer-size query
is bound; :func:`library_path` reports the file actually bound. When no
candidate qualifies, :class:`UnsupportedError` lists every candidate tried and
why it was rejected.
"""

import ctypes
import glob
import importlib.util
import os
import threading

import torch

from .errors import ConfigurationError, UnsupportedError

_CUDA_R_32F = 0
_EIG_MODE_VECTOR = 1
_FILL_LOWER = 0
_SYMBOLS = ("cusolverDnXsyevBatched", "cusolverDnXsyevBatched_bufferSize")
_SONAMES = ("libcusolver.so.12", "libcusolver.so.11", "libcusolver.so")
_lock = threading.Lock()
_lib = None
_lib_info = None
_handles: dict[tuple[int, int], tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}


def _is_cusolver_file(path):
    name = os.path.basename(path)
    return name.startswith("libcusolver.so")


def _mapped_libraries(maps_text=None):
    """libcusolver files already mapped into this process, in mapping order."""
    if maps_text is None:
        try:
            with open("/proc/self/maps") as handle:
                maps_text = handle.read()
        except OSError:
            return []
    seen, found = set(), []
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        path = parts[5].strip()
        if path.startswith("/") and _is_cusolver_file(path) and path not in seen:
            seen.add(path)
            found.append(path)
    return found


def _preferred_major(cuda_version=None):
    """cuSOLVER soname major matching the CUDA major Torch was built with."""
    version = torch.version.cuda if cuda_version is None else cuda_version
    try:
        cuda_major = int(str(version).split(".")[0])
    except (TypeError, ValueError):
        return None
    return {12: 11, 13: 12}.get(cuda_major)


def _rank_files(paths, preferred_major):
    """Exact sonames first, the preferred major first, versioned files last."""
    def key(path):
        name = os.path.basename(path)
        major = name[len("libcusolver.so."):].split(".")[0] if name.startswith("libcusolver.so.") else ""
        return (major != str(preferred_major), name.count("."), name)
    return sorted(paths, key=key)


def _package_library_dirs():
    """Library directories of the Python CUDA wheels and of Torch itself."""
    dirs = []
    spec = importlib.util.find_spec("nvidia")
    for base in (spec.submodule_search_locations or []) if spec else []:
        dirs.append(os.path.join(base, "cusolver", "lib"))
        dirs.extend(sorted(glob.glob(os.path.join(base, "cu[0-9]*", "lib"))))
    dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    return dirs


def _cuda_home_dirs(environ):
    dirs = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = environ.get(variable)
        if root:
            dirs += [os.path.join(root, "lib64"), os.path.join(root, "lib")]
            dirs += sorted(glob.glob(os.path.join(root, "targets", "*", "lib")))
    return dirs


def _search_candidates(*, mapped=None, package_dirs=None, environ=None, cuda_version=None):
    """Ordered ``(candidate, source)`` pairs; paths are de-duplicated by real path."""
    major = _preferred_major(cuda_version)
    environ = os.environ if environ is None else environ
    mapped = _mapped_libraries() if mapped is None else mapped
    package_dirs = _package_library_dirs() if package_dirs is None else package_dirs
    ordered = [(path, "mapped_by_process") for path in _rank_files(mapped, major)]
    for source, dirs in (("python_package", package_dirs), ("cuda_home", _cuda_home_dirs(environ))):
        for directory in dirs:
            files = glob.glob(os.path.join(directory, "libcusolver.so*"))
            ordered += [(path, source) for path in _rank_files(files, major)]
    seen, unique = set(), []
    for path, source in ordered:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append((path, source))
    sonames = sorted(_SONAMES, key=lambda name: name != f"libcusolver.so.{major}")
    return unique + [(name, "loader_soname") for name in sonames]


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]


def _symbol_file(function):
    """Real path of the shared object that defines ``function`` (dladdr), or None."""
    try:
        address = ctypes.cast(function, ctypes.c_void_p)
    except (TypeError, ctypes.ArgumentError):
        return None
    for libc_name in (None, "libc.so.6", "libdl.so.2"):
        try:
            dladdr = ctypes.CDLL(libc_name).dladdr
        except (OSError, AttributeError):
            continue
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        dladdr.restype = ctypes.c_int
        info = _DlInfo()
        if dladdr(address, ctypes.byref(info)) and info.dli_fname:
            return os.path.realpath(info.dli_fname.decode())
        return None
    return None


def _bind(candidates, loader=ctypes.CDLL):
    """Load the first candidate exporting ``XsyevBatched``; return ``(lib, info)``."""
    attempts = []
    for candidate, source in candidates:
        try:
            lib = loader(candidate)
        except OSError as exc:
            attempts.append({"candidate": candidate, "source": source, "rejected": f"load failed: {exc}"})
            continue
        missing = [symbol for symbol in _SYMBOLS if not hasattr(lib, symbol)]
        if missing:
            attempts.append({"candidate": candidate, "source": source,
                             "rejected": f"missing symbols {', '.join(missing)}"})
            continue
        path = _symbol_file(getattr(lib, _SYMBOLS[0])) or (os.path.realpath(candidate) if os.sep in candidate else None)
        attempts.append({"candidate": candidate, "source": source, "bound": path})
        return lib, {"path": path, "candidate": candidate, "source": source, "attempts": attempts}
    tried = "; ".join(f"{a['candidate']} [{a['source']}]: {a['rejected']}" for a in attempts) or "no candidates"
    raise UnsupportedError(
        "cuSOLVER with cusolverDnXsyevBatched (CUDA 12.6+) is unavailable; tried " + tried
        + ". Install the CUDA wheel Torch depends on (nvidia-cusolver) or set CUDA_HOME to a CUDA 12.6+ toolkit."
    )


def _library():
    global _lib, _lib_info
    with _lock:
        if _lib is not None:
            return _lib
        lib, info = _bind(_search_candidates())
        _lib, _lib_info = lib, {**info, "library": lib}
        return _lib


def available() -> bool:
    try:
        _library()
        return True
    except UnsupportedError:
        return False


def library_path() -> str | None:
    """Real path of the bound libcusolver (binds on first use; raises if none qualifies)."""
    return binding_info()["path"]


def binding_info() -> dict:
    """Bound file, how it was found, and every candidate tried before it."""
    lib = _library()
    with _lock:
        if _lib_info is None or _lib_info.get("library") is not lib:
            # The handle was injected directly (e.g. by a test harness); describe it by symbol address.
            info = {"path": _symbol_file(getattr(lib, _SYMBOLS[0])), "candidate": None,
                    "source": "preset_handle", "attempts": []}
        else:
            info = {key: value for key, value in _lib_info.items() if key != "library"}
        return dict(info)


def _handle(device: torch.device):
    lib = _library()
    key = (device.index, threading.get_ident())
    with _lock:
        if key not in _handles:
            handle, params = ctypes.c_void_p(), ctypes.c_void_p()
            with torch.cuda.device(device):
                if lib.cusolverDnCreate(ctypes.byref(handle)) != 0:
                    raise UnsupportedError("cusolverDnCreate failed")
                if lib.cusolverDnCreateParams(ctypes.byref(params)) != 0:
                    raise UnsupportedError("cusolverDnCreateParams failed")
            _handles[key] = (handle, params)
        return lib, *_handles[key]


@torch.no_grad()
def eigh_batched(grams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ascending eigenvalues ``[b, n]`` and row eigenvectors ``[b, n, n]``.

    ``vectors[i, j]`` is the unit eigenvector for ``values[i, j]`` (rows, not
    columns: the column-major cuSOLVER output read in row-major order). The
    input must be a contiguous FP32 CUDA batch of symmetric matrices; it is not
    modified.
    """
    values, vectors, info = eigh_batched_unchecked(grams)
    if bool((info != 0).any()):
        raise ConfigurationError("batched eigensolver did not converge")
    return values, vectors


@torch.no_grad()
def eigh_batched_unchecked(grams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`eigh_batched` without the host read of the convergence flags: ``(values, vectors, info)``.

    ``info`` is the device ``int32 [b]`` cuSOLVER status per matrix (nonzero: not converged); the caller must
    check it before using the results (e.g. once for many calls).
    """
    if (
        not isinstance(grams, torch.Tensor) or grams.device.type != "cuda" or grams.dtype != torch.float32
        or grams.ndim != 3 or grams.shape[1] != grams.shape[2] or grams.shape[0] < 1 or grams.shape[1] < 1
    ):
        raise ConfigurationError("batched eigh requires a nonempty [b, n, n] FP32 CUDA tensor")
    lib, handle, params = _handle(grams.device)
    batch, n = grams.shape[0], grams.shape[1]
    vectors = grams.contiguous().clone()
    values = torch.empty(batch, n, dtype=torch.float32, device=grams.device)
    info = torch.zeros(batch, dtype=torch.int32, device=grams.device)
    stream = torch.cuda.current_stream(grams.device).cuda_stream
    device_bytes, host_bytes = ctypes.c_size_t(), ctypes.c_size_t()
    ptr = ctypes.c_void_p
    with torch.cuda.device(grams.device):
        if lib.cusolverDnSetStream(handle, ptr(stream)) != 0:
            raise UnsupportedError("cusolverDnSetStream failed")
        status = lib.cusolverDnXsyevBatched_bufferSize(
            handle, params, _EIG_MODE_VECTOR, _FILL_LOWER, ctypes.c_int64(n), _CUDA_R_32F,
            ptr(vectors.data_ptr()), ctypes.c_int64(n), _CUDA_R_32F, ptr(values.data_ptr()), _CUDA_R_32F,
            ctypes.byref(device_bytes), ctypes.byref(host_bytes), ctypes.c_int64(batch),
        )
        if status != 0:
            raise UnsupportedError(f"cusolverDnXsyevBatched_bufferSize failed with status {status}")
        workspace = torch.empty(max(1, device_bytes.value), dtype=torch.uint8, device=grams.device)
        host = (ctypes.c_uint8 * max(1, host_bytes.value))()
        status = lib.cusolverDnXsyevBatched(
            handle, params, _EIG_MODE_VECTOR, _FILL_LOWER, ctypes.c_int64(n), _CUDA_R_32F,
            ptr(vectors.data_ptr()), ctypes.c_int64(n), _CUDA_R_32F, ptr(values.data_ptr()), _CUDA_R_32F,
            ptr(workspace.data_ptr()), device_bytes, host, host_bytes, ptr(info.data_ptr()),
            ctypes.c_int64(batch),
        )
        if status != 0:
            raise UnsupportedError(f"cusolverDnXsyevBatched failed with status {status}")
    del workspace
    return values, vectors, info


eigh_batched.unchecked = eigh_batched_unchecked
