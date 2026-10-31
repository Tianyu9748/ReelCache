"""Accel study track C (dev_accel_20261030): bytes-identical archival host-footprint levers (all opt-in).

Every lever is off by default; with :data:`OFF` (the default scope) no code here runs and every caller takes the base
path. An :class:`AccelOptions` value is installed per archival thread by :meth:`ArchivePipeline._archival_thread`
(:func:`enter_thread`), so the levers act only on the archive pipeline's fitter and encode workers.

Levers (class (i): stored bytes, decisions and decoded-coefficient digests identical to the flag-off path):

* ``validation_decoder="nvdec"`` (C1a): the validation decode of a node pack runs on NVDEC (the bridge's
  :class:`~reelcache.decoder_pool.DecoderPool`, or thread-owned canvas sessions without a pool) instead of FFmpeg
  software decoding. HEVC decoding is normative, so the decoded uint8 pictures equal FFmpeg's; the unpacking,
  dequantization and SHA-256 that follow are unchanged. Batches yield to restores through the pipeline's FFmpeg pump
  pause hook (the restore yield the FFmpeg path uses).
* ``hash_once`` (C1b): a payload digest computed by ``codecs._descriptor`` is reused where the very same immutable
  ``bytes`` object is hashed again in this process (``detach_pack_bases``, the store's content address, seal-time
  basis adoption). Bytes read from files are always hashed.
* ``seal_prepare_after_publish`` (C1c): seal-time preparation (``restorer.adopt``) runs after the node's publication
  instead of before it (it only warms restore caches).
* ``nvenc_device_input`` (C1d): NVENC reads the finalized canvas from device memory (no D2H + host frames); see
  :mod:`reelcache.nvenc`.
* ``gil_fastpaths`` (C2): GIL-light equivalents of hot host functions (NAL/access-unit/SEI parsing, unmosaic,
  validated JSON copies, decoded-coefficient digests without a copy, the exact-BLAS handle lookup).
* ``native_ext`` (C3): the Rust extension ``reelcache._native.reelcache_native`` for the remaining holders (outputs
  identical to the Python implementations; calls release the GIL). Falls back to the Python paths when missing.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import math
import threading
import time
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from .errors import ConfigurationError, IntegrityError

VALIDATION_DECODERS = ("ffmpeg", "nvdec")
#: Field names of the opt-in levers (``ReelCacheRuntimeConfig`` carries the same names).
FIELDS = ("validation_decoder", "hash_once", "seal_prepare_after_publish", "nvenc_device_input", "gil_fastpaths",
          "native_ext")


@dataclass(frozen=True)
class AccelOptions:
    validation_decoder: str = "ffmpeg"
    hash_once: bool = False
    seal_prepare_after_publish: bool = False
    nvenc_device_input: bool = False
    gil_fastpaths: bool = False
    native_ext: bool = False

    def __post_init__(self) -> None:
        if self.validation_decoder not in VALIDATION_DECODERS:
            raise ConfigurationError(f"validation_decoder must be one of {VALIDATION_DECODERS}")
        for name in FIELDS[1:]:
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be a bool")

    @property
    def active(self) -> bool:
        return self != OFF

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FIELDS}

    @classmethod
    def from_config(cls, config: Any) -> "AccelOptions":
        return cls(**{name: getattr(config, name, getattr(OFF, name)) for name in FIELDS})


OFF = AccelOptions()
_SCOPE: contextvars.ContextVar[AccelOptions] = contextvars.ContextVar("reelcache_accel_c", default=OFF)


def current() -> AccelOptions:
    return _SCOPE.get()


def enter_thread(options: AccelOptions | None, *, pool: Any = None) -> None:
    """Install ``options`` (and the NVDEC ``pool`` of validation decodes) for the calling archival thread."""
    _SCOPE.set(OFF if options is None else options)
    _LOCAL.pool = pool


def validation_pool() -> Any:
    """The decoder pool installed for this thread's validation decodes (None: thread-owned canvas sessions)."""
    return getattr(_LOCAL, "pool", None)


@contextlib.contextmanager
def scope(options: AccelOptions | None):
    token = _SCOPE.set(OFF if options is None else options)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def fast() -> bool:
    """C2 GIL fast paths enabled on this thread."""
    return _SCOPE.get().gil_fastpaths


def host_paths() -> bool:
    """C2 fast paths or the C3 extension enabled on this thread."""
    options = _SCOPE.get()
    return options.gil_fastpaths or options.native_ext


# ------------------------------------------------------------------------------------------------------------------
# Statistics (records only)
_STATS_LOCK = threading.Lock()
STATS: dict[str, Any] = {"nvdec_validations": 0, "nvdec_validation_seconds": 0.0, "nvdec_validation_pictures": 0,
                         "nvdec_pool_jobs": 0, "nvdec_thread_sessions": 0, "nvdec_yield_calls": 0,
                         "hash_once_hits": 0, "hash_once_bytes": 0, "seal_after_publish": 0,
                         "seal_after_publish_seconds": 0.0, "seal_after_publish_failures": 0,
                         "native_calls": 0, "nvenc_device_frames": 0}


def _count(name: str, value: float = 1) -> None:
    with _STATS_LOCK:
        STATS[name] = STATS.get(name, 0) + value


def stats() -> dict[str, Any]:
    with _STATS_LOCK:
        return dict(STATS)


# ------------------------------------------------------------------------------------------------------------------
# C1a: NVDEC validation decode
_LOCAL = threading.local()
#: Device bytes of one decode batch (pictures per pool job: at least 4, at most 32).
BATCH_BYTES = 32 << 20


def _batch(picture_bytes: int) -> int:
    return max(4, min(32, BATCH_BYTES // max(1, picture_bytes)))


def _private_stream(device):
    import torch

    streams = getattr(_LOCAL, "streams", None)
    if streams is None:
        streams = _LOCAL.streams = {}
    stream = streams.get(device.index)
    if stream is None:
        with torch.cuda.device(device):
            stream = streams[device.index] = torch.cuda.Stream(device=device)
    return stream


def _thread_session(gpu_id: int, width: int, height: int, pix_fmt: str):
    """A canvas session owned by the calling thread (no pool): one per geometry, at most 4 per thread."""
    from .nvdec import CanvasSession

    sessions = getattr(_LOCAL, "sessions", None)
    if sessions is None:
        sessions = _LOCAL.sessions = {}
    key = (gpu_id, width, height, pix_fmt)
    session = sessions.pop(key, None)
    if session is None:
        session = CanvasSession(gpu_id=gpu_id, width=width, height=height, pix_fmt=pix_fmt)
        _count("nvdec_thread_sessions")
    sessions[key] = session
    while len(sessions) > 4:
        sessions.pop(next(iter(sessions))).close()
    return session


def nvdec_validation_decode(stream: bytes, units, *, pictures: int, height: int, width: int, pix_fmt: str,
                            pool=None, gpu_id: int | None = None) -> tuple[np.ndarray, float]:
    """NVDEC decode of a whole all-intra canvas stream: ``([pictures, planes, height, width] uint8, seconds)``.

    ``units`` are the stream's access units (``_hevc.all_intra_access_units``). Equal to
    ``node_packets._validation_decode`` (FFmpeg) picture for picture: HEVC decoding is normative. Decodes in batches
    through ``pool`` (the bridge's decoder pool; each batch is one job ordered by an event on this thread's private
    stream) or, without a pool, on a canvas session owned by this thread; before each batch the pipeline's FFmpeg pump
    pause hook runs (archival yields to restores in flight), then the batch is copied to host memory.
    """
    import torch

    from . import _hevc
    from .nvdec import _canvas_planes, normal_tensors

    started = time.perf_counter()
    planes = _canvas_planes(pix_fmt)
    if len(units) != pictures:
        raise IntegrityError("HEVC stream picture count differs from its declared pictures")
    jobs = [(stream, unit["offset"], unit["length"], 1) for unit in units]
    if pool is not None:
        device = pool.device
    else:
        if gpu_id is None:
            if not torch.cuda.is_available():
                raise ConfigurationError("an NVDEC validation decode needs a decoder pool or a CUDA device")
            gpu_id = torch.cuda.current_device()
        device = torch.device("cuda", gpu_id)
    out = np.empty((pictures, planes, height, width), dtype=np.uint8)
    host = torch.from_numpy(out)
    batch = _batch(planes * height * width)
    private = _private_stream(device)
    pause = _hevc._PUMP_PAUSE.get()
    session = None if pool is not None else _thread_session(device.index, width, height, pix_fmt)
    with normal_tensors(), torch.cuda.device(device), torch.cuda.stream(private):
        buffer = torch.empty((min(batch, pictures), planes, height, width), dtype=torch.uint8, device=device)
        for first in range(0, pictures, batch):
            if pause is not None:
                pause()
                _count("nvdec_yield_calls")
            chunk = jobs[first:first + batch]
            count = len(chunk)
            if pool is not None:
                ready = torch.cuda.Event()
                ready.record(private)
                ticket = pool.submit(chunk, buffer, 0, pix_fmt=pix_fmt, after=[ready])
                ticket.wait(private)
                _count("nvdec_pool_jobs")
            else:
                private.wait_event(session.decode(chunk, buffer, 0))
            host[first:first + count].copy_(buffer[:count])  # on the private stream; returns when copied
    seconds = time.perf_counter() - started
    with _STATS_LOCK:
        STATS["nvdec_validations"] += 1
        STATS["nvdec_validation_seconds"] += seconds
        STATS["nvdec_validation_pictures"] += pictures
    return out, seconds


# ------------------------------------------------------------------------------------------------------------------
# C1d: NVENC device input
def device_input(canvas: Any) -> bool:
    """Keep a finalized canvas on the device for NVENC: the lever is on (this thread), the canvas is a CUDA yuv444p
    canvas of an NVENC specification."""
    if not _SCOPE.get().nvenc_device_input:
        return False
    pictures = getattr(canvas, "device_pictures", None)
    codec = getattr(canvas, "codec", None) or {}
    preset = codec.get("preset")
    if pictures is None or getattr(pictures, "device", None) is None or pictures.device.type != "cuda":
        return False
    if codec.get("pix_fmt") != "yuv444p" or not (isinstance(preset, str) and preset.startswith("nvenc-")):
        return False
    _count("nvenc_device_frames", int(pictures.shape[0]))
    return True


# ------------------------------------------------------------------------------------------------------------------
# C1b: hash once
_DIGESTS = "_reelcache_accel_c_digests"


def record_digests(encoded: Any, payloads: dict, descriptors: dict) -> None:
    """Remember, on the encoded pack, the digest ``codecs._descriptor`` computed for each payload object."""
    if not _SCOPE.get().hash_once:
        return
    record = {name: (payloads[name], descriptors[name]["sha256"]) for name in payloads}
    object.__setattr__(encoded, _DIGESTS, record)


def known_digest(encoded: Any, name: str, raw: bytes) -> str | None:
    """The recorded SHA-256 hex of payload ``name`` when ``raw`` is the very object hashed at encode (else None)."""
    record = getattr(encoded, _DIGESTS, None)
    if record is None or not _SCOPE.get().hash_once:
        return None
    entry = record.get(name)
    if entry is None or entry[0] is not raw:
        return None
    with _STATS_LOCK:
        STATS["hash_once_hits"] += 1
        STATS["hash_once_bytes"] += len(raw)
    return entry[1]


@contextlib.contextmanager
def known_payload_digests(encoded: Any):
    """Within the block, ``ObjectStore._put`` reuses the recorded digest of ``encoded``'s payload objects."""
    record = getattr(encoded, _DIGESTS, None)
    if record is None or not _SCOPE.get().hash_once:
        yield
        return
    previous = getattr(_LOCAL, "known", None)
    _LOCAL.known = {id(raw): (raw, digest) for raw, digest in record.values()}
    try:
        yield
    finally:
        _LOCAL.known = previous


def store_digest(data: bytes) -> str | None:
    """The digest of ``data`` when it is a payload object of the active :func:`known_payload_digests` block."""
    known = getattr(_LOCAL, "known", None)
    if not known:
        return None
    entry = known.get(id(data))
    if entry is None or entry[0] is not data:
        return None
    with _STATS_LOCK:
        STATS["hash_once_hits"] += 1
        STATS["hash_once_bytes"] += len(data)
    return entry[1]


# ------------------------------------------------------------------------------------------------------------------
# C2: GIL fast paths (outputs identical to the base functions; property-tested)
_START = b"\x00\x00\x01"


def _memmem_finder(needle: bytes):
    """glibc ``memmem`` through ctypes (releases the GIL during each search), or a ``bytes.find`` fallback."""
    import ctypes
    import ctypes.util

    try:
        name = ctypes.util.find_library("c")
        function = ctypes.CDLL(name).memmem if name else None
    except (OSError, AttributeError):
        function = None
    if function is None:
        return lambda data, position, end: data.find(needle, position, end)
    function.restype = ctypes.c_void_p
    function.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
    buffer = ctypes.create_string_buffer(needle, len(needle))

    def find(data, position, end):
        if end - position < len(needle):
            return -1
        base = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
        address = function(base + position, end - position, buffer, len(needle))
        return -1 if not address else address - base

    return find


_find_start = _memmem_finder(_START)


def start_code_positions(stream: bytes) -> list[int]:
    """Every offset of ``00 00 01`` in ``stream`` (the matches of ``_hevc._START_CODE.finditer``)."""
    positions, position, end = [], 0, len(stream)
    while True:
        found = _find_start(stream, position, end)
        if found < 0:
            return positions
        positions.append(found)
        position = found + 3


def nal_units(stream):
    """``_hevc.nal_units`` with a GIL-free start-code search: identical units, identical errors (C3: the whole scan
    and its checks in Rust)."""
    if not isinstance(stream, bytes) or not stream:
        raise IntegrityError("HEVC stream must be nonempty bytes")
    native = _native() if _SCOPE.get().native_ext else None
    if native is not None:
        _count("native_calls")
        try:
            return native.nal_units(stream)
        except ValueError as exc:
            raise IntegrityError(str(exc)) from None
    starts = []
    for position in start_code_positions(stream):
        header = position + 3
        if position and stream[position - 1] == 0:
            position -= 1
        starts.append((position, header))
    if not starts or starts[0][0] != 0:
        raise IntegrityError("HEVC stream does not begin with an Annex-B start code")
    units = []
    total = len(stream)
    for index, (start, header) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else total
        if header + 2 > end:
            raise IntegrityError("truncated HEVC NAL unit header")
        first, second = stream[header], stream[header + 1]
        if first & 0x80 or not second & 0x07:
            raise IntegrityError("invalid HEVC NAL unit header")
        nal_type = (first >> 1) & 0x3F
        first_slice = None
        if nal_type < 32:
            if header + 2 >= end:
                raise IntegrityError("truncated HEVC slice segment")
            first_slice = bool(stream[header + 2] & 0x80)
        units.append((start, end - start, nal_type, first_slice))
    return units


def rbsp(payload: bytes) -> bytes:
    """``nvenc._rbsp``: drop emulation-prevention bytes. A ``03`` is removed exactly when the two input bytes before
    it are ``00 00``; such patterns never overlap, so one left-to-right ``replace`` removes exactly those bytes."""
    return bytes(payload).replace(b"\x00\x00\x03", b"\x00\x00")


def unmosaic(images, layout, segment, values) -> None:
    """``_hevc._unmosaic`` as one strided copy (index map ``(row, column, plane)`` -> channel)."""
    slots = layout["channels_per_segment"]
    first = segment * slots
    count = min(slots, layout["rank"] - first)
    columns, rows = layout["grid"]
    planes, height, width = layout["planes"], layout["height"], layout["width"]
    frames = images.shape[0]
    tiled = images[:, :planes, :rows * height, :columns * width].reshape(
        frames, planes, rows, height, columns, width).transpose(0, 3, 5, 2, 4, 1)
    if count == slots:
        target = values[..., first:first + count].reshape(frames, height, width, rows, columns, planes)
        if np.may_share_memory(target, values):  # a view (splitting the contiguous channel axis): write through it
            target[...] = tiled
            return
    values[..., first:first + count] = tiled.reshape(frames, height, width, slots)[..., :count]


def raw_frame_bytes(images, pix_fmt) -> bytes:
    """``_hevc._raw_frame_bytes`` written into one preallocated buffer."""
    if pix_fmt == "yuv444p":
        return images.tobytes()
    frames, _, height, width = images.shape
    out = np.empty((frames, height * width + height * width // 2), dtype=np.uint8)
    out[:, :height * width] = images[:, 0].reshape(frames, -1)
    out[:, height * width:] = 128
    return out.tobytes()


def coefficient_digest(coefficients: np.ndarray) -> str:
    """SHA-256 hex of ``coefficients.astype("<f4", copy=False).tobytes()`` without the copy."""
    values = coefficients.astype("<f4", copy=False)
    if not values.flags.c_contiguous:
        return hashlib.sha256(values.tobytes()).hexdigest()
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


class _Slow(Exception):
    pass


_MAX_INT_BITS = 14000  # well inside int <-> str conversion limits (``json.dumps`` of a larger int raises)


def _copy(item):
    kind = type(item)
    if kind is str or kind is bool or item is None:
        return item
    if kind is dict:
        out = {}
        for key, child in item.items():
            if type(key) is not str:
                raise _Slow
            out[key] = _copy(child)
        return out
    if kind is list or kind is tuple:
        return [_copy(child) for child in item]
    if kind is int:
        if item.bit_length() > _MAX_INT_BITS:
            raise _Slow
        return item
    if kind is float:
        if not math.isfinite(item):
            raise _Slow
        return item
    raise _Slow


def json_copy(value):
    """``types.json_copy`` of plain JSON (exact dict / list / tuple / str / int / bool / None / finite float values) as
    one recursive copy, or None when ``value`` needs the base path (other types, invalid values, deep nesting): the
    base then returns or raises exactly as before. Equal results: JSON round trips keep these values unchanged."""
    try:
        return (_copy(value),)
    except (_Slow, RecursionError):
        return None


def thaw_json(value):
    """``types.thaw_json`` with exact-type checks first (dict and MappingProxyType -> dict, list and tuple -> list)."""
    kind = type(value)
    if kind is dict or kind is MappingProxyType:
        return {key: thaw_json(item) for key, item in value.items()}
    if kind is list or kind is tuple:
        return [thaw_json(item) for item in value]
    if kind is str or kind is int or kind is float or kind is bool or value is None:
        return value
    if isinstance(value, Mapping):  # the base's ABC checks for anything else
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


# ------------------------------------------------------------------------------------------------------------------
# C2: per-stream parse memo (the NVENC encoder's verify_stream units reused by encode_node_pack)
def remember_units(stream: bytes, pictures: int, units: list) -> None:
    if _SCOPE.get().gil_fastpaths:
        _LOCAL.units = (stream, pictures, units)


def cached_units(stream: bytes, pictures: int) -> list | None:
    if not _SCOPE.get().gil_fastpaths:
        return None
    cached = getattr(_LOCAL, "units", None)
    _LOCAL.units = None
    if cached is None or cached[0] is not stream or cached[1] != pictures:
        return None
    return cached[2]


# ------------------------------------------------------------------------------------------------------------------
# C3: the Rust extension
_NATIVE: list = []


def _native():
    """``reelcache._native.reelcache_native`` or None (missing / failed to load: the Python paths run)."""
    if not _NATIVE:
        try:
            from ._native import reelcache_native as module  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - optional
            module = None
        _NATIVE.append(module)
    return _NATIVE[0]


def native_available() -> bool:
    return _native() is not None


def native_unpack_tile(images, tile, first_picture, video):
    """``codecs._unpack_tile`` in Rust (GIL released) for a C-contiguous uint8 canvas; None when not applicable (the
    caller then runs the Python path, with its own errors)."""
    native = _native() if _SCOPE.get().native_ext else None
    if (native is None or not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4
            or not images.flags.c_contiguous or type(first_picture) is not int or first_picture < 0):
        return None
    try:
        x, y, width, height = (int(value) for value in tile["rect"])
        frames, rows_h, cols_w, rank = (int(video[name]) for name in ("frames", "height", "width", "rank"))
        columns, rows = (int(value) for value in video["grid"])
        planes, segments = int(video["planes"]), int(video["segments"])
    except (KeyError, TypeError, ValueError):
        return None
    if (min(x, y) < 0 or columns * cols_w > width or rows * rows_h > height
            or int(video["channels_per_segment"]) != columns * rows * planes):
        return None
    values = np.empty((frames, rows_h, cols_w, rank), dtype=np.uint8)
    try:
        native.unpack_tile(images, values, tuple(int(n) for n in images.shape), first_picture, x, y, frames, rows_h,
                           cols_w, rank, columns, rows, planes, segments)
    except ValueError:
        return None
    _count("native_calls")
    return values.reshape(-1, rank)
