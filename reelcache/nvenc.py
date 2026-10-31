"""Explicit NVENC HEVC encoder for packed all-intra coefficient streams (opt-in, capability-qualified).

libx265 stays the default and the only software encoder. NVENC is used only when a caller constructs an
:class:`NVENCEncoder` (or :class:`NVENCProvider`) and requests NVENC specifications
(``CodecSpec.preset == "nvenc-<p1..p7>-<hq|ll|ull|lossless>"``, built by :func:`nvenc_spec`). It is never
a silent substitute in either direction: libx265 paths reject NVENC specifications and an NVENC encoder
rejects libx265 specifications (``reelcache.codecs._require_encodable``).

Construction probes the actual device: an NVENC HEVC session must open (A100 and H200 have no NVENC; the
driver returns ``NV_ENC_ERR_NO_ENCODE_DEVICE`` and construction raises ``UnsupportedError``), the requested
chroma formats must be supported, and, unless ``qualify=False``, a small deterministic canvas is encoded three
times (fresh, reused, fresh session) and checked on the real bitstream with FFmpeg ``trace_headers``: byte
determinism, only IDR access units that each carry VPS/SPS/PPS, only I slices, every slice at the configured
constant QP, the declared chroma format and picture size, and an FFmpeg software decode. Any failure raises;
nothing falls back. The caller's current CUDA context is restored around the probe and session creation (a
failed probe on H200 left another context current, which broke later NVDEC sessions).

Streams follow the packed v2 contract of ``reelcache.codecs.encode_packed`` (one self-contained IDR access unit
per picture, so tile byte ranges and NVDEC per-picture decoding work unchanged). NVENC prefixes each picture
with an HRD pic_timing SEI whose delays count pictures per session; these timing-only SEI NAL units are dropped
(SEI is not used to decode pixels), which makes the bytes independent of session history and lets sessions
persist (creating one costs 0.1-0.4 s). Packets record the encoder
identity in ``metadata["hevc"]`` (schema ``reelcache.hevc_encoder.nvenc.v1``: settings, the exact
PyNvVideoCodec options, GPU, compute capability, driver and NVENC API version), so decoding, fingerprints
and stored packets stay truthful; NVENC and libx265 packets of the same matrices are different packets.
HEVC decoding is normative: NVENC streams decode on FFmpeg or NVDEC (any GPU) like libx265 streams.
A nominal QP is not an equal-quality guarantee across encoders; measure distortion and native quality.
"""

from __future__ import annotations

import functools
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import _hevc
from . import accel_c as _accel_c
from .errors import ConfigurationError, IntegrityError, UnsupportedError
from .types import CodecSpec

ENCODER_SCHEMA = "reelcache.hevc_encoder.nvenc.v1"
ENCODER_NAME = "nvenc"
PRESETS = ("p1", "p2", "p3", "p4", "p5", "p6", "p7")
TUNINGS = MappingProxyType({
    "hq": "high_quality", "ll": "low_latency", "ull": "ultra_low_latency", "lossless": "lossless",
})
INPUT_FORMATS = MappingProxyType({"yuv444p": "YUV444", "yuv420p": "NV12"})
_PRESET_LABEL = re.compile(r"^nvenc-(p[1-7])-(hq|ll|ull|lossless)$")
_RECORD_FIELDS = {
    "schema", "encoder", "codec", "qp", "preset", "tuning_info", "rate_control", "inter_prediction",
    "picture_structure", "parameter_sets", "pix_fmt", "input_format", "options", "post_processing",
    "implementation",
}
# NVENC (through PyNvVideoCodec) prefixes every picture with an HRD pic_timing SEI whose delay fields count
# pictures per session. SEI is not needed to decode pixels; dropping HRD timing SEI (payload types 0 and 1)
# makes the bytes of a picture independent of session history, so streams are deterministic per content.
POST_PROCESSING = "drop_hrd_timing_sei"
_TIMING_SEI_PAYLOADS = frozenset((0, 1))  # buffering_period, pic_timing
_IMPLEMENTATION_FIELDS = {
    "library", "library_version", "nvenc_api_version", "gpu", "compute_capability", "driver_version",
}
_QUALIFICATION_CANVAS = (2, 160, 160)  # pictures, height, width: at least NVDEC's 144x144 minimum


def is_nvenc_preset(preset):
    return isinstance(preset, str) and preset.startswith("nvenc-")


@dataclass(frozen=True)
class NVENCSettings:
    """Bitstream-relevant NVENC settings: preset, tuning and the constant QP of every (I) slice."""

    preset: str = "p1"
    tuning: str = "hq"
    qp: int = 8

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ConfigurationError(f"NVENC preset must be one of {PRESETS}")
        if self.tuning not in TUNINGS:
            raise ConfigurationError(f"NVENC tuning must be one of {tuple(TUNINGS)}")
        if type(self.qp) is not int or not 0 <= self.qp <= 51:
            raise ConfigurationError("NVENC constant QP must be an integer in [0, 51]")
        if self.tuning == "lossless" and self.qp != 0:
            raise ConfigurationError("NVENC lossless tuning is declared with qp=0")

    @property
    def label(self):
        """The ``CodecSpec.preset`` text of this NVENC contract."""
        return f"nvenc-{self.preset}-{self.tuning}"

    @classmethod
    def from_label(cls, preset, qp):
        match = _PRESET_LABEL.match(preset) if isinstance(preset, str) else None
        if match is None:
            raise UnsupportedError(
                f"{preset!r} is not an NVENC preset label (nvenc-<p1..p7>-<{'|'.join(TUNINGS)}>)",
            )
        return cls(match.group(1), match.group(2), qp)

    def options(self, pix_fmt):
        """Exact PyNvVideoCodec ``CreateEncoder`` options: constant QP, one-picture GOPs, IDR every picture,
        no B pictures, parameter sets repeated on every IDR picture."""
        if pix_fmt not in INPUT_FORMATS:
            raise UnsupportedError(f"NVENC coefficient streams support {tuple(INPUT_FORMATS)}, not {pix_fmt!r}")
        return {
            "codec": "hevc", "preset": self.preset.upper(), "tuning_info": TUNINGS[self.tuning],
            "rc": "constqp", "constqp": str(self.qp), "gop": "1", "idrperiod": "1", "bf": "0",
            "repeatspspps": "1", "fps": "25",
        }


def nvenc_spec(rank, *, qp=8, preset="p1", tuning="hq", pix_fmt="yuv444p", normalization="per_channel",
               protected_fraction=1.0):
    """An all-intra svd_hevc specification that only an explicit NVENC encoder may encode."""
    settings = NVENCSettings(preset, tuning, qp)
    spec = CodecSpec(
        kind="svd_hevc", rank=rank, normalization=normalization, protected_fraction=protected_fraction,
        pix_fmt=pix_fmt, qp=qp, preset=settings.label, inter_prediction=False,
    )
    validate_spec_parameters(spec)
    return spec


def spec_settings(spec):
    """NVENC settings declared by a specification; libx265 specifications raise ``UnsupportedError``."""
    if not isinstance(spec, CodecSpec):
        raise ConfigurationError("spec must be a CodecSpec")
    return NVENCSettings.from_label(spec.preset, spec.qp)


def validate_spec_parameters(spec):
    """Spec-level NVENC contract (used by ``codecs._validate_spec`` for ``nvenc-*`` presets)."""
    settings = spec_settings(spec)
    if spec.kind != "svd_hevc":
        raise UnsupportedError("NVENC applies only to svd_hevc coefficient streams")
    if spec.inter_prediction is not False:
        raise UnsupportedError("the NVENC provider encodes all-intra streams only (inter_prediction=False)")
    if spec.pix_fmt not in INPUT_FORMATS:
        raise UnsupportedError(f"NVENC coefficient streams support {tuple(INPUT_FORMATS)}, not {spec.pix_fmt!r}")
    if spec.ffmpeg_path is not None:
        raise ConfigurationError("NVENC specifications do not carry an FFmpeg executable")
    return settings


def validate_pack_contract(codec, record):
    """Decode-side check of a packed stream that declares the NVENC encoder (``IntegrityError`` on mismatch)."""
    try:
        if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
            raise IntegrityError("NVENC encoder record fields are incomplete or unsupported")
        if (
            record["schema"] != ENCODER_SCHEMA or record["encoder"] != ENCODER_NAME or record["codec"] != "hevc"
            or record["rate_control"] != "constqp" or record["inter_prediction"] is not False
            or record["picture_structure"] != "all_intra_idr" or record["parameter_sets"] != "every_picture"
            or record["post_processing"] != POST_PROCESSING
        ):
            raise IntegrityError("packed stream does not declare the NVENC all-intra constant-QP contract")
        settings = NVENCSettings.from_label(f"nvenc-{record['preset']}-{record['tuning_info']}", record["qp"])
        pix_fmt = record["pix_fmt"]
        if (
            not isinstance(codec, Mapping) or codec.get("kind") != "svd_hevc" or codec.get("inter_prediction") is not False
            or codec.get("preset") != settings.label or codec.get("qp") != settings.qp
            or codec.get("pix_fmt") != pix_fmt
        ):
            raise IntegrityError("packed stream codec differs from its NVENC encoder record")
        options = record["options"]
        if (
            record["input_format"] != INPUT_FORMATS.get(pix_fmt) or not isinstance(options, Mapping)
            or dict(options) != settings.options(pix_fmt)
        ):
            raise IntegrityError("NVENC encoder record options differ from its declared settings")
        implementation = record["implementation"]
        if (
            not isinstance(implementation, Mapping) or set(implementation) != _IMPLEMENTATION_FIELDS
            or any(not isinstance(implementation[name], str) or not implementation[name]
                   for name in _IMPLEMENTATION_FIELDS - {"nvenc_api_version"})
            or type(implementation["nvenc_api_version"]) is not int or implementation["nvenc_api_version"] < 1
        ):
            raise IntegrityError("NVENC encoder record lacks its implementation identity")
    except (ConfigurationError, UnsupportedError, KeyError, TypeError, AttributeError) as exc:
        raise IntegrityError(f"invalid NVENC encoder record: {exc}") from exc
    return settings


@functools.lru_cache(maxsize=1)
def _default_api():
    try:
        import PyNvVideoCodec as api
    except (ImportError, OSError, RuntimeError) as exc:
        raise UnsupportedError(f"NVENC requires a working PyNvVideoCodec installation and NVIDIA driver: {exc}") from exc
    return api


def _api_version(api):
    """NVENC API version the driver supports (PyNvVideoCodec's VersionCheck; injected APIs declare it)."""
    version = getattr(api, "nvenc_api_version", None)
    if version is None:
        try:
            from PyNvVideoCodec.VersionCheck import DriverWrapper
            version = int(DriverWrapper().GetDriverVersion())
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedError(f"cannot determine the driver's NVENC API version: {exc}") from exc
    if type(version) is not int or version < 1:
        raise UnsupportedError("cannot determine the driver's NVENC API version")
    return version


def _device_identity(gpu_id):
    import torch

    if not torch.cuda.is_available() or not 0 <= gpu_id < torch.cuda.device_count():
        raise UnsupportedError(f"NVENC requires a visible CUDA device {gpu_id}")
    properties = torch.cuda.get_device_properties(gpu_id)
    driver = None
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        if isinstance(driver, bytes):
            driver = driver.decode()
    except Exception:  # noqa: BLE001 - nvidia-smi below
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        driver = result.stdout.strip() or None
    if not driver:
        raise UnsupportedError("cannot determine the NVIDIA driver version for the NVENC encoder record")
    return {"gpu": properties.name, "compute_capability": f"{properties.major}.{properties.minor}",
            "driver_version": str(driver)}


@functools.lru_cache(maxsize=1)
def _libcuda():
    import ctypes

    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None


class _ThreadContextGuard:
    """Restore the calling thread's current CUDA driver context around a PyNvVideoCodec call.

    Measured on H200 (no NVENC): a failed ``GetEncoderCaps`` probe leaves a new CUDA context current on the
    thread; a later NVDEC ``CreateDecoder`` given the framework's stream then fails with error 801 (stream does
    not belong to the current context), and CUDA runtime work would run in the wrong context.
    """

    def __enter__(self):
        import ctypes

        self.library, self.saved, self.changed = _libcuda(), None, False
        if self.library is not None:
            context = ctypes.c_void_p()
            if self.library.cuCtxGetCurrent(ctypes.byref(context)) == 0:
                self.saved = context
        return self

    def __exit__(self, *exc):
        import ctypes

        if self.saved is not None:
            context = ctypes.c_void_p()
            if self.library.cuCtxGetCurrent(ctypes.byref(context)) == 0 and context.value != self.saved.value:
                self.changed = True
                self.library.cuCtxSetCurrent(self.saved)
        return False


def _bitstream(value):
    """PyNvVideoCodec 2.2 returns a list of ``{'data': bytes, 'timestamp', 'picture_type'}`` per call."""
    if isinstance(value, (list, tuple)):
        return [piece for item in value for piece in _bitstream(item)]
    if isinstance(value, dict):
        return [bytes(value["data"])]
    return [bytes(value)]


def _rbsp(payload):
    """Remove H.265 emulation-prevention bytes (00 00 03 -> 00 00)."""
    if _accel_c.fast():  # accel C (gil_fastpaths): one bytes.replace, identical output
        return _accel_c.rbsp(payload)
    out, zeros = bytearray(), 0
    for byte in payload:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def sei_payload_types(nal):
    """Payload types of one SEI NAL unit (start code included), or None if it cannot be parsed."""
    start = 4 if nal.startswith(b"\x00\x00\x00\x01") else 3
    rbsp = _rbsp(nal[start + 2:])
    types, position = [], 0
    try:
        while position < len(rbsp) and not (position == len(rbsp) - 1 and rbsp[position] == 0x80):
            kind = 0
            while rbsp[position] == 0xFF:
                kind, position = kind + 255, position + 1
            kind, position = kind + rbsp[position], position + 1
            size = 0
            while rbsp[position] == 0xFF:
                size, position = size + 255, position + 1
            size, position = size + rbsp[position], position + 1
            types.append(kind)
            position += size
    except IndexError:
        return None
    return types if position == len(rbsp) - 1 else None


def drop_timing_sei(access_unit):
    """Drop SEI NAL units that carry only HRD timing messages; every other byte is kept unchanged."""
    kept, dropped = [], 0
    for offset, length, kind, _ in _hevc.nal_units(access_unit):
        nal = access_unit[offset:offset + length]
        if kind in (39, 40):
            types = sei_payload_types(nal)
            if types and set(types) <= _TIMING_SEI_PAYLOADS:
                dropped += 1
                continue
        kept.append(nal)
    return b"".join(kept), dropped


def verify_stream(stream, pictures):
    """Every picture one self-contained IDR access unit led by its VPS (the packed v2 contract)."""
    units = _hevc.all_intra_access_units(stream, pictures)
    if not all(unit["self_contained"] for unit in units):
        raise IntegrityError("NVENC stream has a picture that is not a self-contained IDR access unit")
    _hevc.picture_units(stream, 0, len(stream), pictures)
    _accel_c.remember_units(stream, pictures, units)  # accel C (gil_fastpaths): encode_node_pack reuses the parse
    return units


class NVENCEncoder:
    """One explicitly selected NVENC HEVC configuration on one GPU (see the module docstring).

    ``encode_pictures(pictures, pix_fmt)`` encodes canvases ``uint8[P, planes, H, W]`` (planes 3 for
    yuv444p, 1 for yuv420p with neutral chroma) into one all-intra stream and returns ``(stream, seconds,
    version)`` like ``_hevc.encode_pictures``. Sessions persist per geometry (creation costs 0.1-0.4 s; at
    most ``max_sessions``); HRD timing SEI is dropped from every access unit, so a picture's bytes do not
    depend on session history. ``sessions > 1`` splits the pictures over that many concurrent sessions and
    concatenates them in picture order; the qualification checks byte identity with one session.
    """

    name = ENCODER_NAME

    def __init__(self, *, qp=8, preset="p1", tuning="hq", gpu_id=0, pix_fmts=("yuv444p", "yuv420p"),
                 qualify=True, sessions=1, max_sessions=8, api=None, device=None, ffmpeg_path=None):
        self.settings = NVENCSettings(preset, tuning, qp)
        if type(gpu_id) is not int or gpu_id < 0:
            raise ConfigurationError("gpu_id must be a nonnegative integer")
        if type(sessions) is not int or not 1 <= sessions <= 16:
            raise ConfigurationError("sessions must be an integer in [1, 16]")
        if not isinstance(pix_fmts, (tuple, list)) or not pix_fmts or any(f not in INPUT_FORMATS for f in pix_fmts):
            raise UnsupportedError(f"NVENC coefficient streams support {tuple(INPUT_FORMATS)}")
        if type(qualify) is not bool:
            raise ConfigurationError("qualify must be boolean")
        if type(max_sessions) is not int or max_sessions < 1:
            raise ConfigurationError("max_sessions must be a positive integer")
        self.gpu_id, self.sessions, self.pix_fmts = gpu_id, sessions, tuple(pix_fmts)
        self.max_sessions = max_sessions
        self.ffmpeg_path = None if ffmpeg_path is None else _hevc.discover_ffmpeg(ffmpeg_path)
        self._api = _default_api() if api is None else api
        self.device = dict(_device_identity(gpu_id) if device is None else device)
        try:
            with _ThreadContextGuard() as guard:
                caps = self._api.GetEncoderCaps(gpuid=gpu_id, codec="hevc")
            self.caps = {name: int(caps[name]) for name in caps}
        except Exception as exc:  # noqa: BLE001 - every failure to open NVENC is an explicit unsupported device
            restored = " (the probe's leftover CUDA context was replaced by the caller's)" if guard.changed else ""
            raise UnsupportedError(
                f"NVENC HEVC is unavailable on GPU {gpu_id} ({self.device.get('gpu', 'unknown')}): "
                f"{str(exc).strip()[:400]}; use the libx265 HEVCProvider (CPU encoding) on this device{restored}",
            ) from exc
        for name in ("width_min", "width_max", "height_min", "height_max", "mb_num_max"):
            if self.caps.get(name, 0) < 1:
                raise UnsupportedError(f"NVENC HEVC capability {name} is not reported on this device")
        if "yuv444p" in self.pix_fmts and self.caps.get("support_yuv444_encode") != 1:
            raise UnsupportedError("this NVENC device cannot encode HEVC 4:4:4")
        if self.settings.tuning == "lossless" and self.caps.get("support_lossless_encode") != 1:
            raise UnsupportedError("this NVENC device cannot encode lossless HEVC")
        self.implementation = {
            "library": "PyNvVideoCodec", "library_version": str(getattr(self._api, "__version__", "unknown")),
            "nvenc_api_version": _api_version(self._api), "gpu": str(self.device["gpu"]),
            "compute_capability": str(self.device["compute_capability"]),
            "driver_version": str(self.device["driver_version"]),
        }
        self.version = (
            f"NVENC HEVC via PyNvVideoCodec {self.implementation['library_version']} "
            f"(NVENC API {self.implementation['nvenc_api_version']}, driver {self.implementation['driver_version']}, "
            f"{self.implementation['gpu']})"
        )
        self.stats = {"streams": 0, "pictures": 0, "bytes": 0, "seconds": 0.0, "session_seconds": 0.0,
                      "verify_seconds": 0.0, "dropped_timing_sei": 0, "last": None}
        self._lock = threading.Lock()
        self._sessions = {}
        self.qualification = self._qualify() if qualify else None

    # ---- contract
    def validate_spec(self, spec):
        """Encode-side: the specification must request exactly this encoder's settings and a supported format."""
        settings = spec_settings(spec)
        validate_spec_parameters(spec)
        if settings != self.settings:
            raise ConfigurationError(
                f"specification requests {settings.label} at QP {settings.qp}; this encoder is "
                f"{self.settings.label} at QP {self.settings.qp}",
            )
        if spec.pix_fmt not in self.pix_fmts:
            raise UnsupportedError(f"this NVENC encoder was qualified for {self.pix_fmts}, not {spec.pix_fmt}")
        return settings

    def spec(self, rank, *, pix_fmt="yuv444p", normalization="per_channel", protected_fraction=1.0):
        return nvenc_spec(rank, qp=self.settings.qp, preset=self.settings.preset, tuning=self.settings.tuning,
                          pix_fmt=pix_fmt, normalization=normalization, protected_fraction=protected_fraction)

    def encoder_record(self, pix_fmt):
        if pix_fmt not in self.pix_fmts:
            raise UnsupportedError(f"this NVENC encoder was qualified for {self.pix_fmts}, not {pix_fmt}")
        return {
            "schema": ENCODER_SCHEMA, "encoder": ENCODER_NAME, "codec": "hevc", "qp": self.settings.qp,
            "preset": self.settings.preset, "tuning_info": self.settings.tuning, "rate_control": "constqp",
            "inter_prediction": False, "picture_structure": "all_intra_idr", "parameter_sets": "every_picture",
            "pix_fmt": pix_fmt, "input_format": INPUT_FORMATS[pix_fmt], "options": self.settings.options(pix_fmt),
            "post_processing": POST_PROCESSING, "implementation": dict(self.implementation),
        }

    def check_geometry(self, width, height, pix_fmt):
        caps = self.caps
        if pix_fmt not in self.pix_fmts:
            raise UnsupportedError(f"this NVENC encoder was qualified for {self.pix_fmts}, not {pix_fmt}")
        if (
            not caps["width_min"] <= width <= caps["width_max"] or not caps["height_min"] <= height <= caps["height_max"]
            or ((width + 15) // 16) * ((height + 15) // 16) > caps["mb_num_max"]
            or (pix_fmt == "yuv420p" and (width % 2 or height % 2))
        ):
            raise UnsupportedError(
                f"NVENC cannot encode a {width}x{height} {pix_fmt} picture on this device "
                f"(supported {caps['width_min']}..{caps['width_max']} x {caps['height_min']}..{caps['height_max']}, "
                f"at most {caps['mb_num_max']} macroblocks)",
            )

    # ---- encoding
    def _frames(self, pictures, pix_fmt):
        if pix_fmt == "yuv444p":
            return [np.ascontiguousarray(picture).reshape(-1) for picture in pictures]
        height, width = pictures.shape[2], pictures.shape[3]
        chroma = np.full((height // 2, width), 128, dtype=np.uint8)
        return [np.concatenate((picture[0], chroma), axis=0).reshape(-1) for picture in pictures]

    def _session(self, width, height, pix_fmt, slot):
        """A persistent NVENC session per geometry, format and concurrency slot (creation costs 0.1-0.4 s)."""
        key = (width, height, pix_fmt, slot)
        with self._lock:
            session = self._sessions.pop(key, None)
        created = 0.0
        if session is None:
            started = time.perf_counter()
            with _ThreadContextGuard():
                session = self._api.CreateEncoder(width, height, INPUT_FORMATS[pix_fmt], True,
                                                  **self.settings.options(pix_fmt))
            created = time.perf_counter() - started
        return key, session, created

    def _release(self, key, session):
        with self._lock:
            self._sessions[key] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.pop(next(iter(self._sessions)))

    def _encode_range(self, frames, width, height, pix_fmt, slot=0):
        key, session, created = self._session(width, height, pix_fmt, slot)
        pieces = []
        try:
            for frame in frames:
                pieces.extend(_bitstream(session.Encode(frame)))
            pieces.extend(_bitstream(session.EndEncode()))
        except Exception:
            del session  # a failed session is never reused
            raise
        self._release(key, session)
        if len(pieces) != len(frames):
            raise IntegrityError(f"NVENC returned {len(pieces)} access units for {len(frames)} pictures")
        dropped, units = 0, []
        for piece in pieces:
            unit, count = drop_timing_sei(piece)
            units.append(unit)
            dropped += count
        return b"".join(units), created, dropped

    def close(self):
        """Release the persistent NVENC sessions."""
        with self._lock:
            self._sessions.clear()

    # ---- accel C (C1d, ``nvenc_device_input``): NVENC reads the canvas from device memory
    def _device_range(self, pictures, width, height, pix_fmt):
        """The pictures of a CUDA canvas through a device-input session (PyNvVideoCodec ``usecpuinputbuffer=False``:
        each frame is an object whose ``cuda()`` returns one CUDA array per plane, ``[H, W, 1]``)."""
        key = (width, height, pix_fmt, "device")
        with self._lock:
            session = self._sessions.pop(key, None)
        created = 0.0
        if session is None:
            started = time.perf_counter()
            with _ThreadContextGuard():
                session = self._api.CreateEncoder(width, height, INPUT_FORMATS[pix_fmt], False,
                                                  **self.settings.options(pix_fmt))
            created = time.perf_counter() - started
        pieces = []
        try:
            for index in range(pictures.shape[0]):
                pieces.extend(_bitstream(session.Encode(_DeviceFrame(pictures[index]))))
            pieces.extend(_bitstream(session.EndEncode()))
        except Exception:
            del session  # a failed session is never reused
            raise
        self._release(key, session)
        if len(pieces) != pictures.shape[0]:
            raise IntegrityError(f"NVENC returned {len(pieces)} access units for {pictures.shape[0]} pictures")
        dropped, units = 0, []
        for piece in pieces:
            unit, count = drop_timing_sei(piece)
            units.append(unit)
            dropped += count
        return b"".join(units), created, dropped

    def _qualify_device_input(self, device):
        """Once per encoder: the qualification canvas through host and device input must give identical bytes."""
        import torch

        if getattr(self, "_device_qualified", None) is not None:
            if self._device_qualified is not True:
                raise UnsupportedError(f"NVENC device input failed its qualification: {self._device_qualified}")
            return
        count, height, width = _QUALIFICATION_CANVAS
        rows = np.arange(height, dtype=np.int32)[:, None]
        columns = np.arange(width, dtype=np.int32)[None, :]
        noise = np.random.default_rng(15).integers(0, 48, size=(count, 3, height, width), dtype=np.int32)
        base = np.stack([(rows * 3 + columns * (plane + 1) * 2) % 200 for plane in range(3)])[None]
        canvas = np.clip(base + noise, 0, 255).astype(np.uint8)
        host, _, _ = self._encode_range(self._frames(canvas, "yuv444p"), width, height, "yuv444p")
        on_device = torch.from_numpy(canvas).to(device)
        torch.cuda.synchronize(device)
        stream, _, _ = self._device_range(on_device, width, height, "yuv444p")
        self._device_qualified = True if stream == host else "device-input bytes differ from host input"
        if self._device_qualified is not True:
            raise UnsupportedError(f"NVENC device input failed its qualification: {self._device_qualified}")

    def encode_device_pictures(self, pictures, pix_fmt):
        """:meth:`encode_pictures` of a CUDA uint8 canvas ``[P, 3, H, W]`` (yuv444p) read from device memory; the
        same bytes (qualified once per encoder against host input). The canvas must be complete (host-synchronized)."""
        import torch

        if pix_fmt != "yuv444p":
            raise UnsupportedError("NVENC device input is qualified for yuv444p canvases only")
        if (not isinstance(pictures, torch.Tensor) or pictures.dtype != torch.uint8 or pictures.ndim != 4
                or pictures.device.type != "cuda" or pictures.shape[0] < 1 or pictures.shape[1] != 3
                or not pictures.is_contiguous()):
            raise ConfigurationError("device canvas pictures must be a contiguous CUDA uint8 [P, 3, H, W] tensor")
        count, _, height, width = pictures.shape
        self.check_geometry(width, height, pix_fmt)
        self._qualify_device_input(pictures.device)
        started = time.perf_counter()
        stream, created, dropped = self._device_range(pictures, width, height, pix_fmt)
        seconds = time.perf_counter() - started
        if not stream:
            raise UnsupportedError("NVENC produced an empty HEVC stream")
        checked = time.perf_counter()
        verify_stream(stream, count)
        verify_seconds = time.perf_counter() - checked
        with self._lock:
            self.stats["streams"] += 1
            self.stats["pictures"] += count
            self.stats["bytes"] += len(stream)
            self.stats["seconds"] += seconds
            self.stats["session_seconds"] += created
            self.stats["verify_seconds"] += verify_seconds
            self.stats["dropped_timing_sei"] += dropped
            self.stats["device_input_streams"] = self.stats.get("device_input_streams", 0) + 1
            self.stats["last"] = {"seconds": seconds, "session_seconds": created, "verify_seconds": verify_seconds,
                                  "pictures": count, "bytes": len(stream), "sessions": 1,
                                  "dropped_timing_sei": dropped, "device_input": True}
        return stream, seconds, self.version

    def encode_pictures(self, pictures, pix_fmt):
        if pix_fmt not in INPUT_FORMATS:
            raise UnsupportedError(f"NVENC coefficient streams support {tuple(INPUT_FORMATS)}, not {pix_fmt!r}")
        planes = 3 if pix_fmt == "yuv444p" else 1
        if (
            not isinstance(pictures, np.ndarray) or pictures.dtype != np.uint8 or pictures.ndim != 4
            or pictures.shape[0] < 1 or pictures.shape[1] != planes
        ):
            raise ConfigurationError("canvas pictures differ from the HEVC plane geometry")
        count, _, height, width = pictures.shape
        self.check_geometry(width, height, pix_fmt)
        frames = self._frames(pictures, pix_fmt)
        started = time.perf_counter()
        sessions = min(self.sessions, count)
        if sessions == 1:
            stream, created, dropped = self._encode_range(frames, width, height, pix_fmt)
        else:
            bounds = np.linspace(0, count, sessions + 1).round().astype(int)
            results, failures = [None] * sessions, []

            def work(index):
                try:
                    results[index] = self._encode_range(frames[bounds[index]:bounds[index + 1]], width, height,
                                                        pix_fmt, index)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    failures.append(exc)

            threads = [threading.Thread(target=work, args=(index,)) for index in range(sessions)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if failures or any(result is None for result in results):
                raise IntegrityError(f"an NVENC session failed to produce its pictures: {failures[:1]}")
            stream = b"".join(result[0] for result in results)
            created = max(result[1] for result in results)
            dropped = sum(result[2] for result in results)
        seconds = time.perf_counter() - started
        if not stream:
            raise UnsupportedError("NVENC produced an empty HEVC stream")
        checked = time.perf_counter()
        verify_stream(stream, count)
        verify_seconds = time.perf_counter() - checked
        with self._lock:
            self.stats["streams"] += 1
            self.stats["pictures"] += count
            self.stats["bytes"] += len(stream)
            self.stats["seconds"] += seconds
            self.stats["session_seconds"] += created
            self.stats["verify_seconds"] += verify_seconds
            self.stats["dropped_timing_sei"] += dropped
            self.stats["last"] = {"seconds": seconds, "session_seconds": created, "verify_seconds": verify_seconds,
                                  "pictures": count, "bytes": len(stream), "sessions": sessions,
                                  "dropped_timing_sei": dropped}
        return stream, seconds, self.version

    # ---- qualification
    def _qualify(self):
        """Encode a deterministic canvas twice per format and check the real bitstream (see module docstring)."""
        count, height, width = _QUALIFICATION_CANVAS
        rows = np.arange(height, dtype=np.int32)[:, None]
        columns = np.arange(width, dtype=np.int32)[None, :]
        noise = np.random.default_rng(15).integers(0, 48, size=(count, 3, height, width), dtype=np.int32)
        base = np.stack([(rows * 3 + columns * (plane + 1) * 2) % 200 for plane in range(3)])[None]
        canvas = np.clip(base + noise, 0, 255).astype(np.uint8)
        report = {"canvas": [count, height, width], "formats": {}}
        for pix_fmt in self.pix_fmts:
            planes = 3 if pix_fmt == "yuv444p" else 1
            pictures = np.ascontiguousarray(canvas[:, :planes])
            first, _, _ = self.encode_pictures(pictures, pix_fmt)  # fresh session
            second, _, _ = self.encode_pictures(pictures, pix_fmt)  # the same, reused session
            self.close()
            third, _, _ = self.encode_pictures(pictures, pix_fmt)  # another fresh session
            if not first == second == third:
                raise UnsupportedError(
                    f"NVENC {pix_fmt} encoding is not byte-deterministic across fresh and reused sessions on this device",
                )
            trace = _hevc.trace_headers(first, ffmpeg_path=self.ffmpeg_path)
            expected_qps = None if self.settings.tuning == "lossless" else [self.settings.qp]
            chroma = 3 if pix_fmt == "yuv444p" else 1
            idr = all(picture["nal_unit_type"] in (19, 20) for picture in trace["pictures"])
            if (
                len(trace["pictures"]) != count or trace["slice_type_counts"]["P"] or trace["slice_type_counts"]["B"]
                or not idr or trace["chroma_format_idc"] != [chroma] or trace["picture_sizes"] != [(width, height)]
                or (expected_qps is not None and trace["slice_qps"] != expected_qps)
            ):
                raise UnsupportedError(
                    f"NVENC {pix_fmt} stream violates the declared all-intra constant-QP contract: "
                    f"slices {trace['slice_type_counts']}, QPs {trace['slice_qps']}, chroma {trace['chroma_format_idc']}, "
                    f"sizes {trace['picture_sizes']}, IDR-only {idr}",
                )
            decoded, _ = _hevc.decode_pictures(first, pictures=count, height=height, width=width, pix_fmt=pix_fmt,
                                               ffmpeg_path=self.ffmpeg_path)
            error = np.abs(decoded[:, :planes].astype(np.int16) - pictures.astype(np.int16))
            if self.settings.tuning == "lossless" and error.any():
                raise UnsupportedError(f"NVENC lossless {pix_fmt} stream is not lossless on this device")
            report["formats"][pix_fmt] = {
                "bytes": len(first), "slice_qps": trace["slice_qps"], "general_profile_idc": trace["general_profile_idc"],
                "chroma_format_idc": trace["chroma_format_idc"], "deterministic": True,
                "deterministic_across_session_reuse": True, "max_abs_pixel_error": int(error.max()),
                "dropped_timing_sei_per_picture": self.stats["last"]["dropped_timing_sei"] / count,
            }
        if self.sessions > 1:
            pictures = np.ascontiguousarray(np.concatenate([canvas, canvas[::-1]])[:, :3 if "yuv444p" in self.pix_fmts
                                                                                     else 1])
            pix_fmt = "yuv444p" if "yuv444p" in self.pix_fmts else "yuv420p"
            sessions, self.sessions = self.sessions, 1
            try:
                single, _, _ = self.encode_pictures(pictures, pix_fmt)
            finally:
                self.sessions = sessions
            split, _, _ = self.encode_pictures(pictures, pix_fmt)
            if split != single:
                raise UnsupportedError("NVENC multi-session streams differ from one session; use sessions=1")
            report["multi_session_identical"] = True
        return report


class _DeviceFrame:
    """A planar YUV444 picture ``[3, H, W]`` on the device as PyNvVideoCodec's frame convention (accel C1d)."""

    def __init__(self, picture):
        self._planes = [picture[plane].unsqueeze(-1) for plane in range(picture.shape[0])]

    def cuda(self):
        return self._planes


@dataclass(frozen=True)
class NVENCProvider:
    """Packed-node encoder provider using an explicit :class:`NVENCEncoder`; decoding is the ordinary path.

    ``encode_node`` packs every svd_hevc matrix of a node (``packing="node"``) or each layer group
    (``"layer_group"``) into NVENC all-intra streams; RAW/SVD matrices keep their single-matrix packets.
    Single-matrix NVENC packets are not provided (``encode_tensor`` raises for svd_hevc). Decoding
    delegates to ``reelcache.codecs`` (FFmpeg software or an explicit ``NVDECDecoder``).
    """

    encoder: NVENCEncoder
    ffmpeg_path: str | None = None
    nvdec: object = None
    fitter: object = None
    verify_decoded_coefficients: bool = False
    verify_restored_packets: bool = False
    validate_hevc_encode: bool = False
    packing: str = "node"
    layer_group_size: int | None = None

    def __post_init__(self):
        from . import codecs

        if not isinstance(self.encoder, NVENCEncoder):
            raise ConfigurationError("NVENCProvider requires an explicitly constructed NVENCEncoder")
        if self.packing not in ("node", "layer_group"):
            raise ConfigurationError("NVENCProvider packs node or layer_group streams only")
        codecs._validate_packing(self.packing, self.layer_group_size, "temporal")
        # The HEVCProvider performs the same option checks for the decode-side fields.
        codecs.HEVCProvider(
            ffmpeg_path=self.ffmpeg_path, nvdec=self.nvdec, fitter=self.fitter,
            verify_decoded_coefficients=self.verify_decoded_coefficients,
            verify_restored_packets=self.verify_restored_packets, validate_hevc_encode=self.validate_hevc_encode,
        )
        if self.ffmpeg_path is not None:
            object.__setattr__(self, "ffmpeg_path", _hevc.discover_ffmpeg(self.ffmpeg_path))

    def _decoder(self):
        from . import codecs

        return codecs.HEVCProvider(
            ffmpeg_path=self.ffmpeg_path, nvdec=self.nvdec,
            verify_decoded_coefficients=self.verify_decoded_coefficients,
            verify_restored_packets=self.verify_restored_packets,
        )

    def spec(self, rank, **kwargs):
        return self.encoder.spec(rank, **kwargs)

    def encode_node(self, matrices, contexts, specs, *, node_id=None, reuse_bases=None, reuse_threshold=None,
                    prepared=None):
        from . import codecs

        decoder = self._decoder()
        with decoder._scope():
            return codecs.encode_node(
                matrices, contexts, specs, packing=self.packing, layer_group_size=self.layer_group_size,
                layout="temporal", node_id=node_id, reuse_bases=reuse_bases, reuse_threshold=reuse_threshold,
                prepared=prepared, fitter=self.fitter, validate_hevc=self.validate_hevc_encode, hevc_threads=1,
                encoder=self.encoder,
            )

    def encode_tensor(self, tensor, context, spec, **kwargs):
        from . import codecs

        if isinstance(spec, CodecSpec) and spec.kind == "svd_hevc":
            raise UnsupportedError("NVENCProvider encodes svd_hevc matrices only as packed node streams (encode_node)")
        return codecs.encode_tensor(tensor, context, spec, fitter=self.fitter, **kwargs)

    def decode_tensor(self, encoded, **kwargs):
        return self._decoder().decode_tensor(encoded, **kwargs)

    def decode_column_views(self, encoded, **kwargs):
        return self._decoder().decode_column_views(encoded, **kwargs)

    def decode_packed(self, encoded, **kwargs):
        return self._decoder().decode_packed(encoded, **kwargs)
