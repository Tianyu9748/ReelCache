"""Small, file-free software HEVC transport for coefficient-channel tiles.

The channel/plane order follows ``kv_compression/vcodec.py``. The covering
affine and remaining-column range rules are adapted from
``kv_compression/harness/hybrid_hevc_kv_codec.py`` (source SHA256
1a7d34658b36158395368429148379b1ea60e5e0ddac2f7088cc9a42516885cb).
No historical module, model import, study directory, or temporary-path mutation
is used. FFmpeg communicates exclusively through pipes.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import accel_c as _accel_c
from .errors import ConfigurationError, IntegrityError, UnsupportedError


PIXEL_FORMATS = ("yuv444p", "yuv420p")
MAX_DIMENSION = 8192
_BINARY = contextvars.ContextVar("reelcache_ffmpeg_binary", default=None)
_TIMEOUT_SECONDS = 300
# Size-scaled FFmpeg deadlines (Step 7): a fixed 300 s deadline was within 4-7% of single-thread LongCat node
# encodes on Zen 2 (266-288 s, exceeded once with 16 encodes in flight). Encodes (and the node-pack
# validation decode) may run for ``_TIMEOUT_SECONDS`` or this many seconds per million picture samples,
# whichever is longer: 1.0 s per megasample is 5x the slowest measured libx265 rate (0.198 s/Msample, one
# Zen 2 thread, CausVid node), so a deadline only stops a hung process. Thread counts never lower it.
# Software decodes (``decode_pixels``/``decode_pictures``, as the validation decode) use the decode rate below
# (``decode_timeout``; configurable per call and through REELCACHE_FFMPEG_DECODE_SECONDS_PER_MEGASAMPLE).
_ENCODE_SECONDS_PER_MEGASAMPLE = 1.0
_DECODE_SECONDS_PER_MEGASAMPLE = 0.25


def size_scaled_timeout(samples, *, seconds_per_megasample=_ENCODE_SECONDS_PER_MEGASAMPLE):
    """FFmpeg deadline for ``samples`` picture samples: max(300 s, samples x seconds per megasample)."""
    if type(samples) is not int or samples < 0:
        raise ConfigurationError("picture sample count must be a nonnegative integer")
    return max(float(_TIMEOUT_SECONDS), samples * seconds_per_megasample / 1e6)


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive finite number of seconds")
    return float(value)


def decode_timeout(samples, timeout=None):
    """Deadline (seconds) of one FFmpeg software decode producing ``samples`` picture samples.

    An explicit ``timeout`` is used as given. Otherwise the deadline is size-scaled like the encode side:
    max(300 s, samples x seconds per megasample) at ``REELCACHE_FFMPEG_DECODE_SECONDS_PER_MEGASAMPLE`` (a positive
    number, read at each call) or 0.25 s per megasample, the rate of the node-pack validation decode. Samples count
    canvas picture samples: pictures x (3 for yuv444p, else 1) x height x width.
    """
    if timeout is not None:
        return _positive_seconds(timeout, "an FFmpeg decode timeout")
    rate = _DECODE_SECONDS_PER_MEGASAMPLE
    configured = os.environ.get("REELCACHE_FFMPEG_DECODE_SECONDS_PER_MEGASAMPLE")
    if configured is not None:
        try:
            rate = float(configured)
        except ValueError:
            rate = math.nan
        if not math.isfinite(rate) or rate <= 0:
            raise ConfigurationError("REELCACHE_FFMPEG_DECODE_SECONDS_PER_MEGASAMPLE must be a positive number of seconds")
    return size_scaled_timeout(samples, seconds_per_megasample=rate)

# Version-2 encoder contract. Besides preset=ultrafast and qp=8 (unchanged; the
# ultrafast CTU32/min-CU16 adaptive partitioning stays enabled), every v2 stream
# pins these x265 parameters instead of inheriting preset defaults:
# - ipratio=1/pbratio=1 code every slice at exactly QP8. Version-1 streams used
#   x265's 1.4/1.3 defaults (I slices QP5, P QP8, B QP10).
# - info=0 omits x265's version/options SEI (about 2 KB per stream, thread
#   dependent); the packet metadata records the encoder contract instead.
ENCODER_SCHEMA = "reelcache.hevc_encoder.v2"
_COMMON_X265 = (("ipratio", 1), ("pbratio", 1), ("info", 0))
# inter_prediction=False: every picture is an IDR picture with only I slices.
_ALL_INTRA_GOP = (
    ("keyint", 1), ("min-keyint", 1), ("bframes", 0), ("b-adapt", 0),
    ("scenecut", 0), ("open-gop", 0),
)
# inter_prediction=True: the GOP that version-1 packets inherited from the
# ultrafast preset, now explicit (M10 inter settings): one closed GOP per stream
# (keyint=250 exceeds every chunk packet's picture count), an IDR first picture,
# then a fixed (b-adapt=0) pyramid of up to three B pictures per P picture with
# one reference and no scene-cut insertion.
_INTER_GOP = (
    ("keyint", 250), ("min-keyint", 25), ("bframes", 3), ("b-adapt", 0),
    ("b-pyramid", 1), ("ref", 1), ("scenecut", 0), ("open-gop", 0),
)
LEGACY_ENCODER = {
    "schema": "reelcache.hevc_encoder.v1_implicit", "encoder": "libx265",
    "preset": "ultrafast", "qp": 8, "inter_prediction": None,
    "picture_structure": "x265_ultrafast_preset_default",
    "x265_params": {},
    "observed_semantics": (
        "x265 3.5 ultrafast defaults: keyint=250, bframes=3, b-adapt=0, scenecut=0, "
        "ipratio=1.4/pbratio=1.3 (I QP5, P QP8, B QP10), x265 info SEI"
    ),
}
CANVAS_ALIGNMENT = 32  # x265 ultrafast CTU size; packed canvases hold whole CTUs
# NVDEC's smallest HEVC picture (GetDecoderCaps width_min/height_min, measured on SM80 and SM120);
# packed canvases are at least this size rounded up to whole CTUs so every v2 pack decodes on NVDEC.
NVDEC_MIN_DIMENSION = 144
MIN_CANVAS = -(-NVDEC_MIN_DIMENSION // CANVAS_ALIGNMENT) * CANVAS_ALIGNMENT
_VPS_START = b"\x00\x00\x01\x40\x01"  # start code + VPS NAL header (nuh_layer_id 0, TemporalId 0)
_IRAP_TYPES = frozenset(range(16, 22))  # BLA_W_LP..CRA_NUT: only I slices (H.265 7.4.7.1)
_IDR_TYPES = frozenset((19, 20))
_PARAMETER_SET_TYPES = frozenset((32, 33, 34))
_AU_PREFIX_TYPES = frozenset((32, 33, 34, 35, 39, *range(41, 45), *range(48, 56)))
_START_CODE = re.compile(b"\x00\x00\x01")
_TRACE_FIELD = re.compile(r"^\s*\d+\s+([A-Za-z_][A-Za-z0-9_\[\]]*)\s+[01]+\s+=\s+(-?\d+)\s*$")
_SLICE_TYPES = {0: "B", 1: "P", 2: "I"}


def encoder_record(inter_prediction):
    """The exact, portable x265 contract of a version-2 coefficient stream."""
    if type(inter_prediction) is not bool:
        raise ConfigurationError(
            "version-2 HEVC streams require an explicit boolean inter_prediction; "
            "legacy version-1 specifications are decode-only",
        )
    gop = _INTER_GOP if inter_prediction else _ALL_INTRA_GOP
    return {
        "schema": ENCODER_SCHEMA, "encoder": "libx265", "preset": "ultrafast", "qp": 8,
        "inter_prediction": inter_prediction,
        "picture_structure": "closed_gop_fixed_b3" if inter_prediction else "all_intra_idr",
        "x265_params": dict(_COMMON_X265 + gop),
    }


def x265_parameter_string(inter_prediction, threads, frames):
    """Complete ``-x265-params`` value; thread settings never change pixels."""
    record = encoder_record(inter_prediction)
    pinned = ":".join(f"{name}={value}" for name, value in record["x265_params"].items())
    return f"qp=8:{pinned}:log-level=error:{x265_thread_params(threads, frames)}"


@contextlib.contextmanager
def using_ffmpeg(path):
    """Select an explicit executable in this context, including during reload."""
    resolved = discover_ffmpeg(path)
    token = _BINARY.set(resolved)
    try:
        yield resolved
    finally:
        _BINARY.reset(token)


def discover_ffmpeg(path=None) -> str:
    explicit = path if path is not None else _BINARY.get()
    if explicit is None:
        explicit = os.environ.get("REELCACHE_FFMPEG_BIN")
    if explicit is not None:
        if not isinstance(explicit, (str, os.PathLike)) or not str(explicit):
            raise ConfigurationError("FFmpeg path must be a nonempty path")
        selected = shutil.which(str(explicit))
        if selected is None:
            candidate = Path(explicit).expanduser()
            selected = str(candidate.resolve()) if candidate.is_file() else None
        if selected is None or not os.access(selected, os.X_OK):
            raise UnsupportedError(f"explicit FFmpeg executable is unavailable: {explicit}")
        return str(Path(selected).resolve())
    selected = shutil.which("ffmpeg")
    if selected:
        return str(Path(selected).resolve())
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise UnsupportedError(
            "HEVC requires FFmpeg with libx265; provide using_ffmpeg(path), "
            "REELCACHE_FFMPEG_BIN, or install the optional imageio-ffmpeg dependency"
        ) from exc
    try:
        selected = imageio_ffmpeg.get_ffmpeg_exe()
    except (OSError, RuntimeError) as exc:
        raise UnsupportedError("no usable existing imageio-ffmpeg binary was found") from exc
    return discover_ffmpeg(selected)


# CPU priority of the FFmpeg processes started in this context (``ffmpeg_priority``): None (the caller's), "idle"
# (SCHED_IDLE: run only on otherwise idle CPUs) or a nice increment 1..19. Archival encode workers set it so that x265
# encodes and validation decodes yield CPUs to serving work (restores, generation) without pausing.
_PRIORITY = contextvars.ContextVar("reelcache_ffmpeg_priority", default=None)


def check_cpu_priority(priority):
    """``priority`` if it is None, ``"idle"`` or a nice increment in [1, 19]; ConfigurationError otherwise."""
    if priority is None or priority == "idle" or (type(priority) is int and 1 <= priority <= 19):
        return priority
    raise ConfigurationError("an FFmpeg CPU priority is None, 'idle' (SCHED_IDLE) or a nice increment in [1, 19]")


@contextlib.contextmanager
def ffmpeg_priority(priority):
    """Start this context's FFmpeg processes (encodes, decodes) at a lower CPU priority (``check_cpu_priority``).

    The scheduling policy or nice value is set on the child before it reads its input, so the encoder threads it
    starts inherit it. Priority changes timing only, never output.
    """
    token = _PRIORITY.set(check_cpu_priority(priority))
    try:
        yield
    finally:
        _PRIORITY.reset(token)


def _lower_priority(pid, priority):
    """SCHED_IDLE (``"idle"``; nice 19 where unsupported) or ``priority`` more nice for every thread of ``pid``."""
    try:
        tasks = [int(task) for task in os.listdir(f"/proc/{pid}/task")]
    except (OSError, ValueError):
        tasks = [pid]
    for task in tasks:
        try:
            if priority == "idle" and hasattr(os, "SCHED_IDLE"):
                os.sched_setscheduler(task, os.SCHED_IDLE, os.sched_param(0))
            else:
                increment = 19 if priority == "idle" else priority
                os.setpriority(os.PRIO_PROCESS, task, min(19, os.getpriority(os.PRIO_PROCESS, task) + increment))
        except (OSError, AttributeError):
            pass  # exited already or unsupported: priority changes timing only


#: Pipe capacity requested for FFmpeg's pipes (F_SETPIPE_SZ; the kernel caps it at /proc/sys/fs/pipe-max-size).
# Pipes are exchanged in large blocking reads/writes on helper threads (``_exchange``), not by ``Popen.communicate``,
# whose select loop of 4 KiB writes / 32 KiB reads made ~35k GIL hand-offs per CausVid node encode + validation
# decode, contending with the process's other Python threads (a restore's decoder workers; Step 19 follow-up).
_PIPE_BYTES = 1 << 20
#: Bytes the pipe pump moves between pause checks (``ffmpeg_pump_pause``): a pause stalls the child within ~1 chunk.
_PUMP_CHUNK = 1 << 20
# Called by the pipe pump of this context's FFmpeg exchanges before every chunk it moves (``ffmpeg_pump_pause``).
_PUMP_PAUSE = contextvars.ContextVar("reelcache_ffmpeg_pump_pause", default=None)


@contextlib.contextmanager
def ffmpeg_pump_pause(hook):
    """Call ``hook()`` before every pipe chunk this context's FFmpeg exchanges move (None: no hook).

    An archival pipeline passes its restore yield: while a restore runs, the pump stops feeding and draining the
    child, which then blocks on its pipes (no CPU) and the pump threads stop taking the GIL, for at most the yield's
    bound per restore busy period. Pausing changes timing only.
    """
    if hook is not None and not callable(hook):
        raise ConfigurationError("an FFmpeg pump pause hook must be callable or None")
    token = _PUMP_PAUSE.set(hook)
    try:
        yield
    finally:
        _PUMP_PAUSE.reset(token)


def _enlarge_pipe(stream):
    try:
        import fcntl
        fcntl.fcntl(stream.fileno(), getattr(fcntl, "F_SETPIPE_SZ", 1031), _PIPE_BYTES)
    except (OSError, ValueError, ImportError):
        pass  # capacity changes the number of reads/writes only


def _exchange(process, data, timeout, pause=None):
    """``Popen.communicate`` with few GIL hand-offs: stdin is written by a helper thread in whole-buffer blocking
    writes, stderr read by another, stdout here in blocking reads of up to the (enlarged) pipe capacity; a timer
    kills the child at ``timeout`` (TimeoutExpired, as communicate). With ``pause`` stdin and stdout move in
    ``_PUMP_CHUNK`` pieces, each after ``pause()`` returned."""
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            _enlarge_pipe(stream)
    expired = threading.Event()

    def kill():
        expired.set()
        process.kill()
    timer = threading.Timer(timeout, kill)
    timer.daemon = True
    errors, stderr = [], []

    def write():
        try:
            view = memoryview(data).cast("B")
            fd, offset = process.stdin.fileno(), 0
            while offset < len(view):
                if pause is not None:
                    pause()
                    offset += os.write(fd, view[offset:offset + _PUMP_CHUNK])
                else:
                    offset += os.write(fd, view[offset:])
        except BrokenPipeError:
            pass  # the child stopped reading (exited): its status reports it, as with communicate
        except OSError as exc:
            errors.append(exc)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
    helpers = [threading.Thread(target=lambda: stderr.append(process.stderr.read()), daemon=True)]
    if data is not None:
        helpers.append(threading.Thread(target=write, daemon=True))
    timer.start()
    try:
        for helper in helpers:
            helper.start()
        if pause is None:
            stdout = process.stdout.read()
        else:
            chunks, fd = [], process.stdout.fileno()
            while True:
                pause()
                chunk = os.read(fd, _PIPE_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
            stdout = b"".join(chunks)
        for helper in helpers:
            helper.join()
        process.wait()
    finally:
        timer.cancel()
    if expired.is_set():
        raise subprocess.TimeoutExpired(process.args, timeout)
    if errors:
        raise errors[0]
    return stdout, stderr[0] if stderr else b""


def _run_child(command, data, timeout, priority, pause=None):
    """``subprocess.run`` (same pipes, timeout, cleanup and result) with the child's CPU priority lowered first."""
    with subprocess.Popen(command, stdin=None if data is None else subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        if priority is not None:
            _lower_priority(process.pid, priority)
        try:
            stdout, stderr = _exchange(process, data, timeout, pause)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        except BaseException:
            process.kill()
            raise
        returncode = process.poll()
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _run(command, *, data=None, integrity=False, timeout=None):
    error_type = IntegrityError if integrity else UnsupportedError
    timeout = _TIMEOUT_SECONDS if timeout is None else timeout
    try:
        result = _run_child(command, data, timeout, _PRIORITY.get(), _PUMP_PAUSE.get())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise error_type(f"FFmpeg execution failed: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-1800:]
        raise error_type(f"FFmpeg failed ({result.returncode}): {detail}")
    return result


@functools.lru_cache(maxsize=8)
def _capabilities(binary: str):
    result = _run([binary, "-hide_banner", "-h", "encoder=libx265"])
    text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    match = re.search(r"Supported pixel formats:\s*([^\r\n]+)", text)
    if "Encoder libx265" not in text or match is None:
        raise UnsupportedError("the selected FFmpeg does not provide the libx265 encoder")
    formats = frozenset(match.group(1).split())
    version = _run([binary, "-version"]).stdout.decode("utf-8", errors="replace").splitlines()
    return formats, version[0] if version else "unknown FFmpeg"


def capabilities(ffmpeg_path=None):
    binary = discover_ffmpeg(ffmpeg_path)
    formats, version = _capabilities(binary)
    return {
        "binary": binary, "version": version, "encoder": "libx265",
        "pixel_formats": [fmt for fmt in PIXEL_FORMATS if fmt in formats],
        "qp": 8, "preset": "ultrafast", "hardware_acceleration": False,
    }


def validate_parameters(pix_fmt, qp=8, preset="ultrafast"):
    if pix_fmt not in PIXEL_FORMATS:
        raise UnsupportedError(f"HEVC supports exactly {PIXEL_FORMATS}, not {pix_fmt!r}")
    if type(qp) is not int or qp != 8 or preset != "ultrafast":
        raise UnsupportedError("this HEVC provider supports only QP8 with preset='ultrafast'")


def _covering_affine(lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    span = hi - lo
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scale = (span / 255.0).astype(np.float32)
        scale = np.where(
            scale.astype(np.float64) * 255.0 < span,
            np.nextafter(scale, np.float32(np.inf)), scale,
        )
        offset = lo.astype(np.float32)
        upper = np.float32(255) * scale + offset
        scale = np.where(
            upper.astype(np.float64) < hi,
            np.nextafter(scale, np.float32(np.inf)), scale,
        )
        upper = np.float32(255) * scale + offset
    if (
        not np.isfinite(scale).all() or not np.isfinite(upper).all()
        or np.any(upper.astype(np.float64) < hi)
    ):
        raise ConfigurationError("coefficient range cannot use a finite covering FP32 affine")
    return np.stack((scale, offset)).astype("<f4", copy=False)


def normalization_layout(rank, normalization, protected_fraction):
    if type(rank) is not int or rank < 1:
        raise ConfigurationError("coefficient rank must be a positive integer")
    if normalization not in {"per_channel", "global", "leading_columns"}:
        raise UnsupportedError("normalization must be per_channel, global, or leading_columns")
    if (
        isinstance(protected_fraction, bool)
        or not isinstance(protected_fraction, (int, float))
        or not math.isfinite(protected_fraction) or not 0 <= protected_fraction <= 1
    ):
        raise ConfigurationError("protected_fraction must be in [0, 1]")
    if normalization == "per_channel":
        count, records = rank, rank
    elif normalization == "global":
        count, records = 0, 1
    else:
        count = int(math.ceil(rank * protected_fraction))
        records = count + 1
    return {
        "mode": normalization, "protected_count": count,
        "affine_shape": [2, records], "affine_order": ["scale", "offset"],
        "protected_columns": "leading",
    }


def quantize_coefficients(values, normalization, protected_fraction=1.0):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ConfigurationError("coefficients must be a nonempty finite FP32 matrix")
    layout = normalization_layout(values.shape[1], normalization, protected_fraction)
    count = layout["protected_count"]
    lo, hi = values.min(0), values.max(0)
    if normalization == "per_channel":
        affine = _covering_affine(lo, hi)
    else:
        affine = np.zeros(layout["affine_shape"], dtype="<f4")
        if count < values.shape[1]:
            affine[:, 0] = _covering_affine(lo[count:].min(), hi[count:].max())
        if count:
            affine[:, 1:] = _covering_affine(lo[:count], hi[:count])
    scale, offset = _expanded_affine(affine, layout, values.shape[1])
    ratios = np.zeros(values.shape, dtype=np.float64)
    np.divide(
        values.astype(np.float64) - offset.astype(np.float64),
        scale.astype(np.float64), out=ratios, where=scale > 0,
    )
    quantized = np.clip(np.rint(ratios), 0, 255).astype(np.uint8)
    return quantized, affine, layout


def _expanded_affine(affine, layout, rank):
    if layout["mode"] == "per_channel":
        return affine[0], affine[1]
    scale = np.full(rank, affine[0, 0], dtype=np.float32)
    offset = np.full(rank, affine[1, 0], dtype=np.float32)
    count = layout["protected_count"]
    if count:
        scale[:count], offset[:count] = affine[0, 1:], affine[1, 1:]
    return scale, offset


def validate_affine(affine, layout, rank):
    affine = np.asarray(affine)
    if (
        affine.dtype != np.dtype("<f4") or list(affine.shape) != layout["affine_shape"]
        or not np.isfinite(affine).all() or np.any(affine[0] < 0)
    ):
        raise IntegrityError("invalid coefficient affine payload")
    if (
        layout["mode"] == "leading_columns" and layout["protected_count"] == rank
        and np.any(affine[:, 0] != 0)
    ):
        raise IntegrityError("unused global affine must be zero when all columns are protected")
    with np.errstate(over="ignore", invalid="ignore"):
        upper = np.float32(255) * affine[0] + affine[1]
    if not np.isfinite(upper).all():
        raise IntegrityError("coefficient affine endpoints are nonfinite")
    return affine


def dequantize_coefficients(pixels, affine, layout):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or min(pixels.shape) < 1:
        raise IntegrityError("decoded coefficient pixels must be a nonempty uint8 matrix")
    affine = validate_affine(affine, layout, pixels.shape[1])
    scale, offset = _expanded_affine(affine, layout, pixels.shape[1])
    result = pixels.astype(np.float32) * scale + offset
    if not np.isfinite(result).all():
        raise IntegrityError("decoded coefficients are nonfinite")
    return np.ascontiguousarray(result)


def video_layout(frames, height, width, rank, pix_fmt):
    validate_parameters(pix_fmt)
    if any(type(v) is not int or v < 1 for v in (frames, height, width, rank)):
        raise ConfigurationError("HEVC dimensions must be positive integers")
    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        raise UnsupportedError(f"coefficient tiles cannot exceed {MAX_DIMENSION}x{MAX_DIMENSION}")
    planes = 3 if pix_fmt == "yuv444p" else 1
    columns = min(MAX_DIMENSION // width, max(1, math.ceil(math.sqrt(rank / planes))))
    rows = min(MAX_DIMENSION // height, max(1, math.ceil(rank / (planes * columns))))
    image_width, image_height = max(16, columns * width), max(16, rows * height)
    if pix_fmt == "yuv420p":
        image_width += image_width % 2
        image_height += image_height % 2
    if image_width > MAX_DIMENSION or image_height > MAX_DIMENSION:
        raise UnsupportedError("padded HEVC frame dimensions exceed provider support")
    slots = columns * rows * planes
    return {
        "schema": "reelcache.channel_tiles.v1", "frames": frames,
        "height": height, "width": width, "rank": rank, "pix_fmt": pix_fmt,
        "grid": [columns, rows], "planes": planes,
        "image_height": image_height, "image_width": image_width,
        "channels_per_segment": slots, "segments": math.ceil(rank / slots),
        "frame_padding": [image_height - rows * height, image_width - columns * width],
    }


def _cells(layout, count):
    columns, _ = layout["grid"]
    for index in range(count):
        plane = index % layout["planes"]
        row, column = divmod(index // layout["planes"], columns)
        yield index, plane, row * layout["height"], column * layout["width"]


def _raw_frame_bytes(images, pix_fmt):
    if _accel_c.fast():  # accel C (gil_fastpaths): one preallocated buffer, identical bytes
        return _accel_c.raw_frame_bytes(images, pix_fmt)
    if pix_fmt == "yuv444p":
        return images.tobytes()
    frames, _, height, width = images.shape
    chroma = np.full((frames, height * width // 2), 128, dtype=np.uint8)
    return np.concatenate((images[:, 0].reshape(frames, -1), chroma), axis=1).tobytes()


def x265_thread_params(threads, frames):
    """x265 threading for one encode. Every value >1 reconstructs identical pixels to 1.

    ``pools`` sizes the WPP worker pool; with >=4 threads, frame parallelism
    over the chunk's frames is added. SIMD selection (AVX-512) is left to
    x265 auto-detection: it measured no gain for these picture sizes.
    """
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ConfigurationError("x265 threads must be an integer in [1, 64]")
    frame_threads = 1 if threads < 4 else max(1, min(3, frames))
    return f"pools={threads}:frame-threads={frame_threads}"


def _encode_segment(images, pix_fmt, binary, threads=1, inter_prediction=False):
    frames, _, height, width = images.shape
    command = [
        binary, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}",
        "-framerate", "25", "-i", "pipe:0", "-an", "-c:v", "libx265",
        "-preset", "ultrafast", "-pix_fmt", f"+{pix_fmt}", "-threads", "1",
        "-x265-params", x265_parameter_string(inter_prediction, threads, frames),
        "-frames:v", str(frames), "-f", "hevc", "pipe:1",
    ]
    result = _run(command, data=_raw_frame_bytes(images, pix_fmt), timeout=size_scaled_timeout(int(images.size)))
    if not result.stdout:
        raise UnsupportedError("FFmpeg produced an empty HEVC stream")
    if not inter_prediction:
        all_intra_access_units(result.stdout, frames)
    return result.stdout


def _segment_slots(layout, segment):
    slots = layout["channels_per_segment"]
    first = segment * slots
    return first, min(slots, layout["rank"] - first)


def _mosaic(values, layout, segment):
    """One segment's channel tiles as native planes: [frames, planes, image_h, image_w]."""
    first, count = _segment_slots(layout, segment)
    images = np.zeros((
        layout["frames"], layout["planes"], layout["image_height"], layout["image_width"],
    ), dtype=np.uint8)
    for index, plane, y, x in _cells(layout, count):
        images[:, plane, y:y + layout["height"], x:x + layout["width"]] = values[..., first + index]
    return images


def _unmosaic(images, layout, segment, values):
    first, count = _segment_slots(layout, segment)
    for index, plane, y, x in _cells(layout, count):
        values[..., first + index] = images[:, plane, y:y + layout["height"], x:x + layout["width"]]


def encode_pixels(pixels, layout, *, ffmpeg_path=None, threads=1, inter_prediction=False):
    if (
        not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
        or pixels.shape != (
            layout["frames"] * layout["height"] * layout["width"], layout["rank"]
        )
    ):
        raise ConfigurationError("pixels differ from coefficient video geometry")
    encoder_record(inter_prediction)
    caps = capabilities(ffmpeg_path)
    if layout["pix_fmt"] not in caps["pixel_formats"]:
        raise UnsupportedError(f"selected libx265 cannot encode {layout['pix_fmt']}")
    values = pixels.reshape(layout["frames"], layout["height"], layout["width"], layout["rank"])
    streams = []
    started = time.perf_counter()
    for segment in range(layout["segments"]):
        streams.append(_encode_segment(
            _mosaic(values, layout, segment), layout["pix_fmt"], caps["binary"], threads,
            inter_prediction,
        ))
    return streams, time.perf_counter() - started, caps["version"]


def _decode_segment(stream, layout, binary, timeout=None):
    command = [
        binary, "-hide_banner", "-loglevel", "info", "-nostdin", "-xerror",
        "-err_detect", "explode", "-threads", "1", "-f", "hevc", "-i", "pipe:0",
        "-map", "0:v:0", "-an", "-vf", "showinfo", "-vsync", "0",
        "-pix_fmt", f"+{layout['pix_fmt']}", "-threads", "1",
        "-frames:v", str(layout["frames"] + 1), "-f", "rawvideo", "pipe:1",
    ]
    height, width, frames = layout["image_height"], layout["image_width"], layout["frames"]
    samples = frames * (3 if layout["pix_fmt"] == "yuv444p" else 1) * height * width
    result = _run(command, data=stream, integrity=True, timeout=decode_timeout(samples, timeout))
    per_frame = height * width * (3 if layout["pix_fmt"] == "yuv444p" else 3 / 2)
    if len(result.stdout) != frames * int(per_frame):
        raise IntegrityError("HEVC decoded byte count/frame count differs from declared geometry")
    log = result.stderr.decode("utf-8", errors="replace")
    observed = re.findall(r"\bn:\s*\d+.*?\bfmt:(\S+).*?\bs:(\d+)x(\d+)", log)
    if len(observed) != frames or any(
        fmt != layout["pix_fmt"] or int(w) != width or int(h) != height
        for fmt, w, h in observed
    ):
        raise IntegrityError("HEVC native pixel format or geometry differs from the packet")
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    if layout["pix_fmt"] == "yuv444p":
        return pixels.reshape(frames, 3, height, width)
    return pixels.reshape(frames, int(per_frame))[:, :height * width].reshape(frames, 1, height, width)


def decode_pixels(streams, layout, *, ffmpeg_path=None, timeout=None):
    """FFmpeg software decode of a packet's segment streams to ``[tokens, rank]`` uint8 pixels.

    Each segment's FFmpeg process gets the size-scaled :func:`decode_timeout` deadline, or ``timeout`` seconds.
    """
    if timeout is not None:
        _positive_seconds(timeout, "an FFmpeg decode timeout")
    if len(streams) != layout["segments"] or any(
        not isinstance(stream, bytes) or not stream for stream in streams
    ):
        raise IntegrityError("HEVC streams differ from the declared segment count")
    binary = discover_ffmpeg(ffmpeg_path)
    started = time.perf_counter()
    values = np.empty((
        layout["frames"], layout["height"], layout["width"], layout["rank"],
    ), dtype=np.uint8)
    for segment, stream in enumerate(streams):
        _unmosaic(_decode_segment(stream, layout, binary, timeout), layout, segment, values)
    return values.reshape(-1, layout["rank"]), time.perf_counter() - started


def nal_units(stream):
    """Annex-B NAL units as ``(offset, length, nal_unit_type, first_slice_flag)``.

    Offsets include each start code (and a four-byte start code's zero byte).
    ``first_slice_flag`` is None for non-VCL units. Headers are validated.
    """
    if _accel_c.host_paths():  # accel C (gil_fastpaths / native_ext): GIL-free scan, identical units and errors
        return _accel_c.nal_units(stream)
    if not isinstance(stream, bytes) or not stream:
        raise IntegrityError("HEVC stream must be nonempty bytes")
    starts = []
    for match in _START_CODE.finditer(stream):
        position = match.start()
        if position and stream[position - 1] == 0:
            position -= 1
        starts.append((position, match.end()))
    if not starts or starts[0][0] != 0:
        raise IntegrityError("HEVC stream does not begin with an Annex-B start code")
    units = []
    for index, (start, header) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(stream)
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


def access_units(stream):
    """Split a stream into coded pictures (H.265 7.4.2.4.4 access-unit boundaries)."""
    groups, current, has_vcl = [], [], False
    for unit in nal_units(stream):
        nal_type, first_slice = unit[2], unit[3]
        if has_vcl and (nal_type in _AU_PREFIX_TYPES or (nal_type < 32 and first_slice)):
            groups.append(current)
            current, has_vcl = [], False
        current.append(unit)
        has_vcl = has_vcl or nal_type < 32
    if not has_vcl:
        raise IntegrityError("HEVC stream ends without a coded picture")
    groups.append(current)
    return [{
        "offset": group[0][0], "length": group[-1][0] + group[-1][1] - group[0][0],
        "nal_types": [unit[2] for unit in group],
    } for group in groups]


def all_intra_access_units(stream, pictures):
    """Require exactly ``pictures`` IRAP pictures; IRAP pictures carry only I slices."""
    units = access_units(stream)
    if len(units) != pictures:
        raise IntegrityError("HEVC stream picture count differs from its declared pictures")
    for unit in units:
        vcl = [nal_type for nal_type in unit["nal_types"] if nal_type < 32]
        if not vcl or any(nal_type not in _IRAP_TYPES for nal_type in vcl):
            raise IntegrityError("all-intra HEVC stream contains a non-IRAP picture")
        unit["self_contained"] = (
            _PARAMETER_SET_TYPES <= set(unit["nal_types"])
            and all(nal_type in _IDR_TYPES for nal_type in vcl)
        )
    return units


def _libc_memmem():
    """glibc ``memmem`` through ctypes (the GIL is released during the search), or None."""
    import ctypes
    import ctypes.util

    try:
        name = ctypes.util.find_library("c")
        function = ctypes.CDLL(name).memmem if name else None
    except (OSError, AttributeError):
        return None
    if function is None:
        return None
    function.restype = ctypes.c_void_p
    function.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
    needle = ctypes.create_string_buffer(_VPS_START, len(_VPS_START))

    def find(stream, position, end):
        if end - position < len(_VPS_START):
            return -1
        base = ctypes.cast(ctypes.c_char_p(stream), ctypes.c_void_p).value
        address = function(base + position, end - position, needle, len(_VPS_START))
        return -1 if not address else address - base

    return find


# Picture scans of restore packs run on several threads at once (reelcache.restore_staging): bytes.find holds the GIL,
# memmem does not. Both return the lowest start of the whole pattern inside [position, end).
_find_vps = _libc_memmem() or (lambda stream, position, end: stream.find(_VPS_START, position, end))


def picture_units(stream, offset, length, pictures):
    """Byte ranges ``[(offset, length)]`` of ``pictures`` self-contained all-intra access units.

    Fast boundary scan for streams written under the pinned all-intra x265
    contract, where every picture is an IDR access unit led by its own VPS.
    Emulation prevention makes every ``00 00 01`` a start code, so the VPS
    start codes are exactly the picture boundaries (offsets follow
    :func:`nal_units`, including a four-byte start code's zero byte). Use
    :func:`all_intra_access_units` for full NAL-level validation. The search
    releases the GIL (glibc ``memmem``) where available, so scans of different
    ranges run in parallel threads.
    """
    if not isinstance(stream, bytes) or type(offset) is not int or type(length) is not int:
        raise IntegrityError("picture scan requires bytes and integer offsets")
    if type(pictures) is not int or pictures < 1 or offset < 0 or length < 1 or offset + length > len(stream):
        raise IntegrityError("picture scan range lies outside the stream")
    end = offset + length
    starts, position = [], offset
    while True:
        found = _find_vps(stream, position, end)
        if found < 0:
            break
        starts.append(found - 1 if found > offset and stream[found - 1] == 0 else found)
        if len(starts) > pictures:
            break
        position = found + len(_VPS_START)
    if len(starts) != pictures or starts[0] != offset:
        raise IntegrityError("byte range does not hold the declared self-contained all-intra pictures")
    return [(start, (starts[index + 1] if index + 1 < pictures else end) - start)
            for index, start in enumerate(starts)]


@functools.lru_cache(maxsize=8)
def _bitstream_filters(binary):
    result = _run([binary, "-hide_banner", "-bsfs"])
    return frozenset((result.stdout + result.stderr).decode("utf-8", errors="replace").split())


def trace_headers(stream, *, ffmpeg_path=None):
    """Parse FFmpeg ``trace_headers`` output for one real HEVC stream.

    Diagnostic only: it fails loudly when the selected FFmpeg lacks the filter.
    Slice QP is ``26 + init_qp_minus26 + slice_qp_delta`` (single PPS streams).
    """
    if not isinstance(stream, bytes) or not stream:
        raise ConfigurationError("trace_headers requires nonempty HEVC bytes")
    binary = discover_ffmpeg(ffmpeg_path)
    if "trace_headers" not in _bitstream_filters(binary):
        raise UnsupportedError(f"FFmpeg at {binary} lacks the trace_headers bitstream filter")
    result = _run([
        binary, "-hide_banner", "-nostdin", "-loglevel", "info", "-f", "hevc", "-i", "pipe:0",
        "-map", "0:v:0", "-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-",
    ], data=stream, integrity=True)
    packets, nal_types, pictures, init_qp = 0, [], [], None
    sizes, chroma, profiles, intra_constraint = set(), set(), set(), set()
    nal_type = slice_type = width = None
    for line in result.stderr.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("[trace_headers"):
            continue
        content = line.split("] ", 1)[-1]
        if content.startswith("Packet:"):
            packets += 1
            continue
        match = _TRACE_FIELD.match(content)
        if match is None:
            continue
        name, value = match.group(1), int(match.group(2))
        if name == "nal_unit_type":
            nal_type = value
            nal_types.append(value)
        elif name == "general_profile_idc":
            profiles.add(value)
        elif name == "general_intra_constraint_flag":
            intra_constraint.add(value)
        elif name == "chroma_format_idc":
            chroma.add(value)
        elif name == "pic_width_in_luma_samples":
            width = value
        elif name == "pic_height_in_luma_samples":
            sizes.add((width, value))
        elif name == "init_qp_minus26":
            init_qp = value
        elif nal_type is not None and nal_type < 32:
            if name == "first_slice_segment_in_pic_flag" and value:
                pictures.append({"nal_unit_type": nal_type, "slice_types": [], "slice_qps": []})
            elif name == "slice_type":
                slice_type = value
            elif name == "slice_qp_delta":
                if not pictures or slice_type is None or init_qp is None:
                    raise IntegrityError("trace_headers produced an incomplete slice header")
                pictures[-1]["slice_types"].append(_SLICE_TYPES[slice_type])
                pictures[-1]["slice_qps"].append(26 + init_qp + value)
                slice_type = None
    counts = {kind: 0 for kind in ("I", "P", "B")}
    for picture in pictures:
        for kind in picture["slice_types"]:
            counts[kind] += 1
    return {
        "binary": binary, "packets": packets, "nal_unit_types": nal_types,
        "pictures": pictures, "slice_type_counts": counts,
        "slice_qps": sorted({qp for picture in pictures for qp in picture["slice_qps"]}),
        "picture_sizes": sorted(sizes), "chroma_format_idc": sorted(chroma),
        "general_profile_idc": sorted(profiles),
        "general_intra_constraint_flag": sorted(intra_constraint),
    }


def align(value, alignment=CANVAS_ALIGNMENT):
    return -(-value // alignment) * alignment


def encode_pictures(pictures, pix_fmt, *, ffmpeg_path=None, threads=1):
    """Encode prepared canvas pictures ``[count, planes, height, width]`` all-intra."""
    validate_parameters(pix_fmt)
    planes = 3 if pix_fmt == "yuv444p" else 1
    if (
        not isinstance(pictures, np.ndarray) or pictures.dtype != np.uint8 or pictures.ndim != 4
        or pictures.shape[0] < 1 or pictures.shape[1] != planes
        or not 16 <= pictures.shape[2] <= MAX_DIMENSION or not 16 <= pictures.shape[3] <= MAX_DIMENSION
        or (pix_fmt == "yuv420p" and (pictures.shape[2] % 2 or pictures.shape[3] % 2))
    ):
        raise ConfigurationError("canvas pictures differ from the HEVC plane geometry")
    caps = capabilities(ffmpeg_path)
    if pix_fmt not in caps["pixel_formats"]:
        raise UnsupportedError(f"selected libx265 cannot encode {pix_fmt}")
    started = time.perf_counter()
    stream = _encode_segment(pictures, pix_fmt, caps["binary"], threads, False)
    return stream, time.perf_counter() - started, caps["version"]


def decode_pictures(stream, *, pictures, height, width, pix_fmt, ffmpeg_path=None, timeout=None):
    """Decode a whole canvas stream to ``[pictures, planes, height, width]``.

    The FFmpeg process gets the size-scaled :func:`decode_timeout` deadline, or ``timeout`` seconds.
    """
    validate_parameters(pix_fmt)
    if timeout is not None:
        _positive_seconds(timeout, "an FFmpeg decode timeout")
    binary = discover_ffmpeg(ffmpeg_path)
    started = time.perf_counter()
    images = _decode_segment(stream, {
        "frames": pictures, "image_height": height, "image_width": width, "pix_fmt": pix_fmt,
    }, binary, timeout)
    return images, time.perf_counter() - started
