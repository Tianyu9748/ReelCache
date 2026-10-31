"""Explicit, thread-owned NVDEC sessions for exact native HEVC coefficient pixels.

Inference mode: callers may run inside ``torch.inference_mode`` (SGLang serves requests under it). The mode is
thread-local and decoder threads never inherit it, while PyTorch rejects an in-place update of an inference tensor
outside inference mode. Decode buffers that outlive their call or are written by decoder threads are therefore
created as normal tensors (:func:`normal_tensors`), and writes into a caller-provided canvas run in that canvas's
mode on any thread (:func:`write_mode`).

Process exit (Step 18 job 4429779, Step 11 job 4419034; RTX PRO 6000): a process that used NVDEC can hang in
libnvcuvid's native exit handler after its Python exit completed. The first use of the library arms
:func:`install_exit_guard`, which ends C ``exit(status)`` with ``_exit(status)`` before that handler runs.
``REELCACHE_NVDEC_EXIT_GUARD=0`` disables it.
"""

import atexit
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import ctypes
import io
import os
import sys
import threading
import time
import weakref

import numpy as np

from .errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from .telemetry import cuda_span, span


#: Environment switch of the NVDEC exit guard: "0" disables it (e.g. under a tool that needs the native exit handlers).
EXIT_GUARD_ENV = "REELCACHE_NVDEC_EXIT_GUARD"
#: "1": the exit guard's hook reports its state and the sessions still open to stderr.
EXIT_REPORT_ENV = "REELCACHE_NVDEC_EXIT_REPORT"
_EXIT_GUARD = {"armed": False, "installed": False, "error": None, "open_at_exit": None}
_SESSION_OWNERS = weakref.WeakSet()  # CanvasSession and NVDECDecoder objects (diagnostics: open_nvdec_sessions)


def note_nvdec_use():
    """Arm the exit guard: called before every call into libnvcuvid (decoder caps, decoder creation).

    The state is per address space, so a forked child inherits it with the library's exit handler (whose worker
    threads do not exist in the child).
    """
    _EXIT_GUARD["armed"] = True


def exit_guard_state():
    """``{"armed", "installed", "error", "open_at_exit"}`` of this process's NVDEC exit guard."""
    return dict(_EXIT_GUARD)


def open_nvdec_sessions():
    """NVDEC decoders still open in this process: ``{"canvas_sessions", "decoder_sessions"}``.

    Canvas sessions (decoder pools, NVDECDecoder streams) not closed yet, and NVDECDecoder packet sessions.
    """
    counts = {"canvas_sessions": 0, "decoder_sessions": 0}
    for _ in range(100):
        try:
            owners = list(_SESSION_OWNERS)
            break
        except RuntimeError:  # a session created during the iteration: retry
            continue
    else:
        raise RuntimeError("NVDEC sessions kept changing while counted")
    for owner in owners:
        if isinstance(owner, CanvasSession):
            counts["canvas_sessions"] += owner._decoder is not None
        else:
            counts["decoder_sessions"] += len(owner._sessions)
    return counts


def install_exit_guard():
    """Python exit hook (registered at import) of a process that used NVDEC: C ``exit(status)`` then ends it at once.

    libnvcuvid's native exit handler joins a worker thread of the library, which can wait forever: the process
    has written its results, printed its last line and run every Python exit hook (decoder pools closed, sessions
    destroyed on their workers), then hangs in C ``exit()`` (RTX PRO 6000: Step 18 job 4429779, Step 11 job
    4419034). Native exit handlers run last-registered first, and this hook runs after the library registered its
    handler, so it registers two glibc functions with ``on_exit``: ``fcloseall`` (the C stdio cleanup ``exit``
    itself ends with) and then ``_exit(status)``. Both run before libnvcuvid's handler, which is skipped with the
    other native handlers registered before (C/C++ library teardown; the kernel releases the process's GPU
    contexts). Python runs exit hooks last-registered first, so every hook registered after this module's import
    (decoder pools and bridges closing their sessions) runs before this one; the hooks registered before it,
    interpreter finalization and native handlers registered later still run before the guard's handlers. Without
    glibc's ``on_exit`` nothing is installed and the exit is unchanged (``exit_guard_state()["error"]``). Returns
    whether the guard was installed.
    """
    if not _EXIT_GUARD["armed"] or _EXIT_GUARD["installed"]:
        return False
    try:
        return _install_exit_handlers()
    finally:  # diagnostics, never in the way of the guard
        try:
            _EXIT_GUARD["open_at_exit"] = open_nvdec_sessions()
            if os.environ.get(EXIT_REPORT_ENV) == "1":
                print(f"reelcache.nvdec exit guard {exit_guard_state()}", file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            _EXIT_GUARD["open_at_exit"] = repr(exc)


def _install_exit_handlers():
    if os.environ.get(EXIT_GUARD_ENV, "1") == "0":
        _EXIT_GUARD["error"] = f"disabled by {EXIT_GUARD_ENV}=0"
        return False
    try:
        libc = ctypes.CDLL(None)
        on_exit, fflush = libc.on_exit, libc.fflush
        on_exit.argtypes, on_exit.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
        fflush.argtypes, fflush.restype = [ctypes.c_void_p], ctypes.c_int
        # on_exit calls handler(status, arg): _exit takes the status; fcloseall takes no argument (ignored ones).
        terminate, cleanup = (ctypes.cast(function, ctypes.c_void_p).value for function in (libc._exit, libc.fcloseall))
    except (AttributeError, OSError) as exc:
        _EXIT_GUARD["error"] = f"no glibc exit handlers: {exc}"
        return False
    if on_exit(terminate, None) != 0:
        _EXIT_GUARD["error"] = "on_exit() refused _exit"
        return False
    _EXIT_GUARD["installed"] = True
    if on_exit(cleanup, None) != 0:  # registered after _exit, so it runs before it
        fflush(None)  # C stdio written from here on may then be lost
        _EXIT_GUARD["error"] = "on_exit() refused fcloseall: C stdio flushed early"
    return True


atexit.register(install_exit_guard)  # at import: runs after the exit hooks registered later (pools, bridges)


@contextmanager
def normal_tensors():
    """Create normal (non-inference) tensors without autograd, whatever the caller's ``torch.inference_mode``.

    Used for decode buffers that are reused across calls or written by decoder threads (canvas-arena and
    warm-up canvases, prepared and uploaded bases): a normal tensor may be read and written in either mode.
    """
    import torch

    with torch.inference_mode(False), torch.no_grad():
        yield


def write_mode(tensor):
    """Context for an in-place write of ``tensor`` on any thread: inference mode only where PyTorch requires it.

    An inference tensor (e.g. a canvas allocated by a caller inside ``torch.inference_mode``) is written inside
    inference mode; every other write keeps the thread's current mode.
    """
    import torch

    if tensor.is_inference() and not torch.is_inference_mode_enabled():
        return torch.inference_mode()
    return nullcontext()


class NVDECDecoder:
    """Bound decoder sessions; output returns to owned CPU arrays without conversion.

    NVDEC surfaces/context allocations are driver-owned, not Torch allocator
    memory. Session and host-output limits are not a physical HBM/RSS guarantee.
    No CUDA work occurs at construction. Construction binds the execution thread;
    encode-time software validation intentionally does not use this decoder.
    """

    def __init__(self, *, gpu_id, gpu_uuid, max_sessions=2,
                 max_output_bytes=256 * 1024**2, max_bitstream_bytes=64 * 1024**2,
                 device_output=False):
        if type(gpu_id) is not int or gpu_id < 0:
            raise ConfigurationError("NVDEC gpu_id must be a nonnegative visible-device index")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or len(gpu_uuid) < 8:
            raise ConfigurationError("NVDEC requires an explicit GPU UUID")
        if any(type(value) is not int or value <= 0 for value in (
            max_sessions, max_output_bytes, max_bitstream_bytes,
        )):
            raise ConfigurationError("NVDEC capacities must be positive integers")
        self.gpu_id, self.gpu_uuid = gpu_id, gpu_uuid
        self.max_sessions = max_sessions
        if type(device_output) is not bool:
            raise ConfigurationError("device_output must be boolean")
        self.device_output = device_output
        self.max_output_bytes, self.max_bitstream_bytes = max_output_bytes, max_bitstream_bytes
        self._api = None
        self._owner = (os.getpid(), threading.get_ident())
        self._sessions = OrderedDict()
        self._canvas_sessions = OrderedDict()
        self._closed = False
        _SESSION_OWNERS.add(self)
        self._stats = {
            "packet_calls": 0, "stream_calls": 0, "decoded_frames": 0, "failures": 0,
            "sessions_created": 0, "session_hits": 0, "evictions": 0,
            "peak_sessions": 0, "peak_host_output_reservation": 0,
            "peak_device_output_reservation": 0,
            "session_create_seconds": 0.0, "decode_seconds": 0.0,
        }

    def _initialize(self):
        if self._closed:
            raise StateError("NVDEC decoder is closed")
        current = (os.getpid(), threading.get_ident())
        if self._owner is not None and self._owner != current:
            raise StateError("NVDEC sessions must stay on their owning process/thread")
        if self._api is None:
            try:
                import PyNvVideoCodec as nvc
            except ImportError as exc:
                raise UnsupportedError("NVDEC requires the optional PyNvVideoCodec dependency") from exc
            import torch
            if self.gpu_id >= torch.cuda.device_count():
                raise UnsupportedError("NVDEC device is not visible")
            actual = str(torch.cuda.get_device_properties(self.gpu_id).uuid).removeprefix("GPU-")
            if actual != self.gpu_uuid.removeprefix("GPU-"):
                raise ConfigurationError("NVDEC visible device does not match the granted GPU UUID")
            self._api, self._owner = nvc, current

    def _session(self, layout):
        self._initialize()
        nvc = self._api
        width, height, fmt = layout["image_width"], layout["image_height"], layout["pix_fmt"]
        chroma = nvc.cudaVideoChromaFormat.__members__["444" if fmt == "yuv444p" else "420"]
        stream = 0
        if self.device_output:
            import torch
            stream = torch.cuda.current_stream(self.gpu_id).cuda_stream
        key = (width, height, fmt, stream)
        if key in self._sessions:
            self._sessions.move_to_end(key)
            self._stats["session_hits"] += 1
            return key, self._sessions[key]
        note_nvdec_use()
        caps = nvc.GetDecoderCaps(self.gpu_id, nvc.cudaVideoCodec.HEVC, chroma, 8)
        if not caps["supported"] or not (
            caps["width_min"] <= width <= caps["width_max"]
            and caps["height_min"] <= height <= caps["height_max"]
            and ((width + 15) // 16) * ((height + 15) // 16) <= caps["mb_num_max"]
        ):
            raise UnsupportedError("NVDEC hardware does not support this exact HEVC format/geometry")
        if len(self._sessions) == self.max_sessions:
            self._sessions.popitem(last=False)
            self._stats["evictions"] += 1
        started = time.perf_counter()
        decoder = nvc.CreateDecoder(
            gpuid=self.gpu_id, codec=nvc.cudaVideoCodec.HEVC, usedevicememory=self.device_output,
            cudastream=stream,
            outputColorType=nvc.OutputColorType.NATIVE, maxwidth=width, maxheight=height,
        )
        self._stats["session_create_seconds"] += time.perf_counter() - started
        self._sessions[key] = decoder
        self._stats["sessions_created"] += 1
        self._stats["peak_sessions"] = max(self._stats["peak_sessions"], len(self._sessions))
        return key, decoder

    def _frame(self, frame, decoder, layout):
        height, width = layout["image_height"], layout["image_width"]
        nvc = self._api
        full444 = layout["pix_fmt"] == "yuv444p"
        expected_format = nvc.Pixel_Format.YUV444 if full444 else nvc.Pixel_Format.NV12
        rows = 3 * height if full444 else height * 3 // 2
        expected_bytes = rows * width
        if (
            decoder.GetWidth() != width or decoder.GetHeight() != height
            or decoder.GetPixelFormat() != expected_format or frame.format != expected_format
            or tuple(frame.shape) != (rows, width) or tuple(frame.strides) != (width, 1)
            or np.dtype(frame.dtype) != np.dtype("uint8")
            or frame.framesize() != expected_bytes or decoder.GetFrameSize() != expected_bytes
        ):
            raise IntegrityError("NVDEC changed native pixel format, pitch, bit depth or frame geometry")
        pointer = frame.GetPtrToPlane(0)
        if not isinstance(pointer, int) or pointer <= 0:
            raise IntegrityError("NVDEC returned no owned host output plane")
        planes = 3 if full444 else 2
        if any(frame.GetPtrToPlane(index) != pointer + index * height * width for index in range(planes)):
            raise IntegrityError("NVDEC host planes are not contiguous in native order")
        if self.device_output:
            import torch
            value = torch.from_dlpack(frame)
            if (
                value.device != torch.device("cuda", self.gpu_id)
                or value.dtype != torch.uint8 or tuple(value.shape) != (rows, width)
                or tuple(value.stride()) != (width, 1)
            ):
                raise IntegrityError("NVDEC DLPack frame identity differs from its validated surface")
            return value.reshape(3, height, width) if full444 else value[:height].unsqueeze(0)
        # Host output was explicitly requested; validate its complete extent before reading.
        raw = ctypes.string_at(pointer, expected_bytes)
        array = np.frombuffer(raw, dtype=np.uint8)
        return array.reshape(3, height, width) if full444 else array[:height * width].reshape(1, height, width)

    def _stream(self, stream, layout, output, segment):
        from ._hevc import _cells

        key, decoder = self._session(layout)
        nvc = self._api
        source = io.BytesIO(stream)
        success = False
        try:
            with span("nvdec.demux"):
                demux = nvc.CreateDemuxer(source.readinto)
            if (
                demux.GetNvCodecId() != nvc.cudaVideoCodec.HEVC or demux.BitDepth() != 8
                or demux.Width() != layout["image_width"] or demux.Height() != layout["image_height"]
            ):
                raise IntegrityError("NVDEC stream header differs from the declared HEVC packet")
            count = 0
            first = segment * layout["channels_per_segment"]
            columns = min(layout["channels_per_segment"], layout["rank"] - first)

            def collect(frames):
                nonlocal count
                if self.device_output:
                    import torch
                    decoder.WaitOnCUStream(torch.cuda.current_stream(self.gpu_id).cuda_stream)
                else:
                    decoder.SyncOnCUStream()
                for frame in frames:
                    if count >= layout["frames"]:
                        raise IntegrityError("NVDEC returned extra frames")
                    image = self._frame(frame, decoder, layout)
                    if self.device_output:
                        with span("nvdec.unpack"), cuda_span("nvdec.unpack", output.device):
                            output[count, :, :, first:first + columns].copy_(unpack_device_frame(image, layout, columns))
                    else:
                        with span("nvdec.unpack"):
                            for index, plane, y, x in _cells(layout, columns):
                                output[count, :, :, first + index] = image[
                                    plane, y:y + layout["height"], x:x + layout["width"],
                                ]
                    count += 1
                if self.device_output:
                    # Finish owned tile copies before the SDK can recycle its surfaces.
                    torch.cuda.current_stream(self.gpu_id).synchronize()

            packets = 0
            for packet in demux:
                packets += 1
                if packets > 64 * (layout["frames"] + 1):
                    raise IntegrityError("NVDEC demux packet count exceeds the bounded stream contract")
                with span("nvdec.decode"):
                    decoded = decoder.Decode(packet)
                collect(decoded)
            # PyNvVideoCodec 2.2 uses an empty packet for EOS, preserving display order.
            with span("nvdec.decode"):
                decoded = decoder.Decode(nvc.PacketData())
            collect(decoded)
            if count != layout["frames"]:
                raise IntegrityError("NVDEC returned an incomplete frame sequence")
            self._stats["decoded_frames"] += count
            self._stats["stream_calls"] += 1
            success = True
        finally:
            if not success:
                self._sessions.pop(key, None)

    def decode_pixels(self, streams, layout):
        from ._hevc import video_layout

        expected = video_layout(layout["frames"], layout["height"], layout["width"],
                                layout["rank"], layout["pix_fmt"])
        if expected != layout:
            raise IntegrityError("NVDEC layout differs from the canonical coefficient packing")
        if len(streams) != layout["segments"] or any(not isinstance(s, bytes) or not s for s in streams):
            raise IntegrityError("NVDEC stream count/payload differs from the packet")
        if sum(map(len, streams)) > self.max_bitstream_bytes:
            raise CapacityError("NVDEC compressed input exceeds its admission limit")
        output_bytes = layout["frames"] * layout["height"] * layout["width"] * layout["rank"]
        frame_bytes = layout["image_height"] * layout["image_width"] * (3 if layout["pix_fmt"] == "yuv444p" else 2)
        reservation = output_bytes + (layout["frames"] + 2) * frame_bytes
        if reservation > self.max_output_bytes:
            raise CapacityError("NVDEC host output/workspace exceeds its admission limit")
        self._initialize()
        ledger = "peak_device_output_reservation" if self.device_output else "peak_host_output_reservation"
        self._stats[ledger] = max(self._stats[ledger], reservation)
        started = time.perf_counter()
        success = False
        try:
            shape = (layout["frames"], layout["height"], layout["width"], layout["rank"])
            if self.device_output:
                import torch
                output = torch.empty(shape, dtype=torch.uint8, device=torch.device("cuda", self.gpu_id))
            else:
                output = np.empty(shape, dtype=np.uint8)
            for index, stream in enumerate(streams):
                self._stream(stream, layout, output, index)
            self._stats["packet_calls"] += 1
            success = True
            return output.reshape(-1, layout["rank"]), time.perf_counter() - started
        finally:
            self._stats["decode_seconds"] += time.perf_counter() - started
            if not success:
                self._stats["failures"] += 1

    def stats(self):
        return {
            **self._stats, "backend": "nvdec", "gpu_uuid": self.gpu_uuid, "gpu_id": self.gpu_id,
            "live_sessions": len(self._sessions), "max_sessions": self.max_sessions,
            "canvas_sessions": {f"{w}x{h}:{fmt}": dict(session.stats)
                                for (w, h, fmt), session in self._canvas_sessions.items()},
            "initialized": self._api is not None, "closed": self._closed,
            "output": "owned CUDA uint8 coefficient tiles" if self.device_output else "owned CPU uint8 native planes",
            "device_output": self.device_output,
            "version": None if self._api is None else getattr(self._api, "__version__", "unknown"),
            "physical_driver_memory_included_in_torch_allocator": False,
            "software_fallback": False,
        }

    def decode_canvas(self, units, *, width, height, pix_fmt):
        """Decode all-intra access units ``[(bytes, offset, length, 1)]`` into one canvas.

        Returns ``[pictures, planes, height, width]`` uint8 pictures: a CUDA
        tensor ordered on the current stream with device output, otherwise an
        owned NumPy copy. The zero-latency canvas session is cached per canvas
        geometry on this decoder's owning thread. Only self-contained all-intra
        access units (schema-v2 packets) are accepted.
        """
        import torch

        self._initialize()
        units, pictures, nbytes = check_units(units)
        planes = _canvas_planes(pix_fmt)
        if nbytes > self.max_bitstream_bytes:
            raise CapacityError("NVDEC compressed input exceeds its admission limit")
        reservation = pictures * planes * height * width
        if reservation > self.max_output_bytes:
            raise CapacityError("NVDEC canvas output exceeds its admission limit")
        ledger = "peak_device_output_reservation" if self.device_output else "peak_host_output_reservation"
        self._stats[ledger] = max(self._stats[ledger], reservation)
        key = (width, height, pix_fmt)
        session = self._canvas_sessions.get(key)
        if session is None:
            session = CanvasSession(gpu_id=self.gpu_id, width=width, height=height, pix_fmt=pix_fmt, api=self._api)
            if len(self._canvas_sessions) == self.max_sessions:
                _, evicted = self._canvas_sessions.popitem(last=False)
                evicted.close()
                self._stats["evictions"] += 1
            self._canvas_sessions[key] = session
            self._stats["sessions_created"] += 1
            self._stats["session_create_seconds"] += session.create_seconds
        else:
            self._canvas_sessions.move_to_end(key)
            self._stats["session_hits"] += 1
        device = torch.device("cuda", self.gpu_id)
        canvas = torch.empty((pictures, planes, height, width), dtype=torch.uint8, device=device)
        started = time.perf_counter()
        success = False
        try:
            event = session.decode(units, canvas)
            torch.cuda.current_stream(device).wait_event(event)
            canvas.record_stream(session.stream)
            self._stats["stream_calls"] += 1
            self._stats["decoded_frames"] += pictures
            success = True
            return canvas if self.device_output else canvas.cpu().numpy()
        finally:
            self._stats["decode_seconds"] += time.perf_counter() - started
            if not success:
                self._stats["failures"] += 1
                failed = self._canvas_sessions.pop(key, None)
                if failed is not None:
                    failed.close()

    def close(self):
        if self._owner is not None and self._owner != (os.getpid(), threading.get_ident()):
            raise StateError("close NVDEC on its owning process/thread")
        for session in self._canvas_sessions.values():
            session.close()
        self._canvas_sessions.clear()
        self._sessions.clear()
        self._api = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise StateError("NVDEC decoder is closed")
        return self

    def __exit__(self, *_):
        self.close()


def unpack_device_frame(image, layout, columns):
    """Vectorized native channel-plane tiling, identical to the CPU cell mapping."""
    height, width = layout["height"], layout["width"]
    grid_columns, grid_rows = layout["grid"]
    planes = layout["planes"]
    cropped = image[:planes, :grid_rows * height, :grid_columns * width]
    tiled = cropped.reshape(planes, grid_rows, height, grid_columns, width)
    return tiled.permute(2, 4, 1, 3, 0).reshape(height, width, -1)[:, :, :columns]


def unpack_device_tile(canvas, first, rect, layout):
    """``[tokens, rank]`` uint8 coefficient pixels of one tile-map rectangle.

    ``canvas`` is a decoded ``[pictures, planes, height, width]`` uint8 tensor
    (any device); the tile's segment ``s`` frame ``f`` is picture
    ``first + s * frames + f`` at ``rect = (x, y, width, height)``. The
    channel order equals the CPU cell mapping (:func:`reelcache._hevc._cells`).
    """
    import torch

    x, y = rect[0], rect[1]
    frames, height, width, rank = (layout[name] for name in ("frames", "height", "width", "rank"))
    columns, rows = layout["grid"]
    planes, slots = layout["planes"], layout["channels_per_segment"]
    if canvas.shape[0] < first + frames * layout["segments"] or canvas.shape[1] < planes or (
        canvas.shape[2] < y + rows * height or canvas.shape[3] < x + columns * width
    ):
        raise IntegrityError("tile-map rectangle lies outside the decoded canvas")
    values = torch.empty((frames, height, width, rank), dtype=torch.uint8, device=canvas.device)
    for segment in range(layout["segments"]):
        start = first + segment * frames
        images = canvas[start:start + frames, :planes, y:y + rows * height, x:x + columns * width]
        tiled = images.reshape(frames, planes, rows, height, columns, width).permute(0, 3, 5, 2, 4, 1)
        lo = segment * slots
        count = min(slots, rank - lo)
        values[..., lo:lo + count] = tiled.reshape(frames, height, width, slots)[..., :count]
    return values.reshape(-1, rank)


def load_api():
    """Import PyNvVideoCodec or fail explicitly (there is never a software fallback)."""
    try:
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise UnsupportedError("NVDEC requires the optional PyNvVideoCodec dependency") from exc
    return nvc


def _canvas_planes(pix_fmt):
    if pix_fmt == "yuv444p":
        return 3
    if pix_fmt == "yuv420p":
        return 1
    raise UnsupportedError(f"NVDEC canvases support yuv444p and yuv420p, not {pix_fmt!r}")


def check_canvas(canvas, *, device, pix_fmt, pictures):
    """Validate a uint8 device canvas ``[P, planes, H, W]`` with contiguous rows."""
    import torch

    if (
        not isinstance(canvas, torch.Tensor) or canvas.dtype != torch.uint8 or canvas.ndim != 4
        or canvas.device != device or canvas.shape[1] != _canvas_planes(pix_fmt)
        or canvas.shape[0] < pictures or canvas.stride(3) != 1 or canvas.stride(2) < canvas.shape[3]
    ):
        raise ConfigurationError("NVDEC canvas must be uint8 [pictures, planes, height, width] on the session device")
    return int(canvas.shape[2]), int(canvas.shape[3])


_CAPS = {}
_CAPS_LOCK = threading.Lock()


def decoder_caps(api, gpu_id, pix_fmt):
    """Cached ``GetDecoderCaps`` for 8-bit HEVC in ``pix_fmt`` on one GPU (queried once per process)."""
    key = (id(api), gpu_id, pix_fmt)
    note_nvdec_use()
    with _CAPS_LOCK:
        if key not in _CAPS:
            chroma = api.cudaVideoChromaFormat.__members__["444" if pix_fmt == "yuv444p" else "420"]
            caps = api.GetDecoderCaps(gpu_id, api.cudaVideoCodec.HEVC, chroma, 8)
            _CAPS[key] = {name: caps[name] for name in caps}
        return dict(_CAPS[key])


def check_units(units):
    """Normalize decode units ``(data, offset, length, pictures)``; returns (units, pictures, bytes).

    A unit is a byte range of immutable ``bytes`` holding ``pictures`` complete
    all-intra access units; NVDEC canvas sessions accept one access unit per unit.
    """
    normalized, pictures, nbytes = [], 0, 0
    for unit in units:
        if not isinstance(unit, tuple) or len(unit) != 4:
            raise ConfigurationError("NVDEC units are (bytes, offset, length, pictures) tuples")
        data, offset, length, count = unit
        if not isinstance(data, bytes) or not data:
            raise ConfigurationError("NVDEC input must be nonempty immutable bytes")
        if any(type(value) is not int for value in (offset, length, count)) or count < 1:
            raise ConfigurationError("NVDEC unit offsets, lengths and picture counts must be integers")
        if offset < 0 or length < 1 or offset + length > len(data):
            raise IntegrityError("NVDEC access units lie outside their bitstream")
        normalized.append(unit)
        pictures += count
        nbytes += length
    if not normalized:
        raise ConfigurationError("an NVDEC job needs at least one access unit")
    return normalized, pictures, nbytes


class CanvasSession:
    """One zero-latency NVDEC session decoding all-intra access units of ONE canvas geometry.

    Raw Annex-B access units are fed without a demuxer (``PacketData`` with
    ``ENDOFPICTURE``), one per ``Decode`` call, and each must yield exactly one
    picture (measured: a multi-picture packet returns only its first picture in
    zero-latency mode). A session cannot switch picture geometry (measured on
    PyNvVideoCodec 2.2: neither EOS nor ENDOFSTREAM flushes reconfigure it), so
    it is bound to exactly ``width x height``; use one session per canvas
    geometry. The SDK creates the hardware decoder lazily at the first picture
    (tens of ms): :meth:`warm` it off the critical path. The decoder and every
    canvas copy run on this session's own CUDA stream, so reuse of the SDK's
    output surfaces is stream-ordered after the copies and no host
    synchronization is needed between pictures. ``decode`` returns a CUDA event
    recorded after the last copy. The canvas may be a normal or an inference
    tensor; its copies run in its mode (:func:`write_mode`). Thread-owned.
    Inter-coded or legacy version-1 streams use :class:`NVDECDecoder`; there is
    no software fallback.
    """

    def __init__(self, *, gpu_id, width, height, pix_fmt, api=None):
        import torch

        if type(gpu_id) is not int or gpu_id < 0:
            raise ConfigurationError("NVDEC gpu_id must be a nonnegative visible-device index")
        for value, name in ((width, "width"), (height, "height")):
            if type(value) is not int or not 16 <= value <= 8192:
                raise ConfigurationError(f"{name} must be an integer in [16, 8192]")
        self.planes = _canvas_planes(pix_fmt)
        self.gpu_id, self.pix_fmt, self.width, self.height = gpu_id, pix_fmt, width, height
        nvc = load_api() if api is None else api
        self._api = nvc
        if gpu_id >= torch.cuda.device_count():
            raise UnsupportedError("NVDEC device is not visible")
        self.device = torch.device("cuda", gpu_id)
        started = time.perf_counter()
        caps = decoder_caps(nvc, gpu_id, pix_fmt)
        caps_seconds = time.perf_counter() - started
        if not caps["supported"] or not (
            caps["width_min"] <= width <= caps["width_max"] and caps["height_min"] <= height <= caps["height_max"]
            and ((width + 15) // 16) * ((height + 15) // 16) <= caps["mb_num_max"]
        ):
            raise UnsupportedError(
                f"NVDEC on GPU {gpu_id} does not decode 8-bit {pix_fmt} HEVC at {width}x{height} "
                f"(supported {caps['width_min']}..{caps['width_max']} x {caps['height_min']}..{caps['height_max']})",
            )
        self.caps = {name: caps[name] for name in ("width_min", "height_min", "width_max", "height_max", "mb_num_max")}
        self._owner = threading.get_ident()
        started = time.perf_counter()
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        stream_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self._decoder = nvc.CreateDecoder(
            gpuid=gpu_id, codec=nvc.cudaVideoCodec.HEVC, usedevicememory=True,
            cudastream=self.stream.cuda_stream, outputColorType=nvc.OutputColorType.NATIVE,
            maxwidth=width, maxheight=height, latency=nvc.DisplayDecodeLatencyType.ZERO,
        )
        _SESSION_OWNERS.add(self)
        self.create_seconds = time.perf_counter() - started
        self.setup_seconds = {"caps": caps_seconds, "stream": stream_seconds, "create_decoder": self.create_seconds}
        self._end_of_picture = nvc.VideoPacketFlag.ENDOFPICTURE
        self._format = nvc.Pixel_Format.YUV444 if self.planes == 3 else nvc.Pixel_Format.NV12
        self._failed = None
        self.warm_seconds = None
        self.stats = {"pictures": 0, "jobs": 0, "calls": 0, "bitstream_bytes": 0, "decode_seconds": 0.0,
                      "setup_seconds": self.setup_seconds, "first_picture_seconds": None}

    @property
    def geometry(self):
        return self.width, self.height, self.pix_fmt

    def warm(self, unit):
        """Decode one access unit of this geometry to create the hardware decoder now."""
        import torch

        canvas = torch.empty((1, self.planes, self.height, self.width), dtype=torch.uint8, device=self.device)
        started = time.perf_counter()
        self.decode([unit], canvas).synchronize()
        self.warm_seconds = time.perf_counter() - started
        return self.warm_seconds

    def decode(self, units, canvas, first=0):
        """Decode access units ``[(bytes, offset, length, 1)]`` into ``canvas[first:]`` in order."""
        import torch

        if threading.get_ident() != self._owner:
            raise StateError("an NVDEC canvas session must stay on its owning thread")
        if self._decoder is None:
            raise StateError("the NVDEC canvas session is closed")
        if self._failed is not None:
            raise StateError(f"the NVDEC canvas session failed earlier ({self._failed}); create a new session")
        units, pictures, nbytes = check_units(units)
        if pictures != len(units):
            raise ConfigurationError("zero-latency NVDEC canvas sessions decode exactly one access unit per unit")
        if type(first) is not int or first < 0:
            raise ConfigurationError("the first canvas picture must be a nonnegative integer")
        height, width = check_canvas(canvas, device=self.device, pix_fmt=self.pix_fmt, pictures=first + pictures)
        if (width, height) != (self.width, self.height):
            raise UnsupportedError(f"NVDEC session {self.width}x{self.height} cannot decode a {width}x{height} "
                                   "canvas; sessions are bound to one geometry")
        rows = 3 * height if self.planes == 3 else height * 3 // 2
        nvc = self._api
        started = time.perf_counter()
        cursor = first
        with write_mode(canvas), torch.cuda.stream(self.stream):
            for data, offset, length, count in units:
                packet = nvc.PacketData()
                packet.bsl_data = np.frombuffer(data, dtype=np.uint8).ctypes.data + offset
                packet.bsl = length
                packet.decode_flag = self._end_of_picture
                with span("nvdec.decode"):
                    try:
                        frames = self._decoder.Decode(packet)
                    except Exception as exc:  # noqa: BLE001 - SDK errors become explicit integrity failures
                        self._failed = f"{type(exc).__name__}: {str(exc).strip()}"
                        raise IntegrityError(f"NVDEC rejected an access unit: {self._failed}") from exc
                self.stats["calls"] += 1
                if self.stats["first_picture_seconds"] is None:
                    self.stats["first_picture_seconds"] = time.perf_counter() - started
                if len(frames) != count:
                    self._failed = "picture count"
                    raise IntegrityError(f"zero-latency NVDEC returned {len(frames)} pictures for one "
                                         "all-intra access unit")
                if self._decoder.GetWidth() != width or self._decoder.GetHeight() != height:
                    self._failed = "geometry"
                    raise IntegrityError("NVDEC picture geometry differs from its canvas")
                self._decoder.WaitOnCUStream(self.stream.cuda_stream)
                for frame in frames:
                    if (
                        frame.format != self._format or tuple(frame.shape) != (rows, width)
                        or tuple(frame.strides) != (width, 1)
                    ):
                        self._failed = "format"
                        raise IntegrityError("NVDEC picture format/pitch differs from its canvas")
                    value = torch.from_dlpack(frame)
                    if value.device != self.device or value.dtype != torch.uint8 or tuple(value.shape) != (rows, width):
                        self._failed = "dlpack"
                        raise IntegrityError("NVDEC DLPack surface differs from its validated frame")
                    target = canvas[cursor]
                    if self.planes == 3:
                        target.copy_(value.view(3, height, width), non_blocking=True)
                    else:
                        target[0].copy_(value[:height], non_blocking=True)
                    cursor += 1
            event = torch.cuda.Event()
            event.record(self.stream)
        self.stats["decode_seconds"] += time.perf_counter() - started
        self.stats["pictures"] += pictures
        self.stats["jobs"] += 1
        self.stats["bitstream_bytes"] += nbytes
        return event

    def close(self):
        if threading.get_ident() != self._owner:
            raise StateError("close an NVDEC canvas session on its owning thread")
        if self._decoder is not None:
            self.stream.synchronize()
            self._decoder = None
