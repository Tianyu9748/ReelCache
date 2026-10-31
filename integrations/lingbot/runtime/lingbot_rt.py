"""LingBot-World-Fast runtime helpers for ReelCache (official code, commit a43bec7). API doc: API.md.

Everything here wraps the unmodified official ``wan.WanI2VFast`` pipeline (``wan/image2video_fast.py``):
paths and constants, environment/distributed setup, pipeline construction, the static-camera action path,
start-image preparation, the official uint8 video conversion, and ``ChunkKV`` -- a hook adapter that times
every DiT forward, captures a chunk's pre-RoPE K (``norm_k`` output) and V from that chunk's clean t=0 pass,
re-applies RoPE at the chunk's original positions and injects replacement K/V into the self-attention cache
right after that pass, and exports/imports the raw cache. Works with 1 GPU and with Ulysses SP (torchrun).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------------------- paths / constants
REPO = Path(os.environ.get("LB_REPO", "/models/lingbot-world"))  # official repo @ REPO_COMMIT (placeholder default)
REPO_COMMIT = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
# The directory name must contain 'cam': WanI2VFast derives control_type from the checkpoint path.
CKPT_DIR = REPO / "lingbot-world-base-cam"
FAST_SUBFOLDER = "lingbot_world_fast"
HF_REVISIONS = {"robbyant/lingbot-world-fast": "b7a1db2da9bd3ba675ee01c228f41db0309b981e",
                "robbyant/lingbot-world-base-cam": "6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd"}
RT_ROOT = Path(__file__).resolve().parent
LINGBOT_ROOT = RT_ROOT.parent
WEIGHTS_MANIFEST = LINGBOT_ROOT / "setup" / "weights_manifest.json"
VENV_PYTHON = Path(os.environ.get("LB_PY", sys.executable))
WORK_ROOT = Path(os.environ.get("LB_WORK", str(Path.home() / ".cache" / "reelcache" / "lingbot")))  # large intermediates
STATIC_CAMERA_DIR = WORK_ROOT / "camera" / "static_961"
EXAMPLE_INTRINSICS = REPO / "examples" / "03" / "intrinsics.npy"

TASK = "i2v-A14B"
SIZES = {"480p": (480, 832), "720p": (720, 1280)}               # (height, width)
FPS = 16
CHUNK_LATENTS = 3
LAYERS, HEADS, HEAD_DIM = 40, 40, 128
CHANNELS = HEADS * HEAD_DIM
KV_BYTES_PER_TOKEN = LAYERS * 2 * CHANNELS * 2                     # 40 layers x (K,V) x 5120 x bf16
DEFAULT_SHIFT = 10.0            # generate_fast.py: --sample_shift defaults to cfg.sample_shift = 10.0
TIMESTEPS_INDEX = (0, 179, 358, 679)


def tokens_per_latent(height: int, width: int) -> int:
    return (height // 16) * (width // 16)


def latent_frames(frame_num: int) -> int:
    """generate(): lat_f = (F-1)//4 + 1, truncated to a multiple of the chunk size."""
    lat = (frame_num - 1) // 4 + 1
    return lat - lat % CHUNK_LATENTS


def rgb_frames(lat_f: int) -> int:
    return (lat_f - 1) * 4 + 1


def chunk_rgb_range(chunk: int) -> tuple[int, int]:
    """Decoded RGB frames whose latents belong to chunk ``chunk`` (chunk 0 -> frames 0..8, then 12 per chunk)."""
    return (0, 9) if chunk == 0 else (12 * chunk - 3, 12 * chunk + 9)


def kv_bytes_formula(height: int, width: int, latents: int) -> int:
    return tokens_per_latent(height, width) * latents * KV_BYTES_PER_TOKEN


def official_latent_hw(max_area: int, height: int, width: int) -> tuple[int, int]:
    """generate()'s own lat_h/lat_w for an input image of size width x height (same float expression)."""
    ar = height / width
    lat_h = round(np.sqrt(max_area * ar) // 8 // 2 * 2)
    lat_w = round(np.sqrt(max_area / ar) // 8 // 2 * 2)
    return int(lat_h), int(lat_w)


def exact_max_area(height: int, width: int) -> int:
    """max_area that makes generate() produce exactly height x width for a height x width image.
    The CLI value h*w works for 720x1280 but NOT for 480x832: 399360*(480/832) rounds to just below 230400, the floor
    gives lat_h 58 and the official 480*832 setting produces 464x832 (1508 tokens/latent). h*w+1 yields 60x104."""
    for area in (height * width, height * width + 1):
        if official_latent_hw(area, height, width) == (height // 8, width // 8):
            return area
    raise ValueError(f"no max_area gives {height}x{width}")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(t) -> str:
    import torch
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- environment
def prepare_env(scratch: str | Path | None = None) -> dict:
    """Point every cache/temp dir at node-local scratch (`LB_SCRATCH`, default as in env.sh:
    `${TMPDIR:-/tmp}/reelcache-lingbot-<Slurm job id or pid>`), never a shared home or network filesystem; offline HF."""
    job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    scratch = Path(scratch or os.environ.get("LB_SCRATCH") or Path(os.environ.get("TMPDIR") or "/tmp") / f"reelcache-lingbot-{job}")
    cache = scratch / "cache"
    mapping = {"TMPDIR": cache / "tmp", "XDG_CACHE_HOME": cache / "xdg", "HF_HOME": cache / "hf",
               "TORCH_HOME": cache / "torch", "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
               "MPLCONFIGDIR": cache / "matplotlib", "NUMBA_CACHE_DIR": cache / "numba"}
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    os.environ.setdefault("TRITON_CACHE_DIR", str(WORK_ROOT / "cache" / "triton"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return {k: os.environ[k] for k in [*mapping, "TRITON_CACHE_DIR"]}


def init_distributed(timeout_minutes: int = 45) -> tuple[int, int, int]:
    """(rank, world_size, local_rank). With WORLD_SIZE>1 (torchrun) also sets up the Ulysses SP group."""
    import datetime
    import torch
    import torch.distributed as dist
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=timeout_minutes),
                                device_id=torch.device(f"cuda:{local}"))
    return rank, world, local


def build_pipe(*, local_attn_size: int = -1, sink_size: int = 0, dit_fsdp: bool = False, t5_fsdp: bool = False):
    """Official WanI2VFast (i2v-A14B). SP is enabled iff WORLD_SIZE > 1 (ulysses_size = world size)."""
    import torch.distributed as dist
    rank, world, local = init_distributed()
    prepare_env()
    import wan
    from wan.configs import WAN_CONFIGS
    cfg = WAN_CONFIGS[TASK]
    if world > 1:
        assert cfg.num_heads % world == 0, "40 heads must divide the SP size"
        from wan.distributed.util import init_distributed_group
        init_distributed_group()
    pipe = wan.WanI2VFast(config=cfg, checkpoint_dir=str(CKPT_DIR), device_id=local, rank=rank,
                          t5_fsdp=t5_fsdp, dit_fsdp=dit_fsdp, use_sp=world > 1, t5_cpu=False,
                          convert_model_dtype=False, local_attn_size=local_attn_size, sink_size=sink_size)
    pipe._lb_world = world
    if dist.is_initialized():
        dist.barrier()
    return pipe


def set_attention_window(pipe, local_attn_size: int = -1, sink_size: int = 0):
    """Switch between full history (-1) and a rolling window of ``local_attn_size`` latent frames (incl. the
    current chunk) without reloading weights (same attributes the constructor sets)."""
    pipe.local_attn_size, pipe.sink_size = local_attn_size, sink_size
    model = dit_module(pipe)
    model.local_attn_size = local_attn_size
    for blk in model.blocks:
        blk.local_attn_size = local_attn_size
        blk.self_attn.local_attn_size = local_attn_size
        blk.self_attn.sink_size = sink_size


def dit_module(pipe):
    model = pipe.model
    return getattr(model, "module", model)       # FSDP wrapper -> inner module


def encode_prompt(pipe, prompt: str):
    """generate()'s own T5 path with offload: T5 -> GPU, encode, T5 -> CPU; result cached in pipe._t5_cache so
    generate() never moves T5 to the GPU (saves 11.4 GB). Identical to generate(..., offload_model=True)'s T5."""
    import torch
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if key not in pipe._t5_cache:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        pipe._t5_cache[key] = context
    return pipe._t5_cache[key]


def switch_prompt_note() -> str:
    return ("Prompt switch inside one session: encode the new prompt (encode_prompt) and run the next DiT forward with "
            "cross_attn_first_call=True (rebuilds only the cross-attention cache); the self-attention KV is untouched.")


# ------------------------------------------------------------------------------------------- camera / image
def write_static_action_path(out_dir: str | Path = STATIC_CAMERA_DIR, n_frames: int = 961) -> dict:
    """Static camera: identity OpenCV c2w for every frame + the repo example-03 intrinsics row (given in the
    480x832 frame; generate() rescales intrinsics from 480x832 to the target h x w itself, so passing
    pre-scaled intrinsics would double-scale). n_frames bounds frame_num (generate() truncates to it)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    poses = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    k_row = np.load(EXAMPLE_INTRINSICS)[0].astype(np.float32)
    intr = np.tile(k_row, (n_frames, 1))
    np.save(out / "poses.npy", poses)
    np.save(out / "intrinsics.npy", intr)
    rec = {"kind": "static", "n_frames": n_frames, "intrinsics_480x832": k_row.tolist(),
           "intrinsics_source": str(EXAMPLE_INTRINSICS), "intrinsics_source_sha256": sha256_file(EXAMPLE_INTRINSICS),
           "poses_sha256": sha256_file(out / "poses.npy"), "intrinsics_sha256": sha256_file(out / "intrinsics.npy"),
           "path": str(out)}
    (out / "camera.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def write_action_string_path(out_dir: str | Path, action_string: str) -> dict:
    """Camera from the repo's keyboard DSL with its own converter (wan/utils/wasd_ijkl_to_c2ws.py, the act2cam path of
    image2video.py): action_string -> per-frame WASD/IJKL -> generate_and_save_trajectory (w/s 0.05 per frame along
    the flattened forward axis, j/l yaw 2 deg per frame) -> c2ws (frames + 1 OpenCV c2w, first = identity).
    Intrinsics: the example-03 row in the 480x832 frame (generate() rescales it to h x w)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import importlib.util
    spec = importlib.util.spec_from_file_location("wasd_ijkl_to_c2ws", REPO / "wan/utils/wasd_ijkl_to_c2ws.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)                    # torch + numpy only; avoids importing the wan package
    wasd, ijkl, frames = conv.action_string_to_wasd_ijkl(action_string)
    keys = conv.wasd_array_to_frame_keys(wasd, ijkl)
    c2ws = np.array(conv.generate_and_save_trajectory(keys), dtype=np.float32)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k_row = np.load(EXAMPLE_INTRINSICS)[0].astype(np.float32)
    np.save(out / "poses.npy", c2ws)
    np.save(out / "intrinsics.npy", np.tile(k_row, (len(c2ws), 1)))
    np.save(out / "wasd_action.npy", wasd)
    np.save(out / "ijkl_action.npy", ijkl)
    rec = {"kind": "action_string", "action_string": action_string, "frames": int(frames), "n_poses": int(len(c2ws)),
           "converter": str(REPO / "wan/utils/wasd_ijkl_to_c2ws.py"),
           "converter_sha256": sha256_file(REPO / "wan/utils/wasd_ijkl_to_c2ws.py"),
           "intrinsics_480x832": k_row.tolist(), "path": str(out),
           "poses_sha256": sha256_file(out / "poses.npy"), "intrinsics_sha256": sha256_file(out / "intrinsics.npy")}
    (out / "camera.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def effective_intrinsics(height: int, width: int, k_row_480x832) -> list[float]:
    """The intrinsics generate() actually uses at h x w (official get_Ks_transformed from 480x832, no crop)."""
    fx, fy, cx, cy = [float(v) for v in k_row_480x832]
    sx, sy = width / 832, height / 480
    return [fx * sx, fy * sy, cx * sx, cy * sy]


def write_dolly_action_path(out_dir: str | Path, n_frames: int = 961, step: float = 0.01) -> dict:
    """Fallback only (if the static camera freezes the video): identity rotation, constant +z (forward) c2w
    translation. generate() normalises translations by their max norm, so any constant speed becomes a unit
    framewise step; ``step`` is therefore irrelevant to the model input."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    poses = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    poses[:, 2, 3] = step * np.arange(n_frames, dtype=np.float32)
    k_row = np.load(EXAMPLE_INTRINSICS)[0].astype(np.float32)
    np.save(out / "poses.npy", poses)
    np.save(out / "intrinsics.npy", np.tile(k_row, (n_frames, 1)))
    rec = {"kind": "dolly_forward", "n_frames": n_frames, "step": step, "path": str(out),
           "poses_sha256": sha256_file(out / "poses.npy"), "intrinsics_sha256": sha256_file(out / "intrinsics.npy")}
    (out / "camera.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def center_crop_resize(img, height: int, width: int):
    """Center-crop a PIL image to height:width and resize (LANCZOS) to exactly width x height, so generate()'s
    lat_h/lat_w are exactly height/8, width/8 (480x832 -> 60x104 latents, 720x1280 -> 90x160)."""
    from PIL import Image
    W, H = img.size
    target = width / height
    if W / H > target:
        new_w = int(round(H * target))
        x0 = (W - new_w) // 2
        box = (x0, 0, x0 + new_w, H)
    else:
        new_h = int(round(W / target))
        y0 = (H - new_h) // 2
        box = (0, y0, W, y0 + new_h)
    return img.convert("RGB").crop(box).resize((width, height), Image.LANCZOS), box


def video_to_uint8(video) -> np.ndarray:
    """[3, F, H, W] float in [-1,1] -> uint8 [F, H, W, 3] exactly as wan.utils.utils.save_video (clamp,
    make_grid normalize to [0,1], *255, truncating cast)."""
    import torch
    t = video.clamp(-1, 1).sub(-1).div(2.0)
    return (t * 255).type(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu().numpy()


# ------------------------------------------------------------------------------------------------ KV adapter
class ChunkKV:
    """Hook adapter around ``pipe.model`` (non-invasive; ``remove()`` restores it).

    * ``timeline``: one record per DiT forward (chunk, step, t, seconds, cache end, memory).
    * ``capture_chunks``: chunks whose clean (t == 0) pass is captured: pre-RoPE K = ``self_attn.norm_k`` output,
      V = the cache rows written by that pass; both full heads [N, 5120] (bf16, CPU), gathered over SP ranks.
      ``rope_check[chunk]`` records whether re-RoPE(captured K) equals the cached post-RoPE K bitwise.
    * ``inject``: {chunk: {("K", layer): pre-RoPE [N, 5120], ("V", layer): [N, 5120]}} -- written into the cache
      right after that chunk's clean pass (K re-roped at the chunk's original frame offset with the model's own
      causal_rope_apply; each SP rank writes its own head slice). Later chunks then attend to it.
    * ``export_chunks``: {chunk}: raw per-rank cache snapshot right after that chunk's clean pass (export_kv).
    * ``alloc``: bytes of the self-attention / cross-attention caches allocated by the last generate().
    Every model call must use keyword arguments (as generate() does) or positional ones matching forward().
    """

    def __init__(self, pipe):
        import torch.distributed as dist
        self.pipe = pipe
        self.model = dit_module(pipe)
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.sp = self.world > 1
        if self.sp:
            from wan.distributed.sequence_parallel import causal_rope_apply
        else:
            from wan.modules.model_fast import causal_rope_apply
        self._rope = causal_rope_apply
        self.capture_chunks: set = set()
        self.inject: dict = {}
        self.export_chunks: set = set()
        self.captured: dict = {}
        self.rope_check: dict = {}
        self.exports: dict = {}
        self.timeline: list = []
        self.alloc: dict = {}
        self.last_latents = None
        self.capture_latents = True
        self.keep_captured = True
        # transform(chunk, captured) -> {("K", l): pre-RoPE [N,5120], ("V", l): [N,5120]} for chunks in
        # transform_chunks (which are captured automatically); applied right after that chunk's clean pass.
        # With SP and transform_broadcast, only rank 0's return value is used (broadcast to every rank).
        self.transform = None
        self.transform_chunks: set = set()
        self.transform_broadcast = True
        self.transform_log: list = []
        self._capturing = False
        self._k_local: dict = {}
        self._chunk_t0 = None
        self._orig_forward = self.model.forward
        self._sig = inspect.signature(self._orig_forward)
        self.model.forward = self._forward
        self._hooks = [blk.self_attn.norm_k.register_forward_hook(self._k_hook(i))
                       for i, blk in enumerate(self.model.blocks)]
        self._orig_decode = None
        if self.rank == 0:
            self._orig_decode = pipe.vae.decode
            pipe.vae.decode = self._decode

    # ---- lifecycle
    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if "forward" in self.model.__dict__ and self.model.__dict__["forward"] == self._forward:
            if self.sp:
                self.model.forward = self._orig_forward        # SP installs forward as an instance attribute
            else:
                del self.model.forward
        if self._orig_decode is not None:
            self.pipe.vae.decode = self._orig_decode
            self._orig_decode = None

    def reset(self):
        self.capture_chunks, self.inject, self.export_chunks = set(), {}, set()
        self.captured, self.rope_check, self.exports, self.timeline, self.alloc = {}, {}, {}, [], {}
        self.last_latents = None
        self.transform, self.transform_chunks, self.transform_log = None, set(), []

    # ---- hooks
    def _k_hook(self, layer):
        def hook(_module, _inputs, output):
            if self._capturing:
                self._k_local[layer] = output.detach().clone()      # [B, s_local, 5120] (SP: sequence shard)
        return hook

    def _decode(self, zs):
        if self.capture_latents:
            self.last_latents = zs[0].detach().cpu()
        return self._orig_decode(zs)

    # ---- forward wrapper
    def _forward(self, *args, **kwargs):
        import torch
        a = self._sig.bind(*args, **kwargs).arguments
        t, kv, cs = a["t"], a.get("kv_cache"), int(a.get("current_start", 0))
        x0 = a["x"][0]
        f, hh, ww = int(x0.shape[1]), int(x0.shape[2]) // 2, int(x0.shape[3]) // 2
        fs = a.get("frame_seqlen") or hh * ww
        n_tok = f * hh * ww
        chunk = cs // (fs * CHUNK_LATENTS)
        start_frame = cs // fs                       # the model's own current_start_frame (absolute position)
        clean = bool((t == 0).all().item())
        first_of_chunk = not self.timeline or self.timeline[-1]["clean"] or self.timeline[-1]["current_start"] != cs
        if first_of_chunk and cs == 0 and kv is not None:
            self._record_alloc(kv, a.get("crossattn_cache"))
        torch.cuda.synchronize()
        if first_of_chunk:
            torch.cuda.reset_peak_memory_stats()
            self._chunk_t0 = time.perf_counter()
        capture = clean and (chunk in self.capture_chunks or (self.transform is not None and chunk in self.transform_chunks))
        if capture:
            self._k_local.clear()
            self._capturing = True
        t0 = time.perf_counter()
        try:
            out = self._orig_forward(*args, **kwargs)
        finally:
            self._capturing = False
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rec = {"chunk": chunk, "clean": clean, "t": float(t.flatten()[0].item()), "current_start": cs,
               "tokens": n_tok, "forward_seconds": t1 - t0}
        if kv is not None:
            rolling = self.model.blocks[0].self_attn.local_attn_size != -1
            rec["cache_local_end"] = int(kv[0]["local_end_index"].item()) if rolling else cs + n_tok
        if clean:
            free, total = torch.cuda.mem_get_info()
            rec.update(chunk_seconds=t1 - self._chunk_t0, chunk_peak_allocated=int(torch.cuda.max_memory_allocated()),
                       chunk_peak_reserved=int(torch.cuda.max_memory_reserved()), device_used_bytes=int(total - free))
        self.timeline.append(rec)
        if clean and kv is not None:
            le = rec["cache_local_end"]
            ls = le - n_tok
            if capture:
                self._finish_capture(chunk, kv, ls, le, n_tok, (f, hh, ww), start_frame)
            if self.transform is not None and chunk in self.transform_chunks:
                t2 = time.perf_counter()
                # generate() runs the DiT under torch.amp.autocast(bf16); transforms (factorization/codec arithmetic)
                # must not inherit it (an FP32 matmul would silently become bf16).
                with torch.autocast(device_type="cuda", enabled=False):
                    repl = self.transform(chunk, self.captured[chunk])
                if self.sp and self.transform_broadcast:
                    repl = self._broadcast_values(repl, n_tok)
                t3 = time.perf_counter()
                self._write_values(kv, ls, le, n_tok, (f, hh, ww), start_frame, repl)
                torch.cuda.synchronize()
                self.transform_log.append({"chunk": chunk, "transform_seconds": t3 - t2,
                                           "inject_seconds": time.perf_counter() - t3, "rows": [ls, le]})
                if not self.keep_captured:
                    self.captured.pop(chunk, None)
                del repl
            if chunk in self.inject:
                self._apply_inject(chunk, kv, ls, le, n_tok, (f, hh, ww), start_frame)
            if chunk in self.export_chunks:
                self.exports[chunk] = export_kv(kv)
        return out

    def _record_alloc(self, kv, cross):
        import torch
        import torch.distributed as dist
        self_bytes = sum(d["k"].numel() * d["k"].element_size() + d["v"].numel() * d["v"].element_size() for d in kv)
        cross_bytes = sum(d["k"].numel() * d["k"].element_size() + d["v"].numel() * d["v"].element_size()
                          for d in (cross or []))
        tot = torch.tensor([self_bytes, cross_bytes], dtype=torch.int64, device="cuda")
        if self.sp:
            dist.all_reduce(tot)
        self.alloc = {"self_attn_bytes_this_rank": int(self_bytes), "cross_attn_bytes_this_rank": int(cross_bytes),
                      "self_attn_bytes_all_ranks": int(tot[0]), "cross_attn_bytes_all_ranks": int(tot[1]),
                      "self_attn_shape_this_rank": list(kv[0]["k"].shape), "dtype": str(kv[0]["k"].dtype),
                      "layers": len(kv), "world_size": self.world}

    # ---- capture / inject
    def _head_slice(self):
        hl = HEADS // self.world
        return slice(self.rank * hl, (self.rank + 1) * hl)

    def _reroped(self, k_full, grid, start_frame, like):
        """k_full [N, 5120] pre-RoPE -> post-RoPE [1, N, 40, 128] with the model's own causal_rope_apply."""
        import torch
        n = k_full.shape[0]
        gs = torch.tensor([list(grid)], dtype=torch.long)
        x = k_full.to(device=like.device, dtype=like.dtype).view(1, n, HEADS, HEAD_DIM)
        return self._rope(x, gs, self.model.freqs, start_frame=start_frame).type_as(like)

    def _finish_capture(self, chunk, kv, ls, le, n_tok, grid, start_frame):
        import torch
        import torch.distributed as dist
        hs = self._head_slice()
        ks, vs, exact, max_abs = {}, {}, 0, 0.0
        for layer in range(len(kv)):
            k_loc = self._k_local.pop(layer)[0]                        # [s_local, 5120]
            if self.sp:
                parts = [torch.empty_like(k_loc) for _ in range(self.world)]
                dist.all_gather(parts, k_loc.contiguous())
                k_full = torch.cat(parts, dim=0)[:n_tok]
                v_loc = kv[layer]["v"][0, ls:le].contiguous()         # [N, 40/sp, 128]
                vparts = [torch.empty_like(v_loc) for _ in range(self.world)]
                dist.all_gather(vparts, v_loc)
                v_full = torch.cat(vparts, dim=1).reshape(n_tok, CHANNELS)
            else:
                k_full = k_loc[:n_tok]
                v_full = kv[layer]["v"][0, ls:le].reshape(n_tok, CHANNELS)
            cache_k = kv[layer]["k"][0, ls:le]
            rer = self._reroped(k_full, grid, start_frame, cache_k)[0][:, hs]
            exact += int(torch.equal(rer, cache_k))
            max_abs = max(max_abs, float((rer.float() - cache_k.float()).abs().max().item()))
            ks[("K", layer)] = k_full.reshape(n_tok, CHANNELS).detach().to("cpu", copy=True)
            vs[("V", layer)] = v_full.detach().to("cpu", copy=True)
        cnt = torch.tensor([exact], device="cuda")
        if self.sp:
            dist.all_reduce(cnt)
        self.captured[chunk] = {**ks, **vs}
        self.rope_check[chunk] = {"layers": len(kv), "exact_layer_ranks": int(cnt.item()),
                                  "expected_layer_ranks": len(kv) * self.world, "max_abs_this_rank": max_abs,
                                  "all_exact": int(cnt.item()) == len(kv) * self.world,
                                  "cache_rows": [ls, le], "start_frame": start_frame}

    def _apply_inject(self, chunk, kv, ls, le, n_tok, grid, start_frame):
        self._write_values(kv, ls, le, n_tok, grid, start_frame, self.inject[chunk])

    def _broadcast_values(self, values, n_tok):
        """Rank 0's {(target, layer): [N, 5120]} to every rank (fixed key order; CPU tensors returned)."""
        import torch
        import torch.distributed as dist
        box = [sorted(values) if self.rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        out = {}
        for key in box[0]:
            if self.rank == 0:
                buf = values[key].to(device="cuda", dtype=torch.bfloat16).contiguous()
            else:
                buf = torch.empty((n_tok, CHANNELS), dtype=torch.bfloat16, device="cuda")
            dist.broadcast(buf, src=0)
            out[key] = buf.to("cpu", copy=True)
            del buf
        return out

    def _write_values(self, kv, ls, le, n_tok, grid, start_frame, values):
        hs = self._head_slice()
        for (target, layer), x in values.items():
            assert tuple(x.shape) == (n_tok, CHANNELS), (target, layer, tuple(x.shape))
            if target == "K":
                dst = kv[layer]["k"]
                dst[:, ls:le] = self._reroped(x, grid, start_frame, dst)[:, :, hs]
            elif target == "V":
                dst = kv[layer]["v"]
                dst[:, ls:le] = x.to(device=dst.device, dtype=dst.dtype).view(1, n_tok, HEADS, HEAD_DIM)[:, :, hs]
            else:
                raise ValueError(target)

    # ---- timing summary
    def chunk_summary(self) -> list[dict]:
        """One row per chunk of the timeline (call reset() before each generate())."""
        rows, fw = [], []
        for r in self.timeline:
            fw.append(r["forward_seconds"])
            if r["clean"]:
                rows.append({"chunk": r["chunk"], "chunk_seconds": r["chunk_seconds"], "forwards": len(fw),
                             "forward_seconds_sum": sum(fw), "cache_rows_after": r.get("cache_local_end"),
                             "peak_allocated_bytes": r["chunk_peak_allocated"],
                             "peak_reserved_bytes": r["chunk_peak_reserved"], "device_used_bytes": r["device_used_bytes"]})
                fw = []
        return rows


# --------------------------------------------------------------------------------------------- export/import
def export_kv(kv_cache, rows: tuple[int, int] | None = None) -> dict:
    """Raw per-rank snapshot of the self-attention cache (post-RoPE K and V, cache dtype) to CPU.
    ``rows`` = cache row range (default: everything written so far, [0, local_end))."""
    end = int(kv_cache[0]["local_end_index"].item())
    a, b = rows if rows is not None else (0, end)
    return {"rows": [a, b], "global_end_index": int(kv_cache[0]["global_end_index"].item()), "local_end_index": end,
            "k": [d["k"][:, a:b].detach().to("cpu", copy=True) for d in kv_cache],
            "v": [d["v"][:, a:b].detach().to("cpu", copy=True) for d in kv_cache]}


def import_kv(kv_cache, state: dict, set_indices: bool = True):
    """Inverse of export_kv on the same rank/geometry (copies rows back; optionally restores both indices)."""
    a, b = state["rows"]
    for d, k, v in zip(kv_cache, state["k"], state["v"]):
        d["k"][:, a:b].copy_(k.to(d["k"].device, non_blocking=True))
        d["v"][:, a:b].copy_(v.to(d["v"].device, non_blocking=True))
        if set_indices:
            d["global_end_index"].fill_(state["global_end_index"])
            d["local_end_index"].fill_(state["local_end_index"])


# ------------------------------------------------------------------------------------------------- generate
class _SkipDecode:
    """Replace pipe.vae.decode by a stub (timing/memory profiles that do not need pixels)."""

    def __init__(self, pipe, enabled: bool):
        self.pipe, self.enabled, self.saved = pipe, enabled, None

    def __enter__(self):
        if self.enabled:
            self.saved = self.pipe.vae.decode
            self.pipe.vae.decode = lambda zs: [zs[0].new_zeros(1)]
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.pipe.vae.decode = self.saved


def generate(pipe, prompt: str, image, *, height: int, width: int, frame_num: int = 81, seed: int = 0,
             action_path: str | Path = STATIC_CAMERA_DIR, shift: float = DEFAULT_SHIFT, skip_decode: bool = False,
             max_area: int | None = None):
    """One native WanI2VFast.generate() call (chunk_size 3, timesteps [0,179,358,679], offload_model=False so the
    DiT stays resident). ``image`` must already be width x height (center_crop_resize); max_area defaults to
    exact_max_area(height, width) so the output is exactly height x width (the CLI's 480*832 would give 464x832).
    Returns (video or None, wall seconds); video = [3, F, H, W] float in [-1,1] on rank 0, None elsewhere."""
    import torch
    import torch.distributed as dist
    encode_prompt(pipe, prompt)
    assert image.size == (width, height), f"image {image.size} != {(width, height)}"
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with _SkipDecode(pipe, skip_decode):
        video = pipe.generate(prompt, image, action_path=str(action_path), chunk_size=CHUNK_LATENTS,
                              max_area=max_area or exact_max_area(height, width), frame_num=frame_num, shift=shift,
                              seed=seed, offload_model=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return (None if skip_decode else video), seconds
