"""LingBot-World-Fast chunk-step model (lb_wave_v1).

The official ``WanI2VFast.generate()`` loop is run one chunk at a time against ONE persistent self-attention cache
shared by every session of the run. The per-session conditioning is prepared once at start-up and cached:
- the T5 context of the session's initial prompt;
- the image conditioning ``y`` up to the session's deepest planned chunk;
- the static camera's Plucker embedding.

A request's window is installed into the cache (raw rows, a replayed clean pass, or rows a ReelCache restore writes
through :meth:`destination`), then the 4 denoising steps and the clean pass of the new chunk run, followed by the
decode.

* :class:`LingBotModel`: the real model (``lingbot_rt.build_pipe``; GPU).
* :class:`FakeModel`: a CPU stand-in with the same interface and the same data dependencies:
  - a chunk's KV depends on its x0 and on its attended history;
  - x0 depends on the noise and the attended history;
  so a wrong install, replay or window changes the output (tests).

Cache bookkeeping mirrors ``CausalWanSelfAttention.forward`` (``model_fast.py``):
- installs append a chunk at ``local_end`` with the same sink-aware shift-left eviction;
- ``global_end`` / ``local_end`` are set as the model sets them;
- an installed window therefore has the official rolling layout (gate G2).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

LINGBOT_RT = Path(os.environ.get("REELCACHE_LINGBOT_RT", str(Path(__file__).resolve().parents[1] / "runtime")))
LAYERS, HEADS, HEAD_DIM = 40, 40, 128
CHANNELS = HEADS * HEAD_DIM
CHUNK_LATENTS = 3
LATENT_CHANNELS = 16
TIMESTEPS_INDEX = (0, 179, 358, 679)
SHIFT = 10.0
RESOLUTIONS = {"480x832": (480, 832), "720x1280": (720, 1280)}


def latent_grid(resolution: str) -> tuple[int, int]:
    """``(lat_h, lat_w)`` exactly as ``WanI2VFast.generate()`` derives them from an image of the nominal size and
    ``max_area = h * w`` (float arithmetic included): 480x832 -> (58, 104), i.e. 464x832 frames (sqrt(399360 *
    480/832) = 479.99999...); 720x1280 -> (90, 160). lingbot-runner's Phase-0 smoke shows 4,524 cache rows per 480P
    chunk."""
    h, w = RESOLUTIONS[resolution]
    aspect = h / w
    lat_h = round(np.sqrt(h * w * aspect) // 8 // 2 * 2)
    lat_w = round(np.sqrt(h * w / aspect) // 8 // 2 * 2)
    return int(lat_h), int(lat_w)


def own_device_first(env: Mapping[str, str], rank: int, world: int) -> dict[str, str]:
    """SP rank ``rank``'s CUDA environment: its own GPU first in ``CUDA_VISIBLE_DEVICES`` (devices[rank:] +
    devices[:rank]) and ``LOCAL_RANK=0``, so device 0 of the process is the rank's GPU. Apply before CUDA starts.

    PyNvVideoCodec 2.2 opens an NVENC session on device 0 unless ``gpu_id`` is passed (ReelCache's NVENCEncoder passes
    none), and its capability probe (PyNvEncoderCaps, run by every CreateEncoder) creates a fresh, unchecked CUDA
    context on device 0. With the job-wide order, rank 1's encoder sessions and probes ran on rank 0's GPU, the GPU
    that also runs the VAE decode (smoke 4507402: nvenc_device_bytes 0 on cuda:1, probe 2.7 s; the warm-up's
    archival then failed on both ranks with nvEncOpenEncodeSessionEx error 6 = NV_ENC_ERR_INVALID_PTR)."""
    visible = [d.strip() for d in str(env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d.strip()]
    if not visible:
        visible = [str(i) for i in range(int(world))]
    if not 0 <= int(rank) < len(visible):
        raise ValueError(f"rank {rank} has no device in CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r}")
    rotated = visible[int(rank):] + visible[:int(rank)]
    return {"CUDA_VISIBLE_DEVICES": ",".join(rotated), "LOCAL_RANK": "0", "LBQ_SP_DEVICE": visible[int(rank)]}


def node_seed(video_seed: int, session: str, generation: int) -> int:
    """The per-node noise seed (pre-registered deviation; reports/lbq/DESIGN_lbq.md section 2)."""
    digest = hashlib.sha256(f"lbq|{int(video_seed)}|{session}|{int(generation)}".encode()).hexdigest()
    return int(digest[:16], 16) % (1 << 63)


def sha256_tensor(t) -> str:
    import torch

    x = t.detach()
    if x.dtype == torch.bfloat16:
        x = x.view(torch.int16)
    return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class CacheCursor:
    """``global_end`` / ``local_end`` of the shared cache (identical on every layer, as the model keeps them)."""

    def __init__(self):
        self.global_end = 0
        self.local_end = 0

    def append(self, n_new: int, current_start: int, kv_size: int, sink_tokens: int, local_attn: bool):
        """The model's index update for a forward writing ``n_new`` tokens at absolute ``current_start``: returns
        ``(rows_start, rows_end, evict)`` with ``evict = (shift_from, shift_to, rolled)`` or None."""
        current_end = current_start + n_new
        evict = None
        if not local_attn:
            local_end = current_end
        elif current_end > self.global_end and n_new + self.local_end > kv_size:
            num_evicted = n_new + self.local_end - kv_size
            rolled = self.local_end - num_evicted - sink_tokens
            evict = (sink_tokens + num_evicted, sink_tokens, rolled)
            local_end = self.local_end + current_end - self.global_end - num_evicted
        else:
            local_end = self.local_end + current_end - self.global_end
        self.global_end, self.local_end = current_end, local_end
        return local_end - n_new, local_end, evict


class ModelBase:
    """Shared bookkeeping of the real and the fake model.

    SP (Ulysses, ``world`` ranks): every rank holds the KV of its ``heads / world`` local heads for every token
    (``channels`` = local channels). :meth:`chunk_bytes` is this rank's share; :meth:`chunk_bytes_total` all ranks'."""

    resolution: str
    window_latents: int
    sink_latents: int
    world = 1
    rank = 0

    def geometry(self) -> dict[str, Any]:
        image_h, image_w = RESOLUTIONS[self.resolution]
        lat_h, lat_w = latent_grid(self.resolution)
        fs = lat_h * lat_w // 4
        return {"height": lat_h * 8, "width": lat_w * 8, "image_height": image_h, "image_width": image_w,
                "max_area": image_h * image_w, "lat_h": lat_h, "lat_w": lat_w, "frame_seqlen": fs,
                "chunk_tokens": CHUNK_LATENTS * fs, "token_grid": (CHUNK_LATENTS, lat_h // 2, lat_w // 2),
                "window_latents": self.window_latents, "sink_latents": self.sink_latents}

    @property
    def local_attn(self) -> bool:
        return self.window_latents != -1

    def history_chunks(self) -> int:
        """Attended history chunks of a new chunk (window incl. the current chunk)."""
        if not self.local_attn:
            return 10 ** 6
        return self.window_latents // CHUNK_LATENTS - 1

    def chunk_bytes(self) -> int:
        """Raw KV bytes of one chunk on this rank (all layers, K and V, the rank's local heads)."""
        return self.geometry()["chunk_tokens"] * self.layers * 2 * self.channels * self.elem_bytes

    def chunk_bytes_total(self) -> int:
        """Raw KV bytes of one chunk over every rank (full heads)."""
        return self.chunk_bytes() * self.world


# =================================================================================================== real model
class LingBotModel(ModelBase):
    """The official LingBot-World-Fast pipe (lingbot_rt.build_pipe), driven chunk by chunk."""

    layers, channels, elem_bytes = LAYERS, CHANNELS, 2

    def __init__(self, *, resolution: str = "480x832", window_latents: int = 21, sink_latents: int = 0,
                 max_chunks: int = 9, camera_dir: str | None = None, device_index: int = 0, sp: int = 1):
        self.resolution, self.window_latents, self.sink_latents = resolution, int(window_latents), int(sink_latents)
        self.max_chunks = int(max_chunks)
        self.sp_expected = int(sp)
        self.camera_dir = camera_dir
        self.device_index = device_index
        self.pipe = None
        self.sessions: dict[str, dict[str, Any]] = {}
        self.cursor = CacheCursor()
        self._capturing = False
        self._k_pre: dict[int, Any] = {}
        self._hooks: list = []
        self.gpu_cap: dict[str, Any] | None = None
        self.last_decode: dict[str, Any] | None = None
        self._peak_floor = self._reserved_floor = 0
        self._overlap = None  # accel L: a pending overlapped restore (reelcache_lingbot.lb_overlap.OverlapSession)

    # -- build ---------------------------------------------------------------------------------------------------
    def build(self, alloc_cache: bool = True) -> dict[str, Any]:
        import torch

        if str(LINGBOT_RT) not in sys.path:
            sys.path.insert(0, str(LINGBOT_RT))
        import lingbot_rt as rt  # noqa: PLC0415

        self.rt = rt
        staged = os.environ.get("LBQ_CKPT_DIR")
        if staged:  # jobs/lb_run.sbatch LBQ_STAGE_WEIGHTS=1: a node-local copy of the checkpoint dir (size-verified)
            rt.CKPT_DIR = Path(staged)
        self.ckpt_dir = str(rt.CKPT_DIR)
        started = time.perf_counter()
        self.pipe = rt.build_pipe(local_attn_size=self.window_latents, sink_size=self.sink_latents)
        self.model = rt.dit_module(self.pipe)
        self.device = self.pipe.device
        self.param_dtype = self.pipe.param_dtype
        self.dtype = self.pipe.pipe_dtype  # bf16 caches
        geo = self.geometry()
        self.fs = geo["frame_seqlen"]
        self.chunk_tokens = geo["chunk_tokens"]
        self.world = int(getattr(self.pipe, "sp_size", 1))
        if self.world != self.sp_expected:
            raise RuntimeError(f"SP size {self.world} (WORLD_SIZE) != the requested sp {self.sp_expected}")
        if self.world > 1:
            import torch.distributed as dist  # noqa: PLC0415
            from wan.distributed.util import all_to_all  # noqa: PLC0415

            self.rank = dist.get_rank()
            self._all_to_all = all_to_all
        self.local_heads = HEADS // self.world
        self.channels = self.local_heads * HEAD_DIM  # this rank's KV channels (Ulysses: local heads, every token)
        self.max_seq_len = int(math.ceil(self.chunk_tokens / self.world)) * self.world
        self.kv = None
        if alloc_cache:
            self.alloc_cache()
        self.cross = [{"k": torch.zeros([1, 512, HEADS, HEAD_DIM], dtype=self.dtype, device=self.device),
                       "v": torch.zeros([1, 512, HEADS, HEAD_DIM], dtype=self.dtype, device=self.device),
                       "is_init": torch.tensor(0, dtype=torch.int32, device=self.device)} for _ in range(LAYERS)]
        sched = self.pipe.scheduler
        sched.set_timesteps(self.pipe.num_train_timesteps, shift=SHIFT)
        self.timesteps = sched.timesteps[list(TIMESTEPS_INDEX)]
        if self.world > 1:  # the function the SP attention applies (the same code as model_fast's)
            from wan.distributed.sequence_parallel import causal_rope_apply  # noqa: PLC0415
        else:
            from wan.modules.model_fast import causal_rope_apply  # noqa: PLC0415

        self._rope = causal_rope_apply
        self._hooks = [blk.self_attn.norm_k.register_forward_hook(self._k_hook(i))
                       for i, blk in enumerate(self.model.blocks)]
        self.camera = self._camera_embedding(CHUNK_LATENTS * (self.max_chunks + 1))
        torch.cuda.synchronize()
        return {"model": "lingbot-world-fast", "repo_commit": rt.REPO_COMMIT, "hf_revisions": rt.HF_REVISIONS,
                "ckpt_dir": self.ckpt_dir,
                "resolution": self.resolution, "geometry": geo, "kv_size_rows": getattr(self, "kv_size", None),
                "sp": self.world, "rank": self.rank, "local_heads": self.local_heads, "local_channels": self.channels,
                "cache_bytes": None if self.kv is None else 2 * LAYERS * self.kv_size * self.channels * 2,
                "shift": SHIFT,
                "timesteps": [float(t) for t in self.timesteps], "load_seconds": time.perf_counter() - started,
                "device": str(self.device), "gpu": torch.cuda.get_device_name(self.device),
                "camera": {"dir": self.camera_dir, "latents": CHUNK_LATENTS * (self.max_chunks + 1)}}

    def alloc_cache(self, window_latents: int | None = None, max_chunks: int | None = None) -> int:
        """(Re)allocate the shared self-attention cache; a new window also switches the model's own window
        (lingbot_rt.set_attention_window). Returns the cache bytes."""
        import torch

        if window_latents is not None:
            self.window_latents = int(window_latents)
            self.rt.set_attention_window(self.pipe, self.window_latents, self.sink_latents)
        if max_chunks is not None:
            self.max_chunks = int(max_chunks)
        self.kv = None
        torch.cuda.empty_cache()
        self.kv_size = self.fs * (self.window_latents if self.local_attn else CHUNK_LATENTS * self.max_chunks)
        shape = [1, self.kv_size, self.local_heads, HEAD_DIM]
        self.kv = [{"k": torch.zeros(shape, dtype=self.dtype, device=self.device),
                    "v": torch.zeros(shape, dtype=self.dtype, device=self.device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device)}
                   for _ in range(LAYERS)]
        self.cursor = CacheCursor()
        return 2 * LAYERS * self.kv_size * self.channels * 2

    def _k_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self._capturing:
                if self.world > 1:  # norm_k runs on this rank's sequence shard (every head): [s / sp, 5120]
                    self._k_pre[layer] = output.detach()[0].clone()
                else:
                    self._k_pre[layer] = output.detach()[0, :self.chunk_tokens].clone()  # [N, 5120] pre-RoPE
        return hook

    def _camera_embedding(self, lat_f: int):
        """generate()'s camera path for a static action_path (lingbot_rt.write_static_action_path), lat_f latents."""
        import torch
        from einops import rearrange  # noqa: PLC0415
        from wan.utils.cam_utils import (compute_relative_poses, get_Ks_transformed,  # noqa: PLC0415
                                         get_plucker_embeddings, interpolate_camera_poses)

        geo = self.geometry()
        h, w, lat_h, lat_w = geo["height"], geo["width"], geo["lat_h"], geo["lat_w"]
        cam = Path(self.camera_dir) if self.camera_dir else self.rt.STATIC_CAMERA_DIR
        if not (cam / "poses.npy").is_file():
            self.rt.write_static_action_path(cam)
        frame_num = (lat_f - 1) * 4 + 1
        c2ws = np.load(cam / "poses.npy")
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]
        Ks = torch.from_numpy(np.load(cam / "intrinsics.npy")).float()
        Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h,
                                width_final=w)[0]
        n = len(c2ws)
        n_lat = int((n - 1) // 4) + 1
        n_lat = int(n_lat - (n_lat % CHUNK_LATENTS))
        c2ws_infer = interpolate_camera_poses(src_indices=np.linspace(0, n - 1, n), src_rot_mat=c2ws[:, :3, :3],
                                              src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, n - 1, n_lat))
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)
        emb = get_plucker_embeddings(c2ws_infer.to(self.device), Ks.to(self.device), h, w, only_rays_d=False)
        emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))
        emb = rearrange(emb[None, ...], 'b (f h w) c -> b c f h w', f=n_lat, h=lat_h, w=lat_w).to(self.param_dtype)
        return emb

    # -- sessions --------------------------------------------------------------------------------------------------
    def prepare_session(self, sid: str, prompt: str, image_path: str, max_chunks: int) -> dict[str, Any]:
        """T5 context (cached; T5 offloaded back to the CPU), image conditioning y for ``max_chunks`` chunks."""
        import torch
        import torchvision.transforms.functional as TF  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        started = time.perf_counter()
        geo = self.geometry()
        h, w, lat_h, lat_w = geo["height"], geo["width"], geo["lat_h"], geo["lat_w"]
        context = self.rt.encode_prompt(self.pipe, prompt)
        img = Image.open(image_path).convert("RGB")
        if img.size != (geo["image_width"], geo["image_height"]):  # the nominal size; generate() resizes to (h, w)
            img, _ = self.rt.center_crop_resize(img, geo["image_height"], geo["image_width"])
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        lat_f = CHUNK_LATENTS * int(max_chunks)
        F = (lat_f - 1) * 4 + 1
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        with torch.no_grad():
            y = self.pipe.vae.encode([torch.concat([
                torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)], dim=1).to(self.device)])[0]
        y = torch.concat([msk, y])
        self.sessions[sid] = {"context": context, "y": y, "max_chunks": int(max_chunks)}
        torch.cuda.synchronize()
        return {"session": sid, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "y_sha256": sha256_tensor(y), "context_sha256": sha256_tensor(context[0]), "max_chunks": max_chunks,
                "seconds": time.perf_counter() - started}

    # -- cache -------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        if self._overlap is not None:  # an abandoned overlapped restore: nothing may still write the cache
            self.abort_overlapped_restore()
        self.cursor = CacheCursor()
        for d in self.kv:
            d["global_end_index"].fill_(0)
            d["local_end_index"].fill_(0)

    def _sync_indices(self) -> None:
        for d in self.kv:
            d["global_end_index"].fill_(self.cursor.global_end)
            d["local_end_index"].fill_(self.cursor.local_end)

    def _place(self, chunk_index: int) -> tuple[int, int]:
        """Cursor update for appending chunk ``chunk_index`` (installs), with the model's eviction; returns rows."""
        start, end, evict = self.cursor.append(self.chunk_tokens, chunk_index * self.chunk_tokens, self.kv_size,
                                               self.sink_latents * self.fs, self.local_attn)
        if evict is not None:
            src, dst, rolled = evict
            for d in self.kv:
                d["k"][:, dst:dst + rolled] = d["k"][:, src:src + rolled].clone()
                d["v"][:, dst:dst + rolled] = d["v"][:, src:src + rolled].clone()
        self._sync_indices()
        return start, end

    def rope(self, k_pre, chunk_index: int):
        """Post-RoPE K ``[N, 40, 128]`` of pre-RoPE ``k_pre`` ``[N, 5120]`` at the chunk's absolute frame (the model's own
        causal_rope_apply, in the cache dtype: bitwise the native cache for an exact pre-RoPE K)."""
        import torch

        n = k_pre.shape[0]
        grid = torch.tensor([list(self.geometry()["token_grid"])], dtype=torch.long)
        # any number of whole heads (a channel tile of ablation (b) rotates its own heads; RoPE is per head)
        x = k_pre.to(device=self.device, dtype=self.dtype).reshape(1, n, -1, HEAD_DIM)
        return self._rope(x, grid, self.model.freqs, start_frame=chunk_index * CHUNK_LATENTS).type_as(x)[0]

    def rope_lean(self, k_pre, chunk_index: int):
        """accel L: :meth:`rope` without causal_rope_apply's two full float64 copies (``torch.cat`` with an empty tail
        and ``torch.stack`` of one sample): the same float64 conversion, complex multiply and bf16 rounding kernels on the
        same values, so the result is bitwise :meth:`rope`'s (checked on the CPU and per run by tools/l_overlap_ab.py)
        with about half its memory traffic. One GPU (sp 1) and whole chunks only, else :meth:`rope`."""
        import torch

        n = k_pre.shape[0]
        f, h, w = self.geometry()["token_grid"]
        if self.world != 1 or n != f * h * w:
            return self.rope(k_pre, chunk_index)
        x = k_pre.to(device=self.device, dtype=self.dtype).reshape(n, -1, HEAD_DIM)
        freqs_i = self._lean_freqs(int(chunk_index), f, h, w)
        xi = torch.view_as_complex(x.to(torch.float64).reshape(n, x.shape[1], -1, 2))
        return torch.view_as_real(xi * freqs_i).flatten(2).type_as(x)

    def _lean_freqs(self, chunk_index: int, f: int, h: int, w: int):
        """causal_rope_apply's per-token frequencies of a chunk (``freqs_i``), built once per chunk index."""
        import torch

        cache = self.__dict__.setdefault("_lean_freqs_cache", {})
        freqs_i = cache.get(chunk_index)
        if freqs_i is None:
            c = HEAD_DIM // 2
            parts = self.model.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            start = chunk_index * CHUNK_LATENTS
            freqs_i = torch.cat([parts[0][start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                                 parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                                 parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)],
                                dim=-1).reshape(f * h * w, 1, -1).to(self.device)
            torch.cuda.current_stream(self.device).synchronize()  # complete before any other stream reads it
            cache[chunk_index] = freqs_i
        return freqs_i

    def install_rows(self, chunk_index: int, k_rows, v_rows, *, pre_rope: bool) -> tuple[int, int]:
        """Append one chunk's rows: ``k_rows`` / ``v_rows`` per layer ``[N, 5120]`` (host or device); K post-RoPE
        (raw rows) or pre-RoPE (re-roped here)."""
        start, end = self._place(chunk_index)
        for layer, d in enumerate(self.kv):
            k = k_rows[layer]
            if pre_rope:
                d["k"][0, start:end] = self.rope(k, chunk_index)
            else:
                d["k"][0, start:end] = k.to(self.device, non_blocking=True).view(-1, self.local_heads, HEAD_DIM)
            d["v"][0, start:end] = v_rows[layer].to(self.device, non_blocking=True).view(-1, self.local_heads,
                                                                                        HEAD_DIM)
        return start, end

    def place_for_restore(self, chunk_index: int) -> tuple[int, int]:
        """Reserve the rows of a chunk a ReelCache restore writes (through :meth:`destination`)."""
        return self._place(chunk_index)

    def destination_tensors(self, layer: int) -> tuple[Any, Any]:
        return self.kv[layer]["k"], self.kv[layer]["v"]

    # -- accel L: overlapped restore (reelcache_lingbot.lb_overlap) ---------------------------------------------------------------
    def attach_overlapped_restore(self, session) -> None:
        """A restore into this cache runs layer-overlapped: the next forward's self-attention of block ``l`` waits for
        layer ``l`` (forward pre-hooks, removed after the last layer); :meth:`generate` joins it before x0 leaves."""
        if self._overlap is not None:
            raise RuntimeError("an overlapped restore is already pending on this cache")
        if self.world > 1:
            raise RuntimeError("the overlapped restore supports one GPU (sp 1) only")
        session.attach([block.self_attn for block in self.model.blocks])
        self._overlap = session

    def join_overlapped_restore(self) -> dict | None:
        """Join a pending overlapped restore (its failure re-raised, typed); returns its record (None: none)."""
        session, self._overlap = self._overlap, None
        return None if session is None else session.join()

    def abort_overlapped_restore(self) -> None:
        session, self._overlap = self._overlap, None
        if session is not None:
            session.abort()

    def rows(self, start: int, end: int, layer: int):
        """Views of cache rows ``[start, end)`` of a layer: ``(K post-RoPE, V)`` as ``[N, local channels]``."""
        d = self.kv[layer]
        n = end - start
        return d["k"][0, start:end].reshape(n, self.channels), d["v"][0, start:end].reshape(n, self.channels)

    # -- forwards ----------------------------------------------------------------------------------------------------
    def _kwargs(self, sid: str, chunk_index: int) -> dict[str, Any]:
        s = self.sessions[sid]
        lo, hi = chunk_index * CHUNK_LATENTS, (chunk_index + 1) * CHUNK_LATENTS
        if hi > s["y"].shape[1] or hi > self.camera.shape[2]:
            raise ValueError(f"chunk {chunk_index} beyond the prepared horizon of {sid}")
        return {"context": [s["context"][0]], "seq_len": self.max_seq_len, "y": [s["y"][:, lo:hi]],
                "dit_cond_dict": {"c2ws_plucker_emb": self.camera[:, :, lo:hi].chunk(1, dim=0)},
                "kv_cache": self.kv, "crossattn_cache": self.cross,
                "current_start": chunk_index * self.chunk_tokens, "max_attention_size": self.kv_size,
                "frame_seqlen": self.fs}

    def _forward(self, x, t, kwargs, first: bool):
        """One DiT forward; installs / replays keep the cursor in step with the model's own indices."""
        start, end, _ = self.cursor.append(self.chunk_tokens, kwargs["current_start"], self.kv_size,
                                           self.sink_latents * self.fs, self.local_attn) \
            if kwargs["current_start"] + self.chunk_tokens > self.cursor.global_end else (
                self.cursor.local_end - self.chunk_tokens, self.cursor.local_end, None)
        out = self.model(x=[x], t=t, cross_attn_first_call=first, **kwargs)
        return out, (start, end)

    def replay(self, sid: str, chunk_index: int, x0, *, first: bool) -> tuple[int, int]:
        """The clean pass of chunk ``chunk_index`` on its stored latents x0 (exact recompute of its KV)."""
        import torch

        kwargs = self._kwargs(sid, chunk_index)
        t = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            _, rows = self._forward(x0.to(self.device), t, kwargs, first)
        return rows

    def generate(self, sid: str, chunk_index: int, generator, *, noise=None, capture_pre_k: bool = False,
                 marks: dict | None = None, events: dict | None = None):
        """4 denoising steps + the clean pass of chunk ``chunk_index`` on the installed window. ``generator`` draws
        the initial noise (unless ``noise`` is given: official_rng mode) and the 3 renoise draws. Returns
        ``(x0, rows)``; with ``capture_pre_k`` the clean pass's pre-RoPE K is kept (:meth:`take_pre_k`).

        accel L: a pending overlapped restore (:meth:`attach_overlapped_restore`) is read by the first forward layer
        by layer and joined before returning (on a failure it is quiesced and the error re-raised); the ``marks``
        synchronize then covers the generation stream only. ``events`` (dict): CUDA timing events recorded on the
        generation stream around every forward (``f{i}_start`` / ``f{i}_end``, the clean pass is ``f4``)."""
        import torch

        overlap = self._overlap

        def stamp(name):
            if events is not None:
                events[name] = torch.cuda.Event(enable_timing=True)
                events[name].record(torch.cuda.current_stream(self.device))

        try:
            kwargs = self._kwargs(sid, chunk_index)
            geo = self.geometry()
            if noise is None:
                noise = torch.randn(LATENT_CHANNELS, CHUNK_LATENTS, geo["lat_h"], geo["lat_w"], dtype=torch.float32,
                                    generator=generator, device=self.device)
            latent = noise
            first = True
            with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
                for i in range(len(self.timesteps)):
                    if marks is not None and i == 0:
                        if overlap is not None:
                            torch.cuda.current_stream(self.device).synchronize()
                        else:
                            torch.cuda.synchronize()
                        marks["t2"] = time.monotonic_ns()
                    t = torch.stack([self.timesteps[i]]).to(self.device)
                    stamp(f"f{i}_start")
                    out, rows = self._forward(latent.to(self.device), t, kwargs, first)
                    stamp(f"f{i}_end")
                    first = False
                    x0 = self.pipe._convert_flow_pred_to_x0(flow_pred=out[0], xt=latent, timestep=self.timesteps[i],
                                                            scheduler=self.pipe.scheduler)
                    if i < len(self.timesteps) - 1:
                        latent = self.pipe.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype),
                            self.timesteps[i + 1])
                self._capturing = capture_pre_k
                self._k_pre.clear()
                try:
                    t = torch.stack([self.timesteps[-1] * 0.0]).to(self.device)
                    stamp(f"f{len(self.timesteps)}_start")
                    _, rows = self._forward(x0, t, kwargs, False)
                    stamp(f"f{len(self.timesteps)}_end")
                finally:
                    self._capturing = False
        except BaseException:
            if overlap is not None and self._overlap is overlap:
                self.abort_overlapped_restore()
            raise
        if overlap is not None and self._overlap is overlap:
            self.join_overlapped_restore()  # the integrity gate passed before x0 leaves
        return x0, rows

    def take_pre_k(self) -> list:
        """The captured clean pass's pre-RoPE K per layer, ``[N, local channels]``. SP: the attention's own
        all-to-all (sequence shard, every head -> every token, local heads) on the captured shards: bitwise the
        ``k`` the attention ropes and caches (a collective: every rank calls it)."""
        out = []
        for layer in range(LAYERS):
            k = self._k_pre.pop(layer)
            if self.world > 1:
                k = self._all_to_all(k.view(1, k.shape[0], HEADS, HEAD_DIM), scatter_dim=2, gather_dim=1)
                k = k[0, :self.chunk_tokens].reshape(self.chunk_tokens, self.channels)
            out.append(k)
        self._k_pre = {}
        return out

    def decode(self, parent_x0, x0) -> np.ndarray | None:
        """RGB of the new chunk (pre-registered): the last 12 frames of the VAE decode of [parent, new] latents;
        chunk 0 (no parent) decodes alone (9 frames). uint8 [F, H, W, 3]. SP: rank 0 decodes (as generate()); the
        other ranks return None."""
        import torch

        if self.rank != 0:
            return None
        latents = x0 if parent_x0 is None else torch.cat([parent_x0.to(self.device), x0], dim=1)
        # the decode's own working set (the device's memory peak on rank 0 at 720P): live bytes at entry vs its peak;
        # the request's earlier peak is kept as the floor of memory()
        alloc0 = torch.cuda.memory_allocated(self.device)
        self._peak_floor = max(self._peak_floor, torch.cuda.max_memory_allocated(self.device))
        self._reserved_floor = max(self._reserved_floor, torch.cuda.max_memory_reserved(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.no_grad():
            video = self.pipe.vae.decode([latents])[0]
        frames = self.rt.video_to_uint8(video)
        peak = torch.cuda.max_memory_allocated(self.device)
        self.last_decode = {"latents": int(latents.shape[1]), "alloc_entry_bytes": int(alloc0),
                            "peak_bytes": int(peak), "working_set_bytes": int(peak - alloc0)}
        return frames if parent_x0 is None else frames[-12:]

    def decode_path(self, latents_path: Sequence[Any]) -> np.ndarray | None:
        """Official one-shot decode of a whole path's latents (quality scoring; rank 0)."""
        import torch

        if self.rank != 0:
            return None

        with torch.no_grad():
            video = self.pipe.vae.decode([torch.cat([x.to(self.device) for x in latents_path], dim=1)])[0]
        return self.rt.video_to_uint8(video)

    def generator(self, seed: int):
        import torch

        return torch.Generator(device=self.device).manual_seed(int(seed))

    def official_noise(self, generator, lat_f: int):
        import torch

        geo = self.geometry()
        return torch.randn(LATENT_CHANNELS, lat_f, geo["lat_h"], geo["lat_w"], dtype=torch.float32,
                           generator=generator, device=self.device)

    def memory(self) -> dict[str, Any]:
        import torch

        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        out = {"peak_allocated_gib": max(self._peak_floor, torch.cuda.max_memory_allocated(self.device)) / 2 ** 30,
               "peak_reserved_gib": max(self._reserved_floor, torch.cuda.max_memory_reserved(self.device)) / 2 ** 30,
               "device_used_gib": (total - free) / 2 ** 30,
               # outside PyTorch's pool: CUDA contexts, NCCL, NVENC / NVDEC sessions (what the cap's reserve protects)
               "non_torch_gib": (total - free - reserved) / 2 ** 30}
        if self.gpu_cap is not None:
            out["torch_cap_gib"] = self.gpu_cap["cap_bytes"] / 2 ** 30
        return out

    def reset_peaks(self) -> None:
        import torch

        self._peak_floor = self._reserved_floor = 0
        torch.cuda.reset_peak_memory_stats(self.device)

    def apply_memory_cap(self, reserve_bytes: int) -> dict[str, Any]:
        """Cap PyTorch's caching allocator so that ``reserve_bytes`` of device memory beyond what lives outside PyTorch
        now (contexts, NCCL, the archive's NVENC qualification session) always stays free for the codec sessions
        (NVENC / NVDEC) and PyNvVideoCodec's per-session probe context, which allocate outside PyTorch and fail when
        PyTorch holds the device (smoke 4507402). PyTorch frees its cached blocks and retries before it reports an
        out-of-memory at the cap; no computation changes."""
        import torch

        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        non_torch = total - free - torch.cuda.memory_reserved(self.device)
        cap = int(total - non_torch - int(reserve_bytes))
        torch.cuda.set_per_process_memory_fraction(cap / total, self.device)
        self.gpu_cap = {"total_bytes": int(total), "non_torch_at_cap_bytes": int(non_torch),
                        "reserve_bytes": int(reserve_bytes), "cap_bytes": cap, "fraction": cap / total}
        return dict(self.gpu_cap)

    def decode_room(self) -> int:
        """Device bytes PyTorch can still hand the decode: the cap minus the live allocations (cached blocks are
        released on demand), else the driver's free memory plus the unoccupied cached bytes."""
        import torch

        allocated = torch.cuda.memory_allocated(self.device)
        if self.gpu_cap is not None:
            return int(self.gpu_cap["cap_bytes"] - allocated)
        free, _ = torch.cuda.mem_get_info(self.device)
        return int(free + torch.cuda.memory_reserved(self.device) - allocated)

    def synchronize(self) -> None:
        import torch

        if self._overlap is not None:  # accel L: a pending overlapped restore keeps running beside the generation
            torch.cuda.current_stream(self.device).synchronize()
            return
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        self.abort_overlapped_restore()
        for h in self._hooks:
            h.remove()
        self._hooks = []


# =================================================================================================== fake model
class FakeModel(ModelBase):
    """CPU stand-in: tiny geometry, deterministic float64 math with the real model's data dependencies:
    - x0 = f(noise, the attended history KV);
    - a chunk's KV = g(x0, the attended history KV);
    - RGB = h(latents).
    Installs, evictions, cursor updates and the replay closure are the real model's (shared code). Raw rows,
    pre-RoPE re-roping (a fixed per-frame rotation) and restore destinations behave like the real ones."""

    def __init__(self, *, resolution: str = "fake", window_latents: int = 21, sink_latents: int = 0,
                 max_chunks: int = 9, layers: int = 2, heads: int = 2, head_dim: int = 4, lat_hw: tuple = (4, 4),
                 frame_hw: tuple = (8, 8), elem_bytes: int = 2, sp: int = 1, **_):
        self.resolution, self.window_latents, self.sink_latents = resolution, int(window_latents), int(sink_latents)
        self.max_chunks = int(max_chunks)
        self.layers, self.heads, self.head_dim = int(layers), int(heads), int(head_dim)
        # SP (CPU tests over Gloo): every rank holds its heads / sp local heads; the 'attention' digest sums over
        # every rank's rows (an all-reduce), so a rank's output depends on the other ranks' caches, as with Ulysses
        self.world = int(sp)
        self.rank = 0
        self.group = None
        if self.heads % self.world:
            raise ValueError("heads must divide the SP size")
        self.local_heads = self.heads // self.world
        self.channels = self.local_heads * self.head_dim
        self.elem_bytes = int(elem_bytes)
        self.lat_hw, self.frame_hw = tuple(lat_hw), tuple(frame_hw)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.cursor = CacheCursor()
        self._k_pre: list = []
        self.forwards = 0

    def geometry(self) -> dict[str, Any]:
        lat_h, lat_w = self.lat_hw
        fs = lat_h * lat_w // 4
        return {"height": self.frame_hw[0], "width": self.frame_hw[1], "lat_h": lat_h, "lat_w": lat_w,
                "frame_seqlen": fs, "chunk_tokens": CHUNK_LATENTS * fs, "token_grid": (CHUNK_LATENTS, lat_h // 2,
                                                                                       lat_w // 2),
                "window_latents": self.window_latents, "sink_latents": self.sink_latents}

    def build(self) -> dict[str, Any]:
        import torch

        geo = self.geometry()
        self.fs, self.chunk_tokens = geo["frame_seqlen"], geo["chunk_tokens"]
        self.kv_size = self.fs * (self.window_latents if self.local_attn else CHUNK_LATENTS * self.max_chunks)
        self.device = torch.device("cpu")
        if self.world > 1:
            import torch.distributed as dist  # noqa: PLC0415

            self.rank = dist.get_rank()
        self.kv = [{"k": torch.zeros(1, self.kv_size, self.local_heads, self.head_dim, dtype=torch.float64),
                    "v": torch.zeros(1, self.kv_size, self.local_heads, self.head_dim, dtype=torch.float64)}
                   for _ in range(self.layers)]
        return {"model": "fake", "geometry": geo, "kv_size_rows": self.kv_size, "sp": self.world, "rank": self.rank,
                "local_heads": self.local_heads}

    def prepare_session(self, sid: str, prompt: str, image_path: str, max_chunks: int) -> dict[str, Any]:
        seed = int(hashlib.sha256(f"{prompt}|{image_path}".encode()).hexdigest()[:8], 16)
        self.sessions[sid] = {"cond": float(seed % 997) / 997.0, "max_chunks": int(max_chunks)}
        return {"session": sid, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "max_chunks": max_chunks, "seconds": 0.0}

    def reset(self) -> None:
        self.cursor = CacheCursor()

    def _place(self, chunk_index: int) -> tuple[int, int]:
        start, end, evict = self.cursor.append(self.chunk_tokens, chunk_index * self.chunk_tokens, self.kv_size,
                                               self.sink_latents * self.fs, self.local_attn)
        if evict is not None:
            src, dst, rolled = evict
            for d in self.kv:
                d["k"][:, dst:dst + rolled] = d["k"][:, src:src + rolled].clone()
                d["v"][:, dst:dst + rolled] = d["v"][:, src:src + rolled].clone()
        return start, end

    def rope(self, k_pre, chunk_index: int):
        """A position-dependent, exactly invertible rotation (the fake's 'RoPE')."""
        n = k_pre.shape[0]
        angle = 0.1 * (chunk_index * CHUNK_LATENTS + 1)
        x = k_pre.double().reshape(n, -1, self.head_dim)
        return x * math.cos(angle) + x.roll(1, dims=-1) * math.sin(angle)

    def unrope(self, k_post, chunk_index: int):
        import torch

        n = k_post.shape[0]
        angle = 0.1 * (chunk_index * CHUNK_LATENTS + 1)
        c, s = math.cos(angle), math.sin(angle)
        # x' = c x + s roll(x)  ->  solved exactly in float64 by the circulant inverse
        y = k_post.double().view(n, -1, self.head_dim)
        d = self.head_dim
        m = torch.zeros(d, d, dtype=torch.float64)
        for i in range(d):
            m[i, i] += c
            m[i, (i - 1) % d] += s
        return torch.linalg.solve(m, y.reshape(-1, d).T).T.reshape(n, self.channels)

    def install_rows(self, chunk_index: int, k_rows, v_rows, *, pre_rope: bool) -> tuple[int, int]:
        start, end = self._place(chunk_index)
        for layer, d in enumerate(self.kv):
            k = k_rows[layer]
            d["k"][0, start:end] = self.rope(k, chunk_index) if pre_rope else k.double().view(-1, self.local_heads,
                                                                                               self.head_dim)
            d["v"][0, start:end] = v_rows[layer].double().view(-1, self.local_heads, self.head_dim)
        return start, end

    def place_for_restore(self, chunk_index: int) -> tuple[int, int]:
        return self._place(chunk_index)

    def destination_tensors(self, layer: int):
        return self.kv[layer]["k"], self.kv[layer]["v"]

    def rows(self, start: int, end: int, layer: int):
        d = self.kv[layer]
        return d["k"][0, start:end].reshape(end - start, self.channels), \
            d["v"][0, start:end].reshape(end - start, self.channels)

    def _history_digest(self, upto: int) -> float:
        """A scalar summary of the attended rows [0, upto) of every layer (what the 'attention' sees); SP: summed
        over every rank's local heads (a collective)."""
        total = 0.0
        for d in self.kv:
            if upto:
                total += float(d["k"][0, :upto].sum()) * 1e-3 + float(d["v"][0, :upto].sum()) * 1e-3
        if self.world > 1:
            import torch
            import torch.distributed as dist  # noqa: PLC0415

            t = torch.tensor([total], dtype=torch.float64)
            dist.all_reduce(t, group=self.group)
            total = float(t[0])
        return total

    def _begin(self, chunk_index: int) -> tuple[int, int]:
        """The forward's own index update (first forward of a chunk: append + eviction; later ones: same rows)."""
        start, end, evict = self.cursor.append(self.chunk_tokens, chunk_index * self.chunk_tokens, self.kv_size,
                                               self.sink_latents * self.fs, self.local_attn) \
            if chunk_index * self.chunk_tokens + self.chunk_tokens > self.cursor.global_end else (
                self.cursor.local_end - self.chunk_tokens, self.cursor.local_end, None)
        if evict is not None:
            src, dst, rolled = evict
            for d in self.kv:
                d["k"][:, dst:dst + rolled] = d["k"][:, src:src + rolled].clone()
                d["v"][:, dst:dst + rolled] = d["v"][:, src:src + rolled].clone()
        return start, end

    def _write_chunk(self, sid: str, chunk_index: int, x0) -> tuple[int, int]:
        """The 'clean pass': writes the chunk's K (post-RoPE) / V as a function of x0 and the attended history."""
        import torch

        start, end = self._begin(chunk_index)
        h = self._history_digest(start)
        base = x0.double().reshape(-1)
        self._k_pre = []
        full = self.heads * self.head_dim
        cols = slice(self.rank * self.channels, (self.rank + 1) * self.channels)  # this rank's heads
        for layer, d in enumerate(self.kv):
            n = end - start
            idx = torch.arange(n * full, dtype=torch.float64)
            vals = torch.sin(idx * 0.37 + layer + h) + base[(idx.long() % base.numel())] * 0.5
            k_pre = vals.view(n, full) * (1.0 + 0.01 * layer)
            v = torch.cos(idx * 0.11 - layer + h).view(n, full) + 0.25 * k_pre
            k_pre, v = k_pre[:, cols].contiguous(), v[:, cols].contiguous()
            self._k_pre.append(k_pre.clone())
            d["k"][0, start:end] = self.rope(k_pre, chunk_index)
            d["v"][0, start:end] = v.view(n, self.local_heads, self.head_dim)
        self.forwards += 1
        return start, end

    def replay(self, sid: str, chunk_index: int, x0, *, first: bool) -> tuple[int, int]:
        return self._write_chunk(sid, chunk_index, x0)

    def generate(self, sid: str, chunk_index: int, generator, *, noise=None, capture_pre_k: bool = False,
                 marks: dict | None = None):
        import torch

        geo = self.geometry()
        if noise is None:
            noise = torch.randn(LATENT_CHANNELS, CHUNK_LATENTS, geo["lat_h"], geo["lat_w"], dtype=torch.float64,
                                generator=generator)
        if marks is not None:
            marks["t2"] = time.monotonic_ns()
        start, _ = self._begin(chunk_index)  # the first denoising forward evicts before it attends (as the model)
        h = self._history_digest(start)
        x0 = noise.double() * 0.5 + h * 0.01 + self.sessions[sid]["cond"]
        for _ in range(len(TIMESTEPS_INDEX) - 1):
            x0 = x0 * 0.9 + 0.1 * torch.randn(x0.shape, generator=generator, dtype=torch.float64)
        self.forwards += len(TIMESTEPS_INDEX)
        rows = self._write_chunk(sid, chunk_index, x0)
        if not capture_pre_k:
            self._k_pre = []
        return x0, rows

    def take_pre_k(self) -> list:
        out, self._k_pre = self._k_pre, []
        return out

    def decode(self, parent_x0, x0) -> np.ndarray | None:
        if self.rank != 0:  # SP: rank 0 decodes
            return None
        frames = 9 if parent_x0 is None else 12
        h, w = self.frame_hw
        v = x0.double().reshape(-1)
        idx = np.arange(frames * h * w * 3, dtype=np.float64)
        base = np.tanh(v.numpy()[(idx.astype(np.int64) % v.numel())]) * 127.5 + 127.5
        return base.reshape(frames, h, w, 3).astype(np.uint8)

    def decode_path(self, latents_path: Sequence[Any]) -> np.ndarray | None:
        if self.rank != 0:
            return None
        return np.concatenate([self.decode(None if i == 0 else latents_path[i - 1], x)
                               for i, x in enumerate(latents_path)], axis=0)

    def generator(self, seed: int):
        import torch

        return torch.Generator().manual_seed(int(seed))

    def official_noise(self, generator, lat_f: int):
        import torch

        geo = self.geometry()
        return torch.randn(LATENT_CHANNELS, lat_f, geo["lat_h"], geo["lat_w"], dtype=torch.float64,
                           generator=generator)

    def memory(self) -> dict[str, Any]:
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0, "device_used_gib": 0.0}

    def decode_room(self) -> int:
        return 1 << 62

    def reset_peaks(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        pass
