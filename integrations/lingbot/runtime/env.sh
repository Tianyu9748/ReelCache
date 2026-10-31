# LingBot-World-Fast + ReelCache runtime environment. Source it from the ReelCache root, inside your job:
#   source integrations/lingbot/runtime/env.sh
#   $LB_PY -B my_script.py                                                          # 1 GPU (480P)
#   $LB_PY -B -m torch.distributed.run --standalone --nproc_per_node=2 my_script.py # Ulysses SP=2 (720P)
# Export LB_REPO (the official repository) and, if needed, LB_CKPT, LB_WORK, LB_PY, LB_SCRATCH or REELCACHE_FFMPEG_BIN
# before sourcing; the defaults are placeholders.
LB_RT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REELCACHE_ROOT=${REELCACHE_ROOT:-$(cd "$LB_RT_DIR/../../.." && pwd)}
export LB_WORK=${LB_WORK:-$HOME/.cache/reelcache/lingbot}                       # large intermediates, Triton/CUDA caches
export LB_REPO=${LB_REPO:-/models/lingbot-world}                                # official repo @ a43bec7, unmodified
export LB_CKPT=${LB_CKPT:-$LB_REPO/lingbot-world-base-cam}                     # T5 + VAE + lingbot_world_fast/
export LB_PY=${LB_PY:-$(command -v python3)}                                    # see API.md section 1
export PATH=$(dirname "$LB_PY"):$PATH
export PYTHONPATH=$REELCACHE_ROOT:$REELCACHE_ROOT/integrations/lingbot:$LB_RT_DIR:$LB_REPO
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export FLASH_ATTN_3_AVAILABLE=0
LB_GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)
export LB_SCRATCH=${LB_SCRATCH:-${TMPDIR:-/tmp}/reelcache-lingbot-${SLURM_JOB_ID:-$$}}  # node-local scratch
export XDG_CACHE_HOME=$LB_SCRATCH/cache/xdg TORCH_HOME=$LB_SCRATCH/cache/torch HF_HOME=$LB_SCRATCH/cache/hf
export TMPDIR=$LB_SCRATCH/cache/tmp MPLCONFIGDIR=$LB_SCRATCH/cache/matplotlib NUMBA_CACHE_DIR=$LB_SCRATCH/cache/numba
export TORCHINDUCTOR_CACHE_DIR=$LB_SCRATCH/cache/inductor
export TRITON_CACHE_DIR=$LB_WORK/cache/triton-sm${LB_GPU_CC:-nogpu} CUDA_CACHE_PATH=$LB_WORK/cache/cudacache-sm${LB_GPU_CC:-nogpu}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
[ -n "${REELCACHE_FFMPEG_BIN:-}" ] && export REELCACHE_FFMPEG_BIN  # unset: the FFmpeg of imageio-ffmpeg is used
unset CUBLAS_WORKSPACE_CONFIG
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
