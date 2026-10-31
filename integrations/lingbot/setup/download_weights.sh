#!/bin/bash
# Download LingBot-World-Fast weights + the minimal base-cam files the Fast loader needs (no MoE experts).
# Set LB_CKPT or LB_REPO (destination), and optionally LB_WORK (caches/temp) and HF_BIN (the `hf` CLI); defaults are placeholders.
set -euo pipefail
LB_WORK=${LB_WORK:-$HOME/.cache/reelcache/lingbot}
export HF_HOME=$LB_WORK/hf_home
export HF_XET_CACHE=$HF_HOME/xet TMPDIR=$LB_WORK/tmp
export HF_HUB_DISABLE_TELEMETRY=1
HF=${HF_BIN:-hf}
CKPT=${LB_CKPT:-${LB_REPO:-/models/lingbot-world}/lingbot-world-base-cam}
BASE_REV=6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd
FAST_REV=b7a1db2da9bd3ba675ee01c228f41db0309b981e
mkdir -p "$CKPT/lingbot_world_fast"
echo "[$(date -Is)] base-cam minimal files"
$HF download robbyant/lingbot-world-base-cam --revision $BASE_REV --local-dir "$CKPT" \
  --include "Wan2.1_VAE.pth" --include "models_t5_umt5-xxl-enc-bf16.pth" --include "google/umt5-xxl/*" \
  --include "configuration.json" --include "README.md" --max-workers 8
echo "[$(date -Is)] lingbot-world-fast"
$HF download robbyant/lingbot-world-fast --revision $FAST_REV --local-dir "$CKPT/lingbot_world_fast" --max-workers 8
echo "[$(date -Is)] done"
du -sh "$CKPT"
