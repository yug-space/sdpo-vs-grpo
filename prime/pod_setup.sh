#!/usr/bin/env bash
# Runs ON the Prime Intellect pod after SSH. Bootstraps verl + lasgroup/SDPO,
# preprocesses tooluse, then runs GRPO and SDPO sequentially.
set -euxo pipefail

REPO_URL="${REPO_URL:-https://github.com/yug-space/sdpo-vs-grpo}"
WORKDIR="${WORKDIR:-/workspace/sdpo-vs-grpo}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

# ---- 0. python/pip resolution ----
# Datacrunch/runpod images often ship python3 only and/or no pip — fix both.
if ! command -v python >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "[setup] installing pip via apt..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null 2>&1
  apt-get install -y python3-pip python3-venv git curl >/dev/null 2>&1 || \
    python3 -m ensurepip --upgrade
fi
PIP="python3 -m pip"
$PIP --version

# ---- 1. clone ----
if [[ ! -d "$WORKDIR" ]]; then
  git clone --recurse-submodules "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git submodule update --init --recursive

# ---- 2. install ----
# H200 (Hopper, sm_90) — torch 2.5+cu124 is well tested with verl/lasgroup-SDPO.
# (The Blackwell cu128 path was for B200; H200 uses Hopper toolchain.)
$PIP install --upgrade pip
$PIP install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd upstream
$PIP install -r requirements.txt
$PIP install -e . || echo "[warn] verl editable install had issues"
$PIP install flash-attn --no-build-isolation || echo "[warn] flash-attn install failed; continuing without it"
$PIP install word2number "latex2sympy2" "math-verify[antlr4_9_3]==0.8.0"
$PIP install --upgrade wandb

# vLLM (Hopper supports any modern release)
$PIP install "vllm==0.8.4"

# ---- 2b. fix dependency conflicts that break Ray on this image ----
# The base image ships opentelemetry-exporter-prometheus 0.62b1 which imports
# OtelComponentTypeValues (only in semconv >= 0.50b0). Verl pins semconv=0.47b0
# transitively via Ray, so the prometheus exporter import fails -> Ray
# dashboard dies -> Raylet socket closes -> training worker fails to register.
# Ray's prometheus metrics export is optional; remove the offender.
$PIP uninstall -y opentelemetry-exporter-prometheus 2>&1 | tail -2 || true
# Verl needs numpy<2.0; some deps drag 2.x in. Pin back.
$PIP install "numpy<2.0.0" 2>&1 | tail -3

# Qwen3.5 has model_type="qwen3_5" — needs recent transformers (>=4.46) to recognize.
$PIP install --upgrade "transformers>=4.46.0" 2>&1 | tail -3

# ---- 3. wandb auth ----
if [[ -n "$WANDB_API_KEY" ]]; then
  wandb login "$WANDB_API_KEY"
fi
if [[ -n "$HF_TOKEN" ]]; then
  huggingface-cli login --token "$HF_TOKEN" || python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
fi

# ---- 4. data ----
cd "$WORKDIR"
bash prepare.sh

# ---- 5. run GRPO ----
mkdir -p results
echo "[$(date)] starting GRPO" | tee -a results/run.log
bash configs/grpo.sh 2>&1 | tee -a results/grpo.log

# ---- 6. run SDPO ----
echo "[$(date)] starting SDPO" | tee -a results/run.log
bash configs/sdpo.sh 2>&1 | tee -a results/sdpo.log

echo "[$(date)] both runs complete" | tee -a results/run.log
