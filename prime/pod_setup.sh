#!/usr/bin/env bash
# Runs ON the Prime Intellect pod after SSH. Bootstraps verl + lasgroup/SDPO,
# preprocesses tooluse, then runs GRPO and SDPO sequentially.
set -euxo pipefail

REPO_URL="${REPO_URL:-https://github.com/yug-space/sdpo-vs-grpo}"
WORKDIR="${WORKDIR:-/workspace/sdpo-vs-grpo}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

# ---- 1. clone ----
if [[ ! -d "$WORKDIR" ]]; then
  git clone --recurse-submodules "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git submodule update --init --recursive

# ---- 2. install ----
# B200 (Blackwell, sm_100) needs cu128 + torch 2.7. See upstream/INSTALL.md.
python -m pip install --upgrade pip
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd upstream
pip install -r requirements.txt
pip install -e .
pip install flash-attn --no-build-isolation || echo "[warn] flash-attn install failed; continuing"
pip install word2number "latex2sympy2" "math-verify[antlr4_9_3]==0.8.0"
pip install --upgrade wandb

# Blackwell-tested vLLM
pip install "vllm>=0.12.0"

# ---- 3. wandb auth ----
if [[ -n "$WANDB_API_KEY" ]]; then
  wandb login "$WANDB_API_KEY"
fi
if [[ -n "$HF_TOKEN" ]]; then
  huggingface-cli login --token "$HF_TOKEN" || python -c "from huggingface_hub import login; login('$HF_TOKEN')"
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
