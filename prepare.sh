#!/usr/bin/env bash
# Run on the pod after upstream/ is cloned.
# 1. Pull osunlp/Mind2Web from HF and convert to SDPO format
# 2. Preprocess to parquet (verl's input format)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT/upstream/datasets/mind2web"

export PYTHONPATH="$ROOT/upstream:${PYTHONPATH:-}"

# 1. Install our mind2web reward function into the upstream submodule
bash "$ROOT/data/install_reward.sh"

# 2. Convert Mind2Web -> SDPO JSONL
if [[ -f "$DATA_DIR/train.json" && -f "$DATA_DIR/test.json" ]]; then
  echo "[prepare] mind2web JSONL already present, skipping conversion"
else
  echo "[prepare] converting Mind2Web from HuggingFace to SDPO format..."
  pip install -q "datasets>=2.14.0" pyarrow
  python "$ROOT/data/convert_mind2web.py" \
    --output-dir "$DATA_DIR" \
    --max-train-tasks 800 \
    --max-test-tasks 100
fi

# 3. JSONL -> parquet via upstream preprocessor
if [[ -f "$DATA_DIR/train.parquet" && -f "$DATA_DIR/test.parquet" ]]; then
  echo "[prepare] mind2web parquet already present, skipping preprocess"
else
  echo "[prepare] preprocessing mind2web to parquet..."
  cd "$ROOT/upstream"
  python data/preprocess.py --data_source datasets/mind2web
fi

echo "[prepare] done"
ls -la "$DATA_DIR/"
