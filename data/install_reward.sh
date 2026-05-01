#!/usr/bin/env bash
# Idempotently install our mind2web reward function into the upstream
# (lasgroup/SDPO) submodule. Run once after cloning, before training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_FEEDBACK="$ROOT/upstream/verl/utils/reward_score/feedback"
INIT_PY="$UPSTREAM_FEEDBACK/__init__.py"

if [[ ! -d "$UPSTREAM_FEEDBACK" ]]; then
  echo "[install_reward] error: $UPSTREAM_FEEDBACK does not exist."
  echo "[install_reward] run 'git submodule update --init --recursive' first."
  exit 1
fi

# 1. Copy reward function into upstream
cp "$ROOT/data/mind2web_reward.py" "$UPSTREAM_FEEDBACK/mind2web.py"
echo "[install_reward] copied mind2web.py -> $UPSTREAM_FEEDBACK/"

# 2. Patch __init__.py: import + dispatch entry (idempotent — uses grep guard)
if ! grep -q "from verl.utils.reward_score.feedback import mind2web" "$INIT_PY"; then
  python3 - "$INIT_PY" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
# add import after the last existing feedback-module import
src = re.sub(
    r"(from verl\.utils\.reward_score\.feedback import tooluse\n)",
    r"\1from verl.utils.reward_score.feedback import mind2web\n",
    src, count=1,
)
# add dispatch branch before the trailing 'else' of compute_score
dispatch = (
    '    elif data_source in ["mind2web"]:\n'
    '        results = mind2web.compute_score(solution_str, ground_truth, extra_info)\n'
)
if "mind2web.compute_score" not in src:
    src = src.replace(
        '    else:\n        raise ValueError(f"Reward style {data_source} not found.")',
        dispatch + '    else:\n        raise ValueError(f"Reward style {data_source} not found.")',
    )
p.write_text(src)
print(f"[install_reward] patched {p}")
PY
else
  echo "[install_reward] $INIT_PY already patched, skipping"
fi

# 3. Self-test
echo "[install_reward] running reward self-test..."
python3 "$ROOT/data/mind2web_reward.py"

echo "[install_reward] done"
