#!/usr/bin/env bash
# Local-side orchestrator. Provisions a Prime Intellect B200 180GB pod, ships
# secrets, runs pod_setup.sh, syncs results back, and terminates the pod.
#
# Requires: prime CLI authenticated (`prime login`), and a secrets.env file
# at repo root with WANDB_API_KEY=... and (optionally) HF_TOKEN=...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- config ----
GPU_ID="${GPU_ID:-468e80}"            # 1× H200 141GB datacrunch FI ($1.187/hr)
POD_NAME="${POD_NAME:-sdpo-vs-grpo-$(date +%s)}"
DISK_GB="${DISK_GB:-200}"
SECRETS_FILE="${SECRETS_FILE:-$ROOT/secrets.env}"

if [[ ! -f "$SECRETS_FILE" ]]; then
  cat <<EOF >&2
[error] $SECRETS_FILE not found.
Create it with:
  WANDB_API_KEY=<your wandb key>
  HF_TOKEN=<optional huggingface token>
Then re-run.
EOF
  exit 1
fi

# ---- preflight ----
echo "[launch] preflight checks..."
prime --plain whoami | head -3
prime --plain wallet | grep -i balance

# ---- create pod ----
echo "[launch] creating pod (id=$GPU_ID, name=$POD_NAME, disk=${DISK_GB}GB)..."
POD_INFO=$(prime --plain pods create \
  --id "$GPU_ID" \
  --name "$POD_NAME" \
  --disk-size "$DISK_GB" \
  --image "ubuntu_22_cuda_12_8" \
  -y \
  --output json)

POD_ID=$(echo "$POD_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('id') or d.get('pod',{}).get('id') or '')")
if [[ -z "$POD_ID" ]]; then
  echo "[error] failed to parse pod id from:" >&2
  echo "$POD_INFO" >&2
  exit 1
fi
echo "[launch] pod id: $POD_ID"

# ---- wait until ready ----
echo "[launch] waiting for pod to become ACTIVE..."
for i in {1..60}; do
  STATUS=$(prime --plain pods status "$POD_ID" --output json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" || echo "")
  echo "  [$i] status: $STATUS"
  [[ "$STATUS" == "ACTIVE" || "$STATUS" == "RUNNING" ]] && break
  sleep 15
done

# ---- ship setup script + secrets, kick off DETACHED so we can monitor ----
echo "[launch] uploading secrets + setup script..."
prime pods ssh "$POD_ID" -- "mkdir -p /workspace && cat > /workspace/secrets.env" < "$SECRETS_FILE"
prime pods ssh "$POD_ID" -- "cat > /workspace/pod_setup.sh" < "$ROOT/prime/pod_setup.sh"
prime pods ssh "$POD_ID" -- "chmod +x /workspace/pod_setup.sh"

echo "[launch] starting pod_setup.sh detached on pod (logs -> /workspace/setup.log)..."
prime pods ssh "$POD_ID" -- "set -a; source /workspace/secrets.env; set +a; \
  nohup bash /workspace/pod_setup.sh > /workspace/setup.log 2>&1 < /dev/null & \
  echo \$! > /workspace/setup.pid; \
  echo 'detached pid: '\$(cat /workspace/setup.pid)"

cat <<EOF

[launch] -----------------------------------------------------------------
[launch]  pod is running pod_setup.sh in the background.
[launch]
[launch]  pod id:    $POD_ID
[launch]  cost rate: \$1.187/hr (1× H200 141GB)
[launch]
[launch]  watch logs:
[launch]    prime pods ssh $POD_ID -- 'tail -f /workspace/setup.log'
[launch]
[launch]  sync results:
[launch]    prime pods ssh $POD_ID -- 'tar czf - -C /workspace/sdpo-vs-grpo/results .' | \\
[launch]      tar xzf - -C $ROOT/results
[launch]
[launch]  terminate when done:
[launch]    prime pods terminate $POD_ID -y
[launch] -----------------------------------------------------------------
EOF

echo "$POD_ID" > "$ROOT/.last_pod_id"
