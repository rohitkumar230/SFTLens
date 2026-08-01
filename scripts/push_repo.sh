#!/usr/bin/env bash
# Run this on your LAPTOP. Copies the repo to the pod.
#
#   bash scripts/push_repo.sh <POD_IP> <PORT>
#
# There is no git remote on this repo, so `git clone` on the pod cannot work.
# The source is 0.4 MB; rsync is faster than creating a GitHub repo and needs
# no account. .venv, runs/ and caches are excluded -- the pod builds its own.
set -euo pipefail

IP=${1:?usage: push_repo.sh <POD_IP> <PORT> [SSH_KEY]}
PORT=${2:?usage: push_repo.sh <POD_IP> <PORT> [SSH_KEY]}
KEY=${3:-$HOME/.ssh/id_ed25519}
DEST=${DEST:-/workspace/sftlens}

cd "$(dirname "$0")/.."

echo "==> pushing $(pwd) -> root@$IP:$DEST"
rsync -avz --delete \
  -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" \
  --exclude '.venv/' \
  --exclude 'runs/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude '.DS_Store' \
  ./ "root@$IP:$DEST/"

echo
echo "==> done. On the pod:"
echo "     cd $DEST && bash scripts/runpod_smoke.sh"
