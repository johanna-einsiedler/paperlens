#!/usr/bin/env bash
# Force-push the staged maseminer tree to the public mirror repo.
#
# Usage:    bash .github/scripts/push-maseminer.sh <staged-dir>
# Required: $DEPLOY_KEY env var — SSH private key with write access to
#           the public maseminer repo (a "deploy key" on that repo).
# Required: $MASEMINER_REPO_SSH env var — git@github.com:<owner>/maseminer.git
#
# History is intentionally flat: one commit per release, no inherited
# paperlens history.  Researchers don't need it; we don't want to leak it.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <staged-dir>" >&2
  exit 2
fi

DEST="$1"
TAG="${GITHUB_REF_NAME:-manual-$(date +%Y%m%d-%H%M%S)}"
REMOTE="${MASEMINER_REPO_SSH:-}"

if [[ -z "$REMOTE" ]]; then
  echo "Error: MASEMINER_REPO_SSH env var is required " \
       "(e.g. git@github.com:<owner>/maseminer.git)." >&2
  exit 1
fi
if [[ -z "${DEPLOY_KEY:-}" ]]; then
  echo "Error: DEPLOY_KEY env var is required (SSH private key)." >&2
  exit 1
fi

# ── SSH setup ─────────────────────────────────────────────────────────────
mkdir -p ~/.ssh
printf '%s\n' "$DEPLOY_KEY" > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null

# ── Init fresh repo in the staged dir and push ────────────────────────────
cd "$DEST"
git init -q -b main
git config user.email "ci@maseminer.local"
git config user.name  "maseminer-sync"
git add -A
git commit -q -m "Release $TAG"
git remote add origin "$REMOTE"
git push --force origin main

# Mirror the version tag onto the public repo so users can pin
# ``remotes::install_github('<owner>/maseminer@v0.3.1')`` etc.
git tag -f "$TAG"
git push --force origin "$TAG"

echo "Pushed $TAG to $REMOTE (main + tag)"
