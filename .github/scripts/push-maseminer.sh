#!/usr/bin/env bash
# Force-push the staged maseminer tree to the public mirror repo.
#
# Usage:    bash .github/scripts/push-maseminer.sh <staged-dir>
# Required: $DEPLOY_KEY env var — SSH private key with write access to
#           the public maseminer repo (a "deploy key" on that repo).
# Required: $MASEMINER_REPO_SSH env var — git@github.com:johanna-einsiedler/maseminer.git
#
# History is intentionally flat: one commit per release, no inherited
# paperlens history.  Researchers don't need it; we don't want to leak it.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <staged-dir>" >&2
  exit 2
fi

DEST="$1"
REMOTE="${MASEMINER_REPO_SSH:-}"

# Decide what to call this release and whether to push a version tag.
#   * Tag-triggered run  (refs/tags/v1.2.3)        → TAG="v1.2.3", push tag.
#   * Manual / branch run (refs/heads/main, etc.)  → TAG=snapshot label,
#                                                    do NOT push a tag (avoids
#                                                    "matches more than one"
#                                                    when the would-be tag name
#                                                    clashes with the branch).
GITHUB_REF="${GITHUB_REF:-}"
if [[ "$GITHUB_REF" == refs/tags/* ]]; then
  TAG="${GITHUB_REF#refs/tags/}"
  PUSH_TAG=1
else
  TAG="${GITHUB_REF_NAME:-snapshot}-$(date +%Y%m%d-%H%M%S)"
  PUSH_TAG=0
fi

if [[ -z "$REMOTE" ]]; then
  echo "Error: MASEMINER_REPO_SSH env var is required " \
       "(e.g. git@github.com:johanna-einsiedler/maseminer.git)." >&2
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
# Qualified refspec — never relies on git's name-resolution between
# branches and tags, so we don't trip "matches more than one".
git push --force origin "refs/heads/main:refs/heads/main"

if [[ "$PUSH_TAG" == "1" ]]; then
  # Mirror the version tag onto the public repo so users can pin
  # ``remotes::install_github('johanna-einsiedler/maseminer@v0.3.1')`` etc.
  git tag -f "$TAG"
  git push --force origin "refs/tags/${TAG}:refs/tags/${TAG}"
  echo "Pushed $TAG to $REMOTE (main + tag)"
else
  echo "Pushed snapshot $TAG to $REMOTE (main only — no tag for non-tag-triggered runs)"
fi
