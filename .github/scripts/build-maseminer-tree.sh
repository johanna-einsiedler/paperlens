#!/usr/bin/env bash
# Stage the curated public-mirror tree for the maseminer repo.
#
# Usage:   bash .github/scripts/build-maseminer-tree.sh <dest-dir>
# Env:     GITHUB_REF_NAME (or first git tag at HEAD) is written into VERSION.
#
# Copies an explicit allowlist of files from web/ into <dest-dir>, then
# overlays the public-only files (Procfile, README, LOCAL.md, LICENSE,
# .gitignore) from .github/maseminer-overlay/.  Patches the Dockerfile
# so docker users get PAPERLENS_MASEMINER_ONLY=1 baked in.
#
# Deliberately uses an allowlist (not a denylist + delete pass) so the
# public mirror can only ever contain files we explicitly named.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dest-dir>" >&2
  exit 2
fi

DEST="$1"
SRC_WEB="web"
OVERLAY=".github/maseminer-overlay"
TAG="${GITHUB_REF_NAME:-$(git describe --tags --always 2>/dev/null || echo dev)}"

if [[ ! -d "$SRC_WEB" ]]; then
  echo "Error: $SRC_WEB not found — run this script from the paperlens repo root." >&2
  exit 1
fi
if [[ ! -d "$OVERLAY" ]]; then
  echo "Error: $OVERLAY not found." >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST/static" "$DEST/presets" "$DEST/tests"

# ── Python modules + project metadata ─────────────────────────────────────
for f in \
    server.py db.py jobs.py notifier.py pdf_utils.py \
    prompt_builder.py providers.py presets_loader.py \
    requirements.txt pyproject.toml Dockerfile; do
  cp "$SRC_WEB/$f" "$DEST/$f"
done

# ── Static frontend ───────────────────────────────────────────────────────
for f in index.html app.js masem-builder.js style.css; do
  cp "$SRC_WEB/static/$f" "$DEST/static/$f"
done

# ── MASEMiner presets only — strip non-masem* and archives ────────────────
cp "$SRC_WEB/presets/masem.json"        "$DEST/presets/"
cp "$SRC_WEB/presets/masem-tas20.json"  "$DEST/presets/"
cp "$SRC_WEB/presets/masem.template.md" "$DEST/presets/"

# ── Test suite (smoke-reassurance for users) ──────────────────────────────
for f in "$SRC_WEB"/tests/*.py; do
  cp "$f" "$DEST/tests/"
done

# ── Overlay (public-only files) ───────────────────────────────────────────
cp "$OVERLAY/Procfile"   "$DEST/Procfile"
cp "$OVERLAY/README.md"  "$DEST/README.md"
cp "$OVERLAY/LOCAL.md"   "$DEST/LOCAL.md"
cp "$OVERLAY/LICENSE"    "$DEST/LICENSE"
cp "$OVERLAY/.gitignore" "$DEST/.gitignore"

# ── Patch Dockerfile so docker users get MASEMiner-only by default ────────
# Append ENV after the FROM/COPY layers but before CMD so it's effective.
# Simpler: just append at the end — ENV instructions can appear anywhere.
{
  echo ""
  echo "# Public maseminer build: enable MASEMiner-only mode by default"
  echo "ENV PAPERLENS_MASEMINER_ONLY=1"
} >> "$DEST/Dockerfile"

# ── Write VERSION so users can cite the exact release ─────────────────────
echo "$TAG" > "$DEST/VERSION"

# ── Substitute {{VERSION}} and {{OWNER}} placeholders in overlay docs ─────
# Owner preference order:
#   1. parse from MASEMINER_REPO_SSH (e.g. git@github.com:owner/maseminer.git)
#   2. GITHUB_REPOSITORY_OWNER set by GitHub Actions runner
#   3. literal "<owner>" so a local dry-run shows where the substitution
#      would go, rather than silently injecting an unrelated value.
OWNER=""
if [[ -n "${MASEMINER_REPO_SSH:-}" ]]; then
  _tail="${MASEMINER_REPO_SSH#*:}"     # owner/repo.git
  OWNER="${_tail%/*}"                  # owner
fi
if [[ -z "$OWNER" ]]; then
  OWNER="${GITHUB_REPOSITORY_OWNER:-<owner>}"
fi

# Use perl for portable in-place edits (works on both GNU and BSD sed
# environments — ubuntu-latest runners + local macOS dry-runs).  Values
# are passed via env so they can never break the substitution syntax,
# and \Q...\E disables regex metacharacters in the search side.
for f in "$DEST/README.md" "$DEST/LOCAL.md"; do
  [[ -f "$f" ]] || continue
  OWNER="$OWNER" TAG="$TAG" perl -pi -e '
    s/\Q{{OWNER}}\E/$ENV{OWNER}/g;
    s/\Q{{VERSION}}\E/$ENV{TAG}/g;
  ' "$f"
done

echo "Staged maseminer tree at: $DEST"
echo "Version: $TAG"
echo "Owner:   $OWNER"
echo "File count: $(find "$DEST" -type f | wc -l | tr -d ' ')"
