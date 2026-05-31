#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-PointCLIP-DAG}"
VISIBILITY="${VISIBILITY:-private}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required. Install gh and run: gh auth login" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d .git ]]; then
  git init
fi

git branch -M "${DEFAULT_BRANCH}"
git add .
git commit -m "Initial PointCLIP-DAG project" || true

if ! git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  gh repo create "${REPO_NAME}" "--${VISIBILITY}" --source . --remote "${REMOTE_NAME}" --push
else
  git push -u "${REMOTE_NAME}" "${DEFAULT_BRANCH}"
fi

echo "GitHub repository is ready: ${REPO_NAME}"
