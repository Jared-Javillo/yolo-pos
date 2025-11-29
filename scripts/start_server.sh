#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
export PYTHONUNBUFFERED=1

if [ -d "${VENV_PATH}" ]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

cd "${PROJECT_ROOT}"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
