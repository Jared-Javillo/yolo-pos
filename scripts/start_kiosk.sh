#!/usr/bin/env bash
set -euo pipefail

APP_URL="${APP_URL:-http://127.0.0.1:8000}" 
CHROMIUM_BIN="${CHROMIUM_BIN:-/snap/bin/chromium}"
WAIT_FOR="${WAIT_FOR:-http://127.0.0.1:8000/health}"
MAX_RETRIES=${MAX_RETRIES:-60}
SLEEP_SECONDS=${SLEEP_SECONDS:-2}

export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-"${HOME}/.Xauthority"}

for i in $(seq 1 ${MAX_RETRIES}); do
  if curl --silent --fail "${WAIT_FOR}" >/dev/null; then
    break
  fi
  sleep "${SLEEP_SECONDS}"
  if [ "${i}" -eq "${MAX_RETRIES}" ]; then
    echo "Camera server did not become ready; starting Chromium anyway" >&2
  fi
done

exec "${CHROMIUM_BIN}" \
  --kiosk \
  --start-fullscreen \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  --overscroll-history-navigation=0 \
  --app="${APP_URL}"
