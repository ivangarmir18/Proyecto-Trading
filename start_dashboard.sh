#!/usr/bin/env bash
set -euo pipefail

# Environment hints:
PORT=${PORT:-8501}
HOST=${HOST:-0.0.0.0}
DB_WAIT_RETRIES=${DB_WAIT_RETRIES:-12}
DB_WAIT_SECONDS=${DB_WAIT_SECONDS:-5}
INIT_DB_ON_START=${INIT_DB_ON_START:-1}   # 1 = run scripts/init_db.py on start
LOG_FILE=${LOG_FILE:-""}

echo "Starting dashboard wrapper. PORT=${PORT} HOST=${HOST}"

# Optional: wait for DATABASE_URL to become available (network DB startup)
if [ -n "${DATABASE_URL:-}" ] && [ "${DB_WAIT_RETRIES}" -gt 0 ]; then
  echo "Waiting for DATABASE_URL to be reachable..."
  i=0
  while [ $i -lt "$DB_WAIT_RETRIES" ]; do
    # crude attempt: try to run scripts/init_db.py with a dry-run to check connectivity
    if python -c "import os,sys; from core.storage_postgres import make_storage_from_env; s=make_storage_from_env(); print('OK');" >/dev/null 2>&1; then
      echo "Database reachable."
      break
    fi
    i=$((i+1))
    echo "DB not ready. retry ${i}/${DB_WAIT_RETRIES}. waiting ${DB_WAIT_SECONDS}s..."
    sleep "${DB_WAIT_SECONDS}"
  done
fi

# optional DB init
if [ "${INIT_DB_ON_START}" = "1" ]; then
  echo "Running DB init (scripts/init_db.py) ..."
  python scripts/init_db.py --retries 3 --wait 3 || echo "scripts/init_db.py failed (continuing)"
fi

# Start streamlit (bind to $PORT)
# We recommend disabling CORS for containerized environments if necessary and setting server.fileWatcherType=none in production
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_SERVER_ADDRESS="${HOST}"
export STREAMLIT_SERVER_PORT="${PORT}"
# Optional: send logs to file + stdout
if [ -n "${LOG_FILE}" ]; then
  echo "Logging to ${LOG_FILE}"
  # Use tee to write both stdout and file
  streamlit run dashboard/app_full.py --server.port ${PORT} --server.address ${HOST} 2>&1 | tee -a "${LOG_FILE}"
else
  exec streamlit run dashboard/app_full.py --server.port ${PORT} --server.address ${HOST}
fi
