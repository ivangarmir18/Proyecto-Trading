#!/bin/bash
set -euo pipefail

echo "[start] Inicializando DB..."
python - <<'PY'
from core.storage_postgres import PostgresStorage
try:
    s = PostgresStorage()
    print("[start] DB OK")
except Exception as e:
    print("[start] Error inicializando Postgres connection pool:", e)
    raise
PY

mkdir -p /tmp/watchlist_logs

echo "[start] Arrancando worker en background (nohup)..."
nohup python run_service.py > /tmp/watchlist_logs/worker.log 2>&1 &

echo "[start] Worker log: /tmp/watchlist_logs/worker.log"

echo "[start] Arrancando Streamlit en puerto 10000..."
exec streamlit run dashboard/app.py --server.port 10000 --server.address 0.0.0.0
