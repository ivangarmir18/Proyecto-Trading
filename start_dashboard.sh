#!/usr/bin/env bash
set -euo pipefail

echo "[start] Inicializando DB..."
python - <<'EOF'
from core.storage_postgres import make_storage_from_env
s = make_storage_from_env()
s.init_db()
print("DB init OK")
EOF

mkdir -p /tmp/watchlist_logs

echo "[start] Arrancando worker en background (nohup)..."
nohup python -m core.worker > /tmp/watchlist_logs/worker.log 2>&1 &

echo "[start] Worker log: /tmp/watchlist_logs/worker.log"

echo "[start] Arrancando Streamlit en puerto 10000..."
exec streamlit run ui/Home.py --server.port 10000 --server.enableCORS false
