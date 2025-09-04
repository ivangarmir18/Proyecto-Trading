#!/usr/bin/env bash
# start_render.sh - robust start script for Render
# (compatible con entornos donde 'set -o pipefail' pueda no estar disponible)

# Hacer el script fallar en caso de error, pero manejar pipefail con fallback.
set -e
set -u
# intentar activar pipefail, si no estÃ¡ disponible, ignorar el fallo
set -o pipefail 2>/dev/null || true

# Logs
mkdir -p /tmp/watchlist_logs
WORKER_LOG="/tmp/watchlist_logs/worker.log"

echo "[start] Inicializando DB..."
python - <<'PY'
import traceback
try:
    from core.storage_postgres import PostgresStorage
    PostgresStorage().init_db()
    print("DB init OK")
except Exception:
    traceback.print_exc()
    raise
PY

echo "[start] Arrancando worker en background (nohup)..."
nohup python run_service.py > "${WORKER_LOG}" 2>&1 &

echo "[start] Worker log: ${WORKER_LOG}"
PORT="${PORT:-8501}"
echo "[start] Arrancando Streamlit en puerto ${PORT}..."
exec streamlit run dashboard/app.py --server.port "${PORT}" --server.address "0.0.0.0" --server.enableCORS false


