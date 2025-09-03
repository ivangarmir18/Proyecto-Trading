#!/usr/bin/env bash
set -euo pipefail

# Logs
mkdir -p /tmp/watchlist_logs
export WORKER_LOG="/tmp/watchlist_logs/worker.log"

# 1) Inicializar DB (llama init_db() de PostgresStorage)
echo "[start] Inicializando DB..."
python - <<'PY'
import traceback
try:
    # Import aquí para que use la misma PYTHONPATH del proyecto
    from core.storage_postgres import PostgresStorage
    PostgresStorage().init_db()
    print("DB init OK")
except Exception:
    traceback.print_exc()
    raise
PY

# 2) Lanzar worker en background (nohup para seguir si shell cierra)
echo "[start] Arrancando worker en background..."
nohup python run_service.py > "${WORKER_LOG}" 2>&1 &

# 3) Mostrar ubicación de logs
echo "[start] Worker log: ${WORKER_LOG}"

# 4) Lanzar Streamlit en foreground (Render espera proceso principal en foreground)
PORT="${PORT:-8501}"
echo "[start] Arrancando Streamlit en puerto ${PORT}..."
exec streamlit run dashboard/app.py --server.port "${PORT}" --server.address "0.0.0.0" --server.enableCORS false
