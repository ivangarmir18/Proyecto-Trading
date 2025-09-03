#!/usr/bin/env bash
set -euo pipefail

echo "===== START DASHBOARD: instalar dependencias ====="
pip install -r requirements_prod.txt

echo "===== START DASHBOARD: asegurar DB (opcional) ====="
python - <<'PYCODE'
from core.storage_postgres import PostgresStorage
try:
    s = PostgresStorage()
    s.init_db()
    print("DB init OK (dashboard)")
except Exception as e:
    import sys, traceback
    traceback.print_exc()
    print("DB init FAILED (dashboard)", file=sys.stderr)
PYCODE

echo "===== START DASHBOARD: arrancar streamlit ====="
# Streamlit por defecto escucha en $PORT en Render; usamos opciones para servir correctamente
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false
