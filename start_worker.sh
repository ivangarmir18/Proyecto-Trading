#!/usr/bin/env bash
set -euo pipefail

echo "===== START WORKER: instalar dependencias ====="
# si Render ya instala deps en build, puedes comentar la línea siguiente.
pip install -r requirements_prod.txt

echo "===== START WORKER: inicializar DB (idempotente) ====="
python - <<'PYCODE'
from core.storage_postgres import PostgresStorage
try:
    s = PostgresStorage()
    s.init_db()
    print("DB init OK")
except Exception as e:
    import sys, traceback
    traceback.print_exc()
    print("DB init FAILED", file=sys.stderr)
    # no salir: queremos que el worker intente arrancar y muestre errores
PYCODE

echo "===== START WORKER: arrancar servicio worker (run_service.py) ====="
# Ajusta si tu worker script se llama distinto (p.e. run_worker.py)
python run_service.py
