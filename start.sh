#!/usr/bin/env bash
# start.sh - entrypoint that starts either the Streamlit UI or the FastAPI API
set -euo pipefail

SERVICE=${SERVICE:-web}
PORT=${PORT:-8000}
# Optional: expose the port for Streamlit through STREAMLIT_SERVER_PORT env if needed
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-$PORT}
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}

echo "[$(date --iso-8601=seconds)] Starting service type='$SERVICE' on port $PORT"

if [ "$SERVICE" = "api" ]; then
  # Start fastapi (uvicorn)
  exec uvicorn core.api:app --host 0.0.0.0 --port "$PORT" --proxy-headers --loop auto
else
  # Default: start Streamlit UI
  exec streamlit run dashboard/app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
fi
