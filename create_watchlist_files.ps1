# create_watchlist_files.ps1
# Crea Procfile, Dockerfile, start.sh, .dockerignore, .env.example en el directorio actual.
# Guardar este fichero con nombre create_watchlist_files.ps1 y luego ejecutarlo en PowerShell.

$ErrorActionPreference = "Stop"

# Procfile
@'
# Procfile - define process types
# Web UI (Streamlit)
web: SERVICE=web bash start.sh

# API (FastAPI via Uvicorn)
api: SERVICE=api bash start.sh
'@ | Out-File -Encoding UTF8 -FilePath Procfile -Force

# Dockerfile
@'
# Dockerfile - multienv entrypoint for Streamlit UI or FastAPI API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -s /app/requirements.txt ]; then pip install -r /app/requirements.txt; fi

COPY . /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["bash", "start.sh"]
'@ | Out-File -Encoding UTF8 -FilePath Dockerfile -Force

# start.sh
@'
#!/usr/bin/env bash
set -euo pipefail

SERVICE=${SERVICE:-web}
PORT=${PORT:-8000}
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-$PORT}
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}

echo "[$(date --iso-8601=seconds)] Starting service type='\$SERVICE' on port \$PORT"

if [ "\$SERVICE" = "api" ]; then
  exec uvicorn core.api:app --host 0.0.0.0 --port "\$PORT" --proxy-headers --loop auto
else
  exec streamlit run dashboard/app.py --server.port "\$PORT" --server.address 0.0.0.0 --server.headless true
fi
'@ | Out-File -Encoding UTF8 -FilePath start.sh -Force

# .dockerignore
@'
__pycache__
*.pyc
*.pyo
*.pyd
.env
.env.*
*.sqlite3
data/db/
*.egg-info
dist/
build/
.vscode/
.idea/
.git
.gitignore
.DS_Store
*.log
'@ | Out-File -Encoding UTF8 -FilePath .dockerignore -Force

# .env.example
@'
# .env.example - copia esto a .env y rellena los valores en deploy
DATABASE_URL=postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DBNAME>?sslmode=require
API_TOKEN=changeme_supersecret_token
BINANCE_API_KEY=
BINANCE_API_SECRET=
FINNHUB_API_KEY=
YFINANCE_API_KEY=
WATCHLIST_CONFIG=config.json
LOG_LEVEL=INFO
SERVICE=web
PORT=8000
'@ | Out-File -Encoding UTF8 -FilePath .env.example -Force

Write-Host "Archivos creados: Procfile, Dockerfile, start.sh, .dockerignore, .env.example"
