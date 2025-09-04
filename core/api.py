# core/api.py
"""
FastAPI HTTP API para Watchlist
-------------------------------
Exponer operaciones del Orchestrator (backfill, indicators, scores, train IA, export, health).
Características:
 - Auth opcional por token (env API_TOKEN). Si API_TOKEN no está definido, la API queda abierta.
 - Endpoints asincrónicos que encolan operaciones en background (FastAPI BackgroundTasks) y/o ejecutan síncronamente.
 - Serialización/validación con Pydantic.
 - Endpoints para: health, assets, backfill (start/status), compute_indicators, compute_scores, train_ai, predict_ai,
   export_candles CSV, export_scores CSV.
 - Logging y manejo de errores estándar.
 - Diseñado para correr con Uvicorn/Gunicorn en producción.
"""

from __future__ import annotations
import os
import io
import csv
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime

# core orchestrator factory
from .orchestrator import make_orchestrator

logger = logging.getLogger("watchlist_api")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# Instantiate orchestrator singleton (will read config.json if present and DATABASE_URL)
CFG_PATH = os.getenv("WATCHLIST_CONFIG", "config.json")
_config = {}
if os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH, "r") as fh:
            _config = json.load(fh)
    except Exception:
        logger.exception("No se pudo leer config.json")
ORCH = make_orchestrator(config=_config, db_url=os.getenv("DATABASE_URL"))

# simple token auth dependency
API_TOKEN = os.getenv("API_TOKEN")  # optional
def require_token(authorization: Optional[str] = Header(None)):
    if API_TOKEN:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        # Accept "Bearer <token>"
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid API token")
    return True

app = FastAPI(title="Watchlist API", version="1.0.0")


# -------------------- Pydantic models -------------------- #
class BackfillRequest(BaseModel):
    asset: str = Field(..., description="Símbolo (ej. BTCUSDT, AAPL)")
    interval: str = Field(..., description="Timeframe (5m,15m,30m,1h,4h,12h,1d)")
    start_ts: Optional[int] = Field(None, description="Unix ts (s) inicio, opcional")
    end_ts: Optional[int] = Field(None, description="Unix ts (s) fin, opcional")
    provider: Optional[str] = Field(None, description="Proveedor preferido (binance,yfinance)")


class ComputeReq(BaseModel):
    asset: str
    interval: str
    lookback: Optional[int] = None


class ScoreReq(BaseModel):
    asset: str
    interval: str
    method: Optional[str] = "weighted"
    weights: Optional[Dict[str, float]] = None


class TrainReq(BaseModel):
    asset: str
    interval: str
    forward: Optional[int] = 3
    threshold: Optional[float] = 0.0
    model_path: Optional[str] = None


# -------------------- Utility helpers -------------------- #
def _make_csv_stream(rows: List[Dict[str, Any]], headers: Optional[List[str]] = None):
    """
    Devuelve un generator streaming CSV (utf-8) a partir de lista de dicts.
    """
    buffer = io.StringIO()
    writer = None
    if not rows:
        yield ""
        return
    headers = headers or list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (v if not isinstance(v, (dict, list)) else json.dumps(v)) for k, v in r.items()})
        data = buffer.getvalue()
        yield data
        buffer.seek(0)
        buffer.truncate(0)


# -------------------- Endpoints -------------------- #
@app.get("/health", response_model=Dict[str, Any])
def health(token_ok: bool = Depends(require_token)):
    """Health check: orquestador + db ping"""
    try:
        info = ORCH.health_check()
        return {"ok": True, "timestamp": datetime.utcnow().isoformat(), "orch": info}
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assets", response_model=List[str])
def list_assets(token_ok: bool = Depends(require_token)):
    try:
        assets = ORCH.list_assets()
        return assets
    except Exception as e:
        logger.exception("list_assets error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backfill", response_model=Dict[str, Any])
def start_backfill(req: BackfillRequest, background: BackgroundTasks, token_ok: bool = Depends(require_token)):
    """
    Lanza un backfill en background. Devuelve job info.
    También puedes ejecutar síncrono si no quieres BackgroundTasks (puedes llamar internamente).
    """
    logger.info("API backfill request: %s", req.dict())
    def _task(asset, interval, start_ts, end_ts, provider):
        return ORCH.fetch_and_store(asset, interval, start_ts=start_ts, end_ts=end_ts, provider=provider)
    # Encolar trabajo en background
    background.add_task(_task, req.asset, req.interval, req.start_ts, req.end_ts, req.provider)
    return {"status": "scheduled", "asset": req.asset, "interval": req.interval}


@app.get("/backfill/status", response_model=Dict[str, Any])
def backfill_status(asset: str, interval: str, token_ok: bool = Depends(require_token)):
    try:
        st = ORCH.storage.get_backfill_status(asset, interval)
        return {"asset": asset, "interval": interval, "status": st}
    except Exception as e:
        logger.exception("backfill_status error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indicators", response_model=Dict[str, Any])
def compute_indicators(req: ComputeReq, background: BackgroundTasks, token_ok: bool = Depends(require_token)):
    logger.info("API compute indicators: %s", req.dict())
    if req.lookback is None:
        lookback = ORCH.config.get("indicator_lookback", 500)
    else:
        lookback = req.lookback
    background.add_task(ORCH.compute_and_store_indicators, req.asset, req.interval, lookback)
    return {"status": "scheduled", "asset": req.asset, "interval": req.interval, "lookback": lookback}


@app.post("/scores", response_model=Dict[str, Any])
def compute_scores(req: ScoreReq, background: BackgroundTasks, token_ok: bool = Depends(require_token)):
    logger.info("API compute scores: %s", req.dict())
    weights = req.weights or (ORCH.config.get("score") or {}).get("weights_defaults", {})
    # schedule
    background.add_task(ORCH.compute_and_store_scores, req.asset, req.interval, req.method, weights)
    return {"status": "scheduled", "asset": req.asset, "interval": req.interval, "method": req.method}


@app.post("/train_ai", response_model=Dict[str, Any])
def train_ai(req: TrainReq, background: BackgroundTasks, token_ok: bool = Depends(require_token)):
    """
    Entrena IA en background usando core.ai.train_and_evaluate_from_storage
    """
    logger.info("API train_ai: %s", req.dict())
    # lazy import to avoid heavy deps if not installed
    try:
        from .ai import train_and_evaluate_from_storage
    except Exception as e:
        logger.exception("AI module missing or error")
        raise HTTPException(status_code=500, detail="AI module missing or scikit-learn not installed.")
    # schedule
    background.add_task(train_and_evaluate_from_storage, ORCH.storage.impl if hasattr(ORCH.storage, "impl") else ORCH.storage, req.asset, req.interval, req.forward, req.threshold, req.model_path, None)
    return {"status": "scheduled", "asset": req.asset, "interval": req.interval}


@app.post("/predict_ai", response_model=Dict[str, Any])
def predict_ai(asset: str, interval: str, token_ok: bool = Depends(require_token)):
    """
    Predict probabilities for recent rows using saved model (AI_MODEL_PATH or model_path default).
    Returns list of { ts, prob }.
    """
    try:
        from .ai import load_model, prepare_features_from_indicators, predict_proba
    except Exception:
        raise HTTPException(status_code=500, detail="AI module or sklearn not available.")
    impl = ORCH.storage.impl if hasattr(ORCH.storage, "impl") else ORCH.storage
    # read indicators
    try:
        conn = impl.engine.connect()
        import pandas as pd
        df_ind = pd.read_sql("select ts, value from indicators where asset = :asset and interval = :interval order by ts asc limit 2000", conn, params={"asset": asset, "interval": interval})
        conn.close()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read indicators from storage.")
    if df_ind is None or df_ind.empty:
        raise HTTPException(status_code=404, detail="No indicators found for asset/interval.")
    X, ts = prepare_features_from_indicators(df_ind)
    model = load_model()
    probs = predict_proba(model, X)
    out = [{"ts": int(t), "prob": float(p)} for t, p in zip(ts.values, probs.values)]
    return {"asset": asset, "interval": interval, "predictions": out}


@app.get("/export/candles")
def export_candles(asset: str, interval: str, limit: int = 1000, token_ok: bool = Depends(require_token)):
    """
    Export last N candles as CSV streaming response.
    """
    try:
        rows = ORCH.storage.export_last_candles(asset, interval, limit=limit)
    except Exception as e:
        logger.exception("export_candles error")
        raise HTTPException(status_code=500, detail=str(e))
    generator = _make_csv_stream(rows)
    filename = f"{asset}_{interval}_candles.csv"
    return StreamingResponse(generator, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/export/scores")
def export_scores(asset: str, limit: int = 1000, token_ok: bool = Depends(require_token)):
    """
    Export last N scores as CSV streaming response.
    """
    try:
        impl = ORCH.storage.impl if hasattr(ORCH.storage, "impl") else ORCH.storage
        conn = impl.engine.connect()
        import pandas as pd
        q = "select ts, method, score, details, asset from scores where asset = :asset order by ts desc limit :limit"
        df = pd.read_sql(q, conn, params={"asset": asset, "limit": limit})
        conn.close()
        rows = df.to_dict(orient="records")
    except Exception as e:
        logger.exception("export_scores error")
        raise HTTPException(status_code=500, detail=str(e))
    generator = _make_csv_stream(rows)
    filename = f"{asset}_scores.csv"
    return StreamingResponse(generator, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


# Graceful root
@app.get("/")
def root():
    return {"service": "watchlist-api", "version": "1.0.0", "time": datetime.utcnow().isoformat()}

# Run with: uvicorn core.api:app --host 0.0.0.0 --port 8000
