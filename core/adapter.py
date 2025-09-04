# core/adapter.py
"""
core/adapter.py

Factory + wrapper que expone una API uniforme para persistencia.
- Si existe DATABASE_URL (Postgres/Supabase) -> usa PostgresStorage (core.storage_postgres.PostgresStorage).
- Si no, usa un fallback SQLite ligero (LocalStorage) que implementa la misma API mínima.
- Expone funciones/objeto con métodos:
    - create_schema_if_missing()
    - upsert_asset(symbol, tipo, meta)
    - bulk_insert_candles(rows, batch_size=1000)
    - get_last_candle_ts(asset, interval)
    - bulk_insert_indicators(rows, batch_size=500)
    - bulk_insert_scores(rows, batch_size=500)
    - insert_score(asset, ts, method, score, details)
    - upsert_backfill_status(asset, interval, last_ts, status)
    - get_backfill_status(asset, interval)
    - apply_retention(retention_days)
    - list_assets()
    - export_last_candles(asset, interval, limit)
    - close()
- También exporta normalize_weights(weights, expected_keys=None)
"""

from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Try to import the professional Postgres implementation
try:
    from .storage_postgres import PostgresStorage  # professional implementation
    _HAS_POSTGRES_IMPL = True
except Exception:
    PostgresStorage = None
    _HAS_POSTGRES_IMPL = False

# Lightweight SQLite fallback using SQLAlchemy Core (keeps API parity)
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, BigInteger, String, Float, UniqueConstraint, select, and_, delete, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
import pathlib
import json
from datetime import datetime, timezone

class LocalStorageFallback:
    """Fallback local sqlite storage with a subset of Postgres API (for dev/local)."""
    def __init__(self, db_path: Optional[str] = None):
        db_path = db_path or os.getenv("SQLITE_PATH", "data/db/data.db")
        os.makedirs(pathlib.Path(db_path).parent, exist_ok=True)
        url = f"sqlite:///{db_path}"
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        self.metadata = MetaData()
        # define schema (compatible)
        self._define_tables()
        self.metadata.create_all(self.engine)
        log.info("LocalStorageFallback initialized at %s", db_path)

    def _define_tables(self):
        self.assets = Table(
            "assets", self.metadata,
            Column("symbol", String, primary_key=True),
            Column("type", String, nullable=False, default="crypto"),
            Column("meta", String, nullable=True),
        )
        self.candles = Table(
            "candles", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False, index=True),
            Column("interval", String, nullable=False, index=True),
            Column("ts", BigInteger, nullable=False),
            Column("open", Float, nullable=False),
            Column("high", Float, nullable=False),
            Column("low", Float, nullable=False),
            Column("close", Float, nullable=False),
            Column("volume", Float, nullable=True),
            UniqueConstraint("asset", "interval", "ts", name="u_candle"),
        )
        self.indicators = Table(
            "indicators", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False),
            Column("interval", String, nullable=False),
            Column("ts", BigInteger, nullable=False),
            Column("name", String, nullable=False),
            Column("value", String, nullable=True),
        )
        self.scores = Table(
            "scores", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False),
            Column("ts", BigInteger, nullable=False),
            Column("method", String, nullable=False),
            Column("score", Float, nullable=False),
            Column("details", String, nullable=True),
        )
        self.backfill_status = Table(
            "backfill_status", self.metadata,
            Column("asset", String, nullable=False),
            Column("interval", String, nullable=False),
            Column("last_ts", BigInteger, nullable=True),
            Column("status", String, nullable=True),
            Column("updated_at", String, nullable=True),
            UniqueConstraint("asset", "interval", name="u_backfill"),
        )

    def create_schema_if_missing(self):
        self.metadata.create_all(self.engine)

    def upsert_asset(self, symbol: str, tipo: str = "crypto", meta: Optional[dict] = None):
        meta_s = json.dumps(meta or {})
        stmt = sqlite_insert(self.assets).values(symbol=symbol, type=tipo, meta=meta_s)
        # sqlite insert...on_conflict is supported via dialect extension; fallback simple try/except
        with self.engine.begin() as conn:
            try:
                conn.execute(stmt)
            except IntegrityError:
                conn.execute(self.assets.update().where(self.assets.c.symbol == symbol).values(type=tipo, meta=meta_s))

    def bulk_insert_candles(self, rows: List[dict], batch_size: int = 500):
        if not rows:
            return 0
        inserted = 0
        with self.engine.begin() as conn:
            for r in rows:
                try:
                    conn.execute(self.candles.insert().values(**r))
                    inserted += 1
                except IntegrityError:
                    continue
        return inserted

    def get_last_candle_ts(self, asset: str, interval: str):
        with self.engine.connect() as conn:
            q = select(self.candles.c.ts).where(and_(self.candles.c.asset == asset, self.candles.c.interval == interval)).order_by(self.candles.c.ts.desc()).limit(1)
            r = conn.execute(q).fetchone()
            return int(r[0]) if r else None

    def bulk_insert_indicators(self, rows: List[dict], batch_size: int = 200):
        if not rows:
            return 0
        count = 0
        with self.engine.begin() as conn:
            for r in rows:
                try:
                    r2 = r.copy()
                    r2['value'] = json.dumps(r2.get('value', {}))
                    conn.execute(self.indicators.insert().values(**r2))
                    count += 1
                except Exception:
                    continue
        return count

    def bulk_insert_scores(self, rows: List[dict], batch_size: int = 200):
        if not rows:
            return 0
        count = 0
        with self.engine.begin() as conn:
            for r in rows:
                try:
                    r2 = r.copy()
                    r2['details'] = json.dumps(r2.get('details', {}))
                    conn.execute(self.scores.insert().values(**r2))
                    count += 1
                except Exception:
                    continue
        return count

    def insert_score(self, asset: str, ts: int, method: str, score: float, details: Optional[dict] = None):
        d = json.dumps(details or {})
        with self.engine.begin() as conn:
            conn.execute(self.scores.insert().values(asset=asset, ts=int(ts), method=method, score=float(score), details=d))

    def upsert_backfill_status(self, asset: str, interval: str, last_ts: Optional[int], status: str):
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            try:
                conn.execute(self.backfill_status.insert().values(asset=asset, interval=interval, last_ts=last_ts, status=status, updated_at=now))
            except IntegrityError:
                conn.execute(self.backfill_status.update().where(and_(self.backfill_status.c.asset == asset, self.backfill_status.c.interval == interval)).values(last_ts=last_ts, status=status, updated_at=now))

    def get_backfill_status(self, asset: str, interval: str):
        with self.engine.connect() as conn:
            r = conn.execute(select(self.backfill_status).where(and_(self.backfill_status.c.asset == asset, self.backfill_status.c.interval == interval))).fetchone()
            return dict(r._mapping) if r else None

    def apply_retention(self, retention_days: Dict[str, int]):
        if not retention_days:
            return 0
        now_ts = int(datetime.utcnow().timestamp())
        total = 0
        with self.engine.begin() as conn:
            for tf, days in retention_days.items():
                try:
                    days_i = int(days)
                except Exception:
                    continue
                thresh = now_ts - days_i * 24 * 3600
                res = conn.execute(delete(self.candles).where(and_(self.candles.c.interval == tf, self.candles.c.ts < thresh)))
                try:
                    total += int(res.rowcount or 0)
                except Exception:
                    total += 0
        return total

    def list_assets(self):
        with self.engine.connect() as conn:
            r = conn.execute(select(self.assets.c.symbol).order_by(self.assets.c.symbol)).fetchall()
            return [row[0] for row in r]

    def export_last_candles(self, asset: str, interval: str, limit: int = 500):
        with self.engine.connect() as conn:
            r = conn.execute(select(self.candles).where(and_(self.candles.c.asset == asset, self.candles.c.interval == interval)).order_by(self.candles.c.ts.desc()).limit(limit)).fetchall()
            return [dict(row._mapping) for row in r]

    def close(self):
        try:
            self.engine.dispose()
        except Exception:
            pass

# ---------- Factory ----------
def _use_postgres() -> bool:
    db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DATABASE_URL")
    return bool(db_url and _HAS_POSTGRES_IMPL)

class StorageAdapter:
    """
    Wrapper that exposes a uniform API but delegates to PostgresStorage when available.
    """
    def __init__(self, db_url: Optional[str] = None, config: Optional[dict] = None):
        self.config = config or {}
        db_url = db_url or os.getenv("DATABASE_URL")
        if db_url and _HAS_POSTGRES_IMPL:
            log.info("StorageAdapter -> using PostgresStorage (Supabase).")
            self.impl = PostgresStorage(db_url=db_url)
            # ensure schema
            try:
                self.impl.create_schema_if_missing()
            except Exception:
                log.exception("Error creando schema en PostgresStorage.")
        else:
            log.info("StorageAdapter -> using LocalStorageFallback (SQLite).")
            self.impl = LocalStorageFallback(db_path=os.getenv("SQLITE_PATH", "data/db/data.db"))

    # Delegate methods
    def create_schema_if_missing(self):
        return self.impl.create_schema_if_missing()

    def upsert_asset(self, symbol: str, tipo: str = "crypto", meta: Optional[dict] = None):
        return self.impl.upsert_asset(symbol, tipo, meta)

    def bulk_insert_candles(self, rows: List[dict], batch_size: int = 1000):
        return self.impl.bulk_insert_candles(rows, batch_size=batch_size)

    def get_last_candle_ts(self, asset: str, interval: str):
        return self.impl.get_last_candle_ts(asset, interval)

    def bulk_insert_indicators(self, rows: List[dict], batch_size: int = 500):
        return self.impl.bulk_insert_indicators(rows, batch_size=batch_size)

    def bulk_insert_scores(self, rows: List[dict], batch_size: int = 500):
        return self.impl.bulk_insert_scores(rows, batch_size=batch_size)

    def insert_score(self, asset: str, ts: int, method: str, score: float, details: Optional[dict] = None):
        return self.impl.insert_score(asset, ts, method, score, details)

    def upsert_backfill_status(self, asset: str, interval: str, last_ts: Optional[int], status: str):
        return self.impl.upsert_backfill_status(asset, interval, last_ts, status)

    def get_backfill_status(self, asset: str, interval: str):
        return self.impl.get_backfill_status(asset, interval)

    def apply_retention(self, retention_days: Dict[str, int]):
        return self.impl.apply_retention(retention_days)

    def list_assets(self):
        return self.impl.list_assets()

    def export_last_candles(self, asset: str, interval: str, limit: int = 500):
        return self.impl.export_last_candles(asset, interval, limit=limit)

    def close(self):
        return self.impl.close()


# ---------- utils ----------
def normalize_weights(weights: Dict[str, float], expected_keys: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Normaliza y valida un dict de pesos. Evita problemas como nombres distintos
    y sumas > 1. Se alinea con expected_keys si se pasa.
    """
    if not weights:
        weights = {}
    w = {k.lower(): float(v) for k, v in weights.items() if v is not None}
    # quitar negativos
    w = {k: max(0.0, v) for k, v in w.items()}
    # keys esperadas (normalizar nombres simples)
    keys = [k.lower() for k in (expected_keys or list(w.keys()) or ["ema", "support", "atr", "macd", "rsi", "fibonacci"])]
    # construir base
    base = {k: float(w.get(k, 0.0)) for k in keys}
    total = sum(base.values())
    if total <= 0:
        # repartir uniformemente
        n = len(base)
        if n == 0:
            return {}
        val = 1.0 / n
        base = {k: val for k in base}
        return base
    # normalizar a suma 1.0 (mantener proporciones)
    base = {k: (v / total) for k, v in base.items()}
    return base
