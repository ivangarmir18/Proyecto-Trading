# core/storage_postgres.py
"""
Storage Postgres profesional para Supabase
-----------------------------------------
Responsabilidades:
 - Conectar a Postgres/Supabase usando DATABASE_URL (env) con pooling y SSL opcional.
 - Crear/validar esquema (tablas): assets, candles, indicators, scores, backfill_status, backtests.
 - Bulk inserts optimizados con ON CONFLICT (upsert/no-dup).
 - Operaciones atómicas para backfill_status.
 - Retención de datos por timeframe.
 - Helpers de inspección y salud.

Requisitos recomendados:
 - sqlalchemy >= 1.4
 - psycopg2-binary (para la dialecto postgres en producción)
 - (opcional) pandas para algunos helpers de export

Notas:
 - Timestamps: usamos UNIX seconds (int).
 - Candles: unique(asset, interval, ts) para evitar duplicados.
"""
from __future__ import annotations
import os
import time
import logging
from typing import Optional, List, Dict, Any, Iterable
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, BigInteger, String, Float,
    DateTime, UniqueConstraint, select, and_, delete, text, inspect
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert

log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def _get_db_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DATABASE_URL") or None


def _create_engine(db_url: Optional[str] = None):
    """
    Crea un engine SQLAlchemy con parámetros pensados para Render / Supabase.
    Se pueden configurar vía env:
      - DB_POOL_SIZE (int) default 5
      - DB_MAX_OVERFLOW (int) default 10
      - DB_CONNECT_TIMEOUT (sec) default 10
      - DB_ECHO (bool) default False
    """
    url = db_url or _get_db_url()
    if not url:
        raise RuntimeError("DATABASE_URL no encontrado en el entorno. Define DATABASE_URL apuntando a Supabase/Postgres.")
    pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
    echo = os.getenv("DB_ECHO", "false").lower() in ("1", "true", "yes")

    # Añadir connect_args útiles (sslmode lo controla la URL normalmente)
    connect_args = {"connect_timeout": connect_timeout}

    engine = create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        echo=echo,
        future=True,
        connect_args=connect_args
    )
    log.debug("Engine creado: pool_size=%s max_overflow=%s", pool_size, max_overflow)
    return engine


class PostgresStorage:
    """
    Implementación Postgres/Supabase para persistencia.
    Uso típico:
        storage = PostgresStorage()  # lee DATABASE_URL
        storage.create_schema_if_missing()
        storage.bulk_insert_candles([...])
    """
    def __init__(self, db_url: Optional[str] = None, metadata: Optional[MetaData] = None):
        self.engine = _create_engine(db_url)
        self.metadata = metadata or MetaData()
        self._define_tables()
        # create_all se hace explícitamente con create_schema_if_missing()
        log.info("PostgresStorage inicializado.")

    # ---------- esquema ----------
    def _define_tables(self):
        # assets
        self.assets = Table(
            "assets", self.metadata,
            Column("symbol", String, primary_key=True),
            Column("type", String, nullable=False, default="crypto"),
            Column("meta", JSONB, nullable=True),
        )

        # candles
        self.candles = Table(
            "candles", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False, index=True),
            Column("interval", String, nullable=False, index=True),
            Column("ts", BigInteger, nullable=False),  # unix seconds
            Column("open", Float, nullable=False),
            Column("high", Float, nullable=False),
            Column("low", Float, nullable=False),
            Column("close", Float, nullable=False),
            Column("volume", Float, nullable=True),
            UniqueConstraint("asset", "interval", "ts", name="u_candle"),
        )

        # indicators (store precomputed indicator values as JSONB)
        self.indicators = Table(
            "indicators", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False, index=True),
            Column("interval", String, nullable=False),
            Column("ts", BigInteger, nullable=False, index=True),
            Column("name", String, nullable=False),
            Column("value", JSONB, nullable=True),
        )

        # scores
        self.scores = Table(
            "scores", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False, index=True),
            Column("ts", BigInteger, nullable=False, index=True),
            Column("method", String, nullable=False),
            Column("score", Float, nullable=False),
            Column("details", JSONB, nullable=True),
        )

        # backfill_status
        self.backfill_status = Table(
            "backfill_status", self.metadata,
            Column("asset", String, nullable=False),
            Column("interval", String, nullable=False),
            Column("last_ts", BigInteger, nullable=True),
            Column("status", String, nullable=True),
            Column("updated_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
            UniqueConstraint("asset", "interval", name="u_backfill"),
        )

        # backtests
        self.backtests = Table(
            "backtests", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("asset", String, nullable=False),
            Column("interval", String, nullable=False),
            Column("config", JSONB, nullable=True),
            Column("results", JSONB, nullable=True),
            Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
        )

    def create_schema_if_missing(self):
        """
        Crea las tablas si no existen (idempotente).
        """
        try:
            self.metadata.create_all(self.engine)
            log.info("Schema validado/creado correctamente.")
        except Exception as e:
            log.exception("Error creando esquema: %s", e)
            raise

    # ---------- helpers ----------
    def health_check(self) -> bool:
        """Chequeo rápido de conexión y version."""
        try:
            with self.engine.connect() as conn:
                r = conn.execute(select(text("version()")))
                ver = r.scalar()
                log.debug("Postgres version: %s", ver)
            return True
        except Exception as e:
            log.exception("Health check falló: %s", e)
            return False

    def list_tables(self) -> List[str]:
        insp = inspect(self.engine)
        return insp.get_table_names()

    def count_rows(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            r = conn.execute(text(f"select count(*) from {table_name}"))
            return int(r.scalar() or 0)

    # ---------- assets / upsert ----------
    def upsert_asset(self, symbol: str, tipo: str = "crypto", meta: Optional[dict] = None):
        stmt = pg_insert(self.assets).values(symbol=symbol, type=tipo, meta=meta or {})
        # on conflict do update (simple)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={"type": stmt.excluded.type, "meta": stmt.excluded.meta}
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        log.debug("Asset upserted: %s", symbol)

    # ---------- candles ----------
    def bulk_insert_candles(self, rows: Iterable[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        Inserción en bloque usando ON CONFLICT DO NOTHING.
        rows: iterable de dicts con keys:
            asset, interval, ts (int s), open, high, low, close, volume
        Devuelve número de filas realmente insertadas aproximado (SQLAlchemy no siempre reporta rowcount)
        """
        rows = list(rows)
        if not rows:
            return 0
        inserted = 0
        # Usamos pg_insert (dialect) para ON CONFLICT DO NOTHING
        for i in range(0, len(rows), batch_size):
            block = rows[i:i + batch_size]
            stmt = pg_insert(self.candles).values(block)
            stmt = stmt.on_conflict_do_nothing(index_elements=["asset", "interval", "ts"])
            try:
                with self.engine.begin() as conn:
                    res = conn.execute(stmt)
                    # rowcount en algunos drivers puede ser None; usamos len(block) - estimación de ignorados
                    # No hay forma segura sin comparar antes/after; devolvemos block length como aproximado minimal
                    inserted += int(res.rowcount or len(block))
            except IntegrityError as e:
                log.warning("IntegrityError en bulk_insert_candles: %s", e)
                # intentar insert individualmente para aislar filas problemáticas
                with self.engine.begin() as conn:
                    for r in block:
                        try:
                            conn.execute(pg_insert(self.candles).values(r).on_conflict_do_nothing(index_elements=["asset","interval","ts"]))
                            inserted += 1
                        except Exception:
                            log.exception("Fila problemática: %s", r)
            except Exception as e:
                log.exception("Error bulk_insert_candles: %s", e)
                raise
        log.info("bulk_insert_candles: procesadas=%d, estimadas_insertadas=%d", len(rows), inserted)
        return inserted

    def get_last_candle_ts(self, asset: str, interval: str) -> Optional[int]:
        with self.engine.connect() as conn:
            q = select(self.candles.c.ts).where(
                and_(self.candles.c.asset == asset, self.candles.c.interval == interval)
            ).order_by(self.candles.c.ts.desc()).limit(1)
            r = conn.execute(q).fetchone()
            return int(r[0]) if r else None

    # ---------- indicators ----------
    def bulk_insert_indicators(self, rows: Iterable[Dict[str, Any]], batch_size: int = 500) -> int:
        """
        rows: iterable de dicts con keys asset, interval, ts, name, value (dict)
        """
        rows = list(rows)
        if not rows:
            return 0
        count = 0
        for i in range(0, len(rows), batch_size):
            block = rows[i:i + batch_size]
            stmt = pg_insert(self.indicators).values(block)
            # no duplicates logic by default; you can extend to on_conflict_do_update if quieres versioning
            try:
                with self.engine.begin() as conn:
                    res = conn.execute(stmt)
                    count += int(res.rowcount or len(block))
            except Exception:
                log.exception("Error bulk_insert_indicators en bloque")
        log.info("bulk_insert_indicators: %d filas procesadas", count)
        return count

    # ---------- scores ----------
    def insert_score(self, asset: str, ts: int, method: str, score: float, details: Optional[dict] = None):
        stmt = pg_insert(self.scores).values(asset=asset, ts=int(ts), method=method, score=float(score), details=details or {})
        # Guardar duplicados como entradas separadas (no upsert), por si queremos historico multiple same ts+method
        # Si prefieres upsert por (asset, ts, method) añade unique constraint y on_conflict...
        with self.engine.begin() as conn:
            conn.execute(stmt)
        log.debug("insert_score %s %s -> %s", asset, ts, score)

    def bulk_insert_scores(self, rows: Iterable[Dict[str, Any]], batch_size: int = 500) -> int:
        rows = list(rows)
        if not rows:
            return 0
        count = 0
        for i in range(0, len(rows), batch_size):
            block = rows[i:i + batch_size]
            stmt = pg_insert(self.scores).values(block)
            try:
                with self.engine.begin() as conn:
                    res = conn.execute(stmt)
                    count += int(res.rowcount or len(block))
            except Exception:
                log.exception("Error bulk_insert_scores")
        log.info("bulk_insert_scores: %d filas procesadas", count)
        return count

    # ---------- backfill status ----------
    def upsert_backfill_status(self, asset: str, interval: str, last_ts: Optional[int], status: str):
        stmt = pg_insert(self.backfill_status).values(
            asset=asset, interval=interval, last_ts=last_ts, status=status, updated_at=datetime.now(timezone.utc)
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["asset", "interval"],
            set_={"last_ts": stmt.excluded.last_ts, "status": stmt.excluded.status, "updated_at": stmt.excluded.updated_at}
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        log.debug("upsert_backfill_status %s %s -> %s @%s", asset, interval, status, last_ts)

    def get_backfill_status(self, asset: str, interval: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            q = select(self.backfill_status).where(and_(self.backfill_status.c.asset == asset, self.backfill_status.c.interval == interval))
            r = conn.execute(q).fetchone()
            if not r:
                return None
            return dict(r._mapping)

    # ---------- retention ----------
    def apply_retention(self, retention_days: Dict[str, int]) -> int:
        """
        retention_days: dict { "5m":7, ... }
        Borra filas antiguas por intervalo. Devuelve total aproximado borradas.
        """
        if not retention_days:
            return 0
        total_deleted = 0
        now_ts = int(datetime.now(timezone.utc).timestamp())
        with self.engine.begin() as conn:
            for tf, days in retention_days.items():
                try:
                    days_i = int(days)
                except Exception:
                    continue
                threshold = now_ts - days_i * 24 * 3600
                stmt = delete(self.candles).where(and_(self.candles.c.interval == tf, self.candles.c.ts < threshold))
                res = conn.execute(stmt)
                try:
                    total_deleted += int(res.rowcount or 0)
                except Exception:
                    total_deleted += 0
        log.info("Retention aplicada: ~%d filas borradas", total_deleted)
        return total_deleted

    # ---------- utilities ----------
    def list_assets(self) -> List[str]:
        with self.engine.connect() as conn:
            r = conn.execute(select(self.assets.c.symbol).order_by(self.assets.c.symbol))
            return [row[0] for row in r.fetchall()]

    # ---------- debugging / export ----------
    def export_last_candles(self, asset: str, interval: str, limit: int = 500):
        """
        Devuelve lista de dicts con las últimas `limit` velas para inspección.
        """
        with self.engine.connect() as conn:
            q = select(self.candles).where(and_(self.candles.c.asset == asset, self.candles.c.interval == interval)).order_by(self.candles.c.ts.desc()).limit(limit)
            rows = conn.execute(q).fetchall()
            out = []
            for r in rows:
                m = dict(r._mapping)
                out.append(m)
            return out

    def close(self):
        try:
            self.engine.dispose()
        except Exception:
            pass
        log.info("Engine disposed.")

# ---------- ejemplo mínimo de uso ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    st = PostgresStorage()
    st.create_schema_if_missing()
    print("Tables:", st.list_tables())
    # prueba simple
    now = int(datetime.now(timezone.utc).timestamp())
    st.upsert_asset("BTCUSDT", "crypto", {"note": "ejemplo"})
    st.bulk_insert_candles([
        {"asset": "BTCUSDT", "interval": "5m", "ts": now - 300, "open": 50000, "high": 50010, "low": 49990, "close": 50005, "volume": 1.23},
        {"asset": "BTCUSDT", "interval": "5m", "ts": now, "open": 50005, "high": 50020, "low": 50000, "close": 50010, "volume": 0.5}
    ])
    print("Last candle ts:", st.get_last_candle_ts("BTCUSDT", "5m"))
