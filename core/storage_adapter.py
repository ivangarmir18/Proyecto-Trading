# core/storage_adapter.py
"""
StorageAdapter Postgres-only (reflección segura del esquema real).

Detecta si la tabla 'candles' usa 'ts' o 'timestamp' y adapta los inserts/queries.
Corrige:
 - al insertar, rellena BOTH ts (bigint) y timestamp (timestamptz) si existen,
 - al leer, convierte timestamp a ms con EXTRACT(EPOCH)*1000 para devolver siempre ts bigint.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logger = logging.getLogger("core.storage_adapter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


class StorageAdapter:
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL no definido (env o argumento)")
        self.engine = sa.create_engine(self.database_url, future=True, echo=echo)
        self.Session = sessionmaker(bind=self.engine, future=True)
        logger.info("StorageAdapter inicializado con %s", self.database_url)

    # -------------------------
    # Helpers: reflection
    # -------------------------
    def _reflect_candles_table(self) -> Table:
        meta = MetaData()
        try:
            tbl = Table('candles', meta, autoload_with=self.engine)
            return tbl
        except Exception as e:
            logger.exception("No se pudo reflejar la tabla 'candles': %s", e)
            raise

    def _timestamp_column_name(self, tbl: Table) -> Optional[str]:
        cols = {c.name for c in tbl.columns}
        # prefer 'ts' if exists, else 'timestamp' if exists, else None
        if 'ts' in cols:
            return 'ts'
        if 'timestamp' in cols:
            return 'timestamp'
        return None

    # -------------------------
    # Candles upsert/load (robusto)
    # -------------------------
    def upsert_candles(self, asset: str, interval: str, candles: List[Dict[str, Any]]) -> int:
        if not candles:
            return 0

        try:
            tbl = self._reflect_candles_table()
        except Exception:
            logger.exception("upsert_candles: tabla candles no encontrada")
            return 0

        col_names = {c.name for c in tbl.columns}
        ts_col = self._timestamp_column_name(tbl)

        rows = []
        for c in candles:
            try:
                ts_val = int(c.get('ts') or c.get('timestamp') or c.get('time'))
            except Exception:
                logger.debug("Skipping candle with invalid ts: %s", c)
                continue
            row = {}
            # asset & interval if present
            if 'asset' in col_names:
                row['asset'] = asset
            if 'interval' in col_names:
                row['interval'] = interval
            # set ts if exists
            if 'ts' in col_names:
                row['ts'] = ts_val
            # set timestamp if exists (convert ms->datetime UTC)
            if 'timestamp' in col_names:
                try:
                    row['timestamp'] = datetime.utcfromtimestamp(ts_val / 1000.0)
                except Exception:
                    row['timestamp'] = None
            # OHLCV columns
            if 'open' in col_names:
                row['open'] = float(c.get('open')) if c.get('open') is not None else None
            if 'high' in col_names:
                row['high'] = float(c.get('high')) if c.get('high') is not None else None
            if 'low' in col_names:
                row['low'] = float(c.get('low')) if c.get('low') is not None else None
            if 'close' in col_names:
                row['close'] = float(c.get('close')) if c.get('close') is not None else None
            if 'volume' in col_names:
                row['volume'] = float(c.get('volume')) if c.get('volume') is not None else None
            rows.append(row)

        if not rows:
            return 0

        conn = self.engine.connect()
        trans = conn.begin()
        try:
            stmt = insert(tbl).values(rows)
            # Build update mapping only for OHLCV columns present
            update_cols = {}
            for key in ('open', 'high', 'low', 'close', 'volume'):
                if key in col_names:
                    update_cols[key] = getattr(stmt.excluded, key)
            # Build conflict target: prefer asset+interval+ts if available, else try asset+interval+timestamp if present
            index_elems = []
            if 'asset' in col_names:
                index_elems.append('asset')
            if 'interval' in col_names:
                index_elems.append('interval')
            # If 'ts' in columns prefer it in conflict target
            if 'ts' in col_names:
                index_elems.append('ts')
            elif 'timestamp' in col_names:
                index_elems.append('timestamp')

            if index_elems:
                do_update = stmt.on_conflict_do_update(index_elements=index_elems, set_=update_cols)
                conn.execute(do_update)
            else:
                conn.execute(stmt)
            trans.commit()
            return len(rows)
        except Exception as e:
            trans.rollback()
            logger.exception("upsert_candles failed: %s", e)
            # per-row fallback: be careful to include timestamp when required
            try:
                count = 0
                for r in rows:
                    try:
                        cols = list(r.keys())
                        placeholders = ", ".join([f":{k}" for k in cols])
                        cols_sql = ", ".join(cols)
                        conflict_cols = []
                        if 'asset' in col_names:
                            conflict_cols.append('asset')
                        if 'interval' in col_names:
                            conflict_cols.append('interval')
                        if 'ts' in col_names:
                            conflict_cols.append('ts')
                        elif 'timestamp' in col_names:
                            conflict_cols.append('timestamp')
                        if conflict_cols:
                            set_sql_parts = []
                            for k in ('open', 'high', 'low', 'close', 'volume'):
                                if k in col_names:
                                    set_sql_parts.append(f"{k} = EXCLUDED.{k}")
                            set_sql = ", ".join(set_sql_parts) if set_sql_parts else ""
                            sql = f"INSERT INTO candles ({cols_sql}) VALUES ({placeholders}) ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET {set_sql}" if set_sql else f"INSERT INTO candles ({cols_sql}) VALUES ({placeholders}) ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
                        else:
                            sql = f"INSERT INTO candles ({cols_sql}) VALUES ({placeholders})"
                        conn.execute(text(sql), r)
                        count += 1
                    except Exception:
                        logger.exception("per-row upsert failed for %s", r)
                return count
            except Exception:
                logger.exception("Fallback upsert also failed")
                return 0
        finally:
            conn.close()

    def load_candles(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Devuelve lista de dicts con keys asset, interval, ts, open, high, low, close, volume
        Convertimos timestamp -> ms (bigint) mediante EXTRACT(EPOCH)*1000 para compatibilidad.
        """
        try:
            # Use explicit conversion: if timestamp exists we convert to ms, otherwise use ts
            sql = (
                "SELECT "
                "COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) AS ts, "
                "open, high, low, close, volume, asset, interval "
                "FROM candles "
                "WHERE asset = :asset AND interval = :interval"
            )
            params = {"asset": asset, "interval": interval}
            if start_ts is not None:
                sql += " AND COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) >= :start"
                params["start"] = int(start_ts)
            if end_ts is not None:
                sql += " AND COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) <= :end"
                params["end"] = int(end_ts)
            sql += " ORDER BY ts ASC"
            df = pd.read_sql_query(text(sql), con=self.engine, params=params)
            if df.empty:
                return []
            out = []
            for _, r in df.iterrows():
                out.append({
                    "asset": r.get("asset"),
                    "interval": r.get("interval"),
                    "ts": int(r.get("ts")) if not pd.isna(r.get("ts")) else None,
                    "open": r.get("open"),
                    "high": r.get("high"),
                    "low": r.get("low"),
                    "close": r.get("close"),
                    "volume": r.get("volume")
                })
            return out
        except Exception:
            logger.exception("load_candles failed")
            return []

    def get_ohlcv(self, asset: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        try:
            sql = text(
                "SELECT COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) AS ts, open, high, low, close, volume "
                "FROM candles "
                "WHERE asset = :asset AND interval = :interval AND COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) >= :start AND COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) <= :end "
                "ORDER BY ts ASC"
            )
            df = pd.read_sql_query(sql, con=self.engine, params={"asset": asset, "interval": interval, "start": int(start_ms), "end": int(end_ms)})
            if not df.empty:
                df['ts'] = df['ts'].astype('int64')
            return df
        except Exception as e:
            logger.exception("get_ohlcv error: %s", e)
            return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    def list_assets(self) -> List[str]:
        try:
            tbl = self._reflect_candles_table()
            col_names = {c.name for c in tbl.columns}
            if 'asset' in col_names:
                df = pd.read_sql_query(text("SELECT DISTINCT asset FROM candles"), con=self.engine)
                return [r[0] for r in df.values.tolist()]
            return []
        except Exception:
            logger.exception("list_assets failed")
            return []

    # -------------------------
    # Scores & backtests persistence
    # -------------------------
    def save_scores(self, asset: str, interval: str, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        session = self.Session()
        try:
            for r in rows:
                sql = text("INSERT INTO scores(asset, interval, ts, score, details) VALUES (:asset, :interval, :ts, :score, :details)")
                params = {"asset": asset, "interval": interval, "ts": int(r.get("ts") or 0), "score": float(r.get("score") or 0.0), "details": r.get("details") or {}}
                session.execute(sql, params)
            session.commit()
            return len(rows)
        except Exception:
            session.rollback()
            logger.exception("save_scores failed")
            return 0
        finally:
            session.close()

    def save_backtest(self, asset: str, interval: str, result: Dict[str, Any]) -> bool:
        session = self.Session()
        try:
            sql = text("INSERT INTO backtests(asset, interval, run_ts, result) VALUES (:asset, :interval, :run_ts, :result)")
            params = {"asset": asset, "interval": interval, "run_ts": int(result.get("run_ts") or (pd.Timestamp.now().timestamp() * 1000)), "result": result}
            session.execute(sql, params)
            session.commit()
            return True
        except Exception:
            session.rollback()
            logger.exception("save_backtest failed")
            return False
        finally:
            session.close()

    # -------------------------
    # Misc helpers
    # -------------------------
    def make_save_callback(self):
        def cb(asset: str, interval: str, rows: List[Dict[str, Any]]):
            try:
                return self.upsert_candles(asset, interval, rows)
            except Exception:
                logger.exception("make_save_callback failed for %s %s", asset, interval)
                return 0
        return cb

    def clear_old_candles(self, keep_by_interval: Dict[str, int]):
        try:
            tbl = self._reflect_candles_table()
            col_names = {c.name for c in tbl.columns}
            session = self.Session()
            try:
                assets = [r[0] for r in session.execute(text("SELECT DISTINCT asset FROM candles")).fetchall()] if 'asset' in col_names else []
                for asset in assets:
                    for interval, keep in keep_by_interval.items():
                        res = session.execute(text("SELECT COALESCE(ts, (EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) AS ts FROM candles WHERE asset = :asset AND interval = :interval ORDER BY ts DESC LIMIT :keep"), {"asset": asset, "interval": interval, "keep": keep}).fetchall()
                        ts_keep = [r[0] for r in res if r and r[0] is not None]
                        if ts_keep:
                            min_ts = min(ts_keep)
                            session.execute(text("DELETE FROM candles WHERE asset = :asset AND interval = :interval AND COALESCE(ts,(EXTRACT(EPOCH FROM \"timestamp\") * 1000)::bigint) < :min_ts"), {"asset": asset, "interval": interval, "min_ts": int(min_ts)})
                session.commit()
            finally:
                session.close()
        except Exception:
            logger.exception("clear_old_candles failed")

    def dispose(self):
        try:
            self.engine.dispose()
        except Exception:
            pass


# Backward compatibility name
PostgresStorage = StorageAdapter
