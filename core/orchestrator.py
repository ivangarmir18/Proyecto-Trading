# core/orchestrator.py
"""
core/orchestrator.py

Orquestador que trabaja con StorageAdapter (prefiere Postgres).
Expone funciones de alto nivel que la UI/streamlit debe usar:
 - fetch_and_store(asset, interval, start_ts=None, end_ts=None, provider=None)
 - run_backfill_for(asset, interval, start_ts=None, end_ts=None)
 - compute_and_store_indicators(asset, interval, lookback=1000)
 - compute_and_store_scores(asset, interval, method='weighted', weights=None)
 - apply_retention()
 - list_assets(), get_last_candle_ts()
 - health_check()
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from .adapter import StorageAdapter, normalize_weights

# modules which may be present in core/
try:
    from . import fetch as fetch_mod
except Exception:
    fetch_mod = None

try:
    from . import indicators as indicators_mod
except Exception:
    indicators_mod = None

try:
    from . import score as score_mod
except Exception:
    score_mod = None

class Orchestrator:
    def __init__(self, config: Optional[Dict] = None, db_url: Optional[str] = None):
        self.config = config or {}
        self.storage = StorageAdapter(db_url=db_url, config=self.config)
        # ensure schema
        try:
            self.storage.create_schema_if_missing()
        except Exception:
            log.exception("Error asegurando schema en storage.")
        log.info("Orchestrator inicializado (storage backend listo).")

    # --------- Backfill / fetch ----------
    def fetch_and_store(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None, provider: Optional[str] = None):
        """
        Usa core.fetch.run_backfill para recuperar velas y guarda con storage.bulk_insert_candles.
        Actualiza backfill_status al inicio y al final.
        Devuelve resumen { fetched, inserted, last_ts }
        """
        if fetch_mod is None:
            raise RuntimeError("Módulo fetch no disponible.")
        # mark running
        try:
            self.storage.upsert_backfill_status(asset=asset, interval=interval, last_ts=start_ts, status="running")
        except Exception:
            log.exception("No se pudo actualizar backfill_status al iniciar.")

        # fetcher returns list of candles dicts
        log.info("Starting fetch for %s %s (%s->%s)", asset, interval, start_ts, end_ts)
        candles = fetch_mod.get_candles(asset, interval, start_ts=start_ts, end_ts=end_ts, provider_preference=provider)
        fetched = len(candles or [])
        inserted = 0
        last_ts = None
        if candles:
            # ensure sorted ascending
            candles = sorted(candles, key=lambda x: int(x["ts"]))
            # bulk insert using storage's optimized method
            try:
                inserted = self.storage.bulk_insert_candles(candles)
            except Exception:
                log.exception("Error al insertar velas en storage; intentando insert por filas.")
                # attempt per-row insert (fallback)
                inserted = 0
                for r in candles:
                    try:
                        self.storage.bulk_insert_candles([r], batch_size=1)
                        inserted += 1
                    except Exception:
                        continue
            last_ts = int(candles[-1]["ts"])
            # update backfill status
            try:
                self.storage.upsert_backfill_status(asset=asset, interval=interval, last_ts=last_ts, status="done")
            except Exception:
                log.exception("No se pudo actualizar backfill_status al terminar.")
        else:
            # no candles fetched: mark done with same last_ts
            try:
                self.storage.upsert_backfill_status(asset=asset, interval=interval, last_ts=start_ts, status="done")
            except Exception:
                log.exception("No se pudo actualizar backfill_status (no candles).")

        # apply retention if configured
        try:
            retention = self.config.get("retention_days") or {}
            if retention:
                self.storage.apply_retention(retention)
        except Exception:
            log.exception("apply_retention falló tras fetch.")

        stats = {"fetched": fetched, "inserted": inserted, "last_ts": last_ts}
        log.info("Fetch_and_store finished: %s", stats)
        return stats

    def run_backfill_for(self, asset: str, interval: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None, provider: Optional[str] = None):
        """Alias para fetch_and_store (compat)"""
        return self.fetch_and_store(asset, interval, start_ts=start_ts, end_ts=end_ts, provider=provider)

    # --------- Indicators ----------
    def compute_and_store_indicators(self, asset: str, interval: str, lookback: int = 500):
        """
        Lee velas de storage (export_last_candles) y llama a indicators.apply_indicators(df).
        Guarda los resultados en storage.bulk_insert_indicators.
        """
        if indicators_mod is None:
            raise RuntimeError("Módulo indicators no disponible.")
        # get last N candles. Prefer reading full history if indicator needs more lookback.
        candles = self.storage.export_last_candles(asset, interval, limit=lookback*2)
        if not candles:
            log.info("No candles disponibles para indicadores en %s %s", asset, interval)
            return 0
        # convert to pandas df expected by indicators
        import pandas as pd
        df = pd.DataFrame(candles)
        # ensure proper column names and order
        df = df.sort_values("ts").reset_index(drop=True)
        df_ind = indicators_mod.apply_indicators(df, asset=asset, interval=interval)
        # prepare rows
        rows = []
        for _, row in df_ind.iterrows():
            ts = int(row['ts'])
            # gather indicator columns (everything except OHLCV)
            data = row.drop(labels=['id','asset','interval','open','high','low','close','volume'], errors='ignore').to_dict()
            rows.append({"asset": asset, "interval": interval, "ts": ts, "name": "computed", "value": data})
        if rows:
            inserted = self.storage.bulk_insert_indicators(rows)
            log.info("Inserted %d indicator rows for %s %s", inserted, asset, interval)
            return inserted
        return 0

    # --------- Scores ----------
    def compute_and_store_scores(self, asset: str, interval: str, method: str = "weighted", weights: Optional[Dict] = None):
        """
        Lee indicadores guardados, normaliza weights con normalize_weights, llama a score.compute_scores_from_df
        y guarda scores (bulk_insert_scores or insert_score).
        """
        if score_mod is None:
            raise RuntimeError("Módulo score no disponible.")
        # read indicators: prefer export via storage (we stored name="computed" & value JSON)
        # Simple approach: pull indicators from storage.export_last_candles + compute_indicators, or if indicators table present, read via engine.
        # We'll try read via storage.impl.engine if available to load indicator rows. Otherwise fail fast.
        impl = getattr(self.storage, "impl", None)
        df_ind = None
        try:
            if impl and hasattr(impl, "engine"):
                import pandas as pd
                conn = impl.engine.connect()
                q = f"select ts, value from indicators where asset = :asset and interval = :interval order by ts asc"
                df_ind = pd.read_sql(q, conn, params={"asset": asset, "interval": interval})
                conn.close()
        except Exception:
            log.exception("No se pudo leer indicadores desde storage.engine; intentar recomputar desde velas.")
            df_ind = None

        if df_ind is None or df_ind.empty:
            # fallback: compute indicators from last candles
            log.info("Indicadores no encontrados. Recomputando indicadores desde velas para %s %s", asset, interval)
            self.compute_and_store_indicators(asset, interval, lookback=self.config.get("indicator_lookback", 500))
            # try reading again
            try:
                impl = getattr(self.storage, "impl", None)
                if impl and hasattr(impl, "engine"):
                    import pandas as pd
                    conn = impl.engine.connect()
                    q = f"select ts, value from indicators where asset = :asset and interval = :interval order by ts asc"
                    df_ind = pd.read_sql(q, conn, params={"asset": asset, "interval": interval})
                    conn.close()
            except Exception:
                log.exception("Segunda lectura de indicadores falló.")
                df_ind = None

        if df_ind is None or df_ind.empty:
            raise RuntimeError("No hay indicadores para calcular scores en %s %s" % (asset, interval))

        import pandas as pd
        values = pd.json_normalize(df_ind['value'])
        values['ts'] = df_ind['ts'].values
        df_full = values
        # normalize weights
        cfg_weights = weights or (self.config.get('score') or {}).get('weights_defaults') or {}
        norm = normalize_weights(cfg_weights)
        # compute scores
        results = score_mod.compute_scores_from_df(df_full, weights=norm, method=method)
        # results: list of dicts {ts, score, details}
        rows = []
        for r in results:
            rows.append({"asset": asset, "ts": int(r["ts"]), "method": method, "score": float(r["score"]), "details": r.get("details", {})})
        # bulk insert scores when possible
        try:
            inserted = self.storage.bulk_insert_scores(rows)
            log.info("Inserted %d scores for %s %s", inserted, asset, interval)
            return inserted
        except Exception:
            log.exception("bulk_insert_scores failed; falling back to per-row insert")
            inserted = 0
            for r in rows:
                try:
                    self.storage.insert_score(asset=r["asset"], ts=r["ts"], method=r["method"], score=r["score"], details=r.get("details"))
                    inserted += 1
                except Exception:
                    continue
            return inserted

    # --------- Misc ----------
    def apply_retention(self):
        retention = self.config.get("retention_days") or {}
        return self.storage.apply_retention(retention)

    def list_assets(self):
        return self.storage.list_assets()

    def get_last_candle_ts(self, asset: str, interval: str):
        return self.storage.get_last_candle_ts(asset, interval)

    def health_check(self) -> Dict[str, Any]:
        info = {"ok": True, "time": datetime.utcnow().isoformat()}
        impl = getattr(self.storage, "impl", None)
        if impl and hasattr(impl, "engine"):
            try:
                with impl.engine.connect() as conn:
                    r = conn.execute("select 1").scalar()
                    info["db_ping"] = bool(r)
            except Exception as e:
                info["db_ping"] = False
                info["ok"] = False
                info["db_err"] = str(e)
        return info

def make_orchestrator(config: Optional[Dict] = None, db_url: Optional[str] = None) -> Orchestrator:
    return Orchestrator(config=config, db_url=db_url)
