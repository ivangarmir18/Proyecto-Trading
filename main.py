#!/usr/bin/env python3
"""
main.py - Orquestador del proyecto Watchlist (versión ajustada para compatibilidad)
Basado en el main que me pegaste, con correcciones mínimas:
 - Unifica DB por defecto en data/db/data.db
 - Evita pasar db_path a compute_all_scores (compatibilidad)
 - Pasa db_path a fetch/backfill si la función lo admite
 - Guarda el DataFrame devuelto por compute_all_scores cuando se invoca por asset
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import time

import pandas as pd

# try to import core modules (fallos controlados)
try:
    from core import storage
except Exception:
    storage = None

try:
    import indicators as indicators_pkg
except Exception:
    indicators_pkg = None

try:
    from core import fetch as core_fetch
except Exception:
    core_fetch = None

try:
    from core import score as core_score
except Exception:
    core_score = None

try:
    from core import ai_inference
except Exception:
    ai_inference = None

# utils: try to use core.utils if present, else fallback simple functions here
try:
    from core.utils import load_json, ensure_dir, setup_logging, ts_to_iso  # type: ignore
except Exception:
    def load_json(p: str) -> dict:
        pth = Path(p)
        if not pth.exists():
            return {}
        with pth.open('r', encoding='utf-8') as f:
            return json.load(f)
    def ensure_dir(p: str):
        Path(p).mkdir(parents=True, exist_ok=True)
        return Path(p)
    def setup_logging(cfg: Optional[dict] = None):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    def ts_to_iso(ts: int) -> str:
        return pd.to_datetime(int(ts), unit='s').strftime("%Y-%m-%dT%H:%M:%SZ")


logger = logging.getLogger("main")
setup_logging()

DEFAULT_CONFIG = "config.json"
ROOT = Path(__file__).resolve().parent

# Default unified DB path (aligned to what quieres: data/db/data.db)
DEFAULT_DB_PATH = str(ROOT / "data" / "db" / "data.db")


def _parse_list_arg(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def init_database(db_path: Optional[str]):
    if storage is None:
        logger.error("core.storage no disponible. Copia core/storage.py a core/ o instala el módulo.")
        raise RuntimeError("core.storage no disponible")
    logger.info(f"Inicializando DB en: {db_path}")
    storage.init_db(db_path)


# -----------------------
# Asset loading helpers
# -----------------------
def read_symbols_from_csv(path: str, col: str = "symbol") -> List[str]:
    p = Path(path)
    if not p.exists():
        logger.warning("Asset CSV not found: %s", path)
        return []
    try:
        df = pd.read_csv(p)
    except Exception as e:
        logger.warning("Error reading CSV %s: %s", path, e)
        return []
    # if header absent, read first column
    if col not in df.columns:
        # try first column
        if df.shape[1] >= 1:
            return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        return []
    return df[col].dropna().astype(str).str.strip().tolist()


def load_assets_from_config(config: dict) -> List[str]:
    """
    Read assets from config dict. Accepts:
      - config["asset_files"] as dict {'cryptos': path, 'stocks': path}
      - config["asset_files"] as list of paths
      - config["assets"] as dict {'cripto': [...], 'acciones': [...]}
      - config["assets"] as list of symbols
    Returns merged list (deduplicated) preserving order.
    """
    assets: List[str] = []

    # 1) asset_files: either dict or list
    af = config.get("asset_files") or config.get("assets_files") or config.get("asset_files_paths")
    if af:
        if isinstance(af, dict):
            for k, path in af.items():
                try:
                    assets += read_symbols_from_csv(path)
                except Exception:
                    logger.debug("failed reading asset file %s for key %s", path, k)
        elif isinstance(af, list):
            for path in af:
                assets += read_symbols_from_csv(path)
        elif isinstance(af, str):
            assets += read_symbols_from_csv(af)

    # 2) config["assets"] legacy support
    cfg_assets = config.get("assets")
    if cfg_assets:
        if isinstance(cfg_assets, dict):
            # support keys like 'cripto'/'acciones' or 'cryptos'/'stocks'
            for k in ('cripto','criptoS','cryptos','stocks','acciones'):
                if k in cfg_assets:
                    try:
                        assets += [str(x).strip() for x in cfg_assets[k]]
                    except Exception:
                        pass
            # also support if config["assets"] is a flat list under different key
            for v in cfg_assets.values():
                if isinstance(v, list):
                    assets += [str(x).strip() for x in v]
        elif isinstance(cfg_assets, list):
            assets += [str(x).strip() for x in cfg_assets]

    # dedupe preserving order
    seen = set()
    merged: List[str] = []
    for a in assets:
        if not a:
            continue
        if a not in seen:
            seen.add(a)
            merged.append(a)
    return merged


def load_assets_from_cli_args(assets_arg: Optional[str], assets_file_args: Optional[List[str]]) -> List[str]:
    assets: List[str] = []
    # --assets accepts comma-separated symbols or a single CSV path if .csv extension
    if assets_arg:
        # if it's a single path that endswith .csv treat it as file
        if assets_arg.lower().endswith('.csv') and Path(assets_arg).exists():
            assets += read_symbols_from_csv(assets_arg)
        else:
            assets += _parse_list_arg(assets_arg)

    # --assets-file supports multiple
    if assets_file_args:
        for p in assets_file_args:
            assets += read_symbols_from_csv(p)
    # dedupe
    seen = set()
    merged: List[str] = []
    for a in assets:
        if not a:
            continue
        if a not in seen:
            seen.add(a)
            merged.append(a)
    return merged


# -----------------------
# Backfill / refresh wrappers (improved to pass db_path if possible)
# -----------------------
def run_backfill(config: dict, assets: List[str], intervals: List[str], db_path: str):
    if core_fetch is None:
        logger.warning("core.fetch no disponible: no puedo lanzar backfill.")
        return False, "core.fetch no disponible"
    try:
        if hasattr(core_fetch, "backfill_for_assets"):
            logger.info("Lanzando backfill_for_assets(...)")
            # try to pass db_path if function accepts it
            try:
                core_fetch.backfill_for_assets(assets, intervals, db_path=db_path)
            except TypeError:
                core_fetch.backfill_for_assets(assets, intervals)
            return True, "backfill_for_assets ejecutado"
        elif hasattr(core_fetch, "backfill_historical"):
            logger.info("Lanzando backfill_historical(...)")
            try:
                # prefer to pass db_path if accepted
                core_fetch.backfill_historical(crypto_interval=str(intervals[0]) if intervals else "1h",
                                               stock_interval=str(intervals[0]) if intervals else "1h",
                                               db_path=db_path)
            except TypeError:
                # fallback signature without db_path
                core_fetch.backfill_historical()
            return True, "backfill_historical ejecutado"
        else:
            return False, "core.fetch no implementa backfill callable esperado"
    except Exception as e:
        logger.exception("Backfill falló")
        return False, f"Backfill error: {e}"


def run_refresh_watchlist(config: dict, crypto_list: List[str], stock_list: List[str], crypto_interval: str = "5m", stock_resolution: str = "5", db_path: Optional[str] = None):
    if core_fetch is None:
        logger.warning("core.fetch no disponible: no puedo refrescar watchlist.")
        return False, "core.fetch no disponible"
    try:
        if hasattr(core_fetch, "refresh_watchlist"):
            # try to pass db_path if accepted
            try:
                core_fetch.refresh_watchlist(crypto_list, stock_list, crypto_interval=crypto_interval, stock_resolution=stock_resolution, save_to_db=True, db_path=db_path)
            except TypeError:
                core_fetch.refresh_watchlist(crypto_list, stock_list, crypto_interval=crypto_interval, stock_resolution=stock_resolution, save_to_db=True)
            return True, "refresh_watchlist lanzado"
        else:
            return False, "core.fetch no implementa refresh_watchlist"
    except Exception as e:
        logger.exception("refresh_watchlist falló")
        return False, f"refresh error: {e}"


# -----------------------
# Indicators & Scores wrappers
# -----------------------
def compute_and_save_indicators_for(asset: str, interval: str, db_path: str, fib_lookback: int = 144):
    if storage is None:
        raise RuntimeError("core.storage no disponible")
    if indicators_pkg is None:
        raise RuntimeError("paquete indicators no disponible")

    df = storage.load_candles(asset, interval, db_path=db_path)
    if df is None or df.empty:
        logger.warning(f"No hay candles para {asset} {interval}")
        return False, "no candles"
    for c in ['ts', 'open', 'high', 'low', 'close']:
        if c not in df.columns:
            raise RuntimeError(f"columna {c} faltante en candles para {asset} {interval}")

    df_ind = indicators_pkg.compute_all_indicators(df, symbol=asset, fib_lookback=fib_lookback)
    out = pd.DataFrame({
        'asset': asset,
        'interval': interval,
        'ts': df_ind['ts'],
        'ema9': df_ind.get('ema9'),
        'ema40': df_ind.get('ema40'),
        'atr': df_ind.get('atr'),
        'macd': df_ind.get('macd_line'),
        'macd_signal': df_ind.get('macd_signal'),
        'rsi': df_ind.get('rsi'),
        'support': df_ind.get('support'),
        'resistance': df_ind.get('resistance'),
        'fibonacci_levels': df_ind.get('fibonacci_levels').apply(lambda x: json.dumps(x) if x is not None else None)
    })
    storage.save_indicators(out, db_path=db_path)
    logger.info(f"Indicadores guardados para {asset} {interval} (filas: {len(out)})")
    return True, f"indicators_saved:{len(out)}"


def compute_and_save_scores_for(asset: str, interval: str, db_path: str, config: dict):
    if storage is None:
        raise RuntimeError("core.storage no disponible")
    if core_score is None:
        raise RuntimeError("core.score no disponible")

    # Prefer compute_all_scores single-asset mode if available
    if hasattr(core_score, "compute_all_scores"):
        try:
            logger.info(f"Llamando core.score.compute_all_scores para {asset} {interval}")
            # Llamamos sin el kw db_path para evitar TypeError; la función puede aceptar
            # (asset, interval, config=...) en su modo single-asset.
            df_scores = None
            try:
                # intento con la firma asset, interval, config=...
                df_scores = core_score.compute_all_scores(asset, interval, config=config)
            except TypeError:
                # intento la firma antigua (sin args) -> la función procesa todos los CSV, devolviendo dict
                res_all = core_score.compute_all_scores()  # keep old behavior
                # if old behavior returns dict, try to extract our asset/interval result
                if isinstance(res_all, dict) and asset in res_all:
                    # res_all[asset] may be a dict {"last":..., "rows":...}
                    entry = res_all.get(asset)
                    if isinstance(entry, dict) and "rows" in entry:
                        df_scores = pd.DataFrame(entry["rows"])
                        df_scores["asset"] = asset
                        df_scores["interval"] = interval
                else:
                    df_scores = None

            if df_scores is None:
                logger.info("compute_all_scores no devolvió DataFrame para modo single-asset; fallback a compute_scores si existe")
            else:
                # Persist via storage.save_scores
                # Ensure expected columns and types
                df_scores = df_scores.copy()
                # fill created_at if missing
                if 'created_at' not in df_scores.columns:
                    df_scores['created_at'] = int(time.time())
                # if storage.save_scores expects DataFrame and db_path kw
                storage.save_scores(df_scores, db_path=db_path)
                logger.info(f"Scores guardados para {asset} {interval}: {len(df_scores)} filas")
                return True, f"scores_saved:{len(df_scores)}"
        except Exception as e:
            logger.exception("compute_all_scores falló")
            return False, f"compute_all_scores error: {e}"

    # Fallback: use compute_scores(candles, indicators, config)
    if hasattr(core_score, "compute_scores"):
        try:
            candles = storage.load_candles(asset, interval, db_path=db_path)
            indicators = storage.load_indicators(asset, interval, db_path=db_path)
            df_scores = core_score.compute_scores(candles, indicators, config=config)
            if isinstance(df_scores, pd.DataFrame):
                storage.save_scores(df_scores, db_path=db_path)
                return True, f"scores_saved:{len(df_scores)}"
            else:
                logger.error("compute_scores devolvió un tipo inesperado")
                return False, "compute_scores returned unexpected type"
        except Exception as e:
            logger.exception("fallback compute_scores falló")
            return False, f"fallback error: {e}"

    logger.error("core.score no expone compute_all_scores ni compute_scores; implementar en core/score.py")
    return False, "no scoring function"


# -----------------------
# Pipeline orchestration
# -----------------------
def run_pipeline(assets: List[str], intervals: List[str], db_path: str, config: Dict[str, Any], do_backfill: bool = False, do_refresh: bool = False):
    # ensure DB dir exists and init DB
    ensure_dir(Path(db_path).parent.as_posix())
    init_database(db_path)

    if do_backfill:
        ok, msg = run_backfill(config, assets, intervals, db_path=db_path)
        logger.info("Backfill: %s %s", ok, msg)

    if do_refresh:
        ok, msg = run_refresh_watchlist(config, assets, [], db_path=db_path)
        logger.info("Refresh: %s %s", ok, msg)

    for asset in assets:
        for interval in intervals:
            logger.info("Procesando %s @ %s", asset, interval)
            try:
                r_ind, msg_ind = compute_and_save_indicators_for(asset, interval, db_path, fib_lookback=config.get("fibonacci_lookback", 144))
                logger.info("Indicators result: %s %s", r_ind, msg_ind)
            except Exception as e:
                logger.exception("Error computing indicators for %s %s", asset, interval)
                continue

            try:
                r_sc, msg_sc = compute_and_save_scores_for(asset, interval, db_path, config)
                logger.info("Scores result: %s %s", r_sc, msg_sc)
            except Exception as e:
                logger.exception("Error computing scores for %s %s", asset, interval)
                continue


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=DEFAULT_CONFIG, help='Ruta a config.json')
    p.add_argument('--db', default=None, help='Ruta a sqlite DB (override config)')
    p.add_argument('--assets', default=None, help='Lista CSV de assets para procesar (ej: BTCUSDT,ETHUSDT) - si no se pasa usa config.assets or data/config CSVs')
    p.add_argument('--assets-file', action='append', default=None, help='Path a CSV con columna symbol (puede repetirse para varios archivos)')
    p.add_argument('--intervals', default=None, help='Lista CSV de intervals (ej: 1h,4h) - si no se pasa usa config.intervals')
    p.add_argument('--init-db', action='store_true', help='Inicializa la DB y sale')
    p.add_argument('--backfill', action='store_true', help='Lanza backfill antes de procesar')
    p.add_argument('--refresh', action='store_true', help='Lanza refresh_watchlist antes de procesar')
    p.add_argument('--pipeline', action='store_true', help='Ejecuta pipeline completo (indicators + scores) para assets/intervals')
    p.add_argument('--single-asset', default=None, help='Procesar sólo este asset (shortcut)')
    p.add_argument('--no-confirm', action='store_true', help='Omitir confirm prompts (útil para scripts)')
    args = p.parse_args()

    config = load_json(args.config) if Path(args.config).exists() else {}
    db_path = args.db or config.get("app", {}).get("db_path", DEFAULT_DB_PATH)
    # ensure path string
    db_path = str(db_path)

    # assets from CLI args (either comma list or csv files)
    assets_from_cli = load_assets_from_cli_args(args.assets, args.assets_file)

    # assets from config (csv paths or explicit lists)
    assets_from_config = load_assets_from_config(config)

    # combine: priority CLI files/list > config assets > try data/config CSVs
    assets = assets_from_cli or assets_from_config or []

    # if still empty, attempt to read default data/config/cryptos.csv and actions.csv
    if not assets:
        cfg_dir = ROOT / "data" / "config"
        cryptos_file = cfg_dir / "cryptos.csv"
        actions_file = cfg_dir / "actions.csv"
        assets_tmp: List[str] = []
        if cryptos_file.exists():
            assets_tmp += read_symbols_from_csv(str(cryptos_file))
        if actions_file.exists():
            assets_tmp += read_symbols_from_csv(str(actions_file))
        assets = assets_tmp

    if args.single_asset:
        assets = [args.single_asset.strip()]

    if not assets:
        logger.error("No assets especificados (via --assets, --single-asset o config.json/data/config CSVs)")
        sys.exit(1)

    intervals_arg = _parse_list_arg(args.intervals) if args.intervals else []
    intervals = intervals_arg or config.get("intervals", ["1h"])
    if not intervals:
        intervals = ["1h"]

    if args.init_db:
        init_database(db_path)
        logger.info("DB inicializada en %s", db_path)
        if not args.pipeline:
            return

    if args.backfill or args.refresh:
        if not args.no_confirm:
            msg = "Vas a ejecutar backfill/refresh que pueden escribir en la DB. Continuar? [y/N]: "
            ans = input(msg)
            if ans.strip().lower() not in ("y", "yes"):
                logger.info("Operación cancelada por usuario.")
                return

    if args.pipeline:
        run_pipeline(assets, intervals, db_path, config, do_backfill=args.backfill, do_refresh=args.refresh)
    else:
        for asset in assets:
            for interval in intervals:
                try:
                    compute_and_save_indicators_for(asset, interval, db_path, fib_lookback=config.get("fibonacci_lookback", 144))
                except Exception:
                    logger.exception("Error computing indicators for %s %s", asset, interval)
                try:
                    compute_and_save_scores_for(asset, interval, db_path, config)
                except Exception:
                    logger.exception("Error computing scores para %s %s", asset, interval)


if __name__ == "__main__":
    main()
