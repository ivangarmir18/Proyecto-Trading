# dashboard/utils.py
"""
Utilities para el dashboard:
- storage() singleton (PostgresStorage desde DATABASE_URL)
- load_assets_from_cache() lee data/cache/assets_*.json
- load_user_watchlist_csv()/save_user_watchlist_csv(): manejo CSV atómico
- format_ts(): helper para formatos
- simple helpers para detectar entorno (is_running_render)
"""

from __future__ import annotations

import os
import json
import glob
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# intentamos importar la fábrica de storage (compatible con core/storage_postgres)
try:
    from core.storage_postgres import make_storage_from_env, PostgresStorage
except Exception:
    make_storage_from_env = None
    PostgresStorage = None

# caching simple (module-level)
_storage_instance: Optional[PostgresStorage] = None


def storage(**kwargs) -> Optional[PostgresStorage]:
    """
    Devuelve un singleton PostgresStorage.
    kwargs opcionales sobrescriben la fábrica.
    """
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance
    if make_storage_from_env is None:
        return None
    _storage_instance = make_storage_from_env(**kwargs)
    try:
        # idempotent init
        _storage_instance.init_db()
    except Exception:
        # no fallamos aquí (la app puede seguir funcionando en modo parcial)
        pass
    return _storage_instance


# -----------------------
# Cache / assets helpers
# -----------------------
def load_assets_from_cache(cache_dir: str = "data/cache") -> List[Dict[str, Any]]:
    """
    Lee todos los archivos data/cache/assets_*.json y devuelve lista única de assets:
    [{ "asset": "BTCUSDT", "meta": { ... } }, ...]
    Si no hay archivos, devuelve [].
    """
    out: List[Dict[str, Any]] = []
    p = Path(cache_dir)
    if not p.exists():
        return out
    for path in sorted(glob.glob(str(p / "assets_*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for e in data:
                    if isinstance(e, dict) and "asset" in e:
                        out.append(e)
                    elif isinstance(e, str):
                        out.append({"asset": e})
        except Exception:
            # no paramos por un cache corrupto; lo ignoramos
            continue
    # deduplicate by asset
    seen = set()
    uniq = []
    for e in out:
        a = e.get("asset")
        if a and a not in seen:
            seen.add(a)
            uniq.append(e)
    return uniq


# -----------------------
# Watchlist CSV helpers
# -----------------------
def load_user_watchlist_csv(path: str = "data/user_watchlist.csv") -> List[Dict[str, Any]]:
    """
    Lee CSV simple con columna 'asset' y opcional 'meta' (JSON string) y devuelve lista de dicts.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        # fallback: try to read with utf-8-sig
        df = pd.read_csv(p, encoding="utf-8-sig")
    out: List[Dict[str, Any]] = []
    if "asset" not in df.columns:
        return out
    for _, row in df.iterrows():
        asset = str(row["asset"]).strip()
        meta_raw = None
        if "meta" in df.columns and not pd.isna(row.get("meta", None)):
            mr = row.get("meta")
            try:
                meta_raw = json.loads(mr) if isinstance(mr, str) else mr
            except Exception:
                meta_raw = {"raw": mr}
        out.append({"asset": asset, "meta": meta_raw})
    return out


def save_user_watchlist_csv(data: List[Dict[str, Any]], path: str = "data/user_watchlist.csv") -> None:
    """
    Escritura atómica CSV. `data` = list of dicts with 'asset' and optional 'meta'.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # normalize to DataFrame
    rows = []
    for e in data:
        asset = e.get("asset")
        meta = e.get("meta")
        rows.append({"asset": asset, "meta": json.dumps(meta, ensure_ascii=False) if meta is not None else ""})
    df = pd.DataFrame(rows)
    # atomic write: write to temp file then replace
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_watchlist_", dir=str(p.parent))
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    # atomic replace
    os.replace(tmp_path, str(p))


# -----------------------
# Misc helpers
# -----------------------
def format_ts(ts_ms: Optional[int]) -> str:
    """Convierte ms -> ISO string local (o devuelve '-')"""
    if ts_ms is None:
        return "-"
    try:
        return pd.to_datetime(int(ts_ms), unit="ms", utc=True).isoformat()
    except Exception:
        return str(ts_ms)


def is_running_render() -> bool:
    """Heurística simple: Render defines $RENDER (feature) or $PORT present in env."""
    return bool(os.getenv("RENDER") or os.getenv("PORT"))
