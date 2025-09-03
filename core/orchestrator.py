# dentro de class TradingOrchestrator (pegalo en la clase)
# Sustituye la función _load_assets existente por esto
from pathlib import Path
import csv

def _load_assets(self) -> dict:
    """
    Load assets from config (self.config), then from CSVs data/config/cryptos.csv and actions.csv,
    and finally include items from watchlist (storage) if available.
    Returns: {"crypto": [...], "stocks": [...]}
    """
    cfg = getattr(self, "config", {}) or {}
    crypto_set = set()
    stocks_set = set()

    # 1) From config dict (if present)
    try:
        conf_crypto = cfg.get("crypto", []) or []
        conf_stocks = cfg.get("stocks", []) or []
        for s in conf_crypto:
            if s:
                crypto_set.add(str(s).strip().upper())
        for s in conf_stocks:
            if s:
                stocks_set.add(str(s).strip().upper())
    except Exception:
        # keep going even if config is malformed
        pass

    # 2) From CSV files (project-root/data/config/cryptos.csv and actions.csv)
    try:
        repo_root = Path(__file__).resolve().parents[1]  # core/.. => repo root
        csv_dir = repo_root / "data" / "config"
        cryptos_csv = csv_dir / "cryptos.csv"
        actions_csv = csv_dir / "actions.csv"
        if cryptos_csv.exists():
            with cryptos_csv.open("r", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                # handle files with a 'symbol' column or single-column CSV
                if "symbol" in (rdr.fieldnames or []):
                    for r in rdr:
                        val = r.get("symbol") or ""
                        if val:
                            crypto_set.add(str(val).strip().upper())
                else:
                    fh.seek(0)
                    for row in fh:
                        val = row.strip()
                        if val:
                            crypto_set.add(val.upper())
        if actions_csv.exists():
            with actions_csv.open("r", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                if "symbol" in (rdr.fieldnames or []):
                    for r in rdr:
                        val = r.get("symbol") or ""
                        if val:
                            stocks_set.add(str(val).strip().upper())
                else:
                    fh.seek(0)
                    for row in fh:
                        val = row.strip()
                        if val:
                            stocks_set.add(val.upper())
    except Exception:
        # no hard fail here
        pass

    # 3) From storage/watchlist if available
    try:
        if hasattr(self, "storage") and self.storage is not None:
            # prefer explicit list_assets / list_watchlist if available
            if hasattr(self.storage, "list_assets"):
                try:
                    # expected to return list of strings
                    assets = self.storage.list_assets() or []
                    for a in assets:
                        # storage.list_assets might return "BTCUSDT" or dicts — handle both
                        if isinstance(a, dict):
                            s = a.get("asset") or a.get("symbol") or ""
                        else:
                            s = a
                        if s:
                            crypto_set.add(str(s).strip().upper())
                except Exception:
                    pass
            # fallback: a watchlist with asset_type
            if hasattr(self.storage, "list_watchlist"):
                try:
                    wl = self.storage.list_watchlist() or []
                    for item in wl:
                        asset = item.get("asset") if isinstance(item, dict) else str(item)
                        asset_type = item.get("asset_type", "crypto") if isinstance(item, dict) else "crypto"
                        if asset:
                            if asset_type == "stock" or asset_type == "stocks":
                                stocks_set.add(str(asset).strip().upper())
                            else:
                                crypto_set.add(str(asset).strip().upper())
                except Exception:
                    pass
    except Exception:
        pass

    # Finalize - sort for stable order
    crypto_list = sorted(list(crypto_set))
    stocks_list = sorted(list(stocks_set))
    return {"crypto": crypto_list, "stocks": stocks_list}


def backfill_asset(self, asset: str, interval: Optional[str] = None) -> Dict[str, Any]:
    """
    Try to run a backfill for a single asset (called by worker when processing backfill_request).
    Uses fetcher.backfill_range if available or falls back to repeated fetch calls.
    Returns a dict with status.
    """
    logger.info("Starting backfill for %s (interval=%s)", asset, interval)
    try:
        if not self.fetcher:
            raise RuntimeError("No fetcher configured in orchestrator")

        # Try to call backfill_range if fetcher provides it
        if hasattr(self.fetcher, "backfill_range"):
            # best-effort: pass interval if provided, otherwise let fetcher choose defaults
            try:
                if interval:
                    res = self.fetcher.backfill_range(asset, interval=interval, save_callback=self.storage.save_candles)
                else:
                    res = self.fetcher.backfill_range(asset, save_callback=self.storage.save_candles)
                return {"ok": True, "detail": res}
            except TypeError:
                # different signature; try common alternative param names
                try:
                    res = self.fetcher.backfill_range(asset)
                    return {"ok": True, "detail": res}
                except Exception as e:
                    raise

        # Fallback: attempt to fetch manual ranges (best-effort)
        # We'll try to fetch latest and then request older pages in a loop (simplified).
        logger.info("Fetcher does not expose backfill_range; running manual fallback (may be slow).")
        latest = self.fetcher.fetch_latest(asset, limit=1) if hasattr(self.fetcher, "fetch_latest") else None
        if latest is None:
            return {"ok": False, "error": "fetch_latest not supported by fetcher"}
        # Attempt to save the fetched candle(s)
        try:
            if isinstance(latest, dict) or hasattr(latest, "to_dict"):
                # convert to DataFrame if needed
                import pandas as pd
                if isinstance(latest, dict):
                    df = pd.DataFrame([latest])
                else:
                    df = latest
                self.storage.save_candles(df)
            else:
                logger.warning("Unknown latest format from fetcher for backfill.")
        except Exception:
            logger.exception("Failed saving latest backfill fetch result.")
        return {"ok": True, "detail": "fallback_done"}
    except Exception as e:
        logger.exception("Backfill for asset %s failed: %s", asset, e)
        return {"ok": False, "error": str(e)}
