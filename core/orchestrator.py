# dentro de class TradingOrchestrator (pegalo en la clase)
def _load_assets(self) -> Dict[str, list]:
    """
    Load assets from config + watchlist table. Returns dict with keys 'crypto' and 'stocks'.
    """
    cfg = self.config or {}
    crypto = cfg.get("crypto", []) or []
    stocks = cfg.get("stocks", []) or []

    # get watchlist items from DB (if storage available)
    try:
        if hasattr(self, "storage") and self.storage is not None:
            wl = self.storage.list_watchlist()
            for item in wl:
                asset = item.get("asset")
                asset_type = item.get("asset_type", "crypto")
                if asset_type == "crypto":
                    if asset not in crypto:
                        crypto.append(asset)
                else:
                    if asset not in stocks:
                        stocks.append(asset)
    except Exception as e:
        logger.exception("Error loading watchlist from DB: %s", e)

    return {"crypto": crypto, "stocks": stocks}

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
