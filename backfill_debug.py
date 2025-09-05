# backfill_debug.py
import os, sys, json, traceback
from datetime import datetime
import pandas as pd
import sqlalchemy as sa

print("=== backfill_debug started ===")
print("Time:", datetime.utcnow().isoformat())

# print env relevant
def env_check():
    keys = ["DATABASE_URL", "BINANCE_API_KEY", "BINANCE_SECRET", "FINNHUB_API_KEY", "FINNHUB_KEY", "YFINANCE_ALLOW"]
    print("\n--- ENV VARS ---")
    for k in keys:
        print(k, "=", os.environ.get(k))
env_check()

# DB quick check
print("\n--- DB quick connection & counts ---")
DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    print("ERROR: DATABASE_URL no definido")
else:
    try:
        engine = sa.create_engine(DB_URL)
        with engine.connect() as conn:
            try:
                df = pd.read_sql("select asset, interval, count(*) as cnt from candles group by asset, interval order by cnt desc limit 20", conn)
                print("candles summary:\n", df.to_string(index=False))
            except Exception as e:
                print("No se pudo leer tabla candles (quizá no existe). Error:", e)
    except Exception as e:
        print("No se pudo conectar a DB:", e)
        traceback.print_exc()

# Inspect core modules & functions
print("\n--- core modules & callables ---")
modules = {}
try:
    import importlib
    for m in ["core.orchestrator","core.fetch","core.storage_postgres","core.adapter","core.score","core.indicators","core.ai_train","core.ai_inference"]:
        try:
            mm = importlib.import_module(m)
            modules[m] = mm
            print(f"{m}: OK")
        except Exception as e:
            modules[m] = None
            print(f"{m}: NOT-FOUND ({e})")
except Exception as e:
    print("import error:", e)

# 1) Try orchestrator top-level functions (if module present)
print("\n--- Orchestrator API check ---")
try:
    if modules.get("core.orchestrator"):
        ormod = modules["core.orchestrator"]
        for fn in ("run_full_backfill_safe","run_full_backfill","run_full_backfill_safe_sync","run_full_backfill_sync","run_full_backfill"):
            print("has", fn, getattr(ormod, fn, None) is not None)
        print("Has Orchestrator class:", hasattr(ormod, "Orchestrator"))
        # If functions exist, try calling with small symbol list (non-destructive)
        test_symbols = ["BTCUSDT"]
        called = False
        for fn in ("run_full_backfill_safe","run_full_backfill"):
            f = getattr(ormod, fn, None)
            if callable(f):
                print(f"Calling orchestrator.{fn} with symbols={test_symbols}, per_symbol_limit=50")
                try:
                    out = f(symbols=test_symbols, per_symbol_limit=50) if "symbols" in f.__code__.co_varnames else f()
                    print("-> returned (type):", type(out))
                    try:
                        print("-> summary:", json.dumps(out if not isinstance(out, (pd.DataFrame,)) else {"rows": len(out)}, default=str)[:2000])
                    except Exception:
                        print("-> returned object (repr):", repr(out)[:1000])
                    called = True
                    break
                except Exception as e:
                    print("-> call error:", e)
                    traceback.print_exc()
        # try class instance
        if not called and hasattr(ormod, "Orchestrator"):
            try:
                Or = getattr(ormod, "Orchestrator")
                inst = Or()
                for method in ("run_full_backfill_safe","run_full_backfill"):
                    if hasattr(inst, method):
                        print(f"Calling Orchestrator().{method}(symbols={test_symbols}, per_symbol_limit=50)")
                        try:
                            out = getattr(inst, method)(symbols=test_symbols, per_symbol_limit=50)
                            print("-> returned (type):", type(out))
                            print("-> summary:", str(out)[:2000])
                            called = True
                            break
                        except Exception as e:
                            print("-> call error:", e)
                            traceback.print_exc()
            except Exception as e:
                print("Could not instantiate Orchestrator:", e)
    else:
        print("core.orchestrator module not loaded.")
except Exception as e:
    print("Error testing orchestrator:", e)
    traceback.print_exc()

# 2) Try fetch module network fetch for known symbol (BTCUSDT) with common function names
print("\n--- Fetch network test (BTCUSDT) ---")
try:
    if modules.get("core.fetch"):
        fm = modules["core.fetch"]
        tried = False
        for fn in ("get_candles","fetch_candles","get_latest_candles","fetch","_network_get_candles","get_ohlcv","fetch_multi"):
            f = getattr(fm, fn, None)
            if callable(f):
                print("Trying fetch function:", fn)
                try:
                    # try common signatures
                    try:
                        df = f("BTCUSDT", limit=200)
                    except TypeError:
                        try:
                            df = f("BTCUSDT", "1m", 200)
                        except Exception:
                            df = f("BTCUSDT")
                    print("-> returned type:", type(df))
                    if isinstance(df, (list, dict)):
                        print("-> returned list/dict sample:", repr(df)[:1000])
                    elif hasattr(df, "shape"):
                        print("-> df.shape:", getattr(df, "shape", None))
                        print("-> df.columns:", getattr(df, "columns", None))
                        print("-> head:\n", getattr(df, "head", lambda: None)())
                    tried = True
                    break
                except Exception as e:
                    print("-> fetch call error:", e)
                    traceback.print_exc()
        if not tried:
            print("No suitable fetch function discovered in core.fetch")
    else:
        print("core.fetch not present")
except Exception as e:
    print("Fetch test error:", e)
    traceback.print_exc()

# 3) Check storage adapter upsert path: try to write a small synthetic candle and read it back
print("\n--- Storage upsert test (synthetic row) ---")
try:
    if modules.get("core.storage_postgres"):
        sp = modules["core.storage_postgres"]
        print("core.storage_postgres has:", [n for n in dir(sp) if not n.startswith("_")][:50])
        if hasattr(sp, "PostgresStorage"):
            try:
                S = getattr(sp, "PostgresStorage")
                inst = S()
                print("Instantiated PostgresStorage")
                # build synthetic row; try to call upsert_candles or save_candles
                sample_rows = [{"asset":"DEBUG-TEST","interval":"1m","ts":int(datetime.utcnow().timestamp()*1000),"open":1.0,"high":1.1,"low":0.9,"close":1.05,"volume":1.0}]
                if hasattr(inst, "upsert_candles"):
                    try:
                        r = inst.upsert_candles("DEBUG-TEST","1m", sample_rows)
                        print("upsert_candles returned:", r)
                    except Exception as e:
                        print("upsert_candles error:", e)
                elif hasattr(inst, "save_candles"):
                    try:
                        r = inst.save_candles("DEBUG-TEST", pd.DataFrame(sample_rows))
                        print("save_candles returned:", r)
                    except Exception as e:
                        print("save_candles error:", e)
                else:
                    print("No upsert/save method on PostgresStorage instance")
            except Exception as e:
                print("Could not instantiate PostgresStorage:", e)
    else:
        print("core.storage_postgres not present")
except Exception as e:
    print("Storage test error:", e)
    traceback.print_exc()

print("\n=== backfill_debug finished ===")
