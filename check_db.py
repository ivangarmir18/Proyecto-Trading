# check_db.py
"""
Chequeo rápido de base de datos (Postgres/Supabase o SQLite).

Uso:
  python check_db.py
Requisitos:
  - Si usas Supabase/Render: define DATABASE_URL en el entorno (postgresql://...)
"""

import os
import sys
import pandas as pd
from datetime import datetime

PD_DISPLAY_ROWS = 10

def fmt_ts(ms_or_s):
    # acepta ts en s o ms
    try:
        v = int(ms_or_s)
        if v > 10_000_000_000:  # probablemente ms
            v //= 1000
        return datetime.utcfromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ms_or_s

def print_df(df, name):
    if df is None or df.empty:
        print(f"\n{name}: vacío")
        return
    with pd.option_context("display.max_rows", PD_DISPLAY_ROWS, "display.width", 160):
        print(f"\n{name}:")
        print(df.head(PD_DISPLAY_ROWS))

def main():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # ----- Postgres (Supabase) -----
        try:
            from sqlalchemy import create_engine, text
        except Exception:
            print("Falta sqlalchemy. Añádelo a requirements e instala.", file=sys.stderr)
            sys.exit(1)

        engine = create_engine(db_url)
        with engine.connect() as conn:
            try:
                tables = pd.read_sql(
                    "select table_name from information_schema.tables where table_schema='public' order by 1",
                    conn
                )
                print_df(tables, "Tablas")
            except Exception as e:
                print("Error listando tablas:", e)

            for tbl in ("assets", "candles", "indicators", "scores", "backtests", "backfill_status"):
                try:
                    cnt = pd.read_sql(f"select count(*) as total from {tbl}", conn)
                    print(f"Total filas en {tbl}: {int(cnt.iloc[0]['total'])}")
                except Exception as e:
                    print(f"{tbl}: {e}")

            # últimas 10 velas por asset/interval
            try:
                q = """
                select asset, interval, ts, open, high, low, close, volume
                from candles
                order by ts desc
                limit 10
                """
                last = pd.read_sql(q, conn)
                if not last.empty:
                    last["ts_readable"] = last["ts"].apply(fmt_ts)
                print_df(last, "Últimas velas (10)")
            except Exception as e:
                print("Error leyendo últimas velas:", e)

            # últimos scores
            try:
                q = """
                select asset, ts, method, score
                from scores
                order by ts desc
                limit 10
                """
                sc = pd.read_sql(q, conn)
                if not sc.empty:
                    sc["ts_readable"] = sc["ts"].apply(fmt_ts)
                print_df(sc, "Últimos scores (10)")
            except Exception as e:
                print("Error leyendo últimos scores:", e)

    else:
        # ----- SQLite fallback -----
        import sqlite3
        db_path = "data/db/data.db"
        if not os.path.exists(db_path):
            print("SQLite no encontrado en", db_path)
            sys.exit(0)
        with sqlite3.connect(db_path) as conn:
            def read(q):
                try:
                    return pd.read_sql_query(q, conn)
                except Exception as e:
                    print("SQL error:", e)
                    return pd.DataFrame()

            for tbl in ("assets", "candles", "indicators", "scores", "backtests", "backfill_status"):
                df = read(f"select count(*) as total from {tbl}")
                if not df.empty:
                    print(f"Total filas en {tbl}: {int(df.iloc[0]['total'])}")

            last = read("select asset, interval, ts, open, high, low, close, volume from candles order by ts desc limit 10")
            if not last.empty:
                last["ts_readable"] = last["ts"].apply(fmt_ts)
            print_df(last, "Últimas velas (10)")

            sc = read("select asset, ts, method, score from scores order by ts desc limit 10")
            if not sc.empty:
                sc["ts_readable"] = sc["ts"].apply(fmt_ts)
            print_df(sc, "Últimos scores (10)")

if __name__ == "__main__":
    main()
