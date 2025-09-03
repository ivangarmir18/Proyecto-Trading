# check_db_pg.py - check DB contents preferring Postgres if DATABASE_URL is set
import os
import pandas as pd

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        try:
            df_candles = pd.read_sql_table("candles", conn, columns=["ts","open","high","low","close","volume"])
            print("Candles (Postgres):")
            print(df_candles.head())
            cnt = pd.read_sql_query("SELECT COUNT(*) AS total FROM candles", conn)
            print("\nTotal candles:", int(cnt.iloc[0]['total']))
        except Exception as e:
            print("Error leyendo tablas en Postgres:", e)
else:
    import sqlite3
    db_path = "data/db/data.db"
    if not os.path.exists(db_path):
        print("No existe", db_path)
    else:
        with sqlite3.connect(db_path) as conn:
            df_candles = pd.read_sql_query("SELECT * FROM candles LIMIT 5;", conn)
            print("Candles (SQLite):")
            print(df_candles)
            count_candles = pd.read_sql_query("SELECT COUNT(*) AS total FROM candles;", conn)
            print("\nTotal candles:", count_candles['total'].iloc[0])
