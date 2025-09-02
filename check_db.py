import sqlite3
import pandas as pd

db_path = "data/db/data.db"

with sqlite3.connect(db_path) as conn:
    # Ver las primeras 5 velas
    df_candles = pd.read_sql_query("SELECT * FROM candles LIMIT 5;", conn)
    print("Candles:")
    print(df_candles)

    # Contar cuántas velas hay
    count_candles = pd.read_sql_query("SELECT COUNT(*) AS total FROM candles;", conn)
    print("\nTotal candles:", count_candles['total'].iloc[0])

    # Ver las primeras 5 filas de scores
    df_scores = pd.read_sql_query("SELECT * FROM scores LIMIT 5;", conn)
    print("\nScores:")
    print(df_scores)

    # Contar cuántos scores hay
    count_scores = pd.read_sql_query("SELECT COUNT(*) AS total FROM scores;", conn)
    print("\nTotal scores:", count_scores['total'].iloc[0])
