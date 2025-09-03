#init db
# scripts/init_db.py
import sqlite3
from pathlib import Path

DB_PATH = Path('data/db/watchlist.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset TEXT NOT NULL,
  interval TEXT NOT NULL,
  ts INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL,
  UNIQUE(asset, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_asset_interval_ts ON candles (asset, interval, ts);

CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  asset TEXT NOT NULL,
  interval TEXT NOT NULL,
  ts INTEGER NOT NULL,
  score REAL NOT NULL,
  range_min REAL,
  range_max REAL,
  stop REAL,
  target REAL,
  p_ml REAL,
  signal_quality REAL, 
  multiplier REAL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scores_asset_time ON scores (asset, interval, ts);
"""

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.executescript(DDL)
conn.commit()
conn.close()
print("DB inicializada en:", DB_PATH)
