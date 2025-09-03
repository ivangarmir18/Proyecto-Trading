# init_and_backfill.py
import os
import pandas as pd
import logging
from core.fetch import Fetcher
from core.storage_postgres import PostgresStorage

logger = logging.getLogger("init_and_backfill")
logger.setLevel(os.environ.get("INIT_BACKFILL_LOG_LEVEL", "INFO"))

def main():
    """
    Inicializa Postgres, crea un save_callback (fallback si hace falta),
    y lanza backfill para los CSV: data/config/cryptos.csv y data/config/actions.csv.
    """
    # 1) Inicializar storage y tablas
    storage = PostgresStorage()
    storage.init_db()

    # 2) Crear save_callback: intentamos usar make_save_callback si existiera,
    #    pero como no existe en tu storage_postgres.py, usamos un fallback seguro.
    try:
        # intento rápido: si existiera make_save_callback, úsalo
        from core.storage_postgres import make_save_callback  # si falla, cae al except
        save_callback = make_save_callback(storage)
        logger.info("Usando make_save_callback provisto por storage_postgres")
    except Exception:
        logger.info("make_save_callback no disponible, usando fallback que llama a storage.save_candles")

        def save_callback(df):
            """
            Callback seguro para guardar candles en storage.
            Ignora DataFrames vacíos o malformados y registra excepciones.
            """
            try:
                if df is None:
                    logger.info("save_callback: recibido None, skip")
                    return
        
                if getattr(df, "empty", False):
                    logger.info("save_callback: recibido DataFrame vacío, skip")
                    return
        
                required = {'high', 'close', 'asset', 'interval', 'volume', 'low', 'timestamp', 'open', 'ts'}
                missing = required.difference(set(df.columns))
                if missing:
                    logger.warning(f"save_callback: DataFrame missing required columns {missing}; skipping save. Columns present: {list(df.columns)}")
                    return
        
                # normalizar tipos
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df['ts'] = df['ts'].astype('int64')
        
                # opcional: eliminar duplicados por ts antes de guardar
                df = df.drop_duplicates(subset=['ts'])
        
                # Guardar usando la clase storage (asumiendo storage ya instanciado en el scope)
                storage.save_candles(df)
                logger.info(f"save_callback: guardadas {len(df)} filas para {df['asset'].iat[0]} {df['interval'].iat[0]}")
            except Exception as e:
                logger.exception(f"Error en save_callback al guardar datos: {e}")
        

    # 3) Inicializar fetcher
    fetcher = Fetcher()

    # 4) Rango por defecto: desde 2023-01-01 hasta ahora (ajusta si quieres)
    try:
        start_ts = Fetcher.ms_from_iso("2023-01-01T00:00:00Z")
    except Exception:
        start_ts = Fetcher.now_ms() - 90 * 24 * 3600 * 1000  # 90 días por defecto
    end_ts = Fetcher.now_ms()

    # 5) Localizar CSVs (seguro independientemente del cwd)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "config")

    csv_files = [os.path.join(DATA_DIR, "cryptos.csv"), os.path.join(DATA_DIR, "actions.csv")]

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            logger.warning("No existe: %s — se salta", csv_path)
            continue

        logger.info("Leyendo %s ...", csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.exception("Error leyendo %s: %s — se salta", csv_path, e)
            continue

        # Determinar columna de símbolo (symbol / asset / primera columna)
        if "symbol" in df.columns:
            sym_col = "symbol"
        elif "asset" in df.columns:
            sym_col = "asset"
        else:
            sym_col = df.columns[0]

        interval_col = "interval" if "interval" in df.columns else None

        for _, row in df.iterrows():
            try:
                raw_asset = row.get(sym_col)
                if pd.isna(raw_asset):
                    continue
                asset = str(raw_asset).strip()
                interval = str(row.get(interval_col)).strip() if interval_col and not pd.isna(row.get(interval_col)) else "1h"
                logger.info("Backfilling %s %s ...", asset, interval)
                try:
                    fetcher.backfill_range(
                        asset=asset,
                        interval=interval,
                        start_ts_ms=start_ts,
                        end_ts_ms=end_ts,
                        save_callback=save_callback,
                        progress=True
                    )
                    logger.info("OK %s %s", asset, interval)
                except Exception as e:
                    logger.exception("Error en backfill de %s %s: %s", asset, interval, e)
                    # continuar con siguiente activo
            except Exception as e:
                logger.exception("Error procesando fila en %s: %s", csv_path, e)
                continue

    # 6) Cerrar storage si tiene close()
    try:
        if hasattr(storage, "close"):
            storage.close()
    except Exception:
        pass
    
import sqlalchemy as sa
import pandas as pd
from core.fetch import Fetcher
from core.storage_postgres import PostgresStorage

def get_last_ts_for(asset, interval, engine):
    q = "SELECT MAX(ts) as last_ts FROM candles WHERE asset = %s AND interval = %s"
    with engine.connect() as conn:
        res = conn.execute(sa.text(q), (asset, interval)).fetchone()
        return int(res[0]) if res and res[0] is not None else None

def incremental_backfill_for(asset, interval):
    storage = PostgresStorage()   # usa DATABASE_URL desde env
    engine = storage.engine      # si tu clase expone engine; si no, crea engine con DATABASE_URL
    last_ts = get_last_ts_for(asset, interval, engine)
    # si last_ts es None -> backfill desde inicio (o desde config)
    since = pd.to_datetime(last_ts, unit='s', utc=True) if last_ts else None
    fetcher = Fetcher(storage=storage)
    fetcher.backfill_range(asset=asset, interval=interval, since=since, progress=True)



if __name__ == "__main__":
    main()

