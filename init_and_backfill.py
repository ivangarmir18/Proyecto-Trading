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

        def save_callback(df: pd.DataFrame, asset: str, interval: str, meta: dict):
            """
            Fallback wrapper que adapta el DataFrame y llama a storage.save_candles.
            Trata de cubrir firmas distintas de save_candles.
            """
            try:
                df2 = df.copy()
                if "asset" not in df2.columns:
                    df2["asset"] = asset
                if "interval" not in df2.columns:
                    df2["interval"] = interval

                # Try common signatures
                try:
                    # firma: save_candles(df)
                    storage.save_candles(df2)
                except TypeError:
                    try:
                        # firma: save_candles(df, batch=500)
                        storage.save_candles(df2, batch=500)
                    except TypeError:
                        # firma: save_candles(df, asset, interval) (por si acaso)
                        try:
                            storage.save_candles(df2, asset, interval)
                        except Exception as ee:
                            logger.exception("save_candles: llamada fallback final fallida: %s", ee)
            except Exception as exc:
                logger.exception("Error en save_callback para %s %s: %s", asset, interval, exc)

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


if __name__ == "__main__":
    main()
# %%

