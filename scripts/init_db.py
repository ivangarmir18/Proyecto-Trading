# scripts/init_db.py

import os
import sys
import psycopg2

def init_db():
    """
    Inicializa la base de datos de Supabase.
    Requiere que la variable de entorno DATABASE_URL esté definida.
    """
    try:
        # Obtener la URL de conexión desde el entorno
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL no está definida en las variables de entorno.")

        # Conectar a la base de datos
        print("Intentando conectar a la base de datos...")
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        print("Conexión exitosa. Inicializando la base de datos...")

        # SQL para crear las tablas
        ddl = """
        -- Eliminar tablas si ya existen para un inicio limpio
        DROP TABLE IF EXISTS scores CASCADE;
        DROP TABLE IF EXISTS candles CASCADE;

        -- Crear tabla de velas (candles)
        CREATE TABLE candles (
            id SERIAL PRIMARY KEY,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts BIGINT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            UNIQUE(asset, interval, ts)
        );
        
        -- Crear índice para la tabla de velas
        CREATE INDEX idx_candles_asset_interval_ts ON candles (asset, interval, ts);

        -- Crear tabla de puntuaciones (scores)
        CREATE TABLE scores (
            id SERIAL PRIMARY KEY,
            asset TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts BIGINT NOT NULL,
            score REAL NOT NULL,
            range_min REAL,
            range_max REAL,
            stop REAL,
            target REAL,
            p_ml REAL,
            signal_quality REAL,
            multiplier REAL,
            created_at BIGINT NOT NULL
        );
        
        -- Crear índice para la tabla de puntuaciones
        CREATE INDEX idx_scores_asset_time ON scores (asset, interval, ts);
        """

        # Ejecutar el script SQL
        cur.execute(ddl)
        conn.commit()

        print("Base de datos inicializada correctamente.")
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error al inicializar la base de datos: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    init_db()
