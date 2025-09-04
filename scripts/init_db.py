#!/usr/bin/env python3
# scripts/init_db.py
"""
Inicializar esquema de base de datos (robusto y seguro).

Características:
- Usa make_storage_from_env() (core/storage_postgres.py) si está disponible.
- Opcional --apply-schema <path> para ejecutar un SQL file en la DB (útil si storage no expone init_db).
- Retries y espera entre intentos.
- --force-recreate: si storage expone drop_all_tables(), lo invoca (destructivo).
- Modo verbose para debugging.
- Mensajes claros si faltan variables de entorno o permisos DB.

Ejemplos:
  python scripts/init_db.py
  python scripts/init_db.py --retries 3 --wait 5 --apply-schema schema/init.sql
  python scripts/init_db.py --force-recreate --confirm-force
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import logging
import pathlib
from typing import Optional

# Optional imports (we try to use the project's storage factory)
try:
    from core.storage_postgres import make_storage_from_env
except Exception:
    make_storage_from_env = None

# Optional low-level DB client if we need to run raw SQL
try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None

# Logging
logger = logging.getLogger("init_db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger.setLevel(LOG_LEVEL)
_ch = logging.StreamHandler(sys.stdout := sys.stdout if 'sys' in globals() else None)
_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_ch.setFormatter(_formatter)
logger.addHandler(_ch)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Init DB schema for the project (idempotent, robust).")
    p.add_argument("--retries", type=int, default=1, help="Número de reintentos en caso de fallo.")
    p.add_argument("--wait", type=float, default=3.0, help="Segundos entre reintentos.")
    p.add_argument("--force-recreate", action="store_true", help="Si storage tiene drop_all_tables(), lo invoca (DESTRUCTIVO).")
    p.add_argument("--confirm-force", action="store_true", help="Confirmación explícita para --force-recreate.")
    p.add_argument("--apply-schema", type=str, default=None, help="Ruta a un fichero .sql a ejecutar tras init (opcional).")
    p.add_argument("--verbose", action="store_true", help="Más logging.")
    p.add_argument("--skip-storage-init", action="store_true", help="No llamar a storage.init_db(); solo aplicar --apply-schema si se da.")
    return p.parse_args(argv)


def _env_has_database_url() -> bool:
    return bool(os.getenv("DATABASE_URL") or (os.getenv("PGHOST") and os.getenv("PGUSER") and os.getenv("PGDATABASE")))


def _run_raw_sql_file(database_url: str, sql_path: str) -> None:
    """
    Ejecuta un fichero SQL en la base de datos indicada por DATABASE_URL.
    Usa psycopg2 si está disponible.
    """
    if psycopg2 is None:
        raise RuntimeError("psycopg2 no está instalado; no puedo ejecutar SQL raw. Instala psycopg2-binary o usa storage.init_db().")
    if not sql_path or not pathlib.Path(sql_path).exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    logger.info("Ejecutando fichero SQL: %s", sql_path)
    import urllib.parse as up

    # If DATABASE_URL is a full URL, psycopg2 can accept it; but parse to pass params if needed
    db_url = database_url
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()
        sql_text = pathlib.Path(sql_path).read_text(encoding="utf-8")
        cur.execute(sql_text)
        logger.info("Fichero SQL ejecutado correctamente.")
    finally:
        if conn:
            conn.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel("DEBUG")
    logger.debug("Args: %s", args)

    # sanity checks
    if args.force_recreate and not args.confirm_force:
        logger.error("--force-recreate requiere --confirm-force para evitar borrados accidentales.")
        return 2

    if not _env_has_database_url():
        logger.warning("No se ha detectado DATABASE_URL ni variables PG. Si usas PostgresStorage esto fallará.")
        # still we may attempt storage factory if it encapsulates env handling

    attempts = 0
    last_exc: Optional[Exception] = None

    while attempts < max(1, args.retries):
        attempts += 1
        try:
            # 1) Prefer using storage factory if available and skip-storage-init not requested
            if make_storage_from_env is not None and not args.skip_storage_init:
                logger.info("Creando storage via make_storage_from_env()")
                storage = make_storage_from_env()
                # optional destructive drop
                if args.force_recreate:
                    if hasattr(storage, "drop_all_tables"):
                        logger.warning("Invocando storage.drop_all_tables() (destructivo).")
                        storage.drop_all_tables()
                    else:
                        logger.warning("force-recreate solicitado pero storage.drop_all_tables() no está implementado.")
                # call init_db()
                if hasattr(storage, "init_db"):
                    logger.info("Invocando storage.init_db() ...")
                    storage.init_db()
                    logger.info("storage.init_db() completado.")
                else:
                    logger.warning("storage no expone init_db(); continuar a aplicar SQL si se ha pasado --apply-schema")
                # optionally apply schema SQL if provided
                if args.apply_schema:
                    db_url = os.getenv("DATABASE_URL", None)
                    if db_url:
                        _run_raw_sql_file(db_url, args.apply_schema)
                    else:
                        # try attempting to reuse storage connection if it exposes raw connection
                        if hasattr(storage, "raw_connection"):
                            try:
                                conn = storage.raw_connection()
                                sql_text = pathlib.Path(args.apply_schema).read_text(encoding="utf-8")
                                with conn:
                                    with conn.cursor() as cur:
                                        cur.execute(sql_text)
                                logger.info("SQL file applied via storage.raw_connection().")
                            except Exception:
                                logger.exception("No se pudo aplicar SQL via storage.raw_connection(); intenta pasar DATABASE_URL o usar psycopg2.")
                                raise
                        else:
                            logger.warning("No DATABASE_URL disponible para aplicar --apply-schema; omitiendo apply-schema.")
                # success
                return 0

            # 2) Fallback: try to apply raw SQL using DATABASE_URL if apply_schema given
            if args.apply_schema:
                db_url = os.getenv("DATABASE_URL")
                if not db_url:
                    raise RuntimeError("No DATABASE_URL presente para aplicar --apply-schema.")
                _run_raw_sql_file(db_url, args.apply_schema)
                return 0

            # 3) If make_storage_from_env not present but we have DATABASE_URL, try a basic check
            if make_storage_from_env is None and _env_has_database_url():
                logger.info("make_storage_from_env() no disponible pero DATABASE_URL presente. Intentaré una conexión simple (psycopg2).")
                db_url = os.getenv("DATABASE_URL")
                if psycopg2 is None:
                    raise RuntimeError("psycopg2 no instalado; no puedo inicializar DB sin storage factory.")
                conn = psycopg2.connect(db_url)
                conn.close()
                logger.info("Conexión DB verificada (no hay init_db implícito).")
                if args.apply_schema:
                    _run_raw_sql_file(db_url, args.apply_schema)
                return 0

            # nothing to do
            logger.error("No pude inicializar la DB: ni storage factory disponible ni --apply-schema especificado.")
            return 3

        except KeyboardInterrupt:
            logger.warning("Interrumpido por teclado.")
            return 130
        except Exception as e:
            last_exc = e
            logger.exception("init_db attempt %d failed: %s", attempts, e)
            if attempts >= args.retries:
                logger.error("Se agotaron los reintentos.")
                break
            logger.info("Reintentando en %.1fs ...", args.wait)
            time.sleep(args.wait)

    # final failure
    logger.error("init_db falló definitivamente: %s", repr(last_exc))
    return 1


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
