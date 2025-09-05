# test_db.py - prueba de conexión y upsert/load
import os
import traceback
import time

print("=== test_db.py - inicio ===")
DATABASE_URL = os.environ.get("DATABASE_URL")
print("DATABASE_URL:", DATABASE_URL)

errors = []

# Intentar importar implementación del repo (varias posibilidades)
candidates = [
    "core.storage_postgres",
    "core.storage_postgres.PostgresStorage",
    "core.storage_postgres.Postgres",
    "core.adapter",
    "core.storage_adapter",
]

found_impl = False
impl_name = None

try:
    for mod in ["core.storage_postgres", "core.storage_adapter", "core.adapter"]:
        try:
            m = __import__(mod, fromlist=['*'])
            print(f"Module imported: {mod}")
            # buscar clases posibles
            for attr in dir(m):
                if 'Postgres' in attr or 'Storage' in attr:
                    C = getattr(m, attr)
                    if isinstance(C, type):
                        print("Trying class:", mod + "." + attr)
                        try:
                            # intentar instanciar con DATABASE_URL si lo acepta
                            inst = None
                            try:
                                inst = C(DATABASE_URL)
                            except Exception:
                                try:
                                    inst = C()
                                except Exception:
                                    inst = None
                            if inst is not None:
                                print("Instanciado:", mod + "." + attr)
                                # intentar usar métodos habituales
                                if hasattr(inst, "upsert_candles") and hasattr(inst, "load_candles"):
                                    print("Found usable storage class:", mod + "." + attr)
                                    # preparar dato de prueba
                                    ts = int(time.time() * 1000)
                                    sample = [{'ts': ts, 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.05, 'volume': 10.0}]
                                    try:
                                        inst.upsert_candles("TEST-DB", "1m", sample)
                                        print("upsert_candles OK")
                                        loaded = inst.load_candles("TEST-DB", "1m", start_ts=ts-1000, end_ts=ts+1000)
                                        print("load_candles ->", loaded[-5:] if loaded else loaded)
                                        found_impl = True
                                        impl_name = mod + "." + attr
                                        break
                                    except Exception as e:
                                        print("Error usando métodos upsert/load:", e)
                                else:
                                    print("Clase no expone upsert_candles/load_candles. A revisar.")
                        except Exception as e:
                            print("No se pudo instanciar", mod+"."+attr, ":", e)
            if found_impl:
                break
        except Exception as e:
            print("No se pudo importar", mod, ":", e)
    if not found_impl:
        print("No se encontró clase storage usable en los módulos esperados. Probaremos con SQLAlchemy directo.")
        # Intento con SQLAlchemy directo (verifica conectividad)
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                print("Conectado a la BD con SQLAlchemy")
                # crear tabla de prueba si no existe
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_candles (
                        id serial PRIMARY KEY,
                        asset text,
                        interval text,
                        ts bigint,
                        open double precision,
                        high double precision,
                        low double precision,
                        close double precision,
                        volume double precision
                    );
                """))
                ts = int(time.time() * 1000)
                conn.execute(text("INSERT INTO test_candles (asset, interval, ts, open, high, low, close, volume) VALUES (:asset,:interval,:ts,:o,:h,:l,:c,:v)"),
                             {"asset":"TEST-DB","interval":"1m","ts":ts,"o":1.0,"h":1.1,"l":0.9,"c":1.05,"v":10.0})
                res = conn.execute(text("SELECT asset, interval, ts, open, close FROM test_candles WHERE asset='TEST-DB' ORDER BY ts DESC LIMIT 5"))
                rows = res.fetchall()
                print("Últimas filas test_candles:", rows)
                found_impl = True
                impl_name = "sqlalchemy_direct"
        except Exception as exc:
            print("Error SQLAlchemy directo:", exc)
            traceback.print_exc()
            errors.append(str(exc))
except Exception as e:
    print("EXCEPCIÓN GENERAL:", e)
    traceback.print_exc()
    errors.append(str(e))

print("found_impl:", found_impl, "impl_name:", impl_name)
if errors:
    print("Errores durante la prueba:", errors)
print("=== test_db.py - fin ===")
