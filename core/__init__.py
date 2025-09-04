"""
core/__init__.py
Inicialización del paquete core — exporta helpers y versión.
Mantener ligero: aquí sólo definimos constantes, logging básico y
re-exportamos clases clave del adaptador.
"""

__version__ = "0.1.0"

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def configure_logging(level="INFO"):
    fmt = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=fmt)
    # evitar múltiples handlers en reloads (streamlit dev)
    for logger_name in ("sqlalchemy",):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# export útil
from .adapter import StorageAdapter, normalize_weights  # noqa: E402,F401

# configurar logging por defecto
configure_logging()
