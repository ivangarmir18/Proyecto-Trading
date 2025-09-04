# core/settings.py
"""
Capa mínima para persistir settings usando core.adapter (y sus fallbacks).
Interfaz simple usada por dashboard para guardar toggles y pesos.
"""

from core.adapter import adapter

def save_setting(key: str, value) -> bool:
    return adapter.save_setting(key, value)

def load_setting(key: str, default=None):
    return adapter.load_setting(key, default)

def health_status():
    return adapter.health_status()
