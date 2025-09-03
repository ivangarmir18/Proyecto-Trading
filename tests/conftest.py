"""
tests/conftest.py - Configuración global para tests
"""

import pytest
import os
import sys
from pathlib import Path

# Añadir el directorio root al path para imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Configuración de entorno para tests
os.environ["TESTING"] = "1"
os.environ["DB_PATH"] = ":memory:"  # SQLite en memoria para tests

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configuración automática para cada test"""
    # Setup code here
    yield
    # Teardown code here