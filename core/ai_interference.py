# core/ai_inference.py
"""
Carga y predicción con el modelo IA (LightGBM / sklearn-like).

Comportamiento:
- Carga modelo desde models/ai_score_model.pkl
- Carga lista de features desde models/ai_feature_names.json
- predict_prob(feature_dict) devuelve float 0..1 o None si no hay modelo
- safe: maneja ausencia de fichero, diferentes tipos de modelo y excepciones
"""
from __future__ import annotations
from pathlib import Path
import json
import threading
from typing import Optional, Dict, List

import numpy as np

try:
    import joblib
except Exception:
    joblib = None  # se requiere joblib en runtime

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_score_model.pkl"
FEATURES_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_feature_names.json"

_lock = threading.Lock()
_model = None
_model_features: Optional[List[str]] = None


def _load_model_and_features():
    global _model, _model_features
    with _lock:
        if _model is None:
            if not MODEL_PATH.exists():
                return None, None
            if joblib is None:
                raise RuntimeError("joblib no está instalado. Instálalo (pip install joblib).")
            _model = joblib.load(MODEL_PATH)
        if _model_features is None:
            if FEATURES_PATH.exists():
                with FEATURES_PATH.open("r", encoding="utf-8") as fh:
                    _model_features = json.load(fh)
            else:
                _model_features = None
    return _model, _model_features


def predict_prob(feature_dict: Dict[str, float]) -> Optional[float]:
    """
    Devuelve probabilidad [0,1] o None si no hay modelo.
    feature_dict: mapping feature_name -> numeric value
    """
    mdl, feat_list = _load_model_and_features()
    if mdl is None:
        return None
    # If feature list not available, use keys sorted (risky); prefer to raise
    if feat_list is None:
        # fallback: use keys sorted but warn by returning None to avoid inconsistent inputs
        raise RuntimeError("Faltan feature names (models/ai_feature_names.json). Entrena el modelo primero.")
    # prepare vector in correct order
    try:
        X = np.array([[float(feature_dict.get(fn, 0.0)) for fn in feat_list]])
    except Exception:
        # if conversion fails, return None
        return None
    try:
        if hasattr(mdl, "predict_proba"):
            return float(mdl.predict_proba(X)[0, 1])
        elif hasattr(mdl, "predict"):
            # fallback: map raw prediction to probability via sigmoid
            raw = float(mdl.predict(X)[0])
            return float(1.0 / (1.0 + np.exp(-raw)))
        else:
            return None
    except Exception:
        return None


def get_model_info() -> Dict:
    mdl, feat = _load_model_and_features()
    return {
        "model_exists": mdl is not None,
        "features_exist": feat is not None,
        "n_features": len(feat) if feat else 0,
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
    }


if __name__ == "__main__":
    print(get_model_info())
