"""
core/score.py - Sistema de scoring mejorado con integración de IA
Características:
- Scoring híbrido (heurístico + IA)
- Cálculo automático de stop/target optimizados
- Integración con modelos de ML entrenados
- Métricas de confianza para cada señal
"""

from __future__ import annotations
import logging
import os
import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("core.score")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("SCORE_LOG_LEVEL", "INFO"))

# Default weights (configurables desde config.json)
DEFAULT_WEIGHTS = {
    "trend": 0.25,
    "support": 0.20,
    "momentum": 0.15,
    "volatility": 0.10,
    "volume": 0.10,
    "ai_confidence": 0.20
}

DEFAULT_AI_CONFIG = {
    "model_dir": "models",
    "default_confidence": 0.5,
    "min_confidence_threshold": 0.6,
    "confidence_multiplier": 1.5
}

class AIScoringSystem:
    """Sistema de scoring con integración de IA"""
    
    def __init__(self, storage, config: Optional[Dict[str, Any]] = None):
        self.storage = storage
        self.config = config or {}
        self.ai_config = {**DEFAULT_AI_CONFIG, **self.config.get("ai", {})}
        self.weights = {**DEFAULT_WEIGHTS, **self.config.get("weights", {})}
        
        # Cargar modelos de IA si existen
        self.ai_models = {}
        self.load_ai_models()
    
    def load_ai_models(self):
        """Carga todos los modelos de IA disponibles"""
        model_dir = Path(self.ai_config["model_dir"])
        if not model_dir.exists():
            logger.warning("Directorio de modelos no encontrado: %s", model_dir)
            return
        
        for model_file in model_dir.glob("*.model"):
            try:
                asset_interval = model_file.stem.rsplit('_', 2)[0]  # formato: ASSET_INTERVAL_TIMESTAMP
                asset, interval = asset_interval.split('_', 1)
                
                # Cargar modelo
                model = joblib.load(model_file)
                
                # Cargar scaler
                scaler_file = model_file.with_suffix('.scaler')
                scaler = joblib.load(scaler_file) if scaler_file.exists() else None
                
                # Cargar metadata
                meta_file = model_file.with_suffix('.meta')
                metadata = {}
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                
                self.ai_models[f"{asset}_{interval}"] = {
                    'model': model,
                    'scaler': scaler,
                    'metadata': metadata,
                    'last_used': datetime.now()
                }
                
                logger.info("Modelo cargado para %s %s", asset, interval)
                
            except Exception as e:
                logger.error("Error cargando modelo %s: %s", model_file.name, e)
    
    def get_ai_model(self, asset: str, interval: str) -> Optional[Dict[str, Any]]:
        """Obtiene el modelo de IA para un asset e intervalo específicos"""
        model_key = f"{asset}_{interval}"
        return self.ai_models.get(model_key)
    
    def calculate_ai_confidence(self, df: pd.DataFrame, asset: str, interval: str) -> pd.Series:
        """Calcula la confianza de IA para cada fila del DataFrame"""
        model_info = self.get_ai_model(asset, interval)
        if not model_info:
            return pd.Series([self.ai_config["default_confidence"]] * len(df), index=df.index)
        
        try:
            # Preparar features para el modelo
            feature_columns = model_info['metadata'].get('feature_columns', [])
            available_features = [col for col in feature_columns if col in df.columns]
            
            if not available_features:
                logger.warning("No hay features disponibles para el modelo de IA")
                return pd.Series([self.ai_config["default_confidence"]] * len(df), index=df.index)
            
            X = df[available_features].copy()
            
            # Escalar features si hay scaler
            if model_info['scaler']:
                X_scaled = model_info['scaler'].transform(X)
            else:
                X_scaled = X.values
            
            # Predecir probabilidades
            if hasattr(model_info['model'], 'predict_proba'):
                probabilities = model_info['model'].predict_proba(X_scaled)
                confidence = probabilities[:, 1]  # Probabilidad de clase positiva
            else:
                predictions = model_info['model'].predict(X_scaled)
                confidence = predictions  # Usar predicciones directas
            
            # Ajustar confianza basado en el threshold mínimo
            min_threshold = self.ai_config["min_confidence_threshold"]
            confidence = np.where(confidence < min_threshold, 
                                confidence * 0.5,  # Reducir confianza para predicciones débiles
                                confidence)
            
            return pd.Series(confidence, index=df.index, name='ai_confidence')
            
        except Exception as e:
            logger.error("Error calculando confianza de IA: %s", e)
            return pd.Series([self.ai_config["default_confidence"]] * len(df), index=df.index)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos para scoring"""
        df = df.copy()
        
        # 1. Trend Strength (Fuerza de tendencia)
        if 'ema9' in df.columns and 'ema40' in df.columns:
            price = df['close']
            ema9 = df['ema9']
            ema40 = df['ema40']
            
            # Tendencia alcista si precio > EMA40
            trend_direction = np.where(price > ema40, 1.0, 
                                     np.where(price < ema9, 0.0, 0.5))
            
            # Fuerza de tendencia basada en distancia a EMAs
            ema_distance = (price - ema40) / (ema40 + 1e-9)
            trend_strength = np.tanh(ema_distance * 10) * 0.5 + 0.5  # Normalizar a 0-1
            
            df['trend_score'] = trend_direction * trend_strength
        
        # 2. Support/Resistance Proximity
        if 'support' in df.columns and 'resistance' in df.columns and 'atr' in df.columns:
            price = df['close']
            support = df['support']
            resistance = df['resistance']
            atr = df['atr']
            
            # Proximidad a soporte (0-1, 1 = muy cerca del soporte)
            support_distance = (price - support) / (atr + 1e-9)
            support_proximity = np.exp(-np.abs(support_distance) / 3)
            
            # Proximidad a resistencia
            resistance_distance = (resistance - price) / (atr + 1e-9)
            resistance_proximity = np.exp(-np.abs(resistance_distance) / 3)
            
            # Mejor proximidad (maximizar entrada cerca de soporte)
            df['support_proximity_score'] = support_proximity
        
        # 3. Momentum Indicators
        if 'rsi' in df.columns:
            # RSI normalizado y suavizado
            rsi = df['rsi'].clip(20, 80)  # Evitar extremos
            rsi_score = (rsi - 20) / 60  # Normalizar 20-80 → 0-1
            df['momentum_score'] = rsi_score
        
        if 'macd_hist' in df.columns:
            # MACD histogram strength
            macd_strength = np.tanh(df['macd_hist'] / (df['atr'] + 1e-9) * 10) * 0.5 + 0.5
            df['momentum_score'] = df.get('momentum_score', 0.5) * 0.5 + macd_strength * 0.5
        
        # 4. Volatility Analysis
        if 'atr' in df.columns:
            # Volatilidad normalizada (0-1, 1 = alta volatilidad)
            volatility = df['atr'] / df['close']
            volatility_score = 1 - np.tanh(volatility * 100)  # Invertir: menor volatilidad = mejor
            df['volatility_score'] = volatility_score
        
        # 5. Volume Analysis
        if 'volume' in df.columns:
            # Volume relative to average
            volume_ma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / (volume_ma + 1e-9)
            volume_score = np.tanh(volume_ratio - 1) * 0.5 + 0.5  # Normalizar
            df['volume_score'] = volume_score
        
        return df
    
    def calculate_composite_score(self, df: pd.DataFrame, asset: str, interval: str) -> pd.DataFrame:
        """Calcula el score compuesto (técnico + IA)"""
        df = df.copy()
        
        # Calcular scores técnicos
        df = self.calculate_technical_indicators(df)
        
        # Calcular confianza de IA
        ai_confidence = self.calculate_ai_confidence(df, asset, interval)
        df['ai_confidence'] = ai_confidence
        
        # Inicializar score compuesto
        df['composite_score'] = 0.0
        
        # Componentes del score con pesos
        components = {
            'trend_score': self.weights.get('trend', 0.25),
            'support_proximity_score': self.weights.get('support', 0.20),
            'momentum_score': self.weights.get('momentum', 0.15),
            'volatility_score': self.weights.get('volatility', 0.10),
            'volume_score': self.weights.get('volume', 0.10),
            'ai_confidence': self.weights.get('ai_confidence', 0.20)
        }
        
        # Calcular score ponderado
        total_weight = 0
        for component, weight in components.items():
            if component in df.columns:
                # Rellenar valores missing con neutral (0.5)
                component_values = df[component].fillna(0.5)
                df['composite_score'] += component_values * weight
                total_weight += weight
        
        # Normalizar si algún componente faltó
        if total_weight > 0:
            df['composite_score'] /= total_weight
        
        # Asegurar rango 0-1
        df['composite_score'] = df['composite_score'].clip(0, 1)
        
        return df
    
    def calculate_optimal_stop_target(self, df: pd.DataFrame, asset: str, interval: str) -> pd.DataFrame:
        """Calcula stop y target optimizados basados en score y volatilidad"""
        df = df.copy()
        
        # Multiplicadores base de ATR
        base_stop_multiplier = self.config.get('atr_multipliers', {}).get('stop', 1.3)
        base_target_multiplier = self.config.get('atr_multipliers', {}).get('target', 2.5)
        
        # Ajustar multiplicadores basado en confianza de IA
        if 'ai_confidence' in df.columns and 'atr' in df.columns:
            confidence_multiplier = self.ai_config.get("confidence_multiplier", 1.5)
            
            # Ajustar target multiplier basado en confianza
            target_multipliers = base_target_multiplier * (
                1 + (df['ai_confidence'] - 0.5) * (confidence_multiplier - 1)
            )
            
            # Ajustar stop multiplier (más conservador con alta confianza)
            stop_multipliers = base_stop_multiplier * (
                1 - (df['ai_confidence'] - 0.5) * 0.5  # Reducir stop con alta confianza
            )
            
            # Calcular stop y target
            df['stop'] = df['support'] - stop_multipliers * df['atr']
            df['target'] = df['resistance'] + target_multipliers * df['atr']
            
            # Asegurar stop < price < target
            df['stop'] = np.minimum(df['stop'], df['close'] * 0.99)
            df['target'] = np.maximum(df['target'], df['close'] * 1.01)
        
        else:
            # Fallback a cálculo básico
            df['stop'] = df.get('support', df['close'] * 0.95) - base_stop_multiplier * df.get('atr', 0)
            df['target'] = df.get('resistance', df['close'] * 1.05) + base_target_multiplier * df.get('atr', 0)
        
        return df
    
    def generate_scores(self, df: pd.DataFrame, asset: str, interval: str) -> pd.DataFrame:
        """Genera scores completos con stop/target optimizados"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Calcular score compuesto
        df = self.calculate_composite_score(df, asset, interval)
        
        # Calcular stop/target optimizados
        df = self.calculate_optimal_stop_target(df, asset, interval)
        
        # Rangos para visualización
        df['range_min'] = df.get('support', df['close'] * 0.95)
        df['range_max'] = df.get('resistance', df['close'] * 1.05)
        
        # Métricas de calidad de señal
        df['signal_quality'] = self.calculate_signal_quality(df)
        
        return df
    
    def calculate_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calcula métricas de calidad de señal"""
        quality = pd.Series(0.5, index=df.index, name='signal_quality')
        
        # Factores de calidad
        factors = []
        
        if 'ai_confidence' in df.columns:
            factors.append(df['ai_confidence'])
        
        if 'trend_score' in df.columns:
            factors.append(df['trend_score'])
        
        if 'atr' in df.columns and 'close' in df.columns:
            # Baja volatilidad relativa → mejor calidad
            volatility = df['atr'] / df['close']
            vol_quality = 1 - np.tanh(volatility * 100)
            factors.append(vol_quality)
        
        if len(factors) > 0:
            # Promedio de factores de calidad
            quality = sum(factors) / len(factors)
        
        return quality
    
    def save_scores_to_db(self, df_scores: pd.DataFrame, asset: str, interval: str) -> int:
        """Guarda scores en la base de datos"""
        if df_scores is None or df_scores.empty:
            return 0
        
        required_columns = ['ts', 'composite_score', 'range_min', 'range_max', 
                          'stop', 'target', 'ai_confidence', 'signal_quality']
        
        # Verificar columnas requeridas
        for col in required_columns:
            if col not in df_scores.columns:
                df_scores[col] = np.nan
        
        # Preparar datos para inserción
        df_db = df_scores[required_columns].copy()
        df_db['asset'] = asset
        df_db['interval'] = interval
        df_db['created_at'] = int(datetime.now().timestamp() * 1000)
        
        # Convertir timestamp a ms
        if pd.api.types.is_datetime64_any_dtype(df_db["ts"]):
            df_db["ts_ms"] = (df_db["ts"].astype("int64") // 1_000_000).astype("int64")
        else:
            df_db["ts_ms"] = df_db["ts"].astype("int64")
        
        # Insertar en base de datos
        try:
            with self.storage.get_conn() as conn:
                with conn.cursor() as cur:
                    # UPSERT para scores
                    upsert_sql = """
                    INSERT INTO scores 
                    (asset, interval, ts, score, range_min, range_max, stop, target, 
                     p_ml, multiplier, created_at, signal_quality)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (asset, interval, ts) 
                    DO UPDATE SET
                        score = EXCLUDED.score,
                        range_min = EXCLUDED.range_min,
                        range_max = EXCLUDED.range_max,
                        stop = EXCLUDED.stop,
                        target = EXCLUDED.target,
                        p_ml = EXCLUDED.p_ml,
                        multiplier = EXCLUDED.multiplier,
                        created_at = EXCLUDED.created_at,
                        signal_quality = EXCLUDED.signal_quality
                    """
                    
                    rows = []
                    for _, row in df_db.iterrows():
                        rows.append((
                            asset, interval, int(row["ts_ms"]),
                            float(row["composite_score"]),
                            float(row["range_min"]) if not pd.isna(row["range_min"]) else None,
                            float(row["range_max"]) if not pd.isna(row["range_max"]) else None,
                            float(row["stop"]) if not pd.isna(row["stop"]) else None,
                            float(row["target"]) if not pd.isna(row["target"]) else None,
                            float(row["ai_confidence"]) if not pd.isna(row["ai_confidence"]) else None,
                            float(row["signal_quality"]) if not pd.isna(row["signal_quality"]) else None,
                            int(row["created_at"]),
                            float(row["signal_quality"]) if not pd.isna(row["signal_quality"]) else None
                        ))
                    
                    # Insertar en lotes
                    batch_size = 100
                    inserted = 0
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i + batch_size]
                        cur.executemany(upsert_sql, batch)
                        inserted += cur.rowcount
                    
                    conn.commit()
                    logger.info("Insertados %d scores para %s %s", inserted, asset, interval)
                    return inserted
                    
        except Exception as e:
            logger.exception("Error guardando scores en BD: %s", e)
            return 0

# Función de conveniencia para uso externo
def compute_and_save_scores(storage, asset: str, interval: str, 
                          lookback_bars: int = 500, config: Optional[Dict[str, Any]] = None) -> int:
    """
    Función principal para calcular y guardar scores
    """
    # Obtener datos
    df = storage.get_ohlcv(asset, interval, limit=lookback_bars)
    if df is None or df.empty:
        logger.warning("No hay datos para %s %s", asset, interval)
        return 0
    
    # Obtener indicadores si no están en los datos
    if 'ema9' not in df.columns or 'atr' not in df.columns:
        try:
            from core.score import compute_basic_indicators
            df = compute_basic_indicators(df)
        except Exception as e:
            logger.warning("Error calculando indicadores básicos: %s", e)
    
    # Calcular scores
    scoring_system = AIScoringSystem(storage, config)
    df_scores = scoring_system.generate_scores(df, asset, interval)
    
    # Guardar en BD
    inserted = scoring_system.save_scores_to_db(df_scores, asset, interval)
    return inserted