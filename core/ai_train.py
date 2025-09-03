"""
core/ai_train.py - Entrenamiento de IA con datos de backtesting
Características:
- Entrena modelo para predecir probabilidad de éxito de trades
- Usa datos históricos con labels de éxito/fracaso
- Genera features técnicas avanzadas
- Modelo LightGBM optimizado
"""

from __future__ import annotations
import os
import json
import logging
import warnings
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

# Ignorar warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("core.ai_train")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("AI_TRAIN_LOG_LEVEL", "INFO"))

MODELS_DIR = os.getenv("MODELS_DIR", "models")

def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)

class AITrainer:
    def __init__(self, storage, lookback_days: int = 365, test_size: float = 0.2):
        self.storage = storage
        self.lookback_days = lookback_days
        self.test_size = test_size
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def _load_training_data(self, asset: str, interval: str) -> pd.DataFrame:
        """Carga y prepara datos para entrenamiento"""
        # Obtener velas
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=self.lookback_days)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        df_candles = self.storage.get_ohlcv(asset, interval, start_ms, end_ms)
        if df_candles is None or df_candles.empty:
            raise ValueError(f"No hay datos para {asset} {interval}")
        
        # Obtener scores
        df_scores = self._get_scores(asset, interval, start_ms, end_ms)
        
        # Obtener indicadores
        df_indicators = self._get_indicators(asset, interval, start_ms, end_ms)
        
        # Combinar todos los datos
        df_merged = self._merge_dataframes(df_candles, df_scores, df_indicators)
        return df_merged
    
    def _get_scores(self, asset: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Obtiene scores desde la base de datos"""
        try:
            with self.storage.get_conn() as conn:
                query = """
                    SELECT ts, score, range_min, range_max, stop, target 
                    FROM scores 
                    WHERE asset = %s AND interval = %s AND ts BETWEEN %s AND %s
                    ORDER BY ts
                """
                df = pd.read_sql_query(query, conn, params=(asset, interval, start_ms, end_ms))
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df
        except Exception as e:
            logger.warning("Error obteniendo scores: %s", e)
            return pd.DataFrame()
    
    def _get_indicators(self, asset: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Obtiene indicadores desde la base de datos"""
        try:
            with self.storage.get_conn() as conn:
                query = """
                    SELECT ts, ema9, ema40, atr, macd, macd_signal, rsi, support, resistance
                    FROM indicators 
                    WHERE asset = %s AND interval = %s AND ts BETWEEN %s AND %s
                    ORDER BY ts
                """
                df = pd.read_sql_query(query, conn, params=(asset, interval, start_ms, end_ms))
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df
        except Exception as e:
            logger.warning("Error obteniendo indicadores: %s", e)
            return pd.DataFrame()
    
    def _merge_dataframes(self, df_candles: pd.DataFrame, df_scores: pd.DataFrame, 
                         df_indicators: pd.DataFrame) -> pd.DataFrame:
        """Combina todos los DataFrames en uno solo"""
        df = df_candles.copy()
        
        # Merge con scores
        if not df_scores.empty:
            df = pd.merge_asof(df.sort_values("ts"), df_scores.sort_values("ts"), 
                              on="ts", direction="backward", tolerance=pd.Timedelta("1h"))
        
        # Merge con indicadores
        if not df_indicators.empty:
            df = pd.merge_asof(df.sort_values("ts"), df_indicators.sort_values("ts"), 
                              on="ts", direction="backward", tolerance=pd.Timedelta("1h"))
        
        return df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula features técnicas avanzadas"""
        df = df.copy()
        
        # Precios y returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Volatilidad
        df["volatility_20"] = df["returns"].rolling(20).std()
        df["volatility_50"] = df["returns"].rolling(50).std()
        
        # Medias móviles
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()
        
        # Relación entre medias
        df["sma_20_ratio"] = df["close"] / df["sma_20"]
        df["sma_50_ratio"] = df["close"] / df["sma_50"]
        df["sma_200_ratio"] = df["close"] / df["sma_200"]
        
        # Momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        # RSI (si no está disponible)
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
        
        # Bandas de Bollinger
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        # Volumen
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
        
        # ATR (si no está disponible)
        if "atr" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df["atr"] = true_range.rolling(14).mean()
        
        # MACD (si no está disponible)
        if "macd" not in df.columns:
            exp12 = df["close"].ewm(span=12, adjust=False).mean()
            exp26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp12 - exp26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Diferencias de EMA
        if "ema9" in df.columns and "ema40" in df.columns:
            df["ema_diff"] = df["ema9"] - df["ema40"]
            df["ema_ratio"] = df["ema9"] / df["ema40"]
        
        # Soporte y resistencia
        if "support" in df.columns and "resistance" in df.columns:
            df["support_distance"] = (df["close"] - df["support"]) / df["close"]
            df["resistance_distance"] = (df["resistance"] - df["close"]) / df["close"]
            df["range_width"] = (df["resistance"] - df["support"]) / df["close"]
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        
        # Clean infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _calculate_labels(self, df: pd.DataFrame, horizon_bars: int = 24) -> pd.Series:
        """Calcula labels basados en si el trade fue exitoso"""
        labels = []
        
        for i in range(len(df)):
            if i + horizon_bars >= len(df):
                labels.append(np.nan)
                continue
            
            current_row = df.iloc[i]
            future_df = df.iloc[i+1:i+horizon_bars+1]
            
            # Verificar si tenemos stop y target
            if pd.isna(current_row.get("stop")) or pd.isna(current_row.get("target")):
                labels.append(np.nan)
                continue
            
            entry_price = current_row["close"]
            stop_price = current_row["stop"]
            target_price = current_row["target"]
            
            # Verificar si se alcanzó stop o target
            stop_hit = False
            target_hit = False
            exit_idx = None
            
            for j, future_row in enumerate(future_df.iterrows(), 1):
                idx, row = future_row
                low = row["low"]
                high = row["high"]
                
                if low <= stop_price:
                    stop_hit = True
                    exit_idx = i + j
                    break
                
                if high >= target_price:
                    target_hit = True
                    exit_idx = i + j
                    break
            
            # Determinar resultado
            if stop_hit and target_hit:
                # Ambos en el mismo candle, determinar cuál primero
                open_price = future_df.iloc[exit_idx - i - 1]["open"]
                dist_to_stop = abs(open_price - stop_price)
                dist_to_target = abs(target_price - open_price)
                
                if dist_to_stop <= dist_to_target:
                    labels.append(0)  # Stop hit first
                else:
                    labels.append(1)  # Target hit first
            elif stop_hit:
                labels.append(0)  # Stop hit
            elif target_hit:
                labels.append(1)  # Target hit
            else:
                labels.append(0)  # No target hit within horizon
        
        return pd.Series(labels, index=df.index)
    
    def prepare_dataset(self, asset: str, interval: str, horizon_bars: int = 24) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepara dataset completo con features y labels"""
        logger.info("Preparando dataset para %s %s", asset, interval)
        
        # Cargar datos
        df = self._load_training_data(asset, interval)
        if df.empty:
            raise ValueError("No se pudieron cargar datos para entrenamiento")
        
        # Calcular features
        df_features = self._calculate_features(df)
        
        # Calcular labels
        labels = self._calculate_labels(df_features, horizon_bars)
        
        # Combinar y limpiar
        df_features["label"] = labels
        df_clean = df_features.dropna(subset=["label"]).copy()
        
        # Seleccionar columnas de features
        feature_columns = [
            "returns", "log_returns", "volatility_20", "volatility_50",
            "sma_20", "sma_50", "sma_200", "sma_20_ratio", "sma_50_ratio", "sma_200_ratio",
            "momentum_5", "momentum_10", "momentum_20", "rsi", "bb_middle", "bb_upper",
            "bb_lower", "bb_width", "volume_ma_20", "volume_ratio", "atr", "macd",
            "macd_signal", "macd_hist", "ema_diff", "ema_ratio", "support_distance",
            "resistance_distance", "range_width"
        ]
        
        # Añadir lags
        for lag in [1, 2, 3, 5, 10]:
            feature_columns.extend([f"return_lag_{lag}", f"volume_lag_{lag}"])
        
        # Filtrar columnas disponibles
        available_features = [col for col in feature_columns if col in df_clean.columns]
        
        X = df_clean[available_features].copy()
        y = df_clean["label"].astype(int)
        
        # Escalar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=X.index)
        
        self.feature_columns = available_features
        
        logger.info("Dataset preparado: %d muestras, %d features", len(X_scaled), len(available_features))
        return X_scaled, y, available_features
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Entrena el modelo de IA"""
        logger.info("Entrenando modelo LightGBM")
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, shuffle=False
        )
        
        # Parámetros de LightGBM
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42
        }
        
        # Entrenar modelo
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred_binary)
        
        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "best_iteration": self.model.best_iteration,
            "feature_importance": dict(zip(X.columns, self.model.feature_importance()))
        }
        
        logger.info("Modelo entrenado - Accuracy: %.3f, ROC AUC: %.3f, F1: %.3f", 
                   accuracy, roc_auc, f1)
        
        return metrics
    
    def save_model(self, asset: str, interval: str, metrics: Dict[str, Any]):
        """Guarda el modelo entrenado"""
        ensure_models_dir()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{asset}_{interval}_{timestamp}.model"
        scaler_filename = f"{asset}_{interval}_{timestamp}.scaler"
        meta_filename = f"{asset}_{interval}_{timestamp}.meta"
        
        model_path = os.path.join(MODELS_DIR, model_filename)
        scaler_path = os.path.join(MODELS_DIR, scaler_filename)
        meta_path = os.path.join(MODELS_DIR, meta_filename)
        
        # Guardar modelo
        joblib.dump(self.model, model_path)
        
        # Guardar scaler
        joblib.dump(self.scaler, scaler_path)
        
        # Guardar metadata
        metadata = {
            "asset": asset,
            "interval": interval,
            "timestamp": timestamp,
            "metrics": metrics,
            "feature_columns": self.feature_columns,
            "lookback_days": self.lookback_days,
            "model_type": "lightgbm"
        }
        
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Modelo guardado en %s", model_path)
        
        return {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "meta_path": meta_path,
            "metrics": metrics
        }
    
    def full_training_pipeline(self, asset: str, interval: str) -> Dict[str, Any]:
        """Pipeline completo de entrenamiento"""
        try:
            # Preparar datos
            X, y, features = self.prepare_dataset(asset, interval)
            
            # Entrenar modelo
            metrics = self.train_model(X, y)
            
            # Guardar modelo
            result = self.save_model(asset, interval, metrics)
            
            return result
            
        except Exception as e:
            logger.exception("Error en el pipeline de entrenamiento: %s", e)
            raise

# Función de conveniencia para uso externo
def train_ai_model(storage, asset: str, interval: str, lookback_days: int = 365) -> Dict[str, Any]:
    """Función principal para entrenar modelos de IA"""
    trainer = AITrainer(storage, lookback_days)
    return trainer.full_training_pipeline(asset, interval)