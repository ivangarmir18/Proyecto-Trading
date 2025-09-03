"""
core/orchestrator.py - Sistema central de orquestación
Coordina todos los módulos y proporciona interfaces unificadas
"""

from __future__ import annotations
import logging
import time
import json
import schedule
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import pandas as pd

# Importar módulos del sistema
from core.storage_postgres import PostgresStorage, make_storage_from_env
from core.fetch import Fetcher
from core.score import AIScoringSystem, compute_and_save_scores
from core.ai_train import AITrainer, train_ai_model
from core.utils import get_logger, load_config, retry

logger = get_logger("orchestrator")

class TradingOrchestrator:
    """Sistema central que orquesta todos los componentes de trading"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.storage = make_storage_from_env()
        self.fetcher = self._init_fetcher()
        self.scoring_system = AIScoringSystem(self.storage, self.config)
        self.is_running = False
        
        # Configuración de schedules
        self.schedules = self.config.get("scheduler", {})
        self.assets = self._load_assets()
        
        logger.info("TradingOrchestrator inicializado")
    
    def _init_fetcher(self) -> Fetcher:
        """Inicializa el fetcher con configuración desde config.json"""
        api_config = self.config.get("api", {})
        binance_config = api_config.get("binance", {})
        finnhub_config = api_config.get("finnhub", {})
        
        return Fetcher(
            exchange_name="binance",
            binance_api_key=binance_config.get("api_key"),
            binance_secret=binance_config.get("api_secret"),
            finnhub_keys=finnhub_config.get("keys", []),
            rate_limit_per_min=binance_config.get("rate_limit_per_min", 1200),
            default_limit=self.config.get("app", {}).get("default_limit", 500),
            max_attempts=6,
            backoff_base=1.0,
            backoff_cap=60.0
        )
    
    def _load_assets(self) -> Dict[str, List[str]]:
        """Carga la lista de assets desde configuración"""
        assets_config = self.config.get("assets", {})
        return {
            "crypto": assets_config.get("crypto", []),
            "stocks": assets_config.get("stocks", [])
        }
    
    @retry(Exception, tries=3, delay=1, backoff=2)
    def fetch_data(self, assets: List[str], interval: str) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos para múltiples assets
        Returns: Dict con DataFrames por asset
        """
        results = {}
        save_callback = self.storage.make_save_callback()
        
        for asset in assets:
            try:
                logger.info("Obteniendo datos para %s %s", asset, interval)
                df = self.fetcher.fetch_ohlcv(
                    asset, 
                    interval=interval,
                    save_callback=save_callback,
                    meta={"orchestrator": "scheduled_fetch"}
                )
                results[asset] = df
                logger.info("Datos obtenidos para %s: %d filas", asset, len(df))
                
            except Exception as e:
                logger.error("Error obteniendo datos para %s: %s", asset, e)
                results[asset] = pd.DataFrame()
        
        return results
    
    @retry(Exception, tries=3, delay=1, backoff=2)
    def process_asset(self, asset: str, interval: str, lookback_bars: int = 500) -> bool:
        """
        Procesa completo para un asset: obtiene datos, calcula scores, y entrena IA si es necesario
        Returns: True si el procesamiento fue exitoso
        """
        try:
            # 1. Obtener datos
            df = self.storage.get_ohlcv(asset, interval, limit=lookback_bars)
            if df is None or df.empty:
                logger.warning("No hay datos para %s %s", asset, interval)
                return False
            
            # 2. Calcular y guardar scores
            scores_count = compute_and_save_scores(
                self.storage, asset, interval, lookback_bars, self.config
            )
            logger.info("Scores calculados para %s %s: %d registros", asset, interval, scores_count)
            
            # 3. Entrenar IA si está configurado
            ai_config = self.config.get("ai", {})
            if ai_config.get("enabled", False):
                self.train_ai_model(asset, interval)
            
            return True
            
        except Exception as e:
            logger.exception("Error procesando asset %s: %s", asset, e)
            return False
    
    @retry(Exception, tries=2, delay=1, backoff=2)
    def train_ai_model(self, asset: str, interval: str) -> Optional[Dict[str, Any]]:
        """Entrena un modelo de IA para el asset e intervalo especificados"""
        try:
            ai_config = self.config.get("ai", {})
            lookback_days = ai_config.get("retrain_days", 365)
            
            logger.info("Entrenando modelo IA para %s %s", asset, interval)
            result = train_ai_model(self.storage, asset, interval, lookback_days)
            
            if result and "metrics" in result:
                logger.info(
                    "Modelo entrenado - Accuracy: %.3f, ROC AUC: %.3f", 
                    result["metrics"].get("accuracy", 0),
                    result["metrics"].get("roc_auc", 0)
                )
            
            return result
            
        except Exception as e:
            logger.exception("Error entrenando modelo IA para %s %s: %s", asset, interval, e)
            return None
    
    def run_daily_tasks(self):
        """Ejecuta tareas diarias (entrenamiento IA, mantenimiento)"""
        logger.info("Ejecutando tareas diarias")
        
        # Entrenar modelos IA para todos los assets
        if self.config.get("ai", {}).get("enabled", False):
            for asset_type, assets in self.assets.items():
                for asset in assets:
                    # Usar intervalo por defecto según tipo de asset
                    interval = "5m" if asset_type == "crypto" else "1h"
                    self.train_ai_model(asset, interval)
        
        # Tareas de mantenimiento
        self.run_maintenance_tasks()
    
    def run_maintenance_tasks(self):
        """Ejecuta tareas de mantenimiento del sistema"""
        logger.info("Ejecutando tareas de mantenimiento")
        
        # Purga de datos antiguos
        retention_config = self.config.get("retention_days", {})
        if retention_config:
            self.storage.purge_old_data(
                before_ts_ms=int((datetime.now() - timedelta(days=365)).timestamp() * 1000),
                **retention_config
            )
    
    def run_processing_cycle(self):
        """Ejecuta un ciclo completo de procesamiento para todos los assets"""
        logger.info("Iniciando ciclo de procesamiento")
        
        success_count = 0
        total_count = 0
        
        for asset_type, assets in self.assets.items():
            for asset in assets:
                # Usar intervalo por defecto según tipo de asset
                interval = "5m" if asset_type == "crypto" else "1h"
                total_count += 1
                
                if self.process_asset(asset, interval):
                    success_count += 1
        
        logger.info(
            "Ciclo de procesamiento completado: %d/%d assets procesados exitosamente",
            success_count, total_count
        )
        
        return success_count, total_count
    
    def start_scheduled_tasks(self):
        """Inicia la ejecución de tareas programadas"""
        if self.is_running:
            logger.warning("El orchestrator ya está en ejecución")
            return
        
        self.is_running = True
        logger.info("Iniciando tareas programadas")
        
        # Programar tareas según configuración
        schedule.every().day.at("02:00").do(self.run_daily_tasks)
        
        # Ciclos de procesamiento según configuración
        processing_interval = self.schedules.get("processing_interval_minutes", 15)
        schedule.every(processing_interval).minutes.do(self.run_processing_cycle)
        
        # Bucle principal de scheduling
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Revisar cada minuto
            except KeyboardInterrupt:
                logger.info("Deteniendo orchestrator por interrupción de usuario")
                self.stop()
            except Exception as e:
                logger.exception("Error en bucle de scheduling: %s", e)
                time.sleep(300)  # Esperar 5 minutos antes de reintentar
    
    def stop(self):
        """Detiene el orchestrator"""
        self.is_running = False
        logger.info("Orchestrator detenido")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema"""
        status = {
            "running": self.is_running,
            "assets_loaded": {k: len(v) for k, v in self.assets.items()},
            "next_run": str(schedule.next_run()) if schedule.jobs else "No jobs scheduled",
            "storage_connected": self.storage is not None,
            "fetcher_configured": self.fetcher is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        return status

# Funciones de conveniencia para uso externo
def run_full_pipeline(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo: obtención de datos, procesamiento y entrenamiento IA
    Returns: Dict con resultados y métricas
    """
    orchestrator = TradingOrchestrator(config_path)
    results = {}
    
    try:
        # 1. Obtener datos para todos los assets
        all_assets = orchestrator.assets["crypto"] + orchestrator.assets["stocks"]
        fetch_results = orchestrator.fetch_data(all_assets, "5m")  # Usar 5m como base
        
        results["fetch"] = {
            "assets_processed": len(fetch_results),
            "successful_fetches": sum(1 for df in fetch_results.values() if not df.empty)
        }
        
        # 2. Procesar todos los assets
        success_count, total_count = orchestrator.run_processing_cycle()
        results["processing"] = {
            "successful": success_count,
            "total": total_count,
            "success_rate": success_count / total_count if total_count > 0 else 0
        }
        
        # 3. Entrenar modelos IA si está habilitado
        if orchestrator.config.get("ai", {}).get("enabled", False):
            ai_results = {}
            for asset in all_assets:
                interval = "5m" if asset in orchestrator.assets["crypto"] else "1h"
                ai_result = orchestrator.train_ai_model(asset, interval)
                if ai_result:
                    ai_results[asset] = ai_result.get("metrics", {})
            
            results["ai_training"] = ai_results
        
        logger.info("Pipeline completo ejecutado exitosamente")
        return results
        
    except Exception as e:
        logger.exception("Error en ejecución de pipeline completo: %s", e)
        raise
    finally:
        orchestrator.stop()

def start_service(config_path: str = "config.json"):
    """Inicia el servicio continuo de orchestrator"""
    orchestrator = TradingOrchestrator(config_path)
    orchestrator.start_scheduled_tasks()