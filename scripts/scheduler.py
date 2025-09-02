import schedule
import time
from datetime import datetime
from core.fetch import refresh_watchlist
from main import load_assets_from_config, load_config

def scheduled_refresh():
    print(f"[{datetime.now()}] Ejecutando refresh programado")
    config = load_config()
    assets = load_assets_from_config(config)
    
    # Actualizar cryptos
    cryptos = [a for a in assets if any(a.endswith(x) for x in ['USDT', 'BTC', 'ETH'])]
    stocks = [a for a in assets if a not in cryptos]
    
    refresh_watchlist(cryptos, stocks, save_to_db=True)
    print(f"[{datetime.now()}] Refresh completado")

# Programar ejecuciones cada 5 minutos
schedule.every(5).minutes.do(scheduled_refresh)

if __name__ == "__main__":
    print("Iniciando scheduler de actualizaciones...")
    while True:
        schedule.run_pending()
        time.sleep(1)