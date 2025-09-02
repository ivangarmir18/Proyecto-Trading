# core/utils.py
import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv(".env.production")  # útil para local; en Render no se usa

logger = logging.getLogger("core.utils")

def send_telegram_message(message: str, token: str = None, chat_id: str = None) -> bool:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("Telegram token/chat_id missing")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.exception("Telegram send failed")
        return False
