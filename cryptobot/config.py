from pathlib import Path
import os

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "cryptobot.db"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Default tracked assets (symbols)
DEFAULT_ASSETS = ["SPELL", "ZORA", "PEPE", "LRC", "TOWNS"]  # We'll resolve to CoinGecko IDs

# Polling interval (seconds)
PRICE_POLL_INTERVAL = int(os.getenv("PRICE_POLL_INTERVAL", "15"))
