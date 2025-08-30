# NeuroFi/src/ai_core/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../src
LOG_DIR = PROJECT_ROOT / "logs"
STATUS_DIR = PROJECT_ROOT / "status"
DATA_DIR = PROJECT_ROOT / "data"

for p in (LOG_DIR, STATUS_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
