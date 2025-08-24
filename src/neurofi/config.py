from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8050"))
    symbols: tuple[str, ...] = tuple(os.getenv("SYMBOLS", "BTC,ETH,PEPE,LRC,ZORA,SPELL,TOWN").split(","))

config = Config()
