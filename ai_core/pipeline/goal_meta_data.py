from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

CANONICAL_OHLC_DIR = Path("ai_core/data/ohlc")
CANONICAL_OHLC_DIR.mkdir(parents=True, exist_ok=True)

ApiStyle = Literal["advanced_trade", "exchange"]

@dataclass
class DataPipelineConfig:
    # Historical ingestion (CoinGecko)
    vs_currency: str = "USD"
    backfill_start: str = "2023-01-01"
    backfill_end: Optional[str] = None

    # Unified resample target
    resample_freq: str = "1H"  # "1H" or "1D"

    # WebSocket streaming (Coinbase)
    api_style: ApiStyle = "advanced_trade"
    ws_url: Optional[str] = None
    channels: List[str] = field(default_factory=lambda: ["candles", "ticker", "level2"])
    ensure_heartbeats: bool = True

    # Universe
    seed_products: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    dynamic_top_movers: bool = True
    top_movers_count: int = 20

    # Storage
    canonical_dir: Path = CANONICAL_OHLC_DIR

    # Top Movers tuning
    top_movers_poll_sec: int = 30
    min_candidate_score: float = 0.15
    freshness_sec: int = 15 * 60
    hysteresis_margin: float = 0.03
    max_changes_per_cycle: int = 10
    cooldown_sec: int = 60
    include_seed_products: bool = True
    bootstrap_on_add: bool = True
    bootstrap_lookback_days: int = 7

    def canonical_path(self, product_id: str) -> Path:
        safe = product_id.replace("/", "-")
        return self.canonical_dir / f"{safe}_{self.resample_freq}.csv"
