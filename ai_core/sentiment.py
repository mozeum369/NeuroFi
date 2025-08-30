# NeuroFi/src/ai_core/sentiment.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

@dataclass
class SentimentConfig:
    products: List[str]
    since_ts: Optional[datetime] = None

    @staticmethod
    def from_any(cfg: Any) -> "SentimentConfig":
        if isinstance(cfg, argparse.Namespace):
            cfg = vars(cfg)
        cfg = dict(cfg or {})
        products = cfg.get("products") or ["BTC-USD", "ETH-USD"]
        since = cfg.get("since_ts")
        if isinstance(since, (int, float)):
            since = datetime.fromtimestamp(float(since), tz=timezone.utc)
        return SentimentConfig(products=products, since_ts=since)

# --- very small lexicon just to avoid extra dependencies (swap later with real APIs/ML) ---
POS_WORDS = {"bullish", "surge", "record", "gain", "rally", "adoption", "positive", "upgrade", "growth", "breakout"}
NEG_WORDS = {"bearish", "dump", "hack", "exploit", "ban", "negative", "downgrade", "lawsuit", "selloff", "crash"}

def simple_score(text: str) -> float:
    t = text.lower()
    pos = sum(w in t for w in POS_WORDS)
    neg = sum(w in t for w in NEG_WORDS)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(1, (pos + neg))

def collect_headlines(products: List[str], since_ts: Optional[datetime]) -> List[Dict[str, Any]]:
    """
    Placeholder collector. Replace with real sources:
      - Bing News API
      - Twitter/X API
      - Reddit API
    Return list of dicts: {ts, product, source, text}
    """
    now = datetime.now(timezone.utc)
    items = []
    for p in products:
        items.append({"ts": now, "product": p, "source": "stub", "text": f"{p} technical breakout to new record"})
        items.append({"ts": now, "product": p, "source": "stub", "text": f"{p} faces potential lawsuit over exploit"})
    return items

def run_sentiment_analysis(config: Any, logger) -> pd.DataFrame:
    cfg = SentimentConfig.from_any(config)
    rows = collect_headlines(cfg.products, cfg.since_ts)
    if not rows:
        return pd.DataFrame(columns=["ts", "product", "source", "text", "score", "confidence"]).set_index(["product", "ts"])

    df = pd.DataFrame(rows)
    df["score"] = df["text"].map(simple_score)
    df["confidence"] = np.where(df["score"] == 0, 0.3, 0.6 + 0.4 * np.abs(df["score"]))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index(["product", "ts"]).sort_index()

    logger.info("sentiment_done", extra={"step": "sentiment", "message": f"{len(df)} items"})
    return df[["source", "text", "score", "confidence"]]
