# NeuroFi/src/ai_core/utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

def analyze_market_patterns(ohlcv: pd.DataFrame, ticks: pd.DataFrame, sentiment: pd.DataFrame | None = None) -> Dict[str, Any]:
    """
    Return per-product diagnostics and a coarse signal:
      signal in {-1, 0, +1}, along with momentum and sentiment.
    """
    result: Dict[str, Any] = {}
    if ohlcv is None or ohlcv.empty:
        return result

    # Compute last momentum/zscore from features if present
    # Expect MultiIndex [product, time]
    products = ohlcv.index.get_level_values(0).unique()

    # Reduce sentiment to latest per product
    sent_map = {}
    if sentiment is not None and not sentiment.empty:
        latest = sentiment.reset_index().sort_values("ts").groupby("product").tail(1)
        for _, r in latest.iterrows():
            sent_map[r["product"]] = float(r["score"])

    for p in products:
        dfp = ohlcv.xs(p, level=0).copy()
        if dfp.empty:
            continue
        dfp["ret_1"] = dfp["close"].pct_change()
        mom = dfp["ret_1"].tail(10).mean() / (dfp["ret_1"].tail(10).std(ddof=0) + 1e-12)
        mom = float(np.clip(mom, -5, 5))

        sent = float(sent_map.get(p, 0.0))
        score = 0.7 * mom + 0.3 * sent
        if score > 0.25:
            sig = +1
        elif score < -0.25:
            sig = -1
        else:
            sig = 0

        result[p] = {
            "momentum": mom,
            "sentiment": sent,
            "score": float(score),
            "signal": sig,
        }
    return result
