import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from data_utils import log_message

class RealTimeOHLCResampler:
    def __init__(self, freq: str, canonical_path_func, backtrack_bars: int = 2):
        self.freq = freq.upper()
        self.get_path = canonical_path_func
        self.backtrack_bars = backtrack_bars
        self.buffers: Dict[str, List[dict]] = {}

    def _load_existing(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    def _save_canonical(self, df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy().sort_index()
        out.index = out.index.tz_convert("UTC")
        out = out.reset_index().rename(columns={"index": "timestamp"})
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(path, index=False)
        log_message(f"[RT-OHLC] Saved {len(df)} bars -> {path}")

    def _aggregate_target(self, candle_5m_df: pd.DataFrame) -> pd.DataFrame:
        if candle_5m_df.empty:
            return candle_5m_df
        if candle_5m_df.index.tz is None:
            candle_5m_df.index = candle_5m_df.index.tz_localize("UTC")
        else:
            candle_5m_df.index = candle_5m_df.index.tz_convert("UTC")

        ohlc = candle_5m_df.resample(self.freq, label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        ohlc["ret"] = ohlc["close"].pct_change()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ohlc["close"] / ohlc["close"].shift(1)
            ohlc["log_ret"] = np.log(ratio.replace({0: np.nan}))
        ohlc = ohlc.dropna(subset=["open", "high", "low", "close"], how="all")
        return ohlc

    def ingest_candles_message(self, msg: dict):
        if not msg or msg.get("channel") != "candles":
            return
        events = msg.get("events", [])
        for ev in events:
            for c in ev.get("candles", []):
                try:
                    product = c["product_id"]
                    ts = pd.to_datetime(int(c["start"]), unit="s", utc=True)
                    row = {
                        "timestamp": ts,
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c["volume"]),
                    }
                except Exception as e:
                    log_message(f"[RT-OHLC] Malformed candle: {e}", level="warning")
                    continue

                self.buffers.setdefault(product, []).append(row)
                need = 12 if self.freq == "1H" else 288
                if len(self.buffers[product]) >= need:
                    self._flush_product(product)

    def _flush_product(self, product_id: str):
        buf = self.buffers.get(product_id, [])
        if not buf:
            return
        df5 = pd.DataFrame(buf).drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()

        path = self.get_path(product_id)
        existing = self._load_existing(path)
        if existing is None or existing.empty:
            recompute_start = None
        else:
            last_idx = existing.index.max()
            offset = pd.tseries.frequencies.to_offset(self.freq)
            recompute_start = (last_idx - 2 * offset).tz_convert("UTC")

        if recompute_start is not None:
            df5 = df5.loc[df5.index >= recompute_start]

        new_target = self._aggregate_target(df5)
        if existing is None or existing.empty:
            merged = new_target
        else:
            pre = existing.loc[existing.index < (new_target.index.min() if not new_target.empty else existing.index.max())]
            merged = pd.concat([pre, new_target]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]

        self._save_canonical(merged, path)

        # Trim buffer window (keep last day for 1H, last week for 1D)
        keep_from = merged.index.max() - (pd.Timedelta(days=1) if self.freq == "1H" else pd.Timedelta(days=7))
        self.buffers[product_id] = [r for r in buf if r["timestamp"] >= keep_from]
