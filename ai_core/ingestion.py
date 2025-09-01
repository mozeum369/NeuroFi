# NeuroFi/src/ai_core/ingestion.py
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from queue import Queue, Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Flexible imports for your existing modules (edit if paths differ)
try:
    from ai_core.paths import DATA_DIR
except Exception:
    from pathlib import Path
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- EDIT THESE IF YOUR MODULE NAMES DIFFER ----
# We expect you have *some* ws_listener and data_ingestor modules already.
# If the function/class names differ, update the adapters below.
try:
    import ws_listener as ws_mod  # e.g., NeuroFi/src/ws_listener.py
except Exception:
    ws_mod = None

try:
    import data_ingestor as hist_mod  # e.g., NeuroFi/src/data_ingestor.py
except Exception:
    hist_mod = None
# ------------------------------------------------

DEFAULT_PRODUCTS = ["BTC-USD", "ETH-USD"]
DEFAULT_GRANULARITY = 60  # seconds (1 min candles)
DEFAULT_HIST_LOOKBACK_M = 1000  # minutes to prefill OHLCV, adjust as desired
MAX_TICKS_PER_PRODUCT = 20_000  # ring buffer size

@dataclass
class IngestionConfig:
    products: List[str] = None
    granularity: int = DEFAULT_GRANULARITY
    hist_lookback_minutes: int = DEFAULT_HIST_LOOKBACK_M
    snapshot_window_secs: int = 300  # last 5 minutes of ticks for snapshot
    save_parquet: bool = False  # save normalized outputs for debugging

    @staticmethod
    def from_any(cfg: Any) -> "IngestionConfig":
        if isinstance(cfg, argparse.Namespace):
            cfg = vars(cfg)
        cfg = dict(cfg or {})
        return IngestionConfig(
            products=cfg.get("products") or DEFAULT_PRODUCTS,
            granularity=int(cfg.get("granularity", DEFAULT_GRANULARITY)),
            hist_lookback_minutes=int(cfg.get("hist_lookback_minutes", DEFAULT_HIST_LOOKBACK_M)),
            snapshot_window_secs=int(cfg.get("snapshot_window_secs", 300)),
            save_parquet=bool(cfg.get("save_parquet", False)),
        )

# ----------------------------
# Internal singleton manager
# ----------------------------
class _IngestionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._ws_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._tick_queue: "Queue[dict]" = Queue(maxsize=10000)

        # Per-product tick buffers (ring buffers of normalized dicts)
        self._ticks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TICKS_PER_PRODUCT))
        self._last_hist: Optional[pd.DataFrame] = None  # MultiIndex [product, time] OHLCV
        self._cfg: Optional[IngestionConfig] = None

    # ---------- public API ----------
    def ensure_started(self, cfg: IngestionConfig, logger) -> None:
        with self._lock:
            if self._started:
                return
            self._cfg = cfg
            # 1) Historical bootstrap
            self._last_hist = self._fetch_historical(cfg, logger)
            # 2) Start WS consumer in a background thread
            self._start_ws_consumer(cfg, logger)
            self._started = True
            logger.info(
                "ingestion_started",
                extra={"step": "ingestion", "message": f"Started WS and fetched historical ({len(self._last_hist) if self._last_hist is not None else 0} rows)"},
            )

    def snapshot(self, cfg: IngestionConfig, logger) -> Dict[str, Any]:
        """Return current unified view: ticks (window), OHLCV (latest), derived features."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=cfg.snapshot_window_secs)
        ticks_df = self._ticks_to_df(since=window_start)
        ohlcv_df = self._last_hist.copy() if self._last_hist is not None else pd.DataFrame()

        # Basic feature engineering
        features_df = self._build_features(ohlcv_df)

        result = {
            "ts": now.isoformat(),
            "ticks": ticks_df,
            "ohlcv": ohlcv_df,
            "features": features_df,
            "meta": {"products": cfg.products, "granularity": cfg.granularity},
        }

        if cfg.save_parquet:
            # Helpful for debugging—files rotate due to timestamps
            ts_short = now.strftime("%Y%m%d_%H%M%S")
            base = DATA_DIR / f"snapshot_{ts_short}"
            try:
                if not ticks_df.empty:
                    ticks_df.to_parquet(str(base) + "_ticks.parquet")
                if not ohlcv_df.empty:
                    ohlcv_df.to_parquet(str(base) + "_ohlcv.parquet")
                if not features_df.empty:
                    features_df.to_parquet(str(base) + "_features.parquet")
            except Exception as e:
                logger.warning(f"Failed to save parquet snapshot: {e}")

        return result

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)

    # ---------- internal helpers ----------
    def _fetch_historical(self, cfg: IngestionConfig, logger) -> pd.DataFrame:
        if hist_mod is None:
            logger.warning("data_ingestor module not found; skipping historical bootstrap")
            return pd.DataFrame()

        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=cfg.hist_lookback_minutes)

        # ---- Adapter 1: function get_ohlcv(product, start, end, granularity) -> pd.DataFrame[time, open, high, low, close, volume]
        if hasattr(hist_mod, "get_ohlcv"):
            frames = []
            for p in cfg.products:
                try:
                    df = hist_mod.get_ohlcv(p, start=start, end=end, granularity=cfg.granularity)
                    df = self._normalize_ohlcv(df, p)
                    frames.append(df)
                except Exception as e:
                    logger.error(f"historical error for {p}: {e}", exc_info=True)
            return pd.concat(frames, axis=0).sort_index() if frames else pd.DataFrame()

        # ---- Adapter 2: class HistoricalClient with method fetch_ohlcv(product, start, end, granularity)
        elif hasattr(hist_mod, "HistoricalClient"):
            client = hist_mod.HistoricalClient()
            frames = []
            for p in cfg.products:
                try:
                    df = client.fetch_ohlcv(p, start=start, end=end, granularity=cfg.granularity)
                    df = self._normalize_ohlcv(df, p)
                    frames.append(df)
                except Exception as e:
                    logger.error(f"historical error for {p}: {e}", exc_info=True)
            return pd.concat(frames, axis=0).sort_index() if frames else pd.DataFrame()

        else:
            logger.warning("No known adapter for data_ingestor; please add get_ohlcv(...) or HistoricalClient")
            return pd.DataFrame()

    def _start_ws_consumer(self, cfg: IngestionConfig, logger) -> None:
        if ws_mod is None:
            logger.warning("ws_listener module not found; real-time streaming disabled")
            return

        def target():
            # ---- Adapter A: blocking function stream_ticks(products, queue, stop_event)
            if hasattr(ws_mod, "stream_ticks"):
                ws_mod.stream_ticks(cfg.products, self._tick_queue, self._stop_event)
                return

            # ---- Adapter B: class CoinbaseWS(products).run(queue, stop_event)
            elif hasattr(ws_mod, "CoinbaseWS"):
                client = ws_mod.CoinbaseWS(products=cfg.products)
                client.run(self._tick_queue, self._stop_event)
                return

            # ---- Adapter C: async generator subscribe(products); we poll it in a crude thread loop
            elif hasattr(ws_mod, "subscribe"):
                # Fallback polling loop (requires ws_listener to buffer internally)
                while not self._stop_event.is_set():
                    try:
                        # Expect ws_mod.subscribe to provide a non-blocking pull; if not, replace with your API
                        msg = ws_mod.subscribe().get_nowait()
                        self._tick_queue.put(msg, timeout=0.1)
                    except Exception:
                        time.sleep(0.05)
                return

            else:
                logger.error("No known adapter for ws_listener; please expose stream_ticks(...) or CoinbaseWS")
                return

        self._ws_thread = threading.Thread(target=target, name="ws_consumer", daemon=True)
        self._ws_thread.start()

        # Drain queue into deques
        def drain():
            while not self._stop_event.is_set():
                try:
                    msg = self._tick_queue.get(timeout=0.25)
                except Empty:
                    continue
                norm = self._normalize_tick(msg)
                if norm is None:
                    continue
                self._ticks[norm["product"]].append(norm)

        threading.Thread(target=drain, name="ws_drain", daemon=True).start()

    # ---- Normalizers ----
    def _normalize_tick(self, msg: dict) -> Optional[dict]:
        """
        Normalize WS messages into:
        {ts: pd.Timestamp(tz=UTC), product: str, price: float, size: float|None, bid: float|None, ask: float|None}
        Compatible with Coinbase 'ticker' channel-like payloads.
        """
        try:
            # common Coinbase ticker envelope possibilities:
            # time (ISO), product_id, price, last_size, best_bid, best_ask
            product = msg.get("product_id") or msg.get("product") or msg.get("symbol")
            if not product:
                return None

            # ts
            t = msg.get("time") or msg.get("ts") or msg.get("timestamp")
            if isinstance(t, str):
                ts = pd.to_datetime(t, utc=True, errors="coerce")
            elif isinstance(t, (int, float)):
                ts = pd.to_datetime(t, unit="s", utc=True, errors="coerce")
            else:
                ts = pd.Timestamp.utcnow().tz_localize("UTC")
            if ts is pd.NaT:
                ts = pd.Timestamp.utcnow().tz_localize("UTC")

            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return np.nan

            price = _to_float(msg.get("price"))
            size = _to_float(msg.get("last_size") or msg.get("size"))
            bid = _to_float(msg.get("best_bid") or msg.get("bid"))
            ask = _to_float(msg.get("best_ask") or msg.get("ask"))

            return {
                "ts": ts,
                "product": product,
                "price": price,
                "size": size if np.isfinite(size) else np.nan,
                "bid": bid if np.isfinite(bid) else np.nan,
                "ask": ask if np.isfinite(ask) else np.nan,
            }
        except Exception:
            return None

    def _normalize_ohlcv(self, df: pd.DataFrame, product: str) -> pd.DataFrame:
        """Expect columns like [time, open, high, low, close, volume] (any case); return MultiIndex [product, time]."""
        if df is None or df.empty:
            return pd.DataFrame()
        cols = {c.lower(): c for c in df.columns}
        time_col = cols.get("time") or cols.get("timestamp") or cols.get("date")
        if time_col is None:
            # Attempt to infer index
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("OHLCV frame lacks a time column and DatetimeIndex")
            df.index = df.index.tz_convert("UTC") if df.index.tzinfo else df.index.tz_localize("UTC")
        else:
            df = df.copy()
            df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).set_index("time").sort_index()

        rename = {}
        for want in ["open", "high", "low", "close", "volume"]:
            if want not in df.columns:
                # try case-insensitive
                for c in df.columns:
                    if c.lower() == want:
                        rename[c] = want
                        break
        if rename:
            df = df.rename(columns=rename)

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df.index.name = "time"
        df["product"] = product
        return df.set_index("product", append=True).reorder_levels(["product", "time"])

    def _ticks_to_df(self, since: Optional[datetime] = None) -> pd.DataFrame:
        frames = []
        for product, dq in self._ticks.items():
            if not dq:
                continue
            rows = list(dq)
            df = pd.DataFrame(rows)
            if "ts" in df.columns:
                df = df.set_index("ts").sort_index()
            if since is not None:
                df = df[df.index >= pd.Timestamp(since)]
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).sort_index()

    def _build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv is None or ohlcv.empty:
            return pd.DataFrame()
        df = ohlcv.copy()
        # Simple features
        df["ret_1"] = df.groupby(level=0)["close"].pct_change()
        df["vol_ma_20"] = df.groupby(level=0)["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        df["z_ret_20"] = df.groupby(level=0)["ret_1"].transform(
            lambda s: (s - s.rolling(20, min_periods=5).mean()) / (s.rolling(20, min_periods=5).std(ddof=0) + 1e-12)
        )
        return df

_manager = _IngestionManager()

# ----------------------------
# Public function used by agent.py
# ----------------------------
def run_ingestion_pipeline(config: Any, logger, stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """
    Idempotent: starts WS on first call, returns a unified snapshot each time.
    config can be dict or argparse.Namespace with keys:
      - products: List[str]
      - granularity: int (seconds)
      - hist_lookback_minutes: int
      - snapshot_window_secs: int
      - save_parquet: bool
    """
    cfg = IngestionConfig.from_any(config)
    _manager.ensure_started(cfg, logger)

    if stop_event is not None and stop_event.is_set():
        _manager.stop()

    snapshot = _manager.snapshot(cfg, logger)
    return snapshot
