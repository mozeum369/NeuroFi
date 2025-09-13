import json
import logging
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

# --- Your existing imports (kept) ---
from ai_core import (
    get_next_pending_goal,
    update_goal_status,
    log_strategy_performance,
    load_strategy_logs,
    update_goal_metadata
)
from strategy_selector import pool as strategy_pool
from crawler import gather_data_for_goal
from onchain_scraper import gather_onchain_data_for_goal

# WebSocket listener
from ws_listener import WebSocketListener

# --- NEW: pipeline config + live resampler + historical OHLC helpers ---
from ai_core.pipeline.goal_meta_data import DataPipelineConfig
from ai_core.pipeline.realtime_ohlc import RealTimeOHLCResampler

from ai_core.data_ingestor import (
    fetch_historical_data,
    normalize_market_chart_to_rows,
    rows_to_dataframe,
    aggregate_to_ohlc,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")

# Keep your dirs
SCRAPED_DIR = Path("ai_core/scraped_data")
ONCHAIN_DIR = Path("ai_core/onchain_data")

# Channels we manage (when ws_listener supports multi-channel)
CHANNELS = ["ticker", "candles", "level2", "status"]

# ----------------------------------------------------------------------
# ai_core hook (optional, async)
# ----------------------------------------------------------------------
try:
    from ai_core.ai_core import on_new_bar  # async def on_new_bar(product_id, ts_iso, rowdict)
except Exception:
    async def on_new_bar(product_id: str, bar_ts_iso: str, bar_row: Dict[str, Any]):
        logger.info(f"[ai_core] (fallback) {product_id} new bar {bar_ts_iso} close={bar_row.get('close')}")

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def resolve_symbol_to_product(symbol: str, currency: str = "USD") -> str:
    sym = symbol.upper()
    if "-" in sym:
        return sym
    return f"{sym}-{currency.upper()}"

def save_canonical_ohlc(config: DataPipelineConfig, product_id: str, ohlc_df):
    path = config.canonical_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ohlc_df.copy().sort_index()
    out.index = out.index.tz_convert("UTC")
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=False)
    logger.info(f"[Bootstrap] Wrote canonical OHLC {product_id} ({config.resample_freq}) -> {path}")

# ----------------------------------------------------------------------
# Strategy helpers (yours, unchanged)
# ----------------------------------------------------------------------
def extract_market_conditions(scraped_data: dict, onchain_data: dict) -> dict:
    text_blob = " ".join(scraped_data.get("content", []))
    conditions = {"volatility": 1.0, "sentiment_score": 0.5, "whale_activity": 0.5}

    if "surge" in text_blob or "pump" in text_blob:
        conditions["volatility"] += 0.3
    if "bullish" in text_blob or "positive" in text_blob:
        conditions["sentiment_score"] += 0.3
    if "whale" in text_blob or "large wallet" in text_blob:
        conditions["whale_activity"] += 0.4

    if "whale_concentration" in onchain_data:
        conditions["whale_activity"] += onchain_data["whale_concentration"] * 0.5
    if "transaction_volume" in onchain_data:
        conditions["volatility"] += min(1.0, onchain_data["transaction_volume"] / 1_000_000) * 0.3

    return conditions

def ingest_json_data(directory: Path, goal_text: str) -> dict:
    goal_file = directory / f"{goal_text.replace(' ', '_')}.json"
    if not goal_file.exists():
        logger.warning(f"No data found for goal: {goal_text} in {directory}")
        return {}
    with open(goal_file, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.01) -> float:
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std()

def calculate_strategy_accuracy(strategy_name: str) -> float:
    logs = load_strategy_logs()
    strategy_scores = [entry["score"] for entry in logs if entry["strategy"] == strategy_name]
    if not strategy_scores:
        return 0.0
    threshold = 1.0
    correct_predictions = sum(score >= threshold for score in strategy_scores)
    return correct_predictions / len(strategy_scores)

# ----------------------------------------------------------------------
# Subscriptions (compatible with old/new ws_listener variants)
# ----------------------------------------------------------------------
async def _safe_add_subscription(ws_listener, channel: str, product_ids: List[str]):
    if hasattr(ws_listener, "add_subscription"):
        await ws_listener.add_subscription(channel, product_ids)
    elif hasattr(ws_listener, "subscribe"):
        try:
            await ws_listener.subscribe(channel, product_ids)  # type: ignore
        except TypeError:
            if channel == "ticker":
                await ws_listener.subscribe(product_ids)        # type: ignore
    else:
        logger.warning("ws_listener has no add_subscription/subscribe method.")

async def _safe_remove_subscription(ws_listener, channel: str, product_ids: List[str]):
    if not product_ids:
        return
    if hasattr(ws_listener, "remove_subscription"):
        await ws_listener.remove_subscription(channel, product_ids)
    else:
        logger.warning("ws_listener.remove_subscription not available; cannot unsubscribe in this version.")

async def subscribe_to_tokens(ws_listener, scraped_data: dict):
    token_mentions = scraped_data.get("token_mentions", {})
    tokens = [f"{token.upper()}-USD" for token in token_mentions.keys()]
    if not tokens:
        return
    for channel in CHANNELS:
        await _safe_add_subscription(ws_listener, channel, tokens)
        logger.info(f"Subscribed to {channel} for tokens: {tokens}")

# ----------------------------------------------------------------------
# Goal solving (yours, unchanged, but passes live ws_listener)
# ----------------------------------------------------------------------
async def solve_goal(goal_text: str, ws_listener):
    logger.info(f"🎯 Solving goal: {goal_text}")
    gather_data_for_goal(goal_text)
    gather_onchain_data_for_goal(goal_text)

    scraped = ingest_json_data(SCRAPED_DIR, goal_text)
    onchain = ingest_json_data(ONCHAIN_DIR, goal_text)

    if not scraped and not onchain:
        update_goal_status(goal_text, "failed")
        logger.error(f"❌ Goal '{goal_text}' failed due to missing data.")
        return

    await subscribe_to_tokens(ws_listener, scraped)

    conditions = extract_market_conditions(scraped, onchain)
    logger.info(f"📊 Market conditions: {conditions}")

    best_strategy, score = strategy_pool.select_best_strategy(conditions)
    accuracy = calculate_strategy_accuracy(best_strategy)
    sharpe_ratio = calculate_sharpe_ratio([score])

    logger.info(f"🧠 Selected strategy: {best_strategy} (score: {score:.2f})")
    logger.info(f"📈 Accuracy: {accuracy:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")

    log_strategy_performance(best_strategy, score, context={
        "goal": goal_text,
        "accuracy": accuracy,
        "sharpe_ratio": sharpe_ratio
    })
    update_goal_metadata(goal_text, {
        "selected_strategy": best_strategy,
        "score": score,
        "accuracy": accuracy,
        "sharpe_ratio": sharpe_ratio,
        "market_conditions": conditions
    })
    update_goal_status(goal_text, "completed")
    logger.info(f"✅ Goal '{goal_text}' marked as completed.")

# ----------------------------------------------------------------------
# Bootstrap historical OHLC (CoinGecko -> canonical files)
# ----------------------------------------------------------------------
def bootstrap_backfill(config: DataPipelineConfig, symbols_or_products: List[str]):
    logger.info(f"[Bootstrap] Backfill start={config.backfill_start}, end={config.backfill_end or 'now'}, freq={config.resample_freq}")
    for ident in symbols_or_products:
        if "-" in ident:
            symbol = ident.split("-")[0]
            product_id = ident.upper()
        else:
            symbol = ident.upper()
            product_id = resolve_symbol_to_product(symbol, config.vs_currency)

        raw = fetch_historical_data(symbol, config.vs_currency, config.backfill_start, config.backfill_end or config.backfill_start)
        if not raw:
            logger.warning(f"[Bootstrap] No raw data for {symbol}")
            continue

        rows = normalize_market_chart_to_rows(raw)
        df = rows_to_dataframe(rows)
        ohlc = aggregate_to_ohlc(df, freq=config.resample_freq)
        if ohlc is not None and not ohlc.empty:
            save_canonical_ohlc(config, product_id, ohlc)

# ----------------------------------------------------------------------
# Top Movers Coordinator
# ----------------------------------------------------------------------
class TopMoversCoordinator:
    """
    Periodically reads ai_core/signals/top_movers_candidates.json,
    selects top-N + sticky seeds, and syncs WS subscriptions.
    """
    def __init__(self, config: DataPipelineConfig, ws_listener: WebSocketListener):
        self.cfg = config
        self.ws = ws_listener
        self.state_file = Path("ai_core/signals/top_movers_candidates.json")
        self.current: set[str] = set(config.seed_products) if config.include_seed_products else set()
        self.sticky: set[str] = set(config.seed_products) if config.include_seed_products else set()
        self.last_action_at: dict[str, float] = {}
        self.channels_to_manage = [ch for ch in self.cfg.channels if ch in ("candles", "ticker", "level2")]

    def _now(self) -> float:
        return time.time()

    def _is_fresh(self, iso: str) -> bool:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - dt
            return age.total_seconds() <= self.cfg.freshness_sec
        except Exception:
            return False

    def _load_candidates(self) -> dict:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text())
        except Exception as e:
            logger.warning(f"[TopMovers] Failed to read candidates: {e}")
            return {}

    def _rank_desired(self) -> list[str]:
        data = self._load_candidates()
        items = []
        for pid, info in data.items():
            score = float(info.get("score", 0.0))
            updated = info.get("updated")
            if score < self.cfg.min_candidate_score:
                continue
            if updated and not self._is_fresh(updated):
                continue
            items.append((pid, score))
        items.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in items]

    def _apply_hysteresis(self, ranked: list[str]) -> set[str]:
        limit = max(self.cfg.top_movers_count - len(self.sticky), 0)
        desired_core = set(ranked[:limit])
        desired = set(self.sticky) | desired_core

        data = self._load_candidates()
        cutoff_pid = ranked[limit-1] if limit > 0 and len(ranked) >= limit else None
        cutoff_score = float(data.get(cutoff_pid, {}).get("score", 0.0)) if cutoff_pid else None

        if cutoff_score is not None:
            margin = self.cfg.hysteresis_margin
            for pid in self.current - self.sticky:
                if pid in desired:
                    continue
                my_score = float(data.get(pid, {}).get("score", 0.0))
                if my_score >= (cutoff_score - margin):
                    desired.add(pid)
        return desired

    def _cooldown_ok(self, pid: str) -> bool:
        t = self.last_action_at.get(pid, 0.0)
        return (self._now() - t) >= self.cfg.cooldown_sec

    async def _bootstrap_new(self, add_list: list[str]):
        if not (self.cfg.bootstrap_on_add and add_list):
            return
        start = (datetime.now(timezone.utc) - timedelta(days=self.cfg.bootstrap_lookback_days)).strftime("%Y-%m-%d")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for pid in add_list:
            try:
                sym = pid.split("-")[0]
                raw = fetch_historical_data(sym, self.cfg.vs_currency, start, end)
                if raw:
                    rows = normalize_market_chart_to_rows(raw)
                    df = rows_to_dataframe(rows)
                    ohlc = aggregate_to_ohlc(df, freq=self.cfg.resample_freq)
                    if ohlc is not None and not ohlc.empty:
                        path = self.cfg.canonical_path(pid)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        out = ohlc.copy().sort_index()
                        out.index = out.index.tz_convert("UTC")
                        out = out.reset_index().rename(columns={"index": "timestamp"})
                        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        out.to_csv(path, index=False)
                        logger.info(f"[TopMovers] Bootstrapped OHLC for {pid} -> {path}")
            except Exception as e:
                logger.warning(f"[TopMovers] Bootstrap backfill failed for {pid}: {e}")

    async def _apply_changes(self, to_add: list[str], to_remove: list[str]):
        max_change = max(1, self.cfg.max_changes_per_cycle)
        to_add = [p for p in to_add if self._cooldown_ok(p)][:max_change]
        to_remove = [p for p in to_remove if self._cooldown_ok(p)][:max_change]

        if not to_add and not to_remove:
            return

        await self._bootstrap_new(to_add)

        for ch in self.channels_to_manage:
            if to_add:
                await _safe_add_subscription(self.ws, ch, to_add)
            if to_remove:
                await _safe_remove_subscription(self.ws, ch, to_remove)

        for p in to_add:
            self.current.add(p); self.last_action_at[p] = self._now()
        for p in to_remove:
            if p in self.current and p not in self.sticky:
                self.current.remove(p); self.last_action_at[p] = self._now()

        logger.info(f"[TopMovers] add={to_add} remove={to_remove} current={sorted(self.current)}")

    async def run(self):
        first_run = True
        while True:
            try:
                ranked = self._rank_desired()
                desired = self._apply_hysteresis(ranked)

                if len(desired) > self.cfg.top_movers_count:
                    overflow = len(desired) - self.cfg.top_movers_count
                    rank_map = {pid: i for i, pid in enumerate(ranked)}
                    non_sticky_sorted = sorted([p for p in desired if p not in self.sticky],
                                               key=lambda p: rank_map.get(p, 1_000_000))
                    for p in reversed(non_sticky_sorted[-overflow:]):
                        desired.discard(p)

                to_add = sorted(list(desired - self.current))
                to_remove = sorted([p for p in (self.current - desired) if p not in self.sticky])

                if first_run and self.sticky:
                    for ch in self.channels_to_manage:
                        await _safe_add_subscription(self.ws, ch, sorted(self.sticky))
                    self.current |= set(self.sticky)
                    first_run = False

                await self._apply_changes(to_add, to_remove)

            except Exception as e:
                logger.warning(f"[TopMovers] Loop error: {e}")

            await asyncio.sleep(self.cfg.top_movers_poll_sec)

# ----------------------------------------------------------------------
# Live pipeline: start WS + resampler + notifier + movers (background) and return ws handle
# ----------------------------------------------------------------------
async def start_live_pipeline(config: DataPipelineConfig) -> WebSocketListener:
    # 1) Real-time resampler
    resampler = RealTimeOHLCResampler(freq=config.resample_freq, canonical_path_func=config.canonical_path)

    # 2) WebSocket listener + queue
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    ws_listener = WebSocketListener(
        ws_url=(config.ws_url or "wss://advanced-trade-ws.coinbase.com"),
        out_queue=tick_queue,
        top_movers_count=getattr(config, "top_movers_count", 20),
    )

    # Start listener
    async def start_ws():
        try:
            initial = {ch: (config.seed_products if ch in ("ticker", "ticker_batch", "candles", "level2") else None)
                       for ch in config.channels}
            await ws_listener.run(initial_subscriptions=initial)  # upgraded variant
        except TypeError:
            logger.info("ws_listener.run(initial_subscriptions=...) not supported; falling back to initial_symbols.")
            await ws_listener.run(initial_symbols=config.seed_products)  # older variant

    asyncio.create_task(start_ws())

    # Consumer routes 'candles' frames to resampler
    async def consume():
        while True:
            msg = await tick_queue.get()
            if isinstance(msg, dict) and msg.get("channel") == "candles":
                resampler.ingest_candles_message(msg)

    asyncio.create_task(consume())

    # Notifier: detect new/updated bars by scanning canonical folder (handles dynamic adds)
    async def notify_new_bars():
        import pandas as pd
        last_seen: Dict[str, str] = {}
        freq_suffix = f"_{config.resample_freq}"
        base_dir = config.canonical_dir
        while True:
            for path in base_dir.glob(f"*{freq_suffix}.csv"):
                stem = path.stem  # e.g., "BTC-USD_1H"
                if not stem.endswith(freq_suffix):
                    continue
                product = stem[: -len(freq_suffix)]
                try:
                    df = pd.read_csv(path)
                    if df.empty:
                        continue
                    last_ts = df["timestamp"].iloc[-1]
                    if last_seen.get(product) != last_ts:
                        row = df.iloc[-1].to_dict()
                        ts_iso = row.pop("timestamp")
                        await on_new_bar(product, ts_iso, row)
                        last_seen[product] = last_ts
                except Exception as e:
                    logger.warning(f"[Notifier] Error reading {path}: {e}")
            await asyncio.sleep(10)

    asyncio.create_task(notify_new_bars())

    # Top movers coordinator
    asyncio.create_task(TopMoversCoordinator(config, ws_listener).run())

    return ws_listener

# ----------------------------------------------------------------------
# Main entry: run goals + pipeline together
# ----------------------------------------------------------------------
async def main():
    logger.info("🚀 Agent started.")

    # 0) Pipeline config (move to JSON/YAML later if you prefer)
    config = DataPipelineConfig(
        vs_currency="USD",
        backfill_start="2023-01-01",
        backfill_end=None,            # None = now
        resample_freq="1H",
        api_style="advanced_trade",
        ws_url=None,
        channels=["candles", "ticker", "level2", "status"],
        ensure_heartbeats=True,
        seed_products=["BTC-USD", "ETH-USD", "SOL-USD"],
        dynamic_top_movers=True,
        top_movers_count=20,

        # Top movers tuning
        top_movers_poll_sec=30,
        min_candidate_score=0.15,
        freshness_sec=15*60,
        hysteresis_margin=0.03,
        max_changes_per_cycle=10,
        cooldown_sec=60,
        include_seed_products=True,
        bootstrap_on_add=True,
        bootstrap_lookback_days=7,
    )

    # 1) Bootstrap historical (once)
    bootstrap_backfill(config, config.seed_products)

    # 2) Start live pipeline and get ws handle
    ws_handle = await start_live_pipeline(config)

    # 3) Goal loop (concurrent)
    async def goal_loop():
        while True:
            goal = get_next_pending_goal()
            if not goal:
                logger.info("⏳ No pending goals. Sleeping...")
                await asyncio.sleep(30)
                continue
            # FIX: goal is a dict; pass the string text
            await solve_goal(goal["goal"], ws_handle)

    await goal_loop()  # keep running

if __name__ == "__main__":
    asyncio.run(main())
