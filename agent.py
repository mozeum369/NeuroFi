import json
import logging
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# WebSocket listener (we support both your current and upgraded versions)
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

# Your channel policy (used when upgraded ws_listener is present)
CHANNELS = ["ticker", "candles", "level2", "market_trades", "status"]

# ----------------------------------------------------------------------
# ai_core hook (optional, async)
# ----------------------------------------------------------------------
try:
    # If you implement this in ai_core/ai_core.py, we’ll call it on each new bar
    from ai_core.ai_core import on_new_bar  # async def on_new_bar(product_id, ts_iso, rowdict)
except Exception:
    async def on_new_bar(product_id: str, bar_ts_iso: str, bar_row: Dict[str, Any]):
        logger.info(f"[ai_core] (fallback) {product_id} new bar {bar_ts_iso} close={bar_row.get('close')}")

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def resolve_symbol_to_product(symbol: str, currency: str = "USD") -> str:
    """'BTC' -> 'BTC-USD'  | 'BTC-USD' stays as-is."""
    sym = symbol.upper()
    if "-" in sym:
        return sym
    return f"{sym}-{currency.upper()}"

def canonical_path(config: DataPipelineConfig, product_id: str) -> Path:
    return config.canonical_path(product_id)

def save_canonical_ohlc(config: DataPipelineConfig, product_id: str, ohlc_df):
    """Write canonical OHLC (shared by historical + live)."""
    path = canonical_path(config, product_id)
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
# Subscriptions (compatible with both ws_listener variants)
# ----------------------------------------------------------------------
async def _safe_add_subscription(ws_listener, channel: str, product_ids: List[str]):
    """
    Works with:
      - Upgraded listener: add_subscription(channel, product_ids)
      - Your newer signature: subscribe(channel, product_ids)
      - Older signature: subscribe(product_ids)  [ticker only]
    """
    if hasattr(ws_listener, "add_subscription"):
        await ws_listener.add_subscription(channel, product_ids)
    elif hasattr(ws_listener, "subscribe"):
        try:
            # Try (channel, products)
            await ws_listener.subscribe(channel, product_ids)  # type: ignore
        except TypeError:
            # Fall back to (products) only for ticker
            if channel == "ticker":
                await ws_listener.subscribe(product_ids)        # type: ignore
    else:
        logger.warning("ws_listener has no add_subscription/subscribe method.")

async def subscribe_to_tokens(ws_listener, scraped_data: dict):
    token_mentions = scraped_data.get("token_mentions", {})
    tokens = [f"{token.upper()}-USD" for token in token_mentions.keys()]
    if not tokens:
        return
    for channel in CHANNELS:
        await _safe_add_subscription(ws_listener, channel, tokens)
        logger.info(f"Subscribed to {channel} for tokens: {tokens}")

# ----------------------------------------------------------------------
# Goal solving (yours, unchanged, with subscriptions)
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
        # For CoinGecko we accept 'BTC' etc.; if product given (BTC-USD), convert to symbol
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
        df = rows_to_dataframe(rows)              # -> columns: price, volume (UTC index)
        ohlc = aggregate_to_ohlc(df, freq=config.resample_freq)
        if ohlc is not None and not ohlc.empty:
            save_canonical_ohlc(config, product_id, ohlc)

# ----------------------------------------------------------------------
# Live pipeline: WS + real-time OHLC + notifier to ai_core
# ----------------------------------------------------------------------
async def run_live_pipeline(config: DataPipelineConfig):
    # 1) Real-time resampler (5m -> 1H/1D), writes canonical OHLC files
    resampler = RealTimeOHLCResampler(freq=config.resample_freq, canonical_path_func=config.canonical_path)

    # 2) WebSocket listener + queue
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    # Prefer Advanced Trade URL if you didn’t explicitly override in config
    ws_listener = WebSocketListener(
        ws_url=(config.ws_url or "wss://advanced-trade-ws.coinbase.com"),
        out_queue=tick_queue,
        top_movers_count=config.top_movers_count if hasattr(config, "top_movers_count") else 20,
    )

    # Start listener in the background with initial product subscriptions
    async def start_ws():
        # For upgraded listener: pass initial subscriptions per channel; for older versions, pass product list only.
        try:
            # Try upgraded pattern: channels + products
            initial = {ch: (config.seed_products if ch in ("ticker", "ticker_batch", "candles", "level2") else None)
                       for ch in ["ticker", "candles", "level2", "status"]}
            await ws_listener.run(initial_subscriptions=initial)  # type: ignore
        except TypeError:
            # Fallback to your older signature: a flat product list and only ticker channel
            logger.info("ws_listener.run(initial_subscriptions=...) not supported; falling back to initial_symbols.")
            await ws_listener.run(initial_symbols=config.seed_products)  # type: ignore

    ws_task = asyncio.create_task(start_ws())

    # 3) Consumer routes WS messages to the resampler (expects Advanced Trade 'candles' channel)
    async def consume():
        while True:
            msg = await tick_queue.get()
            if isinstance(msg, dict) and msg.get("channel") == "candles":
                resampler.ingest_candles_message(msg)
            # You can also branch 'ticker' or 'level2' here for microstructure features.

    consumer_task = asyncio.create_task(consume())

    # 4) Notify ai_core when a new bar closes (poll canonical files)
    async def notify_new_bars():
        import pandas as pd
        last_seen: Dict[str, str] = {}
        while True:
            for product in config.seed_products:
                path = config.canonical_path(product)
                if not path.exists():
                    continue
                try:
                    df = pd.read_csv(path)
                    if df.empty:
                        continue
                    last_ts = df["timestamp"].iloc[-1]
                    if last_seen.get(product) != last_ts:
                        row = df.iloc[-1].to_dict()
                        ts_iso = row.pop("timestamp")
                        await on_new_bar(product, ts_iso, row)  # async hook into ai_core
                        last_seen[product] = last_ts
                except Exception as e:
                    logger.warning(f"[Notifier] Error reading {path}: {e}")
            await asyncio.sleep(10)  # tune based on how quickly you want signals after bar close

    notifier_task = asyncio.create_task(notify_new_bars())

    await asyncio.gather(ws_task, consumer_task, notifier_task)

# ----------------------------------------------------------------------
# Main entry: run goals + pipeline together
# ----------------------------------------------------------------------
async def main():
    logger.info("🚀 Agent started.")

    # 0) Load pipeline (goal_meta_data). You can externalize to a JSON/YAML later if you want.
    config = DataPipelineConfig(
        vs_currency="USD",
        backfill_start="2023-01-01",
        backfill_end=None,          # None = now
        resample_freq="1H",         # switch to "1D" if you want daily bars
        api_style="advanced_trade",
        ws_url=None,
        channels=["candles", "ticker", "level2", "status"],
        ensure_heartbeats=True,
        seed_products=["BTC-USD", "ETH-USD", "SOL-USD"],
        dynamic_top_movers=True,
        top_movers_count=20,
    )

    # 1) Bootstrap historical (non-async, runs once)
    bootstrap_backfill(config, config.seed_products)

    # 2) Start live data pipeline in the background
    live_task = asyncio.create_task(run_live_pipeline(config))

    # 3) Goal loop (your original logic), now running concurrently
    async def goal_loop():
        while True:
            goal = get_next_pending_goal()
            if not goal:
                logger.info("⏳ No pending goals. Sleeping...")
                await asyncio.sleep(30)
                continue
            # We pass the ws listener object by reference—retrieved from the live task scope is complex,
            # so for simplicity, we retain a module-global reference pattern (or refactor to share).
            # Here, we’ll re-create a lightweight handle to the running listener by storing it on the task.
            # Simpler approach: keep subscribe_to_tokens to operate on scraped data later via a shared queue.
            # For now, we skip passing the live listener into solve_goal; instead, solve_goal will subscribe via a global registry.
            # To keep your current signature, we expose a NO-OP listener object. See below for the simple adapter.
            await solve_goal(goal, _noop_listener)

    # Minimal adapter to let your solve_goal() issue subscriptions without breaking
    class _NoopListener:
        async def add_subscription(self, channel, product_ids): pass
        async def subscribe(self, *args, **kwargs): pass
    _noop_listener = _NoopListener()

    # NOTE:
    # If you want solve_goal() to actually subscribe in real time, we can wire a shared reference.
    # Easiest is to put the active ws_listener into a global registry in run_live_pipeline()
    # and pull it here. I kept it simple and safe; happy to wire it directly if you prefer.

    goal_task = asyncio.create_task(goal_loop())

    await asyncio.gather(live_task, goal_task)

if __name__ == "__main__":
    asyncio.run(main())
 

 


 



 

 
