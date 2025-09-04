import json
import logging
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
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
from ws_listener import WebSocketListener

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")

SCRAPED_DIR = Path("ai_core/scraped_data")
ONCHAIN_DIR = Path("ai_core/onchain_data")

CHANNELS = ["ticker", "candles", "level2", "market_trades", "status"]

def extract_market_conditions(scraped_data: dict, onchain_data: dict) -> dict:
    text_blob = " ".join(scraped_data.get("content", []))
    conditions = {
        "volatility": 1.0,
        "sentiment_score": 0.5,
        "whale_activity": 0.5,
    }

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

async def subscribe_to_tokens(ws_listener, scraped_data: dict):
    token_mentions = scraped_data.get("token_mentions", {})
    tokens = [f"{token.upper()}-USD" for token in token_mentions.keys()]
    for channel in CHANNELS:
        await ws_listener.subscribe(channel, tokens)
        logger.info(f"Subscribed to {channel} for tokens: {tokens}")

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

async def main():
    logger.info("🚀 Agent started.")
    tick_queue = asyncio.Queue(maxsize=10000)
    ws_listener = WebSocketListener(
        ws_url="wss://advanced-trade-ws.coinbase.com",
        out_queue=tick_queue,
        top_movers_count=20,
    )
    await ws_listener.run(initial_symbols=None)

    while True:
        goal = get_next_pending_goal()
        if not goal:
            logger.info("⏳ No pending goals. Sleeping...")
            await asyncio.sleep(30)
            continue
        await solve_goal(goal, ws_listener)

if __name__ == "__main__":
    asyncio.run(main()) 
