# ai_core/ai_core.py

import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
import math

# -------------------- Paths & Directories --------------------

AI_CORE_DIR = Path("ai_core")
MEMORY_DIR = AI_CORE_DIR / "memory"
GOALS_FILE = MEMORY_DIR / "goals.json"
STRATEGY_LOG_FILE = MEMORY_DIR / "strategy_performance.json"

# New: signal sink
SIGNALS_DIR = AI_CORE_DIR / "signals"
SIGNALS_FILE = SIGNALS_DIR / "signals.jsonl"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Goal Tracking (YOUR ORIGINAL) --------------------

def store_goal(goal_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    metadata = metadata or {}
    goal_entry = {
        "goal": goal_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata,
        "status": "pending"
    }
    goals = load_goals()
    goals.append(goal_entry)
    GOALS_FILE.write_text(json.dumps(goals, indent=2))
    print(f"[AI_CORE] Goal stored: {goal_text}")

def load_goals() -> List[Dict[str, Any]]:
    if GOALS_FILE.exists():
        return json.loads(GOALS_FILE.read_text())
    return []

def update_goal_status(goal_text: str, status: str) -> None:
    goals = load_goals()
    for goal in goals:
        if goal["goal"] == goal_text:
            goal["status"] = status
            goal["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            break
    GOALS_FILE.write_text(json.dumps(goals, indent=2))
    print(f"[AI_CORE] Goal status updated: {goal_text} → {status}")

# -------------------- Goal Metadata (YOUR ORIGINAL) --------------------

GOAL_METADATA_DIR = Path("ai_core/goal_metadata")
GOAL_METADATA_DIR.mkdir(exist_ok=True)

def update_goal_metadata(goal_text: str, metadata: dict):
    goal_file = GOAL_METADATA_DIR / f"{goal_text.replace(' ', '_')}.json"
    with open(goal_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def get_next_pending_goal() -> Optional[Dict[str, Any]]:
    goals = load_goals()
    sorted_goals = sorted(goals, key=lambda g: g.get("metadata", {}).get("priority", "medium"), reverse=True)
    for goal in sorted_goals:
        if goal.get("status") == "pending":
            return goal
    return None

def get_recent_goals(n: int = 5) -> List[Dict[str, Any]]:
    goals = load_goals()
    return goals[-n:]

# -------------------- Strategy Performance Logging (YOUR ORIGINAL) --------------------

def log_strategy_performance(strategy_name: str, score: float, context: Dict[str, Any]) -> None:
    entry = {
        "strategy": strategy_name,
        "score": score,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": context
    }
    logs = load_strategy_logs()
    logs.append(entry)
    STRATEGY_LOG_FILE.write_text(json.dumps(logs, indent=2))
    print(f"[AI_CORE] Strategy performance logged: {strategy_name} → {score:.2f}")

def load_strategy_logs() -> List[Dict[str, Any]]:
    if STRATEGY_LOG_FILE.exists():
        return json.loads(STRATEGY_LOG_FILE.read_text())
    return []

def get_top_strategies(n: int = 5) -> List[str]:
    logs = load_strategy_logs()
    scores: Dict[str, List[float]] = {}
    for entry in logs:
        name = entry["strategy"]
        scores.setdefault(name, []).append(entry["score"])
    avg_scores = {name: sum(vals)/len(vals) for name, vals in scores.items()}
    sorted_strategies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_strategies[:n]]

# -------------------- Performance Analytics (YOUR ORIGINAL) --------------------

def calculate_strategy_accuracy(strategy_name: str, threshold: float = 1.0) -> float:
    logs = load_strategy_logs()
    relevant = [entry for entry in logs if entry["strategy"] == strategy_name]
    if not relevant:
        return 0.0
    successful = [entry for entry in relevant if entry["score"] >= threshold]
    return len(successful) / len(relevant)

def calculate_sharpe_ratio(strategy_name: str, risk_free_rate: float = 0.01) -> float:
    logs = load_strategy_logs()
    returns = [entry["score"] for entry in logs if entry["strategy"] == strategy_name]
    if len(returns) < 2:
        return 0.0
    avg_return = statistics.mean(returns)
    std_dev = statistics.stdev(returns)
    if std_dev == 0:
        return 0.0
    excess_return = avg_return - risk_free_rate
    return excess_return / std_dev

def evaluate_strategy_health(strategy_name: str) -> Dict[str, float]:
    accuracy = calculate_strategy_accuracy(strategy_name)
    sharpe = calculate_sharpe_ratio(strategy_name)
    return {"accuracy": accuracy, "sharpe_ratio": sharpe}

def identify_underperforming_strategies(threshold_accuracy: float = 0.5, threshold_sharpe: float = 0.5) -> List[str]:
    logs = load_strategy_logs()
    strategies = set(entry["strategy"] for entry in logs)
    underperformers = []
    for strategy in strategies:
        metrics = evaluate_strategy_health(strategy)
        if metrics["accuracy"] < threshold_accuracy or metrics["sharpe_ratio"] < threshold_sharpe:
            underperformers.append(strategy)
    return underperformers

def identify_stalled_goals(max_age_seconds: int = 86400) -> List[str]:
    goals = load_goals()
    stalled = []
    now = time.time()
    for goal in goals:
        if goal.get("status") == "pending":
            ts = time.strptime(goal["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
            age = now - time.mktime(ts)
            if age > max_age_seconds:
                stalled.append(goal["goal"])
    return stalled

# -------------------- Goal-Linked RL (YOUR ORIGINAL) --------------------

class GoalLinkedEnv:
    def __init__(self, strategies: List[str], goal: Dict[str, Any]):
        self.strategies = strategies
        self.goal = goal
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self._get_state()

    def step(self, action: str):
        score = calculate_strategy_accuracy(action)
        reward = score
        self.current_index += 1
        done = self.current_index >= len(self.strategies)
        next_state = self._get_state()
        return next_state, reward, done, {}

    def _get_state(self):
        if self.current_index < len(self.strategies):
            return {
                "strategy": self.strategies[self.current_index],
                "goal": self.goal["goal"],
                "priority": self.goal.get("metadata", {}).get("priority", "medium")
            }
        return {"strategy": None, "goal": self.goal["goal"], "priority": self.goal.get("metadata", {}).get("priority", "medium")}

class GoalLinkedAgent:
    def __init__(self, env: GoalLinkedEnv):
        self.env = env
        self.policy: Dict[str, float] = {}

    def select_action(self, state: Dict[str, Any]) -> str:
        strategy = state["strategy"]
        if strategy not in self.policy:
            self.policy[strategy] = 1.0
        return strategy

    def learn(self, state: Dict[str, Any], action: str, reward: float):
        self.policy[action] = self.policy.get(action, 0.0) + 0.1 * (reward - self.policy.get(action, 0.0))

def run_goal_linked_rl_loop(episodes: int = 3):
    goal = get_next_pending_goal()
    if not goal:
        print("[RL] No pending goals found.")
        return

    strategies = get_top_strategies(n=10)
    env = GoalLinkedEnv(strategies, goal)
    agent = GoalLinkedAgent(env)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward)
            state = next_state
            total_reward += reward

        print(f"[RL] Goal: {goal['goal']} | Episode {episode + 1} → Total Reward: {total_reward:.2f}")

    return agent, goal

# -------------------- SHAP (YOUR ORIGINAL) --------------------

import shap
import numpy as np

class SHAPWrapperModel:
    def __init__(self, agent: GoalLinkedAgent):
        self.agent = agent

    def predict(self, X: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([
            self.agent.policy.get(x["strategy"], 1.0) for x in X
        ]).reshape(-1, 1)

def explain_goal_linked_decision(agent: GoalLinkedAgent, strategy_name: str):
    explainer = shap.Explainer(SHAPWrapperModel(agent).predict, feature_names=["strategy"])
    sample_input = [{"strategy": strategy_name}]
    shap_values = explainer(sample_input)
    print(f"\n[SHAP] Explanation for strategy: {strategy_name}")
    shap.plots.text(shap_values[0])

# ==================== NEW: Real-time OHLC Handling & Signals ====================

# Simple in-memory bar history per product (avoids disk I/O for features)
# Each entry: {"ts": iso, "open": float, "high": float, "low": float, "close": float, "volume": float}
HISTORY: Dict[str, deque] = {}
MAX_HISTORY = 600  # keep last ~600 bars (~25 days if 1H)

def _ensure_history(product_id: str):
    if product_id not in HISTORY:
        HISTORY[product_id] = deque(maxlen=MAX_HISTORY)

def _to_float_safe(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def compute_features_from_history(product_id: str) -> Dict[str, Optional[float]]:
    """
    Compute features from the in-memory history for a product.
    Returns a dict with keys: rsi, atr, vol_lookback, vol_zscore, ret_1, ret_5, ret_24,
    breakout_high, breakout_low, close, volume
    """
    hist = HISTORY.get(product_id, deque())
    n = len(hist)
    if n < 20:
        return {"ready": 0}  # not enough data

    closes = np.array([h["close"] for h in hist], dtype=float)
    highs  = np.array([h["high"]  for h in hist], dtype=float)
    lows   = np.array([h["low"]   for h in hist], dtype=float)
    vols   = np.array([h["volume"] for h in hist], dtype=float)

    # --- Returns ---
    def safe_ret(a, k=1):
        if len(a) > k and a[-k-1] > 0:
            return a[-1]/a[-k-1]-1.0
        return None

    ret_1  = safe_ret(closes, 1)
    ret_5  = safe_ret(closes, 5)
    ret_24 = safe_ret(closes, 24)

    # --- Log-return volatility (window=24 by default for 1H) ---
    vol_window = 24
    if len(closes) > vol_window + 1:
        lr = np.diff(np.log(closes[-(vol_window+1):]))
        vol_lookback = float(np.std(lr, ddof=1))
    else:
        vol_lookback = None

    # --- RSI (period=14, SMA variant) ---
    period = 14
    rsi = None
    if len(closes) > period:
        delta = np.diff(closes)
        gains = np.clip(delta, a_min=0, a_max=None)
        losses = np.clip(-delta, a_min=0, a_max=None)
        if len(gains) >= period:
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0 and avg_gain == 0:
                rsi = 50.0
            elif avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

    # --- ATR (period=14) ---
    atr = None
    if len(closes) > period:
        prev_close = closes[:-1]
        tr = np.maximum.reduce([
            highs[1:] - lows[1:],
            np.abs(highs[1:] - prev_close),
            np.abs(lows[1:] - prev_close)
        ])
        if len(tr) >= period:
            atr = float(np.mean(tr[-period:]))

    # --- Volume z-score (window=24) ---
    vol_z = None
    if len(vols) >= vol_window:
        v = vols[-vol_window:]
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if np.std(v, ddof=1) > 0 else None
        if sd and sd > 0:
            vol_z = (float(vols[-1]) - mu) / sd
        elif mu:
            vol_z = (float(vols[-1]) - mu) / (abs(mu) + 1e-9)

    # --- Breakouts (rolling 24 bars) ---
    breakout_high = None
    breakout_low = None
    if len(closes) > 24:
        window = closes[-24:]
        rolling_high = float(np.max(window[:-1]))  # exclude current
        rolling_low = float(np.min(window[:-1]))
        breakout_high = 1.0 if closes[-1] > rolling_high else 0.0
        breakout_low = 1.0 if closes[-1] < rolling_low else 0.0

    feats = {
        "ready": 1,
        "close": float(closes[-1]),
        "volume": float(vols[-1]),
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_24": ret_24,
        "vol_lookback": vol_lookback,
        "rsi": rsi,
        "atr": atr,
        "vol_zscore": vol_z,
        "breakout_high": breakout_high,
        "breakout_low": breakout_low,
    }
    return feats

def _tanh(x: Optional[float], scale: float = 20.0) -> float:
    if x is None:
        return 0.0
    return math.tanh(scale * x)

def generate_signal(product_id: str, ts_iso: str, feats: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """
    Simple, interpretable scoring rule for BUY/SELL/HOLD.
    You can replace this with your model later.
    """
    if not feats.get("ready"):
        return {
            "product_id": product_id, "timestamp": ts_iso,
            "action": "HOLD", "score": 0.0,
            "reason": "Insufficient history",
            "features": feats
        }

    # Components
    momentum = 0.6 * _tanh(feats.get("ret_1")) + 0.3 * _tanh(feats.get("ret_5")) + 0.1 * _tanh(feats.get("ret_24"))
    rsi_term = 0.5 * (((feats.get("rsi") or 50.0) - 50.0) / 50.0)  # [-1, 1] roughly
    vol_term = 0.2 * (feats.get("vol_zscore") or 0.0) / 3.0        # scale z-score a bit
    breakout_term = 0.4 * ((feats.get("breakout_high") or 0.0) - (feats.get("breakout_low") or 0.0))

    raw_score = momentum + rsi_term + vol_term + breakout_term

    # Clamp to [-1, 1]
    score = float(max(-1.0, min(1.0, raw_score)))

    # Decision thresholds
    if score >= 0.25 and (feats.get("rsi") is None or feats["rsi"] <= 75):
        action = "BUY"
    elif score <= -0.25 and (feats.get("rsi") is None or feats["rsi"] >= 25):
        action = "SELL"
    else:
        action = "HOLD"

    reason = (
        f"momentum={momentum:.2f}, rsi_term={rsi_term:.2f}, vol_term={vol_term:.2f}, "
        f"breakout={breakout_term:.2f}, rsi={feats.get('rsi')}, ret_1={feats.get('ret_1')}"
    )

    signal = {
        "product_id": product_id,
        "timestamp": ts_iso,
        "action": action,
        "score": score,
        "reason": reason,
        "features": feats
    }
    return signal

def emit_signal(signal: Dict[str, Any]) -> None:
    """
    Append the signal as JSONL. You can have your dashboard or strategy runner tail this file.
    """
    with open(SIGNALS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(signal) + "\n")
    # Log into your strategy performance log for traceability (optional)
    log_strategy_performance("RealtimeSignal", signal["score"], context=signal)

# (Optional) a simple top-mover score you can use later to drive subscriptions
def update_top_mover_score(product_id: str, feats: Dict[str, Optional[float]]):
    # Example: emphasize short-term return & volume surge
    if not feats.get("ready"):
        return
    short = feats.get("ret_1") or 0.0
    volz = feats.get("vol_zscore") or 0.0
    score = 0.7 * _tanh(short, 50.0) + 0.3 * (volz / 3.0)
    path = SIGNALS_DIR / "top_movers_candidates.json"
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
        data[product_id] = {"score": float(score), "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[AI_CORE] Failed to update top_movers_candidates: {e}")

# -------------------- MAIN LIVE ENTRYPOINT (CALLED BY agent.py) --------------------

async def on_new_bar(product_id: str, bar_ts_iso: str, bar_row: Dict[str, Any]):
    """
    Called by agent.py whenever a new bar is written/closed.
    bar_row should contain: open, high, low, close, volume, ret, log_ret (ret/log_ret optional).
    """
    _ensure_history(product_id)

    # Normalize and append to in-memory history
    o = _to_float_safe(bar_row.get("open"))
    h = _to_float_safe(bar_row.get("high"))
    l = _to_float_safe(bar_row.get("low"))
    c = _to_float_safe(bar_row.get("close"))
    v = _to_float_safe(bar_row.get("volume"))

    if None in (o, h, l, c) or v is None:
        print(f"[AI_CORE] Skipping bar with missing fields for {product_id} @ {bar_ts_iso}")
        return

    HISTORY[product_id].append({
        "ts": bar_ts_iso,
        "open": o, "high": h, "low": l, "close": c, "volume": v
    })

    # Compute features and emit signal
    feats = compute_features_from_history(product_id)
    signal = generate_signal(product_id, bar_ts_iso, feats)
    emit_signal(signal)
    update_top_mover_score(product_id, feats)
    print(f"[AI_CORE] {product_id} {bar_ts_iso} → {signal['action']} (score={signal['score']:.2f})")

# -------------------- Example (kept from your original) --------------------

if __name__ == "__main__":
    store_goal("Find tokens with 20%+ daily surge and explain why.", metadata={"priority": "high"})
    log_strategy_performance("MomentumStrategy", 1.8, {"volatility": 1.2, "sentiment_score": 0.7})
    update_goal_status("Find tokens with 20%+ daily surge and explain why.", "completed")

    print("Recent goals:", get_recent_goals())
    print("Top strategies:", get_top_strategies())
    print("MomentumStrategy accuracy:", calculate_strategy_accuracy("MomentumStrategy"))
    print("MomentumStrategy Sharpe ratio:", calculate_sharpe_ratio("MomentumStrategy"))
    print("Underperforming strategies:", identify_underperforming_strategies())
    print("Stalled goals:", identify_stalled_goals())

    # Run goal-linked RL loop and explain decision
    agent, goal = run_goal_linked_rl_loop(episodes=2)
    if agent:
        explain_goal_linked_decision(agent, "MomentumStrategy")
