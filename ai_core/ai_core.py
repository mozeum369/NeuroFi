import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional

# Define paths for memory and logs
AI_CORE_DIR = Path("ai_core")
MEMORY_DIR = AI_CORE_DIR / "memory"
GOALS_FILE = MEMORY_DIR / "goals.json"
STRATEGY_LOG_FILE = MEMORY_DIR / "strategy_performance.json"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Goal Tracking --------------------

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

# -------------------- Strategy Performance Logging --------------------

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

# -------------------- Performance Analytics --------------------

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

# -------------------- Example Usage --------------------

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
