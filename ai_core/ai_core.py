import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Define paths for memory and logs
AI_CORE_DIR = Path("ai_core")
MEMORY_DIR = AI_CORE_DIR / "memory"
GOALS_FILE = MEMORY_DIR / "goals.json"
STRATEGY_LOG_FILE = MEMORY_DIR / "strategy_performance.json"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Goal Tracking --------------------

def store_goal(goal_text: str, metadata: Dict[str, Any] = None) -> None:
    """Store a new goal with timestamp and optional metadata."""
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
    """Load all stored goals."""
    if GOALS_FILE.exists():
        return json.loads(GOALS_FILE.read_text())
    return []

def update_goal_status(goal_text: str, status: str) -> None:
    """Update the status of a goal (e.g., completed, failed)."""
    goals = load_goals()
    for goal in goals:
        if goal["goal"] == goal_text:
            goal["status"] = status
            goal["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            break
    GOALS_FILE.write_text(json.dumps(goals, indent=2))
    print(f"[AI_CORE] Goal status updated: {goal_text} → {status}")

# -------------------- Strategy Performance Logging --------------------

def log_strategy_performance(strategy_name: str, score: float, context: Dict[str, Any]) -> None:
    """Log the performance of a strategy with context and timestamp."""
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
    """Load all strategy performance logs."""
    if STRATEGY_LOG_FILE.exists():
        return json.loads(STRATEGY_LOG_FILE.read_text())
    return []

# -------------------- Memory Access --------------------

def get_recent_goals(n: int = 5) -> List[Dict[str, Any]]:
    """Return the most recent n goals."""
    goals = load_goals()
    return goals[-n:]

def get_top_strategies(n: int = 5) -> List[str]:
    """Return the top n strategies based on average score."""
    logs = load_strategy_logs()
    scores: Dict[str, List[float]] = {}
    for entry in logs:
        name = entry["strategy"]
        scores.setdefault(name, []).append(entry["score"])
    avg_scores = {name: sum(vals)/len(vals) for name, vals in scores.items()}
    sorted_strategies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_strategies[:n]]

# -------------------- Example Usage --------------------

if __name__ == "__main__":
    # Example goal
    store_goal("Find tokens with 20%+ daily surge and explain why.", metadata={"priority": "high"})

    # Example strategy log
    log_strategy_performance("MomentumStrategy", 1.8, {"volatility": 1.2, "sentiment_score": 0.7})

    # Update goal status
    update_goal_status("Find tokens with 20%+ daily surge and explain why.", "completed")

    # Print recent goals and top strategies
    print("Recent goals:", get_recent_goals())
    print("Top strategies:", get_top_strategies()) 


# -------------------- Goal Retrieval --------------------

def get_next_pending_goal() -> Dict[str, Any] | None:
    """
    Return the next pending goal (FIFO) or None if none exist.
    """
    goals = load_goals()
    for goal in goals:
        if goal.get("status") == "pending":
            return goal
    return None
