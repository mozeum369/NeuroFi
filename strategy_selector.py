import json
import statistics
import numpy as np
from typing import List, Dict, Any

# -------------------- Load Strategy Logs --------------------

def load_strategy_logs(file_path: str = "ai_core/memory/strategy_performance.json") -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Define the Strategy and StrategyPool classes with expanded strategy options
class Strategy:
    def __init__(self, name, description, risk_level, expected_roi, time_horizon):
        self.name = name
        self.description = description
        self.risk_level = risk_level
        self.expected_roi = expected_roi
        self.time_horizon = time_horizon
        self.performance_history = []

    def evaluate(self, market_conditions):
        # Placeholder evaluation logic based on market conditions
        score = (
            market_conditions.get("volatility", 1) * 0.3 +
            market_conditions.get("sentiment_score", 0.5) * 0.4 +
            market_conditions.get("whale_activity", 0.5) * 0.3
        )
        performance = score * self.expected_roi
        self.performance_history.append(performance)
        return performance

    def calculate_accuracy(self):
        # Placeholder accuracy: percentage of scores above a threshold
        if not self.performance_history:
            return 0.0
        threshold = 1.0
        correct = sum(1 for p in self.performance_history if p >= threshold)
        return correct / len(self.performance_history)

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        returns = np.array(self.performance_history)
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

class StrategyPool:
    def __init__(self):
        self.strategies = []

    def register_strategy(self, strategy):
        self.strategies.append(strategy)

    def evaluate_strategies(self, market_conditions):
        evaluations = {
            strategy.name: strategy.evaluate(market_conditions)
            for strategy in self.strategies
        }
        return evaluations

    def select_best_strategy(self, market_conditions):
        evaluations = self.evaluate_strategies(market_conditions)
        best_strategy_name = max(evaluations, key=evaluations.get)
        best_strategy = next(s for s in self.strategies if s.name == best_strategy_name)
        accuracy = best_strategy.calculate_accuracy()
        sharpe_ratio = best_strategy.calculate_sharpe_ratio()
        print(f"Best strategy selected: {best_strategy.name} with score {evaluations[best_strategy.name]:.2f}")
        print(f"Accuracy: {accuracy:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")
        return best_strategy.name, evaluations[best_strategy.name]

# Instantiate the strategy pool
pool = StrategyPool()

# Register multiple strategies
pool.register_strategy(Strategy("MomentumStrategy", "Captures short-term upward trends in asset prices.", "medium", 1.5, "short"))
pool.register_strategy(Strategy("WhaleTrackerStrategy", "Tracks large wallet movements to anticipate market shifts.", "high", 2.0, "short"))
pool.register_strategy(Strategy("BreakoutStrategy", "Detects price breakouts from consolidation zones.", "medium", 1.8, "short"))
pool.register_strategy(Strategy("ArbitrageStrategy", "Exploits price differences across exchanges for profit.", "low", 1.2, "short"))
pool.register_strategy(Strategy("ETFManipulationDetector", "Identifies ETF and hedge fund activity that may distort prices.", "high", 2.5, "medium"))
pool.register_strategy(Strategy("SentimentSurgeStrategy", "Leverages sudden positive sentiment spikes from social media.", "medium", 1.7, "short"))
pool.register_strategy(Strategy("WhaleExitStrategy", "Detects large wallet exits to anticipate price drops.", "high", 2.2, "short"))
pool.register_strategy(Strategy("CorrelationBreakStrategy", "Identifies when correlated assets diverge, signaling opportunity.", "medium", 1.6, "short"))
pool.register_strategy(Strategy("HistoricalEchoStrategy", "Matches current conditions to historical surges for prediction.", "medium", 1.9, "medium"))
pool.register_strategy(Strategy("MacroShiftStrategy", "Uses global financial indicators to anticipate long-term trends.", "low", 1.4, "long"))
pool.register_strategy(Strategy("NewsImpactStrategy", "Analyzes breaking news for immediate market impact.", "medium", 1.6, "short"))
pool.register_strategy(Strategy("TokenRotationStrategy", "Detects capital rotation between tokens for swing trades.", "medium", 1.5, "short"))
pool.register_strategy(Strategy("LiquidityTrapStrategy", "Identifies low-liquidity setups prone to manipulation.", "high", 2.3, "short"))
pool.register_strategy(Strategy("RegulatorySentimentStrategy", "Tracks regulatory news and sentiment for risk management.", "low", 1.3, "medium"))

# -------------------- Risk Metrics --------------------

def calculate_max_drawdown(returns: List[float]) -> float:
    peak = returns[0] if returns else 0.0
    max_drawdown = 0.0
    for r in returns:
        if r > peak:
            peak = r
        drawdown = (peak - r) / peak if peak != 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

def calculate_volatility(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns)

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.01) -> float:
    downside_returns = [r for r in returns if r < risk_free_rate]
    if not downside_returns:
        return 0.0
    downside_std = statistics.stdev(downside_returns)
    avg_return = statistics.mean(returns)
    excess_return = avg_return - risk_free_rate
    return excess_return / downside_std if downside_std != 0 else 0.0

def enforce_capital_exposure_limit(current_exposure: float, max_exposure: float = 0.25) -> bool:
    return current_exposure <= max_exposure

# -------------------- Strategy Risk Evaluation --------------------

def evaluate_strategy_risk(strategy_name: str) -> Dict[str, float]:
    logs = load_strategy_logs()
    returns = [entry["score"] for entry in logs if entry["strategy"] == strategy_name]
    return {
        "volatility": calculate_volatility(returns),
        "max_drawdown": calculate_max_drawdown(returns),
        "sortino_ratio": calculate_sortino_ratio(returns)
    }

# -------------------- Strategy Selection --------------------

def get_all_strategies() -> List[str]:
    logs = load_strategy_logs()
    return list(set(entry["strategy"] for entry in logs))

def filter_strategies_by_risk(strategies: List[str], max_drawdown_threshold: float = 0.3, min_sortino: float = 1.0) -> List[str]:
    safe_strategies = []
    for strategy in strategies:
        risk_metrics = evaluate_strategy_risk(strategy)
        if risk_metrics["max_drawdown"] <= max_drawdown_threshold and risk_metrics["sortino_ratio"] >= min_sortino:
            safe_strategies.append(strategy)
    return safe_strategies

def rank_strategies_by_risk_adjusted_return(strategies: List[str]) -> List[str]:
    strategy_scores = {}
    for strategy in strategies:
        risk_metrics = evaluate_strategy_risk(strategy)
        score = risk_metrics["sortino_ratio"] - risk_metrics["max_drawdown"]
        strategy_scores[strategy] = score
    ranked = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked]

# -------------------- Main Selector --------------------

def select_strategies(max_drawdown_threshold: float = 0.3, min_sortino: float = 1.0, max_exposure: float = 0.25) -> List[str]:
    all_strategies = get_all_strategies()
    filtered = filter_strategies_by_risk(all_strategies, max_drawdown_threshold, min_sortino)
    ranked = rank_strategies_by_risk_adjusted_return(filtered)

    selected = []
    current_exposure = 0.0
    for strategy in ranked:
        if enforce_capital_exposure_limit(current_exposure + 0.05, max_exposure):
            selected.append(strategy)
            current_exposure += 0.05
        else:
            break
    return selected

# Example usage with mock market conditions
if __name__ == "__main__":
    selected = select_strategies()
    print("Selected strategies based on risk profile:")
    for s in selected:
        print(f" - {s}")

mock_market_conditions = {
    "volatility": 1.2,
    "sentiment_score": 0.7,
    "whale_activity": 0.9
}

best_strategy, score = pool.select_best_strategy(mock_market_conditions) 
