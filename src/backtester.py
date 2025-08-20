import numpy as np
import pandas as pd
from loguru import logger

class Backtester:
    def _mock_price_series(self, periods: int = 300, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rets = rng.normal(0, 0.01, periods)
        price = 100 * np.exp(np.cumsum(rets))
        return pd.DataFrame({"close": price})

    def run(self, symbol: str = "BTC-USD") -> pd.DataFrame:
        df = self._mock_price_series()
        df["sma_fast"] = df["close"].rolling(10).mean()
        df["sma_slow"] = df["close"].rolling(30).mean()
        df["signal"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        df["position"] = df["signal"].shift().fillna(0)
        df["rets"] = df["close"].pct_change().fillna(0)
        df["strategy"] = df["position"] * df["rets"]
        equity = (1 + df["strategy"]).cumprod()
        logger.info(f"Backtest complete for {symbol}: final equity x{equity.iloc[-1]:.2f}")
        df_out = df.reset_index().rename(columns={"index": "t"})
        df_out["symbol"] = symbol
        return df_out
