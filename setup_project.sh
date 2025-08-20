#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating Cryptobot project structure with boilerplate..."

mkdir -p src data logs tests data/backtests

# .gitignore
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
.DS_Store
logs/*.log
EOF

# README.md
cat > README.md <<'EOF'
# Cryptobot
A modular crypto research & analysis bot with:
- Async data fetching (CoinGecko snapshot for now)
- Simple strategy engine (SMA crossover placeholder)
- Backtester (mocked prices to validate flow)
- Dash web dashboard for visualization
EOF

# requirements.txt
cat > requirements.txt <<'EOF'
pandas>=2.2
numpy>=1.26
aiohttp>=3.9
requests>=2.31
beautifulsoup4>=4.12
lxml>=5.2
dash>=2.17
plotly>=5.22
scikit-learn>=1.5
python-dotenv>=1.0
loguru>=0.7
pytest>=8.2
EOF

# config.py
cat > config.py <<'PY'
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8050"))
    symbols: tuple[str, ...] = tuple(os.getenv("SYMBOLS", "BTC,ETH,PEPE,LRC,ZORA,SPELL,TOWN").split(","))

config = Config()
PY

# main.py
cat > main.py <<'PY'
import argparse
import asyncio
import json
from pathlib import Path
from loguru import logger
from config import config
from src.utils import setup_logging
from src.data_fetcher import fetch_top_markets
from src.backtester import Backtester
from src.dashboard import run_dashboard

def cmd_fetch(args):
    async def _run():
        markets = await fetch_top_markets(vs_currency="usd", per_page=args.per_page)
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        out = Path(config.data_dir) / "markets.json"
        out.write_text(json.dumps(markets, indent=2))
        logger.info(f"Wrote {len(markets)} markets to {out}")
    asyncio.run(_run())

def cmd_backtest(args):
    bt = Backtester()
    report = bt.run(symbol=args.symbol)
    out_dir = Path("data/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"backtest_{args.symbol.replace('/', '-')}.csv"
    report.to_csv(out, index=False)
    logger.info(f"Backtest saved to {out}")

def cmd_dashboard(_args):
    run_dashboard()

def main():
    setup_logging()
    parser = argparse.ArgumentParser(prog="cryptobot", description="Cryptobot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_fetch = subparsers.add_parser("fetch", help="Fetch top market snapshot")
    p_fetch.add_argument("--per-page", type=int, default=50)
    p_fetch.set_defaults(func=cmd_fetch)

    p_bt = subparsers.add_parser("backtest", help="Run SMA backtest")
    p_bt.add_argument("--symbol", type=str, default="BTC-USD")
    p_bt.set_defaults(func=cmd_backtest)

    p_dash = subparsers.add_parser("dashboard", help="Launch dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
PY

# src files
mkdir -p src
echo '"""Core package for Cryptobot."""' > src/__init__.py

cat > src/utils.py <<'PY'
from pathlib import Path
from loguru import logger
from config import config

def setup_logging():
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(Path(config.logs_dir) / "runtime.log", rotation="1 MB", retention=5, level="INFO", enqueue=True)
    logger.add(lambda m: print(m, end=""))
    logger.info("Logging initialized.")
    return logger
PY

cat > src/data_fetcher.py <<'PY'
import aiohttp
from loguru import logger

async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None):
    async with session.get(url, params=params, timeout=30) as resp:
        resp.raise_for_status()
        return await resp.json()

async def fetch_top_markets(vs_currency: str = "usd", per_page: int = 50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": vs_currency, "order": "market_cap_desc", "per_page": per_page, "page": 1, "sparkline": "false"}
    headers = {"Accept": "application/json"}
    async with aiohttp.ClientSession(headers=headers) as session:
        data = await _fetch_json(session, url, params)
        logger.info(f"Fetched {len(data)} markets from CoinGecko.")
        return data
PY

cat > src/backtester.py <<'PY'
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
PY

cat > src/dashboard.py <<'PY'
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from config import config

def _latest_backtest_csv():
    p = Path("data/backtests")
    files = sorted(p.glob("backtest_*.csv"))
    return files[-1] if files else None

def _load_backtest_df():
    latest = _latest_backtest_csv()
    if latest and latest.exists():
        return pd.read_csv(latest)
    return pd.DataFrame({"t": list(range(100)), "close": [i + 1 for i in range(100)], "symbol": "DEMO"})

def _make_figure(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["close"], mode="lines", name="Close"))
    fig.update_layout(title="Backtest Prices", template="plotly_white", height=500)
    return fig

def run_dashboard():
    app = Dash(__name__)
    app.title = "Cryptobot Dashboard"
    app.layout = html.Div([
        html.H2("Cryptobot Dashboard"),
        dcc.Graph(id="price-graph"),
        dcc.Interval(id="tick", interval=5000, n_intervals=0),
    ])
    @app.callback(Output("price-graph", "figure"), Input("tick", "n_intervals"))
    def _update_graph(_n):
        return _make_figure(_load_backtest_df())
    app.run_server(host=config.dashboard_host, port=config.dashboard_port, debug=True)
PY

echo "==> Boilerplate created."
echo "NEXT STEPS:"
echo "1) python3 -m venv .venv && source .venv/bin/activate"
echo "2) pip install -U pip && pip install -r requirements.txt"
echo "3) python main.py fetch"
echo "4) python main.py backtest --symbol BTC-USD"
echo "5) python main.py dashboard"
