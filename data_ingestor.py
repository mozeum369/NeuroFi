# data_ingestor.py

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from datetime import datetime, timezone
import csv
import time

import pandas as pd
import numpy as np

from data_utils import cached_fetch_json, log_message, save_data_snapshot

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR = Path("ai_core/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COIN_LIST_URL = f"{COINGECKO_BASE}/coins/list?include_platform=false"
MARKET_CHART_RANGE_TMPL = (
    COINGECKO_BASE + "/coins/{coin_id}/market_chart/range?vs_currency={vs}&from={ts_from}&to={ts_to}"
)

# A small, fast-path resolver for common symbols you care about.
COMMON_SYMBOL_TO_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "PEPE": "pepe",
    "LRC": "loopring",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "XRP": "ripple",
    "BNB": "binancecoin",
    "AVAX": "avalanche-2",
    "MATIC": "polygon",
    "DOT": "polkadot",
    "ATOM": "cosmos",
    "OP": "optimism",
    "ARB": "arbitrum",
    "ZORA": "zora",  # verify if needed
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _to_unix_seconds(date_str: str) -> int:
    """
    Convert YYYY-MM-DD to a UTC Unix timestamp (seconds).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _sanitize_symbol(symbol: str) -> str:
    """
    Accepts inputs like 'BTC', 'btc', 'BTC-USD' (Coinbase style) and returns the base symbol.
    """
    s = symbol.strip()
    if "-" in s:
        s = s.split("-")[0]
    return s.upper()


def resolve_coingecko_id(symbol_or_id: str) -> str | None:
    """
    Resolve a trading symbol (e.g., 'BTC') or an already-correct CoinGecko id ('bitcoin')
    to a valid CoinGecko coin id.
    """
    if not symbol_or_id:
        return None

    # If user already passed a CoinGecko id, return it as-is
    # (a simple heuristic: ids are lowercase with hyphens and no spaces).
    if symbol_or_id.islower() and " " not in symbol_or_id:
        return symbol_or_id

    sym = _sanitize_symbol(symbol_or_id)

    # Fast map first
    if sym in COMMON_SYMBOL_TO_ID:
        return COMMON_SYMBOL_TO_ID[sym]

    # Fallback: query the /coins/list and match by symbol
    coins = cached_fetch_json(COIN_LIST_URL)
    if not coins or not isinstance(coins, list):
        log_message(f"Failed to fetch coin list from CoinGecko; cannot resolve '{symbol_or_id}'", level="error")
        return None

    # Prefer exact symbol match; CoinGecko symbols are lowercase.
    matches = [c for c in coins if c.get("symbol", "").lower() == sym.lower()]
    if not matches:
        # Try exact id match (user might have passed a valid id that wasn't lowercase)
        for c in coins:
            if c.get("id", "") == symbol_or_id:
                return symbol_or_id
        log_message(f"No CoinGecko match found for symbol/id '{symbol_or_id}'", level="error")
        return None

    coin_id = matches[0].get("id")
    if len(matches) > 1:
        log_message(
            f"Multiple matches for '{symbol_or_id}' on CoinGecko; choosing '{coin_id}'. "
            f"Consider passing the exact CoinGecko id for disambiguation.",
            level="warning",
        )
    return coin_id


def _build_market_chart_range_url(coin_id: str, vs_currency: str, start_date: str, end_date: str) -> str:
    vs = vs_currency.lower().strip()
    ts_from = _to_unix_seconds(start_date)
    ts_to = _to_unix_seconds(end_date)

    if ts_from >= ts_to:
        raise ValueError(f"start_date '{start_date}' must be before end_date '{end_date}'")

    # Clip 'to' to now (CoinGecko is fine with future but we keep it neat)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if ts_to > now_ts:
        ts_to = now_ts

    return MARKET_CHART_RANGE_TMPL.format(coin_id=coin_id, vs=vs, ts_from=ts_from, ts_to=ts_to)


def _retry_fetch_json(url: str, retries: int = 3, backoff: float = 1.5):
    """
    Thin retry wrapper around cached_fetch_json to handle transient 429/5xx.
    """
    for attempt in range(1, retries + 1):
        data = cached_fetch_json(url)
        if data:  # data_utils should log details; treat falsy as failure
            return data
        sleep_s = backoff ** attempt
        log_message(f"Retry {attempt}/{retries} after failure fetching: {url} (sleep {sleep_s:.1f}s)", level="warning")
        time.sleep(sleep_s)
    return None


# ------------------------------------------------------------------------------
# Main API: Historical data via market_chart/range
# ------------------------------------------------------------------------------
def fetch_historical_data(
    symbol: str = "BTC",
    currency: str = "USD",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
):
    """
    Fetch historical data from CoinGecko using market_chart/range.

    Returns the RAW CoinGecko response:
    {
      "prices": [[t_ms, price], ...],
      "market_caps": [[t_ms, cap], ...],
      "total_volumes": [[t_ms, volume], ...]
    }
    """
    coin_id = resolve_coingecko_id(symbol)
    if not coin_id:
        log_message(f"Could not resolve '{symbol}' to a CoinGecko id.", level="error")
        return None

    try:
        url = _build_market_chart_range_url(coin_id, currency, start_date, end_date)
    except Exception as e:
        log_message(f"Invalid dates for CoinGecko range: {e}", level="error")
        return None

    data = _retry_fetch_json(url)
    if not data or not isinstance(data, dict):
        log_message(f"Failed to fetch CoinGecko range for {symbol}-{currency} {start_date} → {end_date}", level="error")
        return None

    # Minimal sanity check
    if not any(k in data for k in ("prices", "market_caps", "total_volumes")):
        log_message(f"Unexpected CoinGecko response structure for {coin_id}", level="error")
        return None

    log_message(
        f"Fetched CoinGecko market_chart/range for {symbol} (id={coin_id}) vs {currency} "
        f"from {start_date} to {end_date}"
    )
    # Attach small metadata to help downstream
    data["_meta"] = {
        "source": "coingecko",
        "coin_id": coin_id,
        "symbol_input": symbol,
        "vs_currency": currency.lower(),
        "start_date": start_date,
        "end_date": end_date,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return data


def normalize_market_chart_to_rows(data: dict) -> list[dict]:
    """
    Convert CoinGecko market_chart/range response into a list of dict rows suitable for CSV.

    Output columns:
        timestamp_ms, timestamp_iso, price_usd, market_cap_usd, total_volume_usd
    """
    if not data:
        return []

    # Convert arrays into dicts keyed by timestamp for easy alignment.
    def to_map(pairs):
        m = {}
        if isinstance(pairs, list):
            for item in pairs:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    m[int(item[0])] = item[1]
        return m

    prices = to_map(data.get("prices", []))
    caps = to_map(data.get("market_caps", []))
    vols = to_map(data.get("total_volumes", []))

    # Union of all timestamps appearing in any series
    all_ts = sorted(set(prices.keys()) | set(caps.keys()) | set(vols.keys()))

    rows = []
    for ts_ms in all_ts:
        ts_iso = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
        rows.append(
            {
                "timestamp_ms": ts_ms,
                "timestamp_iso": ts_iso,
                "price_usd": prices.get(ts_ms),
                "market_cap_usd": caps.get(ts_ms),
                "total_volume_usd": vols.get(ts_ms),
            }
        )
    return rows


def save_as_csv(rows: list[dict], filename: str):
    """
    Save normalized rows to CSV under ai_core/data.
    """
    if not rows:
        log_message("No data rows to save as CSV.", level="warning")
        return

    csv_path = DATA_DIR / f"{filename}.csv"
    fieldnames = list(rows[0].keys())

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log_message(f"Saved CSV to {csv_path}")
    except Exception as e:
        log_message(f"Failed to save CSV: {e}", level="error")


# ------------------------------------------------------------------------------
# OHLC Aggregation Helpers
# ------------------------------------------------------------------------------

def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Convert normalized rows into a UTC-indexed DataFrame with columns: price, volume.
    Uses 'price_usd' and 'total_volume_usd' from normalize_market_chart_to_rows().
    """
    if not rows:
        return pd.DataFrame(columns=["price", "volume"])

    df = pd.DataFrame(rows)
    if "timestamp_ms" not in df.columns:
        return pd.DataFrame(columns=["price", "volume"])

    # Convert timestamp_ms -> UTC DateTimeIndex
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Standardize column names
    price_col = "price_usd" if "price_usd" in df.columns else "price"
    vol_col = "total_volume_usd" if "total_volume_usd" in df.columns else "volume"

    out = pd.DataFrame(index=df.index)
    out["price"] = df.get(price_col, pd.Series(index=df.index, dtype=float))
    out["volume"] = df.get(vol_col, pd.Series(index=df.index, dtype=float))

    # De-duplicate any overlapping timestamps
    out = out[~out.index.duplicated(keep="last")]
    return out


def aggregate_to_ohlc(price_df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
    """
    Aggregate irregular price (and volume) into OHLCV at the chosen frequency.
    - freq: '1H' (default) or '1D'
    Returns: DataFrame with columns [open, high, low, close, volume, ret, log_ret]
    """
    if price_df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "ret", "log_ret"])

    # Ensure UTC DateTimeIndex
    idx = price_df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Input must be indexed by a pandas DatetimeIndex (UTC).")
    if idx.tz is None:
        price_df = price_df.tz_localize("UTC")
    else:
        price_df = price_df.tz_convert("UTC")

    # OHLC with right-closed/right-labeled intervals
    ohlc = price_df["price"].resample(freq, label="right", closed="right").ohlc()

    # Volume per bucket
    if "volume" in price_df.columns:
        vol = price_df["volume"].resample(freq, label="right", closed="right").sum(min_count=1)
        ohlc["volume"] = vol
    else:
        ohlc["volume"] = np.nan

    # Returns
    ohlc["ret"] = ohlc["close"].pct_change()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ohlc["close"] / ohlc["close"].shift(1)
        ohlc["log_ret"] = np.log(ratio.replace({0: np.nan}))

    # Drop bars where OHLC are all NaN
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"], how="all")
    return ohlc


def save_ohlc_csv(ohlc: pd.DataFrame, filename_base: str, freq: str = "1H"):
    """
    Save OHLC DataFrame to ai_core/data/ohlc/{filename_base}_ohlc_{freq}.csv
    """
    out_dir = DATA_DIR / "ohlc"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = ohlc.copy()
    out = out.sort_index()
    out.index = out.index.tz_convert("UTC")
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    csv_path = out_dir / f"{filename_base}_ohlc_{freq.upper()}.csv"
    out.to_csv(csv_path, index=False)
    log_message(f"Saved OHLC CSV to {csv_path}")


# ------------------------------------------------------------------------------
# CLI / Script Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoinGecko market_chart/range fetch + OHLC aggregation")
    parser.add_argument("--symbol", default="BTC", help="Symbol or CoinGecko id (e.g., BTC or bitcoin)")
    parser.add_argument("--currency", default="USD", help="Quote currency (default: USD)")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--freq", default="1H", help="Resample frequency: 1H or 1D (default: 1H)")
    parser.add_argument("--skip-normalized", action="store_true", help="Skip saving normalized CSV")
    parser.add_argument("--skip-ohlc", action="store_true", help="Skip saving OHLC CSV")
    args = parser.parse_args()

    symbol = args.symbol
    currency = args.currency
    start_date = args.start
    end_date = args.end
    freq = args.freq.upper()

    filename_base = f"{symbol}_{currency}_{start_date}_to_{end_date}".replace("-", "")

    raw = fetch_historical_data(symbol, currency, start_date, end_date)
    if raw:
        # JSON snapshot
        save_data_snapshot(raw, prefix=filename_base)

        # CSV (normalized)
        rows = normalize_market_chart_to_rows(raw)
        if rows and not args.skip_normalized:
            save_as_csv(rows, filename_base)

        # OHLC aggregation
        if rows and not args.skip_ohlc:
            df = rows_to_dataframe(rows)
            ohlc = aggregate_to_ohlc(df, freq=freq)
            if not ohlc.empty:
                save_ohlc_csv(ohlc, filename_base, freq=freq)
            else:
                log_message("OHLC aggregation produced no rows (check date range and symbol).", level="warning")
    else:
        log_message("No raw data returned; nothing to save.", level="error")
 
