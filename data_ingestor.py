from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from datetime import datetime
import csv
from data_utils import cached_fetch_json, log_message, save_data_snapshot

# Load API key from environment
api_key = os.getenv("FREECRYPTOAPI_KEY")

# Configuration
API_URL = "https://freecryptoapi.com/api/v1/getHistory"
DATA_DIR = Path("ai_core/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_historical_data(symbol="BTC", currency="USD", start_date="2023-01-01", end_date="2023-12-31"):
    params = {
        "symbol": symbol,
        "currency": currency,
        "start": start_date,
        "end": end_date
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{API_URL}?{query_string}"
    data = cached_fetch_json(full_url)
    if data:
        log_message(f"Fetched historical data for {symbol}-{currency} from {start_date} to {end_date}")
    else:
        log_message(f"Failed to fetch historical data for {symbol}-{currency}", level='error')
    return data

def save_as_csv(data, filename):
    csv_path = DATA_DIR / f"{filename}.csv"
    if "data" in data and isinstance(data["data"], list) and data["data"]:
        keys = data["data"][0].keys()
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data["data"])
            log_message(f"Saved CSV to {csv_path}")
        except Exception as e:
            log_message(f"Failed to save CSV: {e}", level='error')
    else:
        log_message("No valid data to save as CSV.", level='warning')

if __name__ == "__main__":
    symbol = "BTC"
    currency = "USD"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    filename = f"{symbol}_{currency}_{start_date}_to_{end_date}"

    data = fetch_historical_data(symbol, currency, start_date, end_date)
    if data:
        save_data_snapshot(data, prefix=filename)
        save_as_csv(data, filename) 
