import requests
import json
import csv
from datetime import datetime
from pathlib import Path

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
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

def save_as_json(data, filename):
    json_path = DATA_DIR / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON to {json_path}")

def save_as_csv(data, filename):
    csv_path = DATA_DIR / f"{filename}.csv"
    if "data" in data and isinstance(data["data"], list):
        keys = data["data"][0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data["data"])
        print(f"Saved CSV to {csv_path}")
    else:
        print("No valid data to save as CSV.")

if __name__ == "__main__":
    symbol = "BTC"
    currency = "USD"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    filename = f"{symbol}_{currency}_{start_date}_to_{end_date}"

    data = fetch_historical_data(symbol, currency, start_date, end_date)
    if data:
        save_as_json(data, filename)
        save_as_csv(data, filename)
