# ingestion.py

import requests
import pandas as pd
from datetime import datetime

# Function to fetch market data from Coinbase API
def fetch_coinbase_price(symbol="BTC-USD"):
    url = f"https://api.coinbase.com/v2/prices/{symbol}/spot"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        price = float(data['data']['amount'])
        timestamp = datetime.utcnow()
        return {"symbol": symbol, "price": price, "timestamp": timestamp}
    else:
        raise Exception(f"Failed to fetch data from Coinbase: {response.status_code}")

# Function to fetch market data from CoinGecko API
def fetch_coingecko_price(id="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={id}&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        price = float(data[id]['usd'])
        timestamp = datetime.utcnow()
        return {"id": id, "price": price, "timestamp": timestamp}
    else:
        raise Exception(f"Failed to fetch data from CoinGecko: {response.status_code}")

# Function to clean and normalize data
def clean_data(data):
    df = pd.DataFrame([data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Function to store data to CSV
def store_data(df, filename="market_data.csv"):
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)

# Example usage
if __name__ == "__main__":
    coinbase_data = fetch_coinbase_price()
    coingecko_data = fetch_coingecko_price()

    df_coinbase = clean_data(coinbase_data)
    df_coingecko = clean_data(coingecko_data)

    store_data(df_coinbase, "coinbase_data.csv")
    store_data(df_coingecko, "coingecko_data.csv")

    print("Data ingestion completed and stored.")

