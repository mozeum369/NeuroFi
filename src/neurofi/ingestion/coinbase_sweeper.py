import requests
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

class CoinbaseSweeper:
    BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(self):
        self.session = requests.Session()
        self.products = []

    def fetch_all_products(self) -> List[Dict[str, Any]]:
        """Fetch all tradable products from Coinbase."""
        url = f"{self.BASE_URL}/products"
        response = self.session.get(url)
        if response.status_code == 200:
            self.products = response.json()
            return self.products
        else:
            print(f"Failed to fetch products: {response.status_code}")
            return []

    def fetch_historical_data(self, product_id: str, granularity: int = 3600) -> List[List[float]]:
        """Fetch historical OHLCV data for a product."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        url = f"{self.BASE_URL}/products/{product_id}/candles"
        params = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "granularity": granularity
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch historical data for {product_id}: {response.status_code}")
            return []

    def fetch_ticker(self, product_id: str) -> Dict[str, Any]:
        """Fetch real-time ticker data for a product."""
        url = f"{self.BASE_URL}/products/{product_id}/ticker"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch ticker for {product_id}: {response.status_code}")
            return {}

    def fetch_order_book(self, product_id: str, level: int = 2) -> Dict[str, Any]:
        """Fetch order book depth for a product."""
        url = f"{self.BASE_URL}/products/{product_id}/book"
        params = {"level": level}
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch order book for {product_id}: {response.status_code}")
            return {}

    def sweep(self) -> Dict[str, Dict[str, Any]]:
        """Sweep all products and collect relevant data."""
        data = {}
        products = self.fetch_all_products()
        for product in products:
            product_id = product.get("id")
            if not product_id:
                continue
            print(f"Sweeping data for {product_id}")
            data[product_id] = {
                "historical": self.fetch_historical_data(product_id),
                "ticker": self.fetch_ticker(product_id),
                "order_book": self.fetch_order_book(product_id)
            }
            time.sleep(0.2)  # Rate limiting
        return data

if __name__ == "__main__":
    sweeper = CoinbaseSweeper()
    all_data = sweeper.sweep()

    # Save to file for inspection
    with open("coinbase_sweep.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print("Sweep complete. Data saved to coinbase_sweep.json.")

