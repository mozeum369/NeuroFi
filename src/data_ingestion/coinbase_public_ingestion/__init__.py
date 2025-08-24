import requests

class CoinbasePublicAPI:
    BASE_URL = "https://api.coinbase.com/api/v3"

    def __init__(self, bypass_cache=False):
        self.session = requests.Session()
        if bypass_cache:
            self.session.headers.update({"Cache-Control": "no-cache"})

    def get_server_time(self):
        url = f"{self.BASE_URL}/time"
        response = self.session.get(url)
        return response.json()

    def list_public_products(self):
        url = f"{self.BASE_URL}/market/products"
        response = self.session.get(url)
        return response.json()

    def get_product_details(self, product_id):
        url = f"{self.BASE_URL}/market/products/{product_id}"
        response = self.session.get(url)
        return response.json()

    def get_product_book(self, product_id):
        url = f"{self.BASE_URL}/market/product_book?product_id={product_id}"
        response = self.session.get(url)
        return response.json()

    def get_product_candles(self, product_id, granularity="1h"):
        url = f"{self.BASE_URL}/market/products/{product_id}/candles?granularity={granularity}"
        response = self.session.get(url)
        return response.json()

    def get_product_ticker(self, product_id):
        url = f"{self.BASE_URL}/market/products/{product_id}/ticker"
        response = self.session.get(url)
        return response.json()

# Example usage
if __name__ == "__main__":
    coinbase_api = CoinbasePublicAPI(bypass_cache=True)

    # Fetch server time
    print("Server Time:", coinbase_api.get_server_time())

    # List all products
    products = coinbase_api.list_public_products()
    print("Available Products:", products)

    # Example product ID
    if products:
        product_id = products[0]['product_id']
        print(f"\nDetails for Product {product_id}:")
        print("Product Info:", coinbase_api.get_product_details(product_id))
        print("Order Book:", coinbase_api.get_product_book(product_id))
        print("Candles:", coinbase_api.get_product_candles(product_id))
        print("Ticker:", coinbase_api.get_product_ticker(product_id))


