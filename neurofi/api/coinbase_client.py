import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Coinbase API credentials from environment variables
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
COINBASE_API_PASSPHRASE = os.getenv("COINBASE_API_PASSPHRASE")

def get_account_info():
    headers = {
        "CB-ACCESS-KEY": COINBASE_API_KEY,
        "CB-ACCESS-PASSPHRASE": COINBASE_API_PASSPHRASE,
        # Signature and timestamp logic should be added here for authenticated requests
    }
    response = requests.get("https://api.coinbase.com/v2/accounts", headers=headers)
    return response.json()


if __name__ == "__main__":
    print(get_account_info())
