import os
import requests

# Load JWT from environment
jwt_token = os.getenv("JWT")

# Define the endpoint
url = "https://api.cdp.coinbase.com/platform/v2/evm/token-balances/base-sepolia/0x8fddcc0c5c993a1968b46787919cc34577d6dc5c"

# Set headers
headers = {
    "Authorization": f"Bearer {jwt_token}",
    "Content-Type": "application/json"
}

# Make the request
response = requests.get(url, headers=headers)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
