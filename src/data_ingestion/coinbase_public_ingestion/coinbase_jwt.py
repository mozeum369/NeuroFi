from coinbase import jwt_generator
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and secret from environment
api_key = os.getenv("COINBASE_API_KEY", "organizations/255f8425-8394-4110-a5f3-f8764046239d/apiKeys/b003937c-b133-4dc2-aff5-1adff8a5e40a")
api_secret = os.getenv("COINBASE_API_SECRET", """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIA86tCktmF/Dxs/50tGqVXsw6l1mfFGVzi5H/MSNZMIvoAoGCCqGSM49
AwEHoUQDQgAEypM9tOV3TJ+7pacwZacaZ4SAzMP6oQM6+EB8MOZFdzSC0t4fRxKx
hRtUE7CsNKEL1o+mf3gFSvKZuZ5ihHphMw==
-----END EC PRIVATE KEY-----""")

# Define request method and path
request_method = "GET"
request_path = "/api/v3/brokerage/accounts"

def main():
    # Format the JWT URI
    jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)
    # Build the JWT token
    jwt_token = jwt_generator.build_rest_jwt(jwt_uri, api_key, api_secret)
    # Print the token
    print(jwt_token)

if __name__ == "__main__":
    main()

