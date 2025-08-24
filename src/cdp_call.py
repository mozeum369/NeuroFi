#!/usr/bin/env python3
import os, sys, time, json
import requests
from cdp.auth.utils.jwt import generate_jwt, JwtOptions

# ---- Config you’re calling ----
NETWORK = "base-sepolia"
ADDRESS = "0x8fddcc0c5c993a1968b46787919cc34577d6dc5c"

METHOD = "GET"
HOST   = "api.cdp.coinbase.com"
PATH   = f"/platform/v2/data/evm/token-balances/{NETWORK}/{ADDRESS}"  # NOTE: includes /data

# ---- Read credentials from environment ----
KEY_NAME   = os.getenv("KEY_NAME")
KEY_SECRET = os.getenv("KEY_SECRET")

missing = [k for k in ("KEY_NAME", "KEY_SECRET") if os.getenv(k) in (None, "")]
if missing:
    sys.stderr.write(f"Missing env vars: {', '.join(missing)}\n")
    sys.exit(1)

# ---- Generate short‑lived JWT for THIS exact request ----
jwt_token = generate_jwt(JwtOptions(
    api_key_id=KEY_NAME,
    api_key_secret=KEY_SECRET,
    request_method=METHOD,
    request_host=HOST,
    request_path=PATH,
    expires_in=120  # default; keep short
))

url = f"https://{HOST}{PATH}"
headers = {"Authorization": f"Bearer {jwt_token}"}

# ---- Call the API ----
resp = requests.get(url, headers=headers, timeout=30)

print(f"HTTP {resp.status_code}")
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
