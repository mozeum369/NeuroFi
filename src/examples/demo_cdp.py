# src/examples/demo_cdp.py
import os
from neurofi.services import CdpService

# Ensure these are in your environment or .env sourced into the shell:
#   KEY_NAME, KEY_SECRET
# Host defaults to "api.cdp.coinbase.com"
# Use set -a && source .env && set +a before running if needed.

def main():
    svc = CdpService()
    data = svc.list_evm_token_balances(
        network="base-sepolia",
        address="0x8fddcc0c5c993a1968b46787919cc34577d6dc5c",
    )
    print(data)

if __name__ == "__main__":
    main()
