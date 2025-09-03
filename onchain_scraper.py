import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("ai_core/onchain_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Utility function to clean and extract text
def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

# Scraper for Etherscan token transfers
def scrape_etherscan_token_transfers(address):
    url = f"https://etherscan.io/token/generic-tokenholders2?a={address}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch Etherscan data for {address}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    transfers = []

    if table:
        rows = table.find_all("tr")[1:]  # Skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                transfers.append({
                    "rank": cols[0].text.strip(),
                    "address": cols[1].text.strip(),
                    "quantity": cols[2].text.strip(),
                    "percentage": cols[3].text.strip()
                })
    return transfers

# Scraper for Solscan recent transactions
def scrape_solscan_transactions(address):
    url = f"https://solscan.io/account/{address}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch Solscan data for {address}")
        return []

    text = extract_text_from_html(response.text)
    tx_matches = re.findall(r'Transaction Signature: (\w+)', text)
    return [{"signature": sig} for sig in tx_matches]

# Scraper for Arbiscan contract interactions
def scrape_arbiscan_contracts(address):
    url = f"https://arbiscan.io/address/{address}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch Arbiscan data for {address}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    interactions = []

    tx_table = soup.find("table", {"id": "transactions"})
    if tx_table:
        rows = tx_table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                interactions.append({
                    "hash": cols[0].text.strip(),
                    "method": cols[1].text.strip(),
                    "from": cols[2].text.strip(),
                    "to": cols[3].text.strip(),
                    "value": cols[4].text.strip()
                })
    return interactions

# Save structured data
def save_onchain_data(address, data, source):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{source}_{address}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {source} data for {address} to {filename}")

# Main function to run all scrapers
def run_onchain_scraper(addresses):
    for address in addresses:
        eth_data = scrape_etherscan_token_transfers(address)
        save_onchain_data(address, eth_data, "etherscan")

        sol_data = scrape_solscan_transactions(address)
        save_onchain_data(address, sol_data, "solscan")

        arb_data = scrape_arbiscan_contracts(address)
        save_onchain_data(address, arb_data, "arbiscan")

# Example usage
if __name__ == "__main__":
    test_addresses = [
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Example ETH whale
        "7zN3Yz7ZzZzZzZzZzZzZzZzZzZzZzZzZzZzZzZzZzZ",  # Example Solana address
        "0x0000000000000000000000000000000000000000"   # Example Arbitrum address
    ]
    run_onchain_scraper(test_addresses) 
