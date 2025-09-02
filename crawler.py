import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

# -------------------- Configuration --------------------

SCRAPE_DIR = Path("ai_core/scraped_data")
SCRAPE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NeuroFiBot/1.0; +https://neurofi.local)"
}

SOURCE_MAP = {
    "crypto_news": [
        "https://www.coindesk.com/",
        "https://cryptoslate.com/",
        "https://cointelegraph.com/"
    ],
    "reddit_crypto": [
        "https://www.reddit.com/r/CryptoCurrency/",
        "https://www.reddit.com/r/Bitcoin/",
        "https://www.reddit.com/r/Ethereum/"
    ],
    "github_trending": [
        "https://github.com/trending"
    ]
}


# -------------------- Core Functions --------------------

def accept_goal(goal_text: str) -> dict:
    """Accepts a goal and returns a metadata dictionary."""
    return {
        "goal": goal_text,
        "timestamp": datetime.utcnow().isoformat(),
        "sources": list(SOURCE_MAP.keys())
    }


def scrape_url(url: str) -> str:
    """Scrapes HTML content from a URL and returns cleaned text."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract visible text
        texts = soup.stripped_strings
        return "\n".join(texts)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return ""


def gather_data_for_goal(goal_text: str) -> dict:
    """Scrapes all relevant sources for the given goal and returns a dictionary of results."""
    goal_meta = accept_goal(goal_text)
    results = {}

    for source_key, urls in SOURCE_MAP.items():
        source_results = []
        for url in urls:
            print(f"[SCRAPE] {source_key}: {url}")
            content = scrape_url(url)
            if content:
                source_results.append({
                    "url": url,
                    "content": content[:10000]  # Limit to 10k chars per source
                })
        results[source_key] = source_results

    return {
        "goal": goal_text,
        "timestamp": goal_meta["timestamp"],
        "results": results
    }


def save_scraped_data(goal_text: str, data: dict):
    """Saves scraped data to a JSON file named after the goal."""
    import json
    safe_name = "".join(c if c.isalnum() else "_" for c in goal_text)[:50]
    filename = f"{safe_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
    out_path = SCRAPE_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVE] Scraped data saved to {out_path}")


# -------------------- CLI Entry --------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroFi Crawler")
    parser.add_argument("--goal", type=str, required=True, help="Goal to guide scraping")
    args = parser.parse_args()

    goal_text = args.goal
    scraped = gather_data_for_goal(goal_text)
    save_scraped_data(goal_text, scraped) 
