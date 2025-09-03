import os
import re
import json
from textblob import TextBlob
from pathlib import Path
from bs4 import BeautifulSoup
import requests

# Directory to save scraped data
SCRAPED_DIR = Path("ai_core/scraped_data")
SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

# Local NLP sentiment analysis
def analyze_sentiment(text_list):
    results = []
    for text in text_list:
        blob = TextBlob(text)
        results.append({
            "text": text,
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        })
    return results

# Token mention detection
def extract_token_mentions(text_list, token_list):
    mentions = {token.lower(): 0 for token in token_list}
    for text in text_list:
        lower_text = text.lower()
        for token in token_list:
            if re.search(rf"\b{re.escape(token.lower())}\b", lower_text):
                mentions[token.lower()] += 1
    return mentions

# Scrape visible text from a URL
def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        texts = soup.stripped_strings
        return " ".join(texts)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Accept a goal and determine sources
def accept_goal(goal_text):
    return {
        "goal": goal_text,
        "sources": {
            "news": [
                "https://www.coindesk.com/",
                "https://cryptoslate.com/",
                "https://cointelegraph.com/"
            ],
            "reddit": [
                "https://www.reddit.com/r/CryptoCurrency/",
                "https://www.reddit.com/r/Bitcoin/",
                "https://www.reddit.com/r/Ethereum/"
            ],
            "github": [
                "https://github.com/trending"
            ]
        }
    }

# Gather data for a goal
def gather_data_for_goal(goal_text, token_list=None):
    goal_info = accept_goal(goal_text)
    all_texts = []
    for category, urls in goal_info["sources"].items():
        for url in urls:
            print(f"Scraping {url}...")
            text = scrape_url(url)
            if text:
                all_texts.append(text)

    sentiment_results = analyze_sentiment(all_texts)
    token_mentions = extract_token_mentions(all_texts, token_list or ["Bitcoin", "Ethereum", "Pepe", "Zora"])

    output = {
        "goal": goal_text,
        "content": all_texts,
        "sentiment": sentiment_results,
        "token_mentions": token_mentions
    }

    # Save to file
    out_path = SCRAPED_DIR / f"{goal_text.replace(' ', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuroFi Crawler")
    parser.add_argument("--goal", type=str, required=True, help="Goal to guide scraping")
    args = parser.parse_args()

    gather_data_for_goal(args.goal) 
