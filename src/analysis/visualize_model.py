import os
import json
import matplotlib.pyplot as plt
from textblob import TextBlob
import requests
from datetime import datetime

# Load model performance data from bot_log.json
log_file = 'bot_log.json'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        content = f.read().strip()
        if content:
            log_data = json.loads(content)
        else:
            log_data = {"predictions": [], "accuracy": 0.0, "total": 0, "correct": 0}
else:
    log_data = {"predictions": [], "accuracy": 0.0, "total": 0, "correct": 0}

# Visualize model performance
def visualize_model_performance(log_data):
    total = log_data.get("total", 0)
    correct = log_data.get("correct", 0)
    accuracy = log_data.get("accuracy", 0.0)

    plt.figure(figsize=(6, 4))
    plt.bar(["Correct", "Incorrect"], [correct, total - correct], color=["green", "red"])
    plt.title(f"Model Accuracy: {accuracy:.2%}")
    plt.ylabel("Predictions")
    plt.tight_layout()
    plt.savefig("model_performance.png")
    plt.show()

visualize_model_performance(log_data)

# Real-time sentiment scraping from news and social media (mocked with Bing News API)
TOKEN_IDS = {
    "pepe": "pepe",
    "zora": "zora",
    "spell": "spell-token",
    "lrc": "loopring",
    "townes": "townes"
}

def fetch_sentiment_for_token(token_name):
    # Simulate fetching news headlines (replace with actual API if available)
    query = f"{token_name} crypto"
    url = f"https://api.bing.com/news/search?q={query}&count=5"  # Placeholder URL
    headers = {"Ocp-Apim-Subscription-Key": "YOUR_API_KEY"}  # Replace with actual key

    # Since we can't access external APIs, simulate with sample headlines
    sample_headlines = [
        f"{token_name} surges in market",
        f"Investors cautious about {token_name}",
        f"{token_name} shows strong potential",
        f"Concerns rise over {token_name} volatility",
        f"{token_name} gains traction among traders"
    ]

    sentiments = []
    for headline in sample_headlines:
        blob = TextBlob(headline)
        sentiments.append(blob.sentiment.polarity)

    avg_sentiment = sum(sentiments) / len(sentiments)
    return avg_sentiment

# Collect and display sentiment for each token
sentiment_data = {}
for token in TOKEN_IDS:
    sentiment = fetch_sentiment_for_token(token)
    sentiment_data[token] = sentiment
    print(f"Sentiment for {token}: {sentiment:.2f}")

# Save sentiment data to file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f"sentiment_snapshot_{timestamp}.json", "w") as f:
    json.dump(sentiment_data, f, indent=2)

