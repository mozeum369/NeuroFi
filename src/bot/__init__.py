import json
import time
import requests
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define tokens and their CoinGecko IDs
TOKEN_IDS = {
    "pepe": "pepe",
    "zora": "zora",
    "spell": "spell-token",
    "lrc": "loopring",
    "townes": "townes"  # Assuming 'townes' is a valid token ID; may need adjustment
}

# Load historical log data
def load_log_data(log_file='bot_log.json'):
    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"predictions": [], "accuracy": 0.0, "total": 0, "correct": 0}

# Save log data
def save_log_data(data, log_file='bot_log.json'):
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)

# Fetch current prices from CoinGecko
def fetch_current_prices(token_ids):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(token_ids.values()),
        "vs_currencies": "usd"
    }
    response = requests.get(url, params=params)
    return response.json()

# Train a simple model from historical data
def train_model(log_data):
    X, y = [], []
    for entry in log_data["predictions"]:
        X.append([entry["polarity"], entry["subjectivity"]])
        y.append(1 if entry["outcome"] else 0)
    if len(X) >= 5:
        model = LogisticRegression()
        model.fit(X, y)
        return model
    return None

# Run bot with real price tracking
def run_bot():
    log_data = load_log_data()
    model = train_model(log_data)
    previous_prices = fetch_current_prices(TOKEN_IDS)
    time.sleep(5)  # Simulate delay before checking price movement
    current_prices = fetch_current_prices(TOKEN_IDS)

    # Simulate headline analysis
    headlines = [
        "Pepe surges after meme revival",
        "Zora announces new NFT marketplace",
        "Spell token sees unusual volume spike",
        "Loopring integrates with major exchange",
        "Townes gains traction in DeFi circles"
    ]

    for headline in headlines:
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        mentions = [token for token in TOKEN_IDS if token.lower() in headline.lower()]

        # Predict outcome using model
        if model:
            prediction = model.predict([[polarity, subjectivity]])[0]
            action = "BUY" if prediction == 1 else "HOLD"
        else:
            action = "BUY" if polarity > 0.1 else "HOLD"

        # Evaluate outcome based on price movement
        outcome = False
        for token in mentions:
            token_id = TOKEN_IDS[token]
            if token_id in previous_prices and token_id in current_prices:
                old_price = previous_prices[token_id]["usd"]
                new_price = current_prices[token_id]["usd"]
                if new_price > old_price:
                    outcome = True
                    break

        correct = (action == "BUY" and outcome) or (action == "HOLD" and not outcome)

        # Log prediction
        log_entry = {
            "headline": headline,
            "mentions": mentions,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "prediction": action,
            "outcome": outcome,
            "correct": correct
        }
        log_data["predictions"].append(log_entry)
        log_data["total"] += 1
        log_data["correct"] += int(correct)
        log_data["accuracy"] = round(log_data["correct"] / log_data["total"], 2)

        print(f"{action}: {headline} | Outcome: {'✅' if outcome else '❌'} | Correct: {correct}")

    save_log_data(log_data) 
