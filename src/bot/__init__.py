# Updated src/bot/__init__.py with more sentiment-diverse headlines

import os
import re
import random
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Simulated function to fetch headlines (expanded for sentiment diversity)
def fetch_headlines():
    return [
        "Pepe token surges 20% overnight",
        "Spell token crashes after exploit",
        "Zora announces new NFT marketplace",
        "Pepe faces regulatory scrutiny",
        "Spell gains traction among DeFi users",
        "Zora token remains stable",
        "Pepe hits all-time high",
        "Spell token underperforms market",
        "Zora partners with major exchange",
        "Pepe token plummets after whale sell-off",
        "Spell token praised for innovation",
        "Zora criticized for lack of transparency",
        "Pepe community celebrates major milestone",
        "Spell token delisted from major exchange",
        "Zora token gains 15% in a day",
        "Pepe token faces backlash from investors",
        "Spell token receives positive analyst review",
        "Zora token stagnates amid market uncertainty"
    ]

# Token list to monitor
token_list = ["Pepe", "Spell", "Zora"]

# Sentiment analysis functions
def extract_token_mentions(headlines, token_list):
    mentions = {token.lower(): 0 for token in token_list}
    for headline in headlines:
        text = headline.lower()
        for token in token_list:
            if re.search(rf"\b{re.escape(token.lower())}\b", text):
                mentions[token.lower()] += 1
    return mentions

def analyze_sentiment(headlines):
    results = []
    for headline in headlines:
        blob = TextBlob(headline)
        results.append((headline, blob.sentiment.polarity, blob.sentiment.subjectivity))
    return results

# Convert polarity to sentiment class
def label_sentiment(polarity):
    if polarity > 0.2:
        return 1  # Positive
    elif polarity < -0.2:
        return -1  # Negative
    else:
        return 0  # Neutral

# Train model using sentiment labels
def train_model(headlines):
    sentiment_data = analyze_sentiment(headlines)
    X = []
    y = []

    for headline, polarity, subjectivity in sentiment_data:
        features = [polarity, subjectivity]
        label = label_sentiment(polarity)
        X.append(features)
        y.append(label)

    # Ensure at least two classes are present
    unique_classes = set(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Training data must contain at least two classes. Found: {unique_classes}")

    X = np.array(X)
    y = np.array(y)

    model = LogisticRegression()
    model.fit(X, y)
    return model

# Main bot runner
def run_bot():
    headlines = fetch_headlines()
    mentions = extract_token_mentions(headlines, token_list)
    print("Token Mentions:", mentions)

    try:
        model = train_model(headlines)
        print("✅ Model trained successfully.")
    except ValueError as e:
        print("❌ Model training failed:", e)

# Entry point
def main():
    run_bot()

if __name__ == "__main__":
    main()

