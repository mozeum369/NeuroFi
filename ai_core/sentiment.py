import requests
import re
from textblob import TextBlob

def fetch_bing_headlines(query="crypto", count=10):
    """
    Fetch recent crypto-related headlines using Bing News Search API.
    Returns a list of headline strings.
    """
    api_key = "YOUR_BING_API_KEY"  # Replace with your actual Bing API key
    endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": count,
        "mkt": "en-US",
        "safeSearch": "Moderate"
    }

    response = requests.get(endpoint, headers=headers, params=params)
    headlines = []

    if response.status_code == 200:
        data = response.json()
        for article in data.get("value", []):
            headlines.append(article.get("name", ""))
    else:
        print(f"Error fetching headlines: {response.status_code} - {response.text}")

    return headlines

def extract_token_mentions(headlines, token_list):
    """
    Extract mentions of specific tokens from a list of headlines.
    Returns a dictionary with token names and count of mentions.
    """
    mentions = {token.lower(): 0 for token in token_list}
    for headline in headlines:
        text = headline.lower()
        for token in token_list:
            if re.search(rf"\b{re.escape(token.lower())}\b", text):
                mentions[token.lower()] += 1
    return mentions

def analyze_sentiment(headlines):
    """
    Perform basic sentiment analysis on a list of headlines.
    Returns a list of tuples (headline, polarity, subjectivity).
    """
    results = []
    for headline in headlines:
        blob = TextBlob(headline)
        results.append((headline, blob.sentiment.polarity, blob.sentiment.subjectivity))
    return results

# Example usage
if __name__ == "__main__":
    headlines = fetch_bing_headlines()
    tokens = ["Bitcoin", "Ethereum", "Pepe", "Zora"]
    sentiment_results = analyze_sentiment(headlines)
    token_mentions = extract_token_mentions(headlines, tokens)

    print("Sentiment Analysis:")
    for result in sentiment_results:
        print(result)

    print("\nToken Mentions:")
    print(token_mentions)

