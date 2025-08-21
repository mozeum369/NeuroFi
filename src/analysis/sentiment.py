import re
from textblob import TextBlob

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
