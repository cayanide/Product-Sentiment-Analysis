# sentiment_analysis.py
from textblob import TextBlob

def analyze_sentiment(review):
    blob = TextBlob(review)
    sentiment = blob.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)
    return sentiment
