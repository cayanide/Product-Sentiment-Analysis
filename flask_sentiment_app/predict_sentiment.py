import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
from collections import Counter
import requests

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load trained model and vectorizer
NB_model = joblib.load('NB_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in word_tokenize(text) if word not in stop_words])

# Predict sentiment
def predict_sentiment(reviews):
    if not reviews:
        print("No reviews to analyze.")
        return []

    processed_reviews = [preprocess_text(review) for review in reviews]
    reviews_TFIDF = vectorizer.transform(processed_reviews)
    predictions = NB_model.predict(reviews_TFIDF)
    return predictions

# Generate AI-based summary
def generate_ai_review(reviews, sentiments, overall_rating):
    combined_reviews = "\n".join(reviews)
    prompt = (
        f"The product has received the following reviews:\n{combined_reviews}\n\n"
        f"The overall sentiment is '{overall_rating}'. Write a detailed summary highlighting common praises and complaints."
    )
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [
                    {"role": "system", "content": "Summarize customer reviews."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 150,
            },
        )
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"AI Summary Generation Failed: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error generating summary: {e}"

# Generate summary and evaluate AI-generated paragraph
def generate_summary(reviews, sentiments):
    sentiment_count = Counter(sentiments)
    total_reviews = len(sentiments)

    # Sentiment counts
    positive_count = sentiment_count.get('positive', 0)
    neutral_count = sentiment_count.get('neutral', 0)
    negative_count = sentiment_count.get('negative', 0)

    # Overall sentiment
    if positive_count > negative_count and positive_count > neutral_count:
        overall_rating = "Positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        overall_rating = "Negative"
    else:
        overall_rating = "Neutral"

    # Call OpenAI API to generate AI-based review
    ai_review = generate_ai_review(reviews, sentiments, overall_rating)

    return ai_review, overall_rating

# Example Usage
if __name__ == "__main__":
    # Example reviews
    example_reviews = [
        "This product is amazing! I love the quality and durability.",
        "It's okay, not the best but does the job.",
        "Terrible experience. It broke within a week of use."
    ]

    # Predict sentiments for reviews
    sentiments = predict_sentiment(example_reviews)

    # Generate summary and re-evaluate AI-generated paragraph
    ai_review, overall_rating, ai_sentiment = generate_summary(example_reviews, sentiments)

    # Print results
    print("AI Summary:")
    print(ai_review)
    print("\nOverall Sentiment (from reviews):", overall_rating)
    print("Sentiment of AI-generated summary:", ai_sentiment)
