import joblib
from nltk.corpus import stopwords
from collections import Counter
from scrape_reviews import scrape_reviews
import openai

# Set up OpenAI API key
openai.api_key = "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"

# Load the model and vectorizer
NB_model = joblib.load('NB_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Predict sentiment for reviews
def predict_sentiment(reviews):
    if not reviews:
        print("No reviews to analyze.")
        return []

    processed_reviews = [preprocess_text(review) for review in reviews]
    reviews_TFIDF = vectorizer.transform(processed_reviews)
    predictions = NB_model.predict(reviews_TFIDF)
    return predictions

# Generate summary based on sentiments
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

# Function to generate AI-based review using OpenAI API
def generate_ai_review(reviews, sentiments, overall_rating):
    # Combine reviews for AI input
    combined_reviews = "\n".join(reviews)

    # Create prompt for the AI model
    prompt = (
        f"The product has the following reviews:\n{combined_reviews}\n\n"
        f"The overall sentiment is '{overall_rating.lower()}'. "
        f"Based on the reviews, write a detailed summary describing customers' opinions, "
        f"highlighting common praises and complaints."
    )

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or any other available engine
            prompt=prompt,
            max_tokens=300,  # Adjust token limit as needed
            temperature=0.7
        )
        ai_review = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating AI review: {e}")
        ai_review = "Could not generate AI-based review due to an error."

    return ai_review
