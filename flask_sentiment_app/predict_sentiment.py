import joblib
from nltk.corpus import stopwords
from collections import Counter
import openai
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Fetch API key and Organization ID from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

# Load the model and vectorizer
NB_model = joblib.load('NB_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess the text
def preprocess_text(text):
    try:
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return ""

# Predict sentiment for reviews
def predict_sentiment(reviews):
    try:
        if not reviews:
            print("No reviews to analyze.")
            return []

        processed_reviews = [preprocess_text(review) for review in reviews]
        reviews_TFIDF = vectorizer.transform(processed_reviews)
        predictions = NB_model.predict(reviews_TFIDF)
        return predictions
    except Exception as e:
        print(f"Error in predicting sentiment: {e}")
        return []

# Generate summary based on sentiments
def generate_summary(reviews, sentiments):
    try:
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
    except Exception as e:
        print(f"Error in generating summary: {e}")
        return "Error generating summary.", "Unknown"

# Function to generate AI-based review using OpenAI API with error handling
def generate_ai_review(reviews, sentiments, overall_rating):
    combined_reviews = "\n".join(reviews)
    prompt = (
        f"The product has the following reviews:\n{combined_reviews}\n\n"
        f"The overall sentiment is '{overall_rating.lower()}'. "
        f"Based on the reviews, write a detailed summary describing customers' opinions, "
        f"highlighting common praises and complaints."
    )

    try:
        # Use the new ChatCompletion endpoint
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Adjust the model based on your usage
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes customer reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Adjust the max token limit as per your needs
            temperature=0.7,  # Control randomness (0.0-1.0)
            top_p=1.0,  # Control diversity (1.0 means full diversity)
            n=1,  # Number of responses to generate
            stop=None  # Optionally, set stop sequences
        )

        # Return the generated content
        return response['choices'][0]['message']['content'].strip()

    except openai.error.RateLimitError:
        print("Rate limit exceeded, retrying...")
          # Wait for 60 seconds before retrying
        return generate_ai_review(reviews, sentiments, overall_rating)

    except openai.error.AuthenticationError:
        print("Authentication error: Please check your API key and organization ID.")
        return "Authentication error occurred. Please check your credentials."

    except openai.error.OpenAIError as e:
        # General OpenAI API error handling
        print(f"OpenAI API error: {e}")
        return "No detailed summary available for this product."

    except Exception as e:
        # Catch all other exceptions
        print(f"An unexpected error occurred: {e}")
        return "An error occurred while generating the review."

# Example .env file content:
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_ORG_ID=your_openai_org_id
