from flask import Flask, render_template, request
from scrape_reviews import scrape_reviews
from predict_sentiment import predict_sentiment, generate_summary

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_data = None
    overall_rating = None
    ai_review = None

    if request.method == "POST":
        # Get the product URL from the form
        product_url = request.form["product_url"]

        # Scrape reviews for the product
        reviews = scrape_reviews(product_url)

        if reviews:
            # Predict sentiments for the scraped reviews
            sentiments = predict_sentiment(reviews)
            sentiment_data = list(zip(reviews, sentiments))  # Convert zip object to list for template rendering

            # Generate AI-based summary and overall rating
            ai_review, overall_rating = generate_summary(reviews, sentiments)

    # Render the index.html template with the processed data
    return render_template("index.html", sentiment_data=sentiment_data, overall_rating=overall_rating, ai_review=ai_review)


if __name__ == "__main__":
    app.run(debug=True)
