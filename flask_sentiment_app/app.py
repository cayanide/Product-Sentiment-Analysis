import os
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request, session, make_response, redirect
from scrape_reviews import scrape_reviews  # Import your scraping logic
from predict_sentiment import predict_sentiment, generate_summary  # Import your sentiment analysis logic
from sklearn.metrics import confusion_matrix, classification_report

# Set up the non-GUI backend for saving plots in Flask environment
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session key

# Folder for saving static images (graphs)
STATIC_DATA_FOLDER = os.path.join(os.getcwd(), 'static', 'Data')
if not os.path.exists(STATIC_DATA_FOLDER):
    os.makedirs(STATIC_DATA_FOLDER)

def get_cookie_data(request, key, default=None):
    """Helper function to load data from cookies."""
    cookie = request.cookies.get(key)
    return json.loads(cookie) if cookie else default

def save_cookie_data(response, key, data):
    """Helper function to save data in cookies."""
    response.set_cookie(key, json.dumps(data), max_age=30*24*60*60)  # 30 days

@app.route('/', methods=['GET', 'POST'])
def index():
    ai_review = None
    overall_rating = None
    sentiment_data = None
    product_details = None
    class_report = None
    cm_image = None
    class_dist_image = None

    # Load user history and previous analysis from cookies
    history = get_cookie_data(request, "history", [])
    analysis_reports = get_cookie_data(request, "analysis_reports", {})

    if request.is_json:
            data = request.get_json()
            product_url = data.get("text_input")
            # Logic to analyze the product and return results as JSON
            # Your analysis logic here...
            response = render_template('index.html', ai_review=ai_review)
            return response

    if request.method == 'POST':
        product_url = request.form['text_input']

        # Check if product has been analyzed before
        if product_url in analysis_reports:
            report = analysis_reports[product_url]
            return render_template(
                'index.html',
                ai_review=report['ai_review'],
                overall_rating=report['overall_rating'],
                sentiment_data=report['sentiment_data'],
                cm_image=report['cm_image'],
                class_dist_image=report['class_dist_image'],
                class_report=report['class_report'],
                history=history,
                product_details=report['product_details']
            )

        # Scrape reviews and analyze
        reviews = scrape_reviews(product_url)
        if reviews:
            # Predict sentiments
            sentiments = predict_sentiment(reviews)

            # Generate AI summary and evaluate sentiment
            ai_review, overall_rating = generate_summary(reviews, sentiments)

            # Combine reviews with their respective sentiments
            sentiment_data = list(zip(reviews, sentiments))

            # Generate and save visualization graphs
            cm_image, class_dist_image = generate_graphs(sentiments)

            # Generate classification report
            class_report = generate_classification_report(sentiments)

            # Extract product details
            product_details = extract_product_details(product_url)

            # Save the analysis to analysis_reports
            analysis_reports[product_url] = {
                'ai_review': ai_review,
                'overall_rating': overall_rating,
                'sentiment_data': sentiment_data,
                'cm_image': cm_image,
                'class_dist_image': class_dist_image,
                'class_report': class_report,
                'product_details': product_details
            }

        # Update user history
        if product_url not in history:
            history.append(product_url)

        # Prepare the response with updated cookies
        response = make_response(
            render_template(
                'index.html',
                ai_review=ai_review,
                overall_rating=overall_rating,
                sentiment_data=sentiment_data,
                cm_image=cm_image,
                class_dist_image=class_dist_image,
                class_report=class_report,
                history=history,
                product_details=product_details
            )
        )
        save_cookie_data(response, "history", history)
        save_cookie_data(response, "analysis_reports", analysis_reports)
        return response

    return render_template(
        'index.html',
        history=history
    )

def extract_product_details(product_url):
    """
    Placeholder logic for extracting product details from the URL.
    Add scraping logic here to extract details dynamically.
    """
    return {
        'name': 'Sample Product Name',
        'category': 'Electronics',
        'brand': 'Sample Brand',
        'model': 'Model XYZ'
    }

def get_sentiment_int(sentiment):
    """Map sentiment strings to integers."""
    sentiment_mapping = {
        'Negative': 0,
        'Neutral': 1,
        'Positive': 2
    }
    return sentiment_mapping.get(sentiment, -1)

def generate_classification_report(sentiments):
    """Generate a classification report from sentiment predictions."""
    sentiment_ints = [get_sentiment_int(sentiment) for sentiment in sentiments]
    report = classification_report(
        sentiment_ints, sentiment_ints,
        labels=[0, 1, 2],
        target_names=['Negative', 'Neutral', 'Positive'],
        zero_division=0
    )
    return report

def generate_graphs(sentiments):
    """Generate and save confusion matrix and class distribution plots."""
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    sentiment_ints = [sentiment_map[s] for s in sentiments]

    # Confusion Matrix
    cm = confusion_matrix(sentiment_ints, sentiment_ints)
    cm_path = os.path.join(STATIC_DATA_FOLDER, 'cm_plot.png')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(cm_path)
    plt.close()

    # Class Distribution
    class_counts = np.bincount(sentiment_ints, minlength=3)
    class_dist_path = os.path.join(STATIC_DATA_FOLDER, 'class_dist_plot.png')
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["Negative", "Neutral", "Positive"], y=class_counts)
    plt.title("Class Distribution")
    plt.ylabel("Frequency")
    plt.savefig(class_dist_path)
    plt.close()

    return cm_path, class_dist_path

@app.route('/graph')
def view_graph():
    """Render the graph view page."""
    cm_image = os.path.join('static', 'Data', 'cm_plot.png')
    class_dist_image = os.path.join('static', 'Data', 'class_dist_plot.png')

    # Load classification report from session
    class_report = session.get('class_report', None)

    return render_template(
        'graph.html',
        cm_image=cm_image,
        class_dist_image=class_dist_image,
        class_report=class_report
    )

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3030)
