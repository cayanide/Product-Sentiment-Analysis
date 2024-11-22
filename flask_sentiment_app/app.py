import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request, session
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

@app.route('/', methods=['GET', 'POST'])
def index():
    ai_review = None
    overall_rating = None
    sentiment_data = None
    product_details = None
    class_report = None  # Variable for the classification report

    # Initialize session history
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        product_url = request.form['text_input']
        reviews = scrape_reviews(product_url)

        # Add product URL to session history
        if product_url not in session['history']:
            session['history'].append(product_url)

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

            # Persist the classification report in the session
            session['class_report'] = class_report

            # Render the main page with results
            return render_template(
                'index.html',
                ai_review=ai_review,
                overall_rating=overall_rating,
                sentiment_data=sentiment_data,
                cm_image=cm_image,
                class_dist_image=class_dist_image,
                class_report=class_report,
                history=session['history'],
                product_details=product_details
            )

    return render_template(
        'index.html',
        ai_review=ai_review,
        overall_rating=overall_rating,
        sentiment_data=sentiment_data,
        class_report=class_report,
        history=session['history'],
        product_details=product_details
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
    """
    Map sentiment strings to integers.
    """
    sentiment_mapping = {
        'Negative': 0,
        'Neutral': 1,
        'Positive': 2
    }
    return sentiment_mapping.get(sentiment, -1)

def generate_classification_report(sentiments):
    """
    Generate a classification report from sentiment predictions.
    """
    sentiment_ints = [get_sentiment_int(sentiment) for sentiment in sentiments]
    report = classification_report(
        sentiment_ints, sentiment_ints,
        labels=[0, 1, 2],
        target_names=['Negative', 'Neutral', 'Positive'],
        zero_division=0
    )
    return report


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

STATIC_DATA_FOLDER = "static/data"

def generate_graphs(sentiments):
    """
    Generate and save confusion matrix and class distribution plots.

    :param sentiments: List of sentiment labels (e.g., ["positive", "negative"]).
    :return: Paths to the confusion matrix plot and class distribution plot.
    """
    # Map sentiment labels to integers
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    # Normalize input sentiments to title case to match sentiment_map
    normalized_sentiments = [s.capitalize() for s in sentiments]

    # Check for unexpected sentiment values
    if not all(s in sentiment_map for s in normalized_sentiments):
        raise ValueError(f"Unexpected sentiment found in the input: {sentiments}")

    # Convert sentiments to integers
    sentiment_ints = [sentiment_map[s] for s in normalized_sentiments]

    # Ensure there's data to plot
    if not sentiment_ints:
        raise ValueError("No sentiment data provided for graph generation.")

    # Confusion Matrix
    # For demo purposes, using predictions identical to true labels
    cm = confusion_matrix(sentiment_ints, sentiment_ints)
    cm_path = os.path.join(STATIC_DATA_FOLDER, 'cm_plot.png')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Neutral", "Positive"],
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
    sns.barplot(x=["Negative", "Neutral", "Positive"], y=class_counts, palette="muted")
    plt.title("Class Distribution")
    plt.ylabel("Frequency")
    plt.savefig(class_dist_path)
    plt.close()

    return cm_path, class_dist_path



@app.route('/graph')
def view_graph():
    """
    Render the graph view page.
    """
    cm_image = os.path.join('static', 'Data', 'cm_plot.png')
    class_dist_image = os.path.join('static', 'Data', 'class_dist_plot.png')

    # Ensure the classification report is available in the session
    class_report = session.get('class_report', None)

    return render_template(
        'graph.html',
        cm_image=cm_image,
        class_dist_image=class_dist_image,
        class_report=class_report,
        history=session['history']
    )

if __name__ == '__main__':
    app.run(debug=True)
