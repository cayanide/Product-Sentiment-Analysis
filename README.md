Product Sentiment Analysis
This project provides a sentiment analysis tool for product reviews using machine learning (Naive Bayes classifier). It allows users to predict the sentiment of a given product review (positive, negative, or neutral). This web application is built using Flask, Python, and a Naive Bayes model to classify product sentiments.

Username: cayanide
Table of Contents
Features
Installation
Usage
API
Deployment
Technologies
License
Features
Sentiment Prediction: Classifies product reviews as either Positive, Negative, or Neutral.
Real-time Web Interface: A Flask-based web app where users can input a review to get instant sentiment analysis.
Data Scraping: Scrapes product reviews from external sources using BeautifulSoup.
Model Training: Trains a Naive Bayes model using historical review data to predict sentiment.
Pretrained Model: Utilizes a pretrained model (NB_model.pkl) to predict sentiment for unseen reviews.
Installation
1. Clone the Repository
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/cayanide/Product-Sentiment-Analysis.git
cd Product-Sentiment-Analysis
2. Create a Virtual Environment
Create a virtual environment to manage the dependencies:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
3. Install Dependencies
Install all the necessary Python dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
1. Run the Flask Web App Locally
To run the app locally, you can use the following command:

bash
Copy code
python app.py
This will start the Flask application, and you can access it via your browser at http://127.0.0.1:5000/.

2. Web Interface
Once the app is running, open your browser and go to http://127.0.0.1:5000/. Here you can input a product review, and the app will predict the sentiment (positive, negative, or neutral).

API
The web app exposes an API to predict sentiment based on the provided text (review).

Endpoint: /predict_sentiment
Method: POST
Input JSON:

json
Copy code
{
  "review": "This product is amazing, I love it!"
}
Response JSON:

json
Copy code
{
  "sentiment": "positive"
}
Deployment
Deploy on Heroku
To deploy the app on Heroku, follow these steps:

Create a Heroku App:

bash
Copy code
heroku create your-app-name
Push to Heroku: After creating the app, push your local repository to Heroku:

bash
Copy code
git push heroku main
Open the app: Once deployed, you can open your app in the browser using:

bash
Copy code
heroku open
For detailed deployment instructions, check out Heroku's guide.

Technologies
Flask: A lightweight Python web framework.
Scikit-learn: A Python library used to build the Naive Bayes classifier.
BeautifulSoup: A Python library for web scraping.
NLTK: A library for natural language processing and text classification.
Heroku: A cloud platform for deploying and managing applications.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Scikit-learn for machine learning models.
Flask for building the web application.
BeautifulSoup for scraping product reviews.
This README provides an overview of how to set up and use your sentiment analysis application. If you have any other questions or need further assistance, feel free to open an issue or contribute to the repository.
