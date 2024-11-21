# ---------------- MAIN LIBRARIES ------------------
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK data
nltk.download('stopwords')

# ---------------- LOAD DATA ------------------
# Load datasets
train_dataset = 'train.csv'
test_dataset = 'test.csv'

# Read the datasets
train_df = pd.read_csv(train_dataset, encoding='ISO-8859-1')
test_df = pd.read_csv(test_dataset, encoding='ISO-8859-1')

# Drop rows with missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# ---------------- TEXT PREPROCESSING ------------------
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# ---------------- VECTORIZE TEXT ------------------
vectorizer = TfidfVectorizer(max_features=5000)
train_TFIDF = vectorizer.fit_transform(train_df['processed_text'])
test_TFIDF = vectorizer.transform(test_df['processed_text'])

# ---------------- TRAIN NAIVE BAYES MODEL ------------------
X = train_TFIDF
y = train_df['sentiment']  # Assuming 'sentiment' column contains labels

NB_model = MultinomialNB()
NB_model.fit(X, y)

# Save the vectorizer and model for reuse
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(NB_model, 'NB_model.pkl')

# ---------------- TEST THE MODEL ------------------
# Test the model on the test dataset
predictions = NB_model.predict(test_TFIDF)
print(f"Accuracy: {accuracy_score(test_df['sentiment'], predictions)}")
print(classification_report(test_df['sentiment'], predictions))
