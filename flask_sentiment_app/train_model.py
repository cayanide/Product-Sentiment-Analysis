import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

# Ensure NLTK downloads are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ---------------- LOAD DATA ------------------
# Paths to datasets
datasets = [
    'Data/amazon_cells_labelled.csv',
    'Data/imdb_labelled.txt',
    'Data/yelp_labelled.txt',
    'Data/training.1600000.processed.noemoticon.csv'
]

# Column names for the datasets
amazon_columns = ['text', 'sentiment']
imdb_columns = ['text', 'sentiment']
yelp_columns = ['text', 'sentiment']
sentiment140_columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Load datasets
amazon_data = pd.read_csv(datasets[0], names=amazon_columns)
imdb_data = pd.read_csv(datasets[1], names=imdb_columns, sep='\t')
yelp_data = pd.read_csv(datasets[2], names=yelp_columns, sep='\t')
sentiment140_data = pd.read_csv(datasets[3], encoding='ISO-8859-1', names=sentiment140_columns)

# Convert sentiment in Sentiment140 dataset (0: Negative, 4: Positive)
sentiment140_data['sentiment'] = sentiment140_data['sentiment'].map({0: 0, 4: 1})

# Combine all datasets
data = pd.concat([amazon_data[['text', 'sentiment']], imdb_data[['text', 'sentiment']],
                  yelp_data[['text', 'sentiment']], sentiment140_data[['text', 'sentiment']]], ignore_index=True)

# ---------------- TEXT PREPROCESSING ------------------
def preprocess_text(text):
    # Check if the text is a string (handle missing or non-string values)
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    # Remove URLs, mentions, hashtags, special characters, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in tokens])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

# Handle missing values (replace NaN or None with an empty string)
data['text'] = data['text'].fillna("")

# Handle missing values in target variable ('sentiment') by dropping rows with NaN
data = data.dropna(subset=['sentiment'])

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess_text)

# ---------------- VECTORIZE TEXT ------------------
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 1))  # Adjusted max_features and ngram_range
X = vectorizer.fit_transform(data['processed_text'])
y = data['sentiment']

# ---------------- TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- TRAIN NAIVE BAYES MODEL ------------------
NB_model = MultinomialNB(alpha=1.0)  # Adjusted alpha for smoothing
NB_model.fit(X_train, y_train)

# ---------------- SAVE THE MODEL AND VECTORIZER ------------------
joblib.dump(NB_model, 'NB_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# ---------------- TEST THE MODEL ------------------
predictions = NB_model.predict(X_test)

# Model evaluation
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))

# Confusion matrix to understand misclassifications
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
