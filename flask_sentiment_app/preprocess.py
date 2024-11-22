# preprocess.py

import re
import string

# Example of a preprocessing function
def preprocess_text(text):
    """
    Preprocess the input text for both prediction and training consistency.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase conversion
    text = text.lower()
    # Remove special characters, URLs, mentions, hashtags, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    # Tokenization and Lemmatization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in tokens if word not in stop_words])
