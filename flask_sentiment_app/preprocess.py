# preprocess.py

import re
import string

# Example of a preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove any extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # You can add more preprocessing steps as required
    return text
