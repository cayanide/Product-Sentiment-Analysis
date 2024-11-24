FROM python:3.10-slim

# Install system dependencies for building h5py and nltk requirements
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code and datasets into the container
COPY . /app

RUN pip install nltk
# Install Python dependencies


# Install NLTK separately to ensure it is available for downloading resources


# Ensure NLTK resources are downloaded during build
RUN python -m nltk.downloader punkt stopwords wordnet punkt_tab

RUN pip install -r requirements.txt

# Copy the datasets into the container
COPY flask_sentiment_app/Data /app/Data

# Run the train_model.py to train the model and save it as a .pkl file
RUN python flask_sentiment_app/train_model.py

# Expose port for Flask app
EXPOSE 3636

# Ensure Flask binds to 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=3636

# Command to run the Flask app after training
CMD ["python", "flask_sentiment_app/app.py"]
