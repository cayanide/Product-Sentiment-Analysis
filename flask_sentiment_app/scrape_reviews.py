import requests
from bs4 import BeautifulSoup
import time
import random
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to scrape Amazon reviews
def scrape_amazon_reviews(product_url):
    """
    Scrapes reviews from an Amazon product page.
    Args:
    product_url (str): URL of the Amazon product page.

    Returns:
    list: List of reviews as text strings.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(product_url, headers=headers)
        response.raise_for_status()  # Will raise an HTTPError if the response was not successful

        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = []

        # Scrape reviews based on Amazon's review container class
        review_elements = soup.find_all('span', {'data-hook': 'review-body'})
        for review_element in review_elements:
            review_text = review_element.get_text(strip=True)
            reviews.append(review_text)

        if not reviews:
            logging.warning("No reviews found on the page.")
        return reviews

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Amazon: {e}")
        return []

# Function to scrape Flipkart reviews
def scrape_flipkart_reviews(product_url):
    """
    Scrapes reviews from a Flipkart product page.
    Args:
    product_url (str): URL of the Flipkart product page.

    Returns:
    list: List of reviews as text strings.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(product_url, headers=headers)
        response.raise_for_status()  # Will raise an HTTPError if the response was not successful

        soup = BeautifulSoup(response.content, 'html.parser')
        reviews = []

        # Flipkart review container class
        review_elements = soup.find_all('div', {'class': 't-ZTKy'})
        for review_element in review_elements:
            review_text = review_element.get_text(strip=True)
            reviews.append(review_text)

        if not reviews:
            logging.warning("No reviews found on the page.")
        return reviews

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Flipkart: {e}")
        return []

# General function to decide which platform to scrape from
def scrape_reviews(product_url):
    """
    Determines the platform (Amazon or Flipkart) based on the URL and calls the appropriate scraping function.
    Args:
    product_url (str): URL of the product (Amazon or Flipkart).

    Returns:
    list: List of reviews.
    """
    if "amazon" in product_url:
        logging.info("Scraping Amazon reviews...")
        return scrape_amazon_reviews(product_url)
    elif "flipkart" in product_url:
        logging.info("Scraping Flipkart reviews...")
        return scrape_flipkart_reviews(product_url)
    else:
        logging.error("Unsupported URL. Please provide an Amazon or Flipkart product link.")
        return []

# Optional function to simulate human-like delays between requests (politeness layer)
def delay_request():
    """
    Introduces a random delay between requests to avoid overloading servers.
    """
    time.sleep(random.uniform(1, 3))  # Random sleep between 1-3 seconds
