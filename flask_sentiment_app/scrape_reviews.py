import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from itertools import cycle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# User-agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
]

# ScraperAPI key
SCRAPER_API_KEY = "d15a2436252282daa51df17d0d964d0b"

# Function to make requests using ScraperAPI
def make_request(url, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            scraper_api_url = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={url}"
            response = requests.get(scraper_api_url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response
        except requests.exceptions.RequestException as e:
            retries += 1
            logging.warning(f"Request failed ({retries}/{max_retries}): {e}")
            time.sleep(random.uniform(2, 5))  # Randomized retry delay
    logging.error("Max retries reached. Returning None.")
    return None

# Function to parse reviews from Amazon
def scrape_amazon_reviews(product_url):
    reviews = []
    response = make_request(product_url)
    if not response:
        return reviews  # Return empty if request fails

    soup = BeautifulSoup(response.content, 'html.parser')

    # Handle potential variations in review container
    review_containers = soup.find_all('span', {'data-hook': 'review-body'})
    if not review_containers:
        logging.warning("No reviews found using 'data-hook'. Trying alternative parsing.")
        review_containers = soup.find_all('span', class_='a-size-base review-text')  # Backup class

    for container in review_containers:
        reviews.append(container.get_text(strip=True))

    if not reviews:
        logging.warning("No reviews extracted.")
    return reviews

# Function to parse reviews from Flipkart
def scrape_flipkart_reviews(product_url):
    reviews = []
    response = make_request(product_url)
    if not response:
        return reviews  # Return empty if request fails

    soup = BeautifulSoup(response.content, 'html.parser')

    # Flipkart review container
    review_containers = soup.find_all('div', {'class': 't-ZTKy'})
    if not review_containers:
        logging.warning("No reviews found in 't-ZTKy' class. Trying alternative parsing.")
        review_containers = soup.find_all('div', class_='col _2wzgFH')  # Backup class

    for container in review_containers:
        reviews.append(container.get_text(strip=True))

    if not reviews:
        logging.warning("No reviews extracted.")
    return reviews

# General scraping function
def scrape_reviews(product_url):
    if "amazon" in product_url:
        logging.info("Scraping Amazon reviews...")
        return scrape_amazon_reviews(product_url)
    elif "flipkart" in product_url:
        logging.info("Scraping Flipkart reviews...")
        return scrape_flipkart_reviews(product_url)
    else:
        logging.error("Unsupported URL. Please provide an Amazon or Flipkart product link.")
        return []

# Add polite delay
def delay_request():
    time.sleep(random.uniform(1, 3))

# Example usage
if __name__ == "__main__":
    amazon_url = "https://www.amazon.in/dp/B09G9D8KRQ"
    flipkart_url = "https://www.flipkart.com/apple-iphone-13/p/itmca361dc03f8cf"

    logging.info("Fetching Amazon reviews...")
    amazon_reviews = scrape_reviews(amazon_url)
    logging.info(f"Amazon reviews: {amazon_reviews}")

    delay_request()

    logging.info("Fetching Flipkart reviews...")
    flipkart_reviews = scrape_reviews(flipkart_url)
    logging.info(f"Flipkart reviews: {flipkart_reviews}")
