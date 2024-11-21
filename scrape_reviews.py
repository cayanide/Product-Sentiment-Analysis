import requests
from bs4 import BeautifulSoup

# Function to scrape Amazon reviews
def scrape_amazon_reviews(product_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(product_url, headers=headers)

    if response.status_code != 200:
        print("Failed to retrieve the Amazon page.")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []

    # Amazon's review container class
    review_elements = soup.find_all('span', {'data-hook': 'review-body'})

    for review_element in review_elements:
        reviews.append(review_element.get_text(strip=True))

    return reviews

# Function to scrape Flipkart reviews
def scrape_flipkart_reviews(product_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(product_url, headers=headers)

    if response.status_code != 200:
        print("Failed to retrieve the Flipkart page.")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []

    # Flipkart's review container class
    review_elements = soup.find_all('div', {'class': 't-ZTKy'})

    for review_element in review_elements:
        reviews.append(review_element.get_text(strip=True))

    return reviews

# General function to decide which platform to scrape from
def scrape_reviews(product_url):
    if "amazon" in product_url:
        print("Scraping Amazon reviews...")
        return scrape_amazon_reviews(product_url)
    elif "flipkart" in product_url:
        print("Scraping Flipkart reviews...")
        return scrape_flipkart_reviews(product_url)
    else:
        print("Unsupported URL. Please provide an Amazon or Flipkart product link.")
        return []
