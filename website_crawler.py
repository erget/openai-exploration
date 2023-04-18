import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tldextract import extract
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)


def extract_domain(url):
    domain_parts = extract(url)
    domain = f"{domain_parts.domain}.{domain_parts.suffix}"
    return domain


def crawl_website(url, domain):
    visited_urls = set()
    urls_to_visit = [url]
    all_text = ""
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp")

    while urls_to_visit:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        # Skip URLs with common image extensions
        if current_url.lower().endswith(image_extensions):
            logging.debug(f"Skipping image URL: {current_url}")
            continue

        logging.info(f"Visiting: {current_url}")
        visited_urls.add(current_url)

        try:
            response = requests.get(current_url)
        except (requests.exceptions.SSLError, requests.exceptions.RequestException) as e:
            logging.warning(f"Request Error for {current_url}: {e}")
            continue

        if response.status_code != 200:
            logging.warning(f"Failed to fetch {current_url}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        all_text += f"{text}\n\n"

        # Find all the links on the current page and add them to the queue if they are in the same domain
        for link in soup.find_all("a", href=True):
            new_url = urljoin(current_url, link["href"])
            new_domain = extract_domain(new_url)

            if new_domain == domain and new_url not in visited_urls:
                logging.debug(f"Adding {new_url} to the queue")
                urls_to_visit.append(new_url)

    return all_text


def save_to_file(filename, text):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python website_crawler.py <URL> <output_filename>")
        sys.exit(1)

    url = sys.argv[1]
    domain = extract_domain(url)
    output_filename = sys.argv[2]

    logging.info(f"Starting crawl of {url}")
    extracted_text = crawl_website(url, domain)
    save_to_file(output_filename, extracted_text)
    logging.info(f"Extracted text saved to {output_filename}")
