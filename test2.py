import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# def setup_chromedriver():
#     try:
#         # Path to the ChromeDriver in the virtual environment
#         chromedriver_path = (
#             "C:/Users/Jonathan/Documents/kbdocs/env/Lib/site-packages/chromedriver.exe"
#         )

#         # Add the ChromeDriver path to the system PATH
#         os.environ["PATH"] += os.pathsep + os.path.dirname(chromedriver_path)

#         return chromedriver_path
#     except Exception as e:
#         logger.error(f"Error setting up ChromeDriver: {e}")
#         logger.error(traceback.format_exc())
#         return None


def setup_browser():
    try:
        # Setup Chrome browser with the specified ChromeDriver path
        options = webdriver.ChromeOptions()
        # Comment out or remove the headless option to show the browser
        # options.add_argument('--headless')  # Disable headless mode to show the browser
        # options.add_argument("--disable-gpu")  # Disable GPU acceleration
        browser = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        )
        return browser
    except Exception as e:
        logger.error(f"Error setting up the browser: {e}")
        logger.error(traceback.format_exc())
        return None


def perform_search(browser, query):
    try:
        # Open Google
        browser.get("https://www.google.com")
        logger.info("Opened Google homepage")

        # Wait for the search box to be available and perform a search
        search_box = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        logger.info(f"Performed search for query: {query}")
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise


def extract_search_results(browser):
    try:
        # Wait for the search results to load and display the results
        results = WebDriverWait(browser, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h3"))
        )

        # Extract and print the titles of the search results
        titles = [result.text for result in results]
        for i, title in enumerate(titles):
            logger.info(f"{i+1}: {title}")

        # Click on the first search result link
        if results:
            first_result = results[0]
            first_result.click()
            logger.info(f"Clicked on the first search result: {titles[0]}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error extracting search results: {e}")
        raise


def extract_page_content(browser):
    try:
        # Wait for the page content to load and extract some text from the page
        content = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        logger.info("Extracted page content")
        logger.info(
            content.text[:500]
        )  # Print the first 500 characters of the page content
    except Exception as e:
        logger.error(f"Error extracting page content: {e}")
        raise


def main():
    browser = setup_browser()
    if browser is not None:
        try:
            perform_search(browser, "Selenium WebDriver")
            if extract_search_results(browser):
                extract_page_content(browser)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Close the browser after the operation
            time.sleep(5)  # Sleep for a while to see the results
            browser.quit()
            logger.info("Browser closed")
    else:
        logger.error("Failed to initialize the browser.")


if __name__ == "__main__":
    main()
