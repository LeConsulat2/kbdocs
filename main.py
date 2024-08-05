# import time
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from webdriver_manager.chrome import ChromeDriverManager

# # Specify the path to your ChromeDriver executable
# chromedriver_path = (
#     "C:/Users/Jonathan/Documents/kbdocs/env/Lib/site-packages/chromedriver.exe"
# )

# browser = webdriver.Chrome(service=ChromeService(executable_path=chromedriver_path))
# # browser = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))


# browser.get("https://google.com")

# time.sleep(20)

# search_bar = browser.find_element_by_class_name("gLFyf")

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# Set Chrome options
chrome_options = Options()
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
chrome_options.add_experimental_option("detach", True)

# Initialize the Chrome browser using webdriver_manager to handle ChromeDriver installation
service = Service(ChromeDriverManager().install())
browser = webdriver.Chrome(service=service, options=chrome_options)

# Navigate to Google
browser.get("https://google.com")

# Find the search bar element
search_bar = browser.find_element(By.CLASS_NAME, "gLFyf")
search_bar.send_keys("hello!")
search_bar.send_keys(Keys.ENTER)

# Wait for the search results to load and display them
search_results = WebDriverWait(browser, 20).until(
    EC.presence_of_all_elements_located((By.CLASS_NAME, "g"))
)

print(search_results)
