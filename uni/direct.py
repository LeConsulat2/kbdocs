import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from shareplum import Site
from shareplum.site import Version
import pandas as pd
from docx import Document

# SharePoint site credentials
username = ""
password = ""
sharepoint_site = "https://autuni.sharepoint.com"
sharepoint_list_url = "https://autuni.sharepoint.com/sites/knowledgebase-master/Lists/Knowledge%20Base/AllItems.aspx"

# File paths
excel_file_path = r"C:\Users\Jonathan\Documents\kbdocs\3-items.xlsx"
base_save_directory = r"C:\Users\Jonathan\Documents\kbdocs\staff"

# Initialize Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")  # Ensure the browser is maximized
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)


def login_to_sharepoint(username, password):
    driver.get(sharepoint_site)
    time.sleep(2)

    try:
        # Enter username
        username_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.NAME, "loginfmt"))
        )
        username_input.send_keys(username)
        username_input.send_keys(Keys.ENTER)
        time.sleep(5)
        driver.save_screenshot("after_entering_username.png")

        # Enter password
        password_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "i0118"))
        )
        password_input.send_keys(password)
        password_input.send_keys(Keys.ENTER)
        time.sleep(5)
        driver.save_screenshot("after_entering_password.png")

        # Handle "Stay signed in?" prompt
        stay_signed_in = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "idSIButton9"))
        )
        stay_signed_in.click()
        time.sleep(5)
        driver.save_screenshot("after_stay_signed_in.png")

        # Duo Mobile authentication
        WebDriverWait(driver, 60).until(EC.title_contains("Duo Mobile"))
        print("Please complete the Duo Mobile authentication on your device.")
        while "Duo Mobile" in driver.title:
            time.sleep(10)
        driver.save_screenshot("after_duo_authentication.png")

        # Finalize login process
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.ID, "idSIButton9"))
        )
        driver.find_element(By.ID, "idSIButton9").click()
        time.sleep(30)
        driver.save_screenshot("after_finalizing_login.png")

        print(
            "Please manually complete any additional login steps. Waiting for 60 seconds..."
        )
        time.sleep(60)

    except Exception as e:
        print(f"An error occurred: {e}")
        driver.save_screenshot("error_screenshot.png")
        driver.quit()
        raise


login_to_sharepoint(username, password)

# Verify login
current_url = driver.current_url
print(f"Current URL after login: {current_url}")

if "Access Denied" in driver.page_source:
    print("Access Denied. Check permissions and try again.")
    driver.quit()
    exit()

# Perform additional actions
driver.get(sharepoint_list_url)
time.sleep(20)
driver.save_screenshot("after_navigating_to_sharepoint_list.png")

input("Press Enter to continue after verifying access...")

# Get authentication cookies
cookies = driver.get_cookies()
authcookie = {cookie["name"]: cookie["value"] for cookie in cookies}

# Optionally close the browser
# driver.quit()

# Use cookies with requests session
session = requests.Session()
for cookie in cookies:
    session.cookies.set(cookie["name"], cookie["value"])

# Login to SharePoint with cookies
site = Site(
    f"{sharepoint_site}/sites/ssastudentadvisory",
    version=Version.v365,
    authcookie=authcookie,
)
sp_list = site.List("Service Knowledge Content uploader")

# Get list items
list_items = sp_list.GetListItems("All Items")

# Create output directory
output_dir = "sharepoint_downloads"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to download file from SharePoint
def download_file(file_url, output_dir, session):
    response = session.get(file_url, stream=True)
    file_name = os.path.basename(file_url)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


# Process list items and download files
for item in list_items:
    attachments = item.get("Attachments")
    if attachments:
        for attachment in attachments:
            file_url = f"{sharepoint_site}{attachment['ServerRelativeUrl']}"
            download_file(file_url, output_dir, session)

print("Files downloaded successfully.")

# Load Excel data
excel_data = pd.read_excel(excel_file_path)

# Ensure base save directory exists
if not os.path.exists(base_save_directory):
    os.makedirs(base_save_directory)

# Iterate through DataFrame rows and save Word documents
for index, row in excel_data.iterrows():
    content = str(row["Content"]) if pd.notna(row["Content"]) else ""
    name = str(row["Name"]) if pd.notna(row["Name"]) else "Unnamed"
    business_owner = (
        str(row["Business owner"]) if pd.notna(row["Business owner"]) else "Unknown"
    )

    business_owner_directory = os.path.join(base_save_directory, business_owner)
    if not os.path.exists(business_owner_directory):
        os.makedirs(business_owner_directory)

    doc = Document()
    doc.add_paragraph(content)

    valid_filename = "".join(x for x in name if x.isalnum() or x in "._- ")
    file_path = os.path.join(business_owner_directory, f"{valid_filename}.docx")
    doc.save(file_path)

print("Word documents created successfully.")
