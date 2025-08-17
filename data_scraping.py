from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
import os
import requests
from selenium.webdriver.chrome.options import Options

# PDF folder (inside Docker container)
download_dir = "/app/input_file/{project_type}"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Selenium Grid URL (replace with the actual host and port of your Selenium server)
# selenium_grid_url = "http://localhost:4444/wd/hub"  # Change this if necessary

# Set Chrome options for running inside Docker with Selenium Grid
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Running in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

chrome_driver_path = "/usr/bin/chromedriver"
service = Service(executable_path=chrome_driver_path)


# Initialize Remote WebDriver (communicating with the Selenium server)
driver = webdriver.Chrome(service=service, options=chrome_options)


# Download function
def download_pdf(pdf_url, file_name, project_type):
    print(f"Attempting to download {file_name} from {pdf_url}...")
    try:
        # Ensure that the project-specific directory exists
        project_dir = download_dir.format(project_type=project_type)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)  # Create the directory if it doesn't exist
        
        # Construct the full path for the PDF file
        pdf_path = os.path.join(project_dir, file_name)
        
        # Download the PDF file
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check for HTTP errors
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded: {file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")
    except Exception as e:
        print(f"Error saving {file_name}: {e}")


# Function to scrape and download patterns from Yarnspirations
def download_yarnspirations(project_type, total_page_num):
    base_url = "https://www.yarnspirations.com/collections/patterns?filter.p.m.global.project_type={project_type}&filter.p.m.global.skill_type=Crochet&page={page_num}"

    # Loop through pages
    for page_num in range(1, total_page_num + 1):  # page range
        print(f"Scraping page {page_num}...")
        driver.get(base_url.format(project_type=project_type, page_num=page_num))
        
        # Wait until the "Free Pattern" buttons are loaded
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@class, 'card-button--full')]"))
            )
        except Exception as e:
            print(f"Error while waiting for patterns to load: {e}")
            continue
        
        # Find "Free Pattern" buttons by XPath or class
        pattern_links = driver.find_elements(By.XPATH, "//a[contains(@class, 'card-button--full')]")
        
        print(f"Found {len(pattern_links)} pattern links on page {page_num}")
        
        for pattern_link in pattern_links:
            pdf_url = pattern_link.get_attribute('href')
            
            # Check if the link is a PDF link
            if pdf_url and pdf_url.endswith('.pdf'):
                # Extract the file name from the URL
                file_name = pdf_url.split('/')[-1]
                
                # Download the PDF
                download_pdf(pdf_url, file_name, project_type)
            else:
                print(f"Skipping non-PDF or invalid link: {pdf_url}")

# Quit WebDriver after the task is done
def quit_driver():
    driver.quit()

# Example: Uncomment this line to scrape patterns
# download_yarnspirations('Tops', 2)

# Quit WebDriver
# quit_driver()
