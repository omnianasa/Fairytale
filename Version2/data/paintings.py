#####################
#web scraping for paintings data
#enjoy!
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import time

#chrome headless
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)
driver.get("https://vangoghworldwide.org/search/?q=van+gogh&size=n_100_n")

#scroll down to get all photos
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  #wait for new images to load
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break  #reached the bottom
    last_height = new_height

#get HTML after all images have loaded
html = driver.page_source
driver.quit()

soup = BeautifulSoup(html, "html.parser")

#extract all image URLs
imgs = [img["src"] for img in soup.find_all("img") if img.get("src")]
print("Number of images after scroll:", len(imgs))

#download images
out = Path("data/B")
out.mkdir(exist_ok=True)

for i, url in enumerate(tqdm(imgs[:100], desc="Downloading")):
    try:
        response = requests.get(url, timeout=30)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(out / f"painting_{i:03d}.jpg", "JPEG", quality=95)
    except Exception as e:
        print(f"Failed to download image {i}: {e}")
        continue

print("Images have been downloaded to:", out.resolve())
