import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Directory containing JSON files
JSON_DIR = "text_json"

# Base URLs for image links
IMAGE_URL_TEMPLATES = [
    "https://www.yarnspirations.com/cdn/shop/files/{id}.jpg",
    "https://www.yarnspirations.com/cdn/shop/products/{id}.jpg",
    "https://www.yarnspirations.com/cdn/shop/products/{id}-1.jpg",
    "https://www.yarnspirations.com/cdn/shop/files/{id}-1.jpg",
    "https://www.yarnspirations.com/cdn/shop/products/{id}-2.jpg",
    "https://www.yarnspirations.com/cdn/shop/files/{id}-2.jpg",
]

def check_image_url(id):
    """
    Check which URL template works for the given ID by making HEAD requests.
    Returns the working URL or None if neither works.
    """
    for template in IMAGE_URL_TEMPLATES:
        url = template.format(id=id)
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None

def process_json_file(file_path):
    """
    Process a single JSON file to add image links.
    """
    print(f"Processing {file_path}...")
    
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track changes
    updated_count = 0
    already_has_link = 0
    failed_count = 0
    
    # Process each pattern in the file
    for pattern in tqdm(data, desc=f"Patterns in {os.path.basename(file_path)}"):
        # Skip if already has image_link
        if "image_link" in pattern and pattern["image_link"]:
            already_has_link += 1
            continue
            
        # Get the ID
        if "id" in pattern and pattern["id"]:
            id = pattern["id"]
            # Check which URL works
            image_url = check_image_url(id)
            
            if image_url:
                pattern["image_link"] = image_url
                updated_count += 1
            else:
                failed_count += 1
                print(f"  Failed to find image for ID: {id}")
    
    # Save the updated JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    return {
        "file": os.path.basename(file_path),
        "updated": updated_count,
        "already_had_link": already_has_link,
        "failed": failed_count
    }

def main():
    # Get the absolute path to the JSON directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(script_dir, JSON_DIR)
    
    # Get all JSON files
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(json_dir, f))]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_json_file, json_files))
    
    # Print summary
    print("\nSummary:")
    total_updated = sum(r["updated"] for r in results)
    total_already = sum(r["already_had_link"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    
    print(f"Total patterns updated: {total_updated}")
    print(f"Patterns already with links: {total_already}")
    print(f"Failed to find images: {total_failed}")
    
    # Print details for each file
    print("\nDetails by file:")
    for r in results:
        print(f"{r['file']}: Updated {r['updated']}, Already had links {r['already_had_link']}, Failed {r['failed']}")

if __name__ == "__main__":
    main()
