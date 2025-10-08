import json
import os
import random

# Get all JSON files in the directory
files = os.listdir('/store01/nchawla/pli9/crochet/text_json/')
patterns_with_images = {}
total_patterns = 0
total_with_images = 0

# Process each file
for file in files:
    try:
        with open(f'/store01/nchawla/pli9/crochet/text_json/{file}', 'r') as f:
            data = json.load(f)
            count = len(data)
            with_images = sum(1 for item in data if 'image_link' in item and item['image_link'])
            patterns_with_images[file] = {'total': count, 'with_images': with_images}
            total_patterns += count
            total_with_images += with_images
            print(f'{file}: {with_images}/{count} patterns with image links')
    except Exception as e:
        print(f'Error with {file}: {e}')

print(f'Total: {total_with_images}/{total_patterns} patterns with image links')
