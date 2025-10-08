import os
import json
import base64
import time
import random
import sys
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

SYSTEM_PROMPT = """
You are a professional crochet pattern writer. 
Examine the image of the finished crochet product carefully. 
Write a complete set of crochet instructions in the standard style used in published patterns.

Requirements:
- Use standard abbreviations: sc (single crochet), hdc (half double crochet), dc (double crochet), tr (treble), ch (chain), sl st (slip stitch), rep (repeat).
- Organize the instructions row by row or round by round (e.g., "Rnd 1: ...", "Row 2: ...").
- If color changes are visible in the image, include them in the pattern.
- Keep the instructions concise and precise, as if for experienced crocheters.
- Output only the crochet pattern. Do not add any explanations, commentary, or extra text."""

# --- API setup ---
# Get API key from environment variable for security
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# API configuration
GPT_API_URL = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# --- Directories ---
json_dir = "/store01/nchawla/pli9/crochet/text_json"  # input JSONs
output_dir = "./generated_instructions_gpt4v/"  # output directory
os.makedirs(output_dir, exist_ok=True)

# Progress tracking file
progress_file = os.path.join(output_dir, "progress.json")

# File for tracking missing patterns
missing_patterns_file = os.path.join(output_dir, "missing_patterns.json")

def encode_image_to_base64(image_path_or_url):
    """Convert image to base64 encoding for API submission"""
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image from {image_path_or_url}")
        image_content = response.content
    else:
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
        with open(image_path_or_url, "rb") as image_file:
            image_content = image_file.read()
    
    # Convert to base64
    return base64.b64encode(image_content).decode('utf-8')

def load_progress():
    """Load progress from file to resume where we left off"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Error reading progress file. Starting from scratch.")
            return {}
    return {}

def save_progress(progress):
    """Save progress to file"""
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def call_api_with_retry(payload, max_retries=5, initial_delay=1):
    """Call the API with exponential backoff retry logic"""
    delay = initial_delay
    
    for retry in range(max_retries):
        try:
            response = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=30)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                # Get retry-after header or use exponential backoff
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else delay
                print(f"Rate limited. Waiting {wait_time} seconds before retry {retry+1}/{max_retries}...")
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
                continue
                
            # Handle other common API errors
            elif response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                
                if response.status_code >= 500:
                    # Server error, retry
                    print(f"Server error, retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    # Client error, don't retry
                    raise Exception(error_message)
            
            # Success
            return response.json()
            
        except (requests.RequestException, TimeoutError) as e:
            print(f"Request error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
            
    raise Exception(f"Failed after {max_retries} retries")

def find_missing_instructions(specific_file=None):
    """Find patterns that don't have generated instructions"""
    missing_patterns = {}
    
    # List JSON files or use the specific file if provided
    if specific_file:
        json_files = [specific_file]
    else:
        json_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "progress.json" and f != "missing_patterns.json"]
    
    total_missing = 0
    total_patterns = 0
    
    for json_file in tqdm(json_files, desc="Analyzing files for missing patterns"):
        output_path = os.path.join(output_dir, json_file)
        original_path = os.path.join(json_dir, json_file)
        
        # Skip if the output file doesn't exist
        if not os.path.exists(output_path):
            continue
            
        try:
            # Load output file
            with open(output_path, "r") as f:
                data = json.load(f)
                
            # Load original file for comparison if needed
            original_data = None
            if os.path.exists(original_path):
                with open(original_path, "r") as f:
                    original_data = json.load(f)
            
            # Convert to list if data is a dictionary
            if isinstance(data, dict):
                # If the dictionary has a list of entries, use that
                if any(isinstance(v, list) for v in data.values()):
                    for key, value in data.items():
                        if isinstance(value, list):
                            entries = value
                            break
                else:
                    # Otherwise treat the dictionary as a single entry
                    entries = [data]
            else:
                # It's already a list
                entries = data
            
            # Collect entries without generated instructions
            file_missing = []
            for idx, entry in enumerate(entries):
                total_patterns += 1
                
                if isinstance(entry, dict):
                    if not entry.get("generated_instructions") and (entry.get("image_link") or entry.get("image_path")):
                        pattern_name = entry.get("pattern_name", f"Pattern {idx}")
                        file_missing.append({"index": idx, "name": pattern_name})
                        total_missing += 1
            
            if file_missing:
                missing_patterns[json_file] = file_missing
                print(f"{json_file}: {len(file_missing)} patterns missing instructions out of {len(entries)}")
        
        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")
    
    print(f"Total: {total_missing} patterns missing instructions out of {total_patterns} patterns")
    
    # Save to file
    with open(missing_patterns_file, "w") as f:
        json.dump(missing_patterns, f, indent=2)
        
    return missing_patterns

def process_missing_patterns(specific_file=None):
    """Process only patterns that are missing instructions"""
    # First, find which patterns are missing instructions
    if os.path.exists(missing_patterns_file):
        with open(missing_patterns_file, "r") as f:
            try:
                missing_patterns = json.load(f)
                print(f"Loaded existing missing patterns file with {sum(len(v) for v in missing_patterns.values())} patterns to process")
            except json.JSONDecodeError:
                print("Error loading missing patterns file. Generating new one...")
                missing_patterns = find_missing_instructions(specific_file)
    else:
        missing_patterns = find_missing_instructions(specific_file)
    
    # Filter by specific file if requested
    if specific_file and specific_file in missing_patterns:
        missing_patterns = {specific_file: missing_patterns[specific_file]}
    
    # Load progress
    progress = load_progress()
    
    # Process each file with missing patterns
    for json_file, missing_entries in missing_patterns.items():
        print(f"Processing {len(missing_entries)} missing patterns in {json_file}")
        
        # Initialize progress for this file if needed
        json_key = json_file
        if json_key not in progress:
            progress[json_key] = {"completed": False, "processed_entries": {}}
        
        json_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        # Load the output file
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                try:
                    data = json.load(f)
                    print(f"Loaded existing output file: {output_path}")
                except json.JSONDecodeError:
                    print(f"Error loading existing output file: {output_path}. Loading original instead.")
                    with open(json_path, "r") as orig_f:
                        data = json.load(orig_f)
        else:
            with open(json_path, "r") as f:
                data = json.load(f)
        
        # Convert to list if data is a dictionary
        if isinstance(data, dict):
            # If the dictionary has a list of entries, use that
            if any(isinstance(v, list) for v in data.values()):
                for key, value in data.items():
                    if isinstance(value, list):
                        entries = value
                        break
            else:
                # Otherwise treat the dictionary as a single entry
                entries = [data]
        else:
            # It's already a list
            entries = data
        
        # Process only the missing entries
        for missing_entry in tqdm(missing_entries, desc=f"{json_file} missing patterns", leave=False):
            idx = missing_entry["index"]
            entry_id = str(idx)
            entry = entries[idx]
            
            # Skip if already processed in this run
            if entry_id in progress[json_key]["processed_entries"]:
                print(f"Skipping processed entry: {idx}")
                continue
            
            # Double-check if entry still needs processing
            if isinstance(entry, dict) and entry.get("generated_instructions"):
                progress[json_key]["processed_entries"][entry_id] = True
                continue
                
            if not isinstance(entry, dict):
                print(f"Skipping non-dictionary entry at index {idx}")
                continue
                
            image_source = entry.get("image_link") or entry.get("image_path")
            if not image_source:
                print(f"No image source found for entry {idx}")
                continue
            
            try:
                # Load and encode image
                base64_image = ""
                try:
                    base64_image = encode_image_to_base64(image_source)
                except Exception as img_err:
                    print(f"Error processing image {image_source}: {img_err}")
                    continue
                    
                # Create the message with the image
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Generate step-by-step crochet instructions for this image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
                
                # Prepare API payload
                payload = {
                    "model": "gpt-4o",  # Updated to the latest model with vision capabilities
                    "messages": messages,
                    "max_tokens": 1024
                }
                
                # Call the API with retry logic
                print(f"Processing {json_file} entry {idx} - {entry.get('pattern_name', 'Unnamed')}")
                response_data = call_api_with_retry(payload)
                
                # Parse the response
                generated_instructions = response_data["choices"][0]["message"]["content"]
                
                # Save the instructions to the entry
                entry["generated_instructions"] = generated_instructions
                print(f"Generated instructions for {json_file} entry {idx}")
                
                # Mark as processed
                progress[json_key]["processed_entries"][entry_id] = True
                save_progress(progress)
                
                # Save updated JSON after each successful entry
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Add a random delay between API calls (0.5-2 seconds)
                time.sleep(0.5 + 1.5 * random.random())
                
            except Exception as e:
                print(f"Error processing {json_file} [{idx}]: {e}")
                # Save progress even on error
                save_progress(progress)
                continue
        
        # Mark file as completed if all patterns have been processed
        all_processed = True
        for idx, entry in enumerate(entries):
            if isinstance(entry, dict) and not entry.get("generated_instructions") and (entry.get("image_link") or entry.get("image_path")):
                all_processed = False
                break
        
        if all_processed:
            progress[json_key]["completed"] = True
            print(f"File {json_file} is now complete!")
        
        save_progress(progress)
    
    print(f"Finished processing missing patterns. Results saved to {output_dir}")
    
    # Generate updated missing patterns file
    find_missing_instructions(specific_file)

def main():
    """Main function with command line argument handling"""
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Just analyze and find missing patterns
        if len(sys.argv) > 2:
            find_missing_instructions(sys.argv[2])
        else:
            find_missing_instructions()
    elif len(sys.argv) > 1 and sys.argv[1] == "--force":
        # Force regeneration of missing patterns file and process
        print("Forcing regeneration of missing patterns file...")
        if os.path.exists(missing_patterns_file):
            os.remove(missing_patterns_file)
        
        # Get specific file if provided
        specific_file = None
        if len(sys.argv) > 2:
            specific_file = sys.argv[2]
            print(f"Processing only file: {specific_file}")
            
        # Find and process missing patterns
        missing_patterns = find_missing_instructions(specific_file)
        process_missing_patterns(specific_file)
    else:
        # Process missing patterns
        specific_file = None
        if len(sys.argv) > 1:
            specific_file = sys.argv[1]
            print(f"Processing only file: {specific_file}")
        
        # Always regenerate the missing patterns file
        print("Regenerating missing patterns file to ensure up-to-date processing...")
        missing_patterns = find_missing_instructions(specific_file)
        process_missing_patterns(specific_file)

if __name__ == "__main__":
    main()
