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
You are a crochet stitch expert. Identify which stitches appear in a crochet product image. 
Always answer using standard U.S. crochet abbreviations only (sc, hdc, dc, tr, ch, sl st, pop, etc.). 
Output only a comma-separated list of abbreviations—no explanations or extra commentary."""

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

# Default model settings (can be overridden by command line arguments)
SELECTED_MODEL = "gpt-4o"  # Default model

# --- Directories ---
json_dir = "../data/crochet_pattern_by_project/"  # input JSONs
output_dir = "./task_a_gpt4o/"  # output directory
os.makedirs(output_dir, exist_ok=True)

# Progress tracking file
progress_file = os.path.join(output_dir, "abbreviation_progress.json")

# File for tracking missing patterns
missing_patterns_file = os.path.join(output_dir, "missing_abbreviations.json")

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

def call_api_with_retry(payload, max_retries=15, initial_delay=10):
    """Call the API with adaptive exponential backoff retry logic"""
    delay = initial_delay
    base_jitter_factor = 0.5  # For randomizing wait times
    
    # Keep track of consecutive rate limit errors
    consecutive_rate_limits = 0
    max_consecutive_rate_limits = 3
    
    for retry in range(max_retries):
        try:
            # Apply jitter to prevent synchronized retries
            jitter = random.uniform(0.8, 1.2) * delay
            
            if retry > 0:
                # For longer waits, provide more feedback
                if jitter > 60:
                    minutes = int(jitter // 60)
                    seconds = int(jitter % 60)
                    print(f"Attempt {retry+1}/{max_retries}: Waiting {minutes}m {seconds}s before retry...")
                else:
                    print(f"Attempt {retry+1}/{max_retries}: Waiting {jitter:.1f}s before retry...")
                    
                time.sleep(jitter)
                
            response = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=90)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                consecutive_rate_limits += 1
                
                # Extract retry-after header or use exponential backoff
                retry_after = response.headers.get("Retry-After")
                base_wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
                
                # More aggressive backoff for consecutive rate limits
                if consecutive_rate_limits >= max_consecutive_rate_limits:
                    print(f"Too many consecutive rate limits ({consecutive_rate_limits}). Adding extra backoff.")
                    # Add extra time proportional to consecutive failures
                    extra_time = 30 * consecutive_rate_limits
                    base_wait += extra_time
                    
                # Add randomized buffer based on retry number
                buffer = random.uniform(5, 15) + (retry * 10)
                wait_time = base_wait + buffer
                
                # Format wait time for better readability
                if wait_time > 60:
                    minutes = int(wait_time // 60)
                    seconds = int(wait_time % 60)
                    print(f"Rate limited. Waiting {minutes}m {seconds}s before retry {retry+1}/{max_retries}...")
                else:
                    print(f"Rate limited. Waiting {wait_time:.1f}s before retry {retry+1}/{max_retries}...")
                    
                time.sleep(wait_time)
                delay = min(delay * 2.5, 600)  # More aggressive backoff, capped at 10 minutes
                continue
            else:
                # Reset consecutive rate limits counter on non-429 response
                consecutive_rate_limits = 0
                
            # Handle other common API errors
            if response.status_code != 200:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                
                if response.status_code >= 500:
                    # Server error, retry with backoff
                    wait_time = delay + random.uniform(2, 8)
                    print(f"Server error, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    delay *= 2
                    continue
                else:
                    # Client error, handle specific cases
                    if response.status_code == 400:
                        try:
                            error_data = response.json() if response.text else {}
                            if error_data.get('error', {}).get('code') in ['invalid_api_key', 'invalid_organization']:
                                raise Exception(f"Authentication error: {error_message}")
                        except (ValueError, KeyError):
                            pass  # JSON parsing error, continue with normal retry logic
                            
                    # For other client errors, we can still retry with less aggressive backoff
                    if retry < max_retries - 1:
                        wait_time = delay + random.uniform(2, 5)
                        print(f"Client error, retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        delay *= 1.5  # Less aggressive for non-rate limit errors
                        continue
                    else:
                        raise Exception(error_message)
            
            # Success case
            return response.json()
            
        except (requests.RequestException, TimeoutError) as e:
            print(f"Network/timeout error: {e}")
            # Add jitter to backoff time
            wait_time = delay * random.uniform(0.8, 1.2)
            if wait_time > 60:
                minutes = int(wait_time // 60)
                seconds = int(wait_time % 60)
                print(f"Retrying in {minutes}m {seconds}s...")
            else:
                print(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            delay = min(delay * 2, 300)  # Cap at 5 minutes
            
    raise Exception(f"Failed after {max_retries} retries. Consider manually restarting with the --resume flag.")

def find_missing_abbreviations(specific_file=None):
    """Find patterns that don't have generated abbreviations"""
    missing_patterns = {}
    
    # List JSON files or use the specific file if provided
    if specific_file:
        json_files = [specific_file]
    else:
        # First check if output directory has any JSON files
        output_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and f != "abbreviation_progress.json" and f != "missing_abbreviations.json"]
        
        # If output directory is empty, use files from text_json directory
        if not output_files:
            print("No files found in output directory. Using source files from text_json directory...")
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        else:
            json_files = output_files
    
    total_missing = 0
    total_patterns = 0
    
    for json_file in tqdm(json_files, desc="Analyzing files for missing abbreviations"):
        output_path = os.path.join(output_dir, json_file)
        original_path = os.path.join(json_dir, json_file)
        
        # If the output file doesn't exist, use the original file as both source and target
        if not os.path.exists(output_path):
            print(f"Creating new output file for {json_file} based on the source file")
            # Load the original file as our starting point
            if os.path.exists(original_path):
                with open(original_path, "r") as f:
                    data = json.load(f)
                # Initialize the output file by copying the original
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)
            
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
            
            # Collect entries without generated abbreviations
            file_missing = []
            for idx, entry in enumerate(entries):
                total_patterns += 1
                
                if isinstance(entry, dict):
                    # Check if entry has image link but no generated instructions
                    if (entry.get("image_link") or entry.get("image_path")) and not entry.get("generated_instructions"):
                        pattern_name = entry.get("pattern_name", f"Pattern {idx}")
                        file_missing.append({"index": idx, "name": pattern_name})
                        total_missing += 1
            
            if file_missing:
                missing_patterns[json_file] = file_missing
                print(f"{json_file}: {len(file_missing)} patterns missing abbreviations out of {len(entries)}")
        
        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")
    
    print(f"Total: {total_missing} patterns missing abbreviations out of {total_patterns} patterns")
    
    # Save to file
    with open(missing_patterns_file, "w") as f:
        json.dump(missing_patterns, f, indent=2)
        
    return missing_patterns

def process_missing_abbreviations(specific_file=None):
    """Process only patterns that are missing abbreviations"""
    # First, find which patterns are missing abbreviations
    if os.path.exists(missing_patterns_file):
        with open(missing_patterns_file, "r") as f:
            try:
                missing_patterns = json.load(f)
                print(f"Loaded existing missing patterns file with {sum(len(v) for v in missing_patterns.values())} patterns to process")
            except json.JSONDecodeError:
                print("Error loading missing patterns file. Generating new one...")
                missing_patterns = find_missing_abbreviations(specific_file)
    else:
        missing_patterns = find_missing_abbreviations(specific_file)
    
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
                                "text": "Look at this crochet product image and list the stitches used."
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
                    "model": SELECTED_MODEL,  # Use the model selected via command line
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.2,
                    # Add unique request ID to help with debugging
                    "user": f"crochet-abbreviation-{json_file}-{idx}-{time.time()}"
                }
                
                # Call the API with retry logic
                print(f"Processing {json_file} entry {idx} - {entry.get('pattern_name', 'Unnamed')}")
                response_data = call_api_with_retry(payload)
                
                # Parse the response
                generated_instructions = response_data["choices"][0]["message"]["content"]
                
                # Save the instructions to the entry
                entry["generated_instructions"] = generated_instructions
                print(f"Generated abbreviations for {json_file} entry {idx}")
                print(generated_instructions)
                
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
    find_missing_abbreviations(specific_file)

def main():
    """Main function with command line argument handling"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process crochet pattern images with GPT-4V to extract abbreviations.')
    parser.add_argument('file', nargs='?', help='Specific JSON file to process')
    parser.add_argument('--analyze', action='store_true', help='Only analyze and find patterns missing abbreviations')
    parser.add_argument('--force', action='store_true', help='Force regeneration of missing patterns file')
    parser.add_argument('--model',default='gpt-4o',
                        help='OpenAI model to use for vision processing (default: gpt-4o)')
    parser.add_argument('--resume', action='store_true', help='Resume processing from last checkpoint without regenerating missing files')
    
    args = parser.parse_args()
    
    # No need for cooldown delays with new rate limiting approach
    
    # Create a global for the selected model
    global SELECTED_MODEL
    SELECTED_MODEL = args.model
    print(f"Using OpenAI model: {SELECTED_MODEL}")
    
    if args.analyze:
        # Just analyze and find missing patterns
        find_missing_abbreviations(args.file)
    elif args.force:
        # Force regeneration of missing patterns file and process
        print("Forcing regeneration of missing abbreviations file...")
        if os.path.exists(missing_patterns_file):
            os.remove(missing_patterns_file)
        
        # Find and process missing patterns
        missing_patterns = find_missing_abbreviations(args.file)
        process_missing_abbreviations(args.file)
    elif args.resume:
        # Skip regenerating missing patterns file
        print("Resuming from last checkpoint...")
        process_missing_abbreviations(args.file)
    else:
        # Process missing patterns with default behavior
        specific_file = args.file
        if specific_file:
            print(f"Processing only file: {specific_file}")
        
        # Regenerate the missing patterns file unless resuming
        print("Regenerating missing abbreviations file to ensure up-to-date processing...")
        missing_patterns = find_missing_abbreviations(specific_file)
        process_missing_abbreviations(specific_file)

if __name__ == "__main__":
    main()
