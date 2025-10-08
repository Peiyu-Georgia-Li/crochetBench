#!/usr/bin/env python3
"""
Test GPT-4o-mini model on multiple-choice dataset (mc_dataset.json).
Uses direct option selection (A/B/C/D) for evaluation.
Implements rate limiting and exponential backoff for API calls.
"""

import os
import json
import base64
import argparse
import requests
import time
import random
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Configure argument parser
parser = argparse.ArgumentParser(description='Test GPT-4o-mini model on multiple-choice dataset')
parser.add_argument('--input_file', type=str, default='../mc_dataset.json',
                    help='Input multiple-choice dataset file')
parser.add_argument('--output_file', type=str, default='gpt4omini_mc_results.json',
                    help='Output results file')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (None for all)')
args = parser.parse_args()

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

def load_failed_sample_ids(failed_samples_path="/store01/nchawla/pli9/crochet/crochet/failed_samples.txt"):
    """Load the list of failed sample IDs from failed_samples.txt"""
    try:
        with open(failed_samples_path, 'r', encoding='utf-8') as f:
            failed_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(failed_ids)} failed sample IDs")
        return failed_ids
    except FileNotFoundError:
        print(f"Failed samples file not found at {failed_samples_path}")
        return set()

def load_dataset(file_path, max_samples=None, only_failed=True):
    """Load the multiple-choice dataset from a JSON file."""
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    if only_failed:
        # Filter to include only failed samples from previous run
        failed_ids = load_failed_sample_ids()
        if failed_ids:
            dataset = [sample for sample in full_dataset if sample['id'] in failed_ids]
            print(f"Filtered dataset to {len(dataset)} failed samples out of {len(full_dataset)} total")
        else:
            print("No failed sample IDs found, using full dataset")
            dataset = full_dataset
    else:
        dataset = full_dataset
    
    if max_samples is not None:
        if max_samples == 0:  # 0 means all samples
            pass
        else:
            dataset = dataset[:max_samples]
    
    print(f"Final dataset size: {len(dataset)} samples")
    return dataset

def encode_image_to_base64(image_path_or_url):
    """Convert image to base64 encoding for API submission"""
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url, timeout=10)
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

def call_api_with_backoff(payload, max_retries=5, base_delay=2):
    """
    Make an API call with exponential backoff for rate limiting.
    Adds jitter to avoid thundering herd problem.
    """
    retry = 0
    while retry <= max_retries:
        try:
            response = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=60)
            
            # Handle rate limit errors
            if response.status_code == 429:
                error_data = response.json().get('error', {})
                wait_time = float(error_data.get('message', '').split('try again in ')[1].split('s.')[0]) if 'try again in' in error_data.get('message', '') else None
                
                if wait_time is None:  # If wait time not specified in the error
                    wait_time = (2 ** retry) * base_delay + random.uniform(0, 1)
                else:
                    # Add a buffer to the suggested wait time
                    wait_time += 0.5
                
                print(f"Rate limited, waiting {wait_time:.2f} seconds (retry {retry+1}/{max_retries})...")
                time.sleep(wait_time)
                retry += 1
                continue
            
            return response
            
        except requests.RequestException as e:
            # For network errors, also use backoff
            wait_time = (2 ** retry) * base_delay + random.uniform(0, 1)
            print(f"Request error: {e}. Retrying in {wait_time:.2f} seconds (retry {retry+1}/{max_retries})")
            time.sleep(wait_time)
            retry += 1
            
    # If we've exhausted all retries
    raise Exception(f"Failed after {max_retries} retries due to rate limiting or connection issues")

def process_sample(sample):
    """Process a single sample from the dataset using GPT-4o-mini."""
    try:
        # Format options for the prompt
        options_text = "\n".join([f"({opt['label']}) {opt['instructions']}" for opt in sample['options']])
        
        # Load and encode image
        base64_image = ""
        try:
            base64_image = encode_image_to_base64(sample['image_url']) 
        except Exception as img_err:
            print(f"Error processing image {sample['image_url']}: {img_err}")
            return {
                'id': sample['id'],
                'error': f"Image loading error: {str(img_err)}",
                'is_correct': False,
                'success': False
            }
        
        # Create the system prompt
        system_prompt = """You are a crochet expert. Your task is to determine which option (A, B, C, or D) contains the correct crochet instructions for creating the item shown in the image."""
        
        # Create the message with the image and options
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Look at this crochet image and choose which option best matches the instructions for making it.

Options:
{options_text}

Choose exactly ONE option. Your answer must be only one letter: A, B, C, or D."""
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
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 20,
            "temperature": 0.0  # Deterministic output
        }
        
        # Call the API with exponential backoff for rate limiting
        try:
            response = call_api_with_backoff(payload)
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return {
                    'id': sample['id'],
                    'error': f"API error: {response.status_code} - {response.text}",
                    'is_correct': False,
                    'success': False
                }
        except Exception as api_err:
            print(f"API call failed: {api_err}")
            return {
                'id': sample['id'],
                'error': f"API call failed: {str(api_err)}",
                'is_correct': False,
                'success': False
            }
        
        # Parse the response
        response_data = response.json()
        answer = response_data["choices"][0]["message"]["content"].strip()
        
        # Normalize the answer
        if answer in ["A", "B", "C", "D"]:
            chosen_label = answer
        elif "a" in answer.lower():
            chosen_label = "A"
        elif "b" in answer.lower():
            chosen_label = "B"
        elif "c" in answer.lower():
            chosen_label = "C"
        elif "d" in answer.lower():
            chosen_label = "D"
        else:
            chosen_label = None
        
        # Check if the chosen option is correct
        is_correct = chosen_label == sample['correct_label'] if chosen_label else False
        
        return {
            'id': sample['id'],
            'pattern_name': sample.get('pattern_name', ''),
            'project_type': sample.get('project_type', ''),
            'image_url': sample['image_url'],
            'correct_label': sample['correct_label'],
            'model_choice_label': chosen_label,
            'raw_response': answer,
            'is_correct': is_correct,
            'success': chosen_label is not None
        }
        
    except Exception as e:
        import traceback
        print(f"Error processing sample {sample['id']}: {e}")
        print(traceback.format_exc())
        return {
            'id': sample['id'],
            'error': str(e),
            'is_correct': False,
            'success': False
        }

def main():
    # Load only the failed samples from the previous run
    dataset = load_dataset(args.input_file, args.max_samples, only_failed=True)
    
    # Process samples
    results = []
    correct_count = 0
    total_count = 0
    
    print("Processing samples...")
    for i, sample in enumerate(tqdm(dataset)):
        # Add a small delay between API calls (helps with rate limiting)
        if i > 0:
            # Base delay of 1 second + small random jitter
            delay = 1 + random.uniform(0, 0.5)  
            time.sleep(delay)
            
        result = process_sample(sample)
        results.append(result)
        
        if result.get('success', False):
            total_count += 1
            if result.get('is_correct', False):
                correct_count += 1
                
        # Save progress periodically (every 10 samples)
        if (i + 1) % 10 == 0:
            # Create intermediate summary
            intermediate_accuracy = correct_count / total_count if total_count > 0 else 0
            intermediate_summary = {
                'total_processed_so_far': total_count,
                'correct_count_so_far': correct_count,
                'accuracy_so_far': intermediate_accuracy,
                'results': results
            }
            
            # Save intermediate results
            temp_output_file = args.output_file.replace('.json', f'_progress_{i+1}.json')
            print(f"\nSaving intermediate results to {temp_output_file}...")
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_summary, f, indent=2)
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Create summary
    summary = {
        'total_processed': total_count,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'results': results
    }
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    # Save successfully processed samples for future reference
    successful_ids = [item['id'] for item in results if item.get('success', False)]
    if successful_ids:
        with open("/store01/nchawla/pli9/crochet/crochet/successful_samples.txt", "w") as f:
            f.write("\n".join(successful_ids))
        print(f"Saved {len(successful_ids)} successful sample IDs")
    
    # Print summary
    print(f"Evaluation completed:")
    print(f"  Total samples processed: {total_count}")
    print(f"  Correct predictions: {correct_count}")
    print(f"  Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    main()
