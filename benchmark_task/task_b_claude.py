#!/usr/bin/env python3
"""
Test Claude model on multiple-choice dataset (mc_dataset.json).
Uses direct option selection (A/B/C/D) for evaluation.
"""

import os
import json
import argparse
import base64
import re
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import traceback
import anthropic

# Configure argument parser
parser = argparse.ArgumentParser(description='Test Claude model on multiple-choice dataset')
parser.add_argument('--input_file', type=str, default='../mc_dataset.json',
                    help='Input multiple-choice dataset file')
parser.add_argument('--output_file', type=str, default='claude_mc_results_300.json',
                    help='Output results file')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (None for all)')
parser.add_argument('--model', type=str, default="claude-sonnet-4-20250514",
                    help='Claude model to use')
args = parser.parse_args()

# Initialize Claude client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def load_dataset(file_path, max_samples=None):
    """Load the multiple-choice dataset from a JSON file."""
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples is not None:
        if max_samples == 0:  # 0 means all samples
            pass
        else:
            dataset = dataset[:max_samples]
    
    print(f"Loaded {len(dataset)} samples")
    return dataset

def load_image_from_source(image_source):
    """Load an image from URL or local path with enhanced error handling"""
    if not image_source:
        raise ValueError("Empty image source provided")
        
    if isinstance(image_source, str) and image_source.strip():
        image_source = image_source.strip()
        if image_source.startswith("http"):
            try:
                response = requests.get(image_source, timeout=10)
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch image: {response.status_code}")
                image = Image.open(BytesIO(response.content)).convert("RGB")
                # Validate image has valid dimensions
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Invalid image dimensions")
                return image
            except requests.RequestException as e:
                raise Exception(f"Request error: {str(e)}")
            except Exception as e:
                raise Exception(f"Error processing image from URL: {str(e)}")
        else:
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            try:
                image = Image.open(image_source).convert("RGB")
                # Validate image has valid dimensions
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Invalid image dimensions")
                return image
            except Exception as e:
                raise Exception(f"Error loading local image: {str(e)}")
    else:
        raise ValueError(f"Invalid image source: {type(image_source)}")

def image_to_base64(image):
    """Convert a PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def normalize_answer(raw_response):
    """Normalize the model response to extract the answer choice (A, B, C, or D)."""
    # Look for the last single character (A, B, C, or D) after double newline
    match = re.search(r'\n\n([A-D])\s*$', raw_response)
    if match:
        return match.group(1)
    
    # If no match with the double newline pattern, try to find the last standalone A, B, C, or D
    match = re.search(r'([A-D])\s*$', raw_response)
    if match:
        return match.group(1)
    
    # If still no match, return None
    return None

def process_sample(sample, model_name):
    """Process a single sample from the dataset using Claude."""
    try:
        # Load image from URL
        image = load_image_from_source(sample['image_url'])
        
        # Convert image to base64
        image_base64 = image_to_base64(image)
        
        # Format options for the prompt
        options_text = "\n".join([f"({opt['label']}) {opt['instructions'][:300]}..." for opt in sample['options']])
        
        # Create the system prompt
        system_prompt = """You are a crochet expert. Your task is to determine which option (A, B, C, or D) contains the correct instructions for creating the crochet item shown in the image."""
        
        # Create the user prompt
        user_text = f"""Look at this crochet image and choose which option best matches the instructions for making it.

Options:
{options_text}

Choose exactly ONE option. Your answer must be only one letter: A, B, C, or D."""

        # Call Claude API
        message = client.messages.create(
            model=model_name,
            max_tokens=800,  
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": user_text
                        }
                    ]
                }
            ]
        )
        
        # Get the response text
        answer = message.content[0].text.strip()
        print(answer)
        
        # Normalize the answer
        chosen_label = normalize_answer(answer)
        
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
        print(f"Error processing sample {sample['id']}: {e}")
        print(traceback.format_exc())
        return {
            'id': sample['id'],
            'error': str(e),
            'is_correct': False,
            'success': False
        }

def main():
    # Load the dataset
    dataset = load_dataset(args.input_file, args.max_samples)
    
    # Process samples
    results = []
    correct_count = 0
    total_count = 0
    
    print(f"Processing samples using {args.model}...")
    for sample in tqdm(dataset):
        result = process_sample(sample, args.model)
        results.append(result)
        
        if result.get('success', False):
            total_count += 1
            if result.get('is_correct', False):
                correct_count += 1
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Create summary
    summary = {
        'model': args.model,
        'total_processed': total_count,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'results': results
    }
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"Evaluation completed:")
    print(f"  Total samples processed: {total_count}")
    print(f"  Correct predictions: {correct_count}")
    print(f"  Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    main()
