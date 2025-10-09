#!/usr/bin/env python3
"""
Test Gemini model on multiple-choice dataset (mc_dataset.json).
Uses direct option selection (A/B/C/D) for evaluation.
"""

import os
import json
import argparse
import re
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import traceback
from google import genai

# Configure argument parser
parser = argparse.ArgumentParser(description='Test Gemini model on multiple-choice dataset')
parser.add_argument('--input_file', type=str, default='../data/mc_dataset.json',
                    help='Input multiple-choice dataset file')
parser.add_argument('--output_file', type=str, default='task_b_gemini.json',
                    help='Output results file')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (None for all)')
parser.add_argument('--model', type=str, default="gemini-2.5-flash-lite",
                    help='Gemini model to use')
args = parser.parse_args()

# Initialize Gemini client
client = genai.Client()

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

def extract_boxed_answer(raw_response):
    """Extract the answer from $\\boxed{X}$ format in the raw response."""
    # Pattern to match $\boxed{X}$, where X is a single letter (A, B, C, or D)
    boxed_pattern = r'\$\\boxed\{([A-D])\}\$'
    match = re.search(boxed_pattern, raw_response)
    
    if match:
        return match.group(1)  # Return the letter inside the boxed format
    
    # If no boxed format found, return None
    return None

def process_sample(sample, model_name):
    """Process a single sample from the dataset using Gemini."""
    try:
        # Load image from URL
        image = load_image_from_source(sample['image_url'])
        
        # Format options for the prompt
        options_text = "\n".join([f"({opt['label']}) {opt['instructions']}" for opt in sample['options']])
        
        # Create the system prompt
        system_prompt = """You are a crochet expert. Your task is to determine which option (A, B, C, or D) contains the correct instructions for creating the crochet item shown in the image."""

        
        # Create the user prompt with options
        user_prompt = f"""Look at this crochet image and choose which option best matches the instructions for making it.

Options:
{options_text}

Choose exactly ONE option. Your answer must be only one letter: A, B, C, or D."""


        # Call Gemini API
        response = client.models.generate_content(
            model=model_name,
            contents=[system_prompt, image, user_prompt],
        )
        
        # Get the response text
        answer = response.text.strip()
        
        # Extract the answer using boxed format
        chosen_label = extract_boxed_answer(answer)
        
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
