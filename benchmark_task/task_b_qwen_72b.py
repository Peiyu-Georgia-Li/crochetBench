#!/usr/bin/env python3
"""
Test Qwen2-VL model on multiple-choice dataset (mc_dataset.json).
Uses direct option selection (A/B/C/D) for evaluation.
"""

import os
import json
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import traceback

# Configure argument parser
parser = argparse.ArgumentParser(description='Test Qwen2-VL model on multiple-choice dataset')
parser.add_argument('--input_file', type=str, default='../data/mc_data.json',
                    help='Input multiple-choice dataset file')
parser.add_argument('--output_file', type=str, default='task_b_qwen_72b.json',
                    help='Output results file')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (None for all)')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-72B-Instruct',
                    help='Qwen model name')
args = parser.parse_args()

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def process_sample(sample, model, processor):
    """Process a single sample from the dataset using Qwen2-VL."""
    try:
        # Load image from URL
        image = load_image_from_source(sample['image_url'])
        
        # Format options for the prompt
        options_text = "\n".join([f"({opt['label']}) {opt['instructions']}" for opt in sample['options']])
        
        # Create the prompt
        system_prompt = """You are a crochet expert. Your task is to determine which option (A, B, C, or D) contains the correct instructions for creating the crochet item shown in the image."""
        
        # Prepare the conversation with system prompt and image
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"""Look at this crochet image and choose which option best matches the instructions for making it.

Options:
{options_text}

Choose exactly ONE option. Your answer must be only one letter: A, B, C, or D."""}
                ]
            }
        ]
        
        # Apply chat template to get formatted text prompt
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,  # Only need a short response
                do_sample=False
            )
        
        # Trim the prompt tokens before decoding
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
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
    
    # Initialize model
    print(f"Loading model {args.model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    print(f"Model loaded on {model.device}")
    
    # Process samples
    results = []
    correct_count = 0
    total_count = 0
    
    print("Processing samples...")
    for sample in tqdm(dataset):
        result = process_sample(sample, model, processor)
        results.append(result)
        
        if result.get('success', False):
            total_count += 1
            if result.get('is_correct', False):
                correct_count += 1
    
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
    
    # Print summary
    print(f"Evaluation completed:")
    print(f"  Total samples processed: {total_count}")
    print(f"  Correct predictions: {correct_count}")
    print(f"  Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    main()
