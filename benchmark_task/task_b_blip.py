#!/usr/bin/env python3
"""
Simple test script for BLIP2-FLAN-T5-XL model on multiple-choice dataset.
Uses direct option selection (A/B/C/D) for evaluation.
"""

import json
import os
import argparse
import torch
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Configure argument parser
parser = argparse.ArgumentParser(description='Test BLIP2-FLAN-T5-XL model on multiple-choice dataset')
parser.add_argument('--input_file', type=str, default='../data/mc_dataset.json',
                    help='Input multiple-choice dataset file')
parser.add_argument('--output_file', type=str, default='task_b_blip.json',
                    help='Output results file')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (None for all)')
args = parser.parse_args()

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

def process_sample(sample, processor, model, device):
    """Process a single sample from the dataset."""
    try:
        # Load image from URL
        response = requests.get(sample['image_url'], timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Format options for the prompt
        options_text = "\n".join([f"({opt['label']}) {opt['instructions']}" for opt in sample['options']])
        
        # Create the prompt
        prompt = f"""You are a crochet expert.

Question: Which of the following options best matches the crochet instructions for the product image?

Options:
{options_text}

Rules:
- Choose exactly ONE option.
- Output only: A, B, C, or D.

Answer:"""
        
        # Process image and text
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False
            )
        
        # Decode the answer
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        
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
    # Load the dataset
    dataset = load_dataset(args.input_file, args.max_samples)
    
    # Initialize model
    print("Loading model Salesforce/blip2-flan-t5-xl...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    device = model.device
    print(f"Model loaded on {device}")
    
    # Process samples
    results = []
    correct_count = 0
    total_count = 0
    
    print("Processing samples...")
    for sample in tqdm(dataset):
        result = process_sample(sample, processor, model, device)
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
