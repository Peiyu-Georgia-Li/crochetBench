#!/usr/bin/env python3
"""
Test DeepSeek VL model on multiple-choice dataset (mc_dataset.json).
Uses direct option selection (A/B/C/D) for evaluation.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from PIL import Image
import requests
from io import BytesIO

# Configure argument parser
parser = argparse.ArgumentParser(description='Test DeepSeek VL model on multiple-choice dataset')
parser.add_argument('--input_file', type=str, default='../mc_dataset.json',
                    help='Input multiple-choice dataset file')
parser.add_argument('--output_file', type=str, default='dsvl_mc_results.json',
                    help='Output results file')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to evaluate (None for all)')
parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-vl-7b-chat',
                    help='DeepSeek VL model name')
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

def process_sample(sample, vl_chat_processor, vl_model, tokenizer):
    """Process a single sample from the dataset using DeepSeek VL."""
    try:
        # Load image from URL
        response = requests.get(sample['image_url'], timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Format options for the prompt
        options_text = "\n".join([f"({opt['label']}) {opt['instructions'][:300]}..." for opt in sample['options']])
        
        # Create the prompt
        system_prompt = """You are a crochet expert. Your task is to determine which of the given options (A, B, C, or D) contains the correct crochet instructions for the image shown."""
        
        user_content = f"""Look at this crochet image and choose which option best matches the instructions for making it.

Options:
{options_text}

Choose exactly ONE option. Your answer should be only one letter: A, B, C, or D."""

        # Prepare conversation
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "User",
                "content": "<image_placeholder>" + user_content,
                "images": [image]  # Pass the actual image object
            },
            {"role": "Assistant", "content": ""}
        ]
        
        # Process inputs
        pil_images = [image]
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_model.device)
        
        # Prepare embeddings
        inputs_embeds = vl_model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response
        outputs = vl_model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=20,  # Only need a short response
            do_sample=False,
            use_cache=True
        )
        
        # Decode response
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
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
    print(f"Loading model {args.model_name}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_name)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_model = MultiModalityCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    vl_model = vl_model.to(torch.bfloat16).cuda().eval()
    print(f"Model loaded on {vl_model.device}")
    
    # Process samples
    results = []
    correct_count = 0
    total_count = 0
    
    print("Processing samples...")
    for sample in tqdm(dataset):
        result = process_sample(sample, vl_chat_processor, vl_model, tokenizer)
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
