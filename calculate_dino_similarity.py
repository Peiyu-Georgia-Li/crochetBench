#!/usr/bin/env python3
"""
Calculate DINO similarity between product images and rendered images.
Processes images from valid_*_img directories and compares them with product images
from selected_crochet_patterns_3.json.
"""

import json
import argparse
from pathlib import Path
import csv
from datetime import datetime
import traceback
from typing import Dict, List, Optional

import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cosine

# DINO
import timm


def load_dino_model(device):
    """Load DINO model."""
    print("Loading DINO (ViT-Small)...")
    dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    dino_model = dino_model.to(device)
    dino_model.eval()
    print(f"DINO model loaded on {device}")
    return dino_model


def load_image_from_url(url):
    """Download and load image from URL."""
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    image = Image.open(response.raw).convert("RGB")
    return image


def load_image_from_path(path):
    """Load image from local path."""
    image = Image.open(path).convert("RGB")
    return image


def preprocess_for_dino(image, device):
    """Preprocess image for DINO."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def compute_dino_similarity(image1, image2, dino_model, device):
    """
    Compute DINO feature similarity.
    Uses CLS token from ViT for global semantic representation.
    """
    img1_tensor = preprocess_for_dino(image1, device)
    img2_tensor = preprocess_for_dino(image2, device)
    
    with torch.no_grad():
        # Get features from DINO
        feat1 = dino_model.forward_features(img1_tensor)
        feat2 = dino_model.forward_features(img2_tensor)
        
        # Use CLS token (first token)
        feat1 = feat1[:, 0, :]
        feat2 = feat2[:, 0, :]
        
        # Normalize
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
    
    return similarity.item()


def load_patterns_data(json_path):
    """Load pattern data from JSON file."""
    with open(json_path, 'r') as f:
        patterns = json.load(f)
    return patterns


def find_rendered_images(img_dir: Path) -> Dict[str, Path]:
    """
    Find all rendered images in the directory.
    Returns dict mapping pattern_name -> image_path
    """
    rendered_images = {}
    
    if not img_dir.exists():
        print(f"Warning: Directory {img_dir} does not exist")
        return rendered_images
    
    for img_file in img_dir.glob("*.png"):
        # Extract pattern name from filename (remove _full.png or _last_valid.png suffix)
        pattern_name = img_file.stem
        if pattern_name.endswith('_full'):
            pattern_name = pattern_name[:-5]
        elif pattern_name.endswith('_last_valid'):
            pattern_name = pattern_name[:-11]
        
        rendered_images[pattern_name] = img_file
    
    return rendered_images


def process_model_directory(model_name: str, img_dir: Path, patterns: List[Dict], 
                           dino_model, device) -> List[Dict]:
    """
    Process all images in a model's directory and compute DINO similarities.
    """
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Image directory: {img_dir}")
    print(f"{'='*80}")
    
    rendered_images = find_rendered_images(img_dir)
    print(f"Found {len(rendered_images)} rendered images")
    
    results = []
    
    # Create pattern lookup by name
    pattern_lookup = {}
    for pattern in patterns:
        # Normalize pattern name for matching
        name = pattern['pattern_name'].replace(' ', '_').replace('/', '_')
        pattern_lookup[name] = pattern
    
    successful = 0
    failed = 0
    
    for pattern_name, rendered_path in rendered_images.items():
        # Find matching pattern in JSON
        matching_pattern = None
        for norm_name, pattern in pattern_lookup.items():
            if norm_name.upper() == pattern_name.upper():
                matching_pattern = pattern
                break
        
        if not matching_pattern:
            print(f"⚠️  No matching pattern found for: {pattern_name}")
            failed += 1
            continue
        
        try:
            # Load product image from URL
            product_url = matching_pattern['image_link']
            print(f"\n📊 Processing: {pattern_name}")
            print(f"   Product URL: {product_url}")
            print(f"   Rendered: {rendered_path.name}")
            
            product_img = load_image_from_url(product_url)
            rendered_img = load_image_from_path(rendered_path)
            
            # Compute DINO similarity
            dino_sim = compute_dino_similarity(product_img, rendered_img, dino_model, device)
            
            print(f"   ✅ DINO Similarity: {dino_sim:.4f}")
            
            results.append({
                'model': model_name,
                'pattern_name': matching_pattern['pattern_name'],
                'pattern_id': matching_pattern.get('id', 'N/A'),
                'project_type': matching_pattern.get('project_type', 'N/A'),
                'dino_similarity': dino_sim,
                'product_url': product_url,
                'rendered_path': str(rendered_path),
            })
            
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {pattern_name}: {str(e)}")
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"Model {model_name} Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total: {len(rendered_images)}")
    print(f"{'='*80}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return
    
    fieldnames = ['model', 'pattern_name', 'pattern_id', 'project_type', 
                  'dino_similarity', 'product_url', 'rendered_path']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to: {output_path}")


def compute_statistics(results: List[Dict]):
    """Compute and display statistics."""
    if not results:
        print("No results to analyze")
        return
    
    # Group by model
    by_model = {}
    for result in results:
        model = result['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(result['dino_similarity'])
    
    print(f"\n{'='*80}")
    print("DINO Similarity Statistics")
    print(f"{'='*80}")
    
    for model, similarities in by_model.items():
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        print(f"\n{model}:")
        print(f"  Average: {avg_sim:.4f}")
        print(f"  Std Dev: {std_sim:.4f}")
        print(f"  Min:     {min_sim:.4f}")
        print(f"  Max:     {max_sim:.4f}")
        print(f"  Count:   {len(similarities)}")
    
    # Overall statistics
    all_similarities = [r['dino_similarity'] for r in results]
    print(f"\nOverall:")
    print(f"  Average: {np.mean(all_similarities):.4f}")
    print(f"  Std Dev: {np.std(all_similarities):.4f}")
    print(f"  Min:     {np.min(all_similarities):.4f}")
    print(f"  Max:     {np.max(all_similarities):.4f}")
    print(f"  Count:   {len(all_similarities)}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate DINO similarity between product and rendered images"
    )
    parser.add_argument(
        '--patterns-json',
        type=str,
        default='/store01/nchawla/pli9/crochet_origin/split_dataset/test.json',
        help='Path to patterns JSON file'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/store01/nchawla/pli9/crochet_origin/crochet',
        help='Base directory containing valid_*_img folders'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['qwen'],
        help='Model names to process (e.g., qwen claude gemini)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dino_similarity_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load DINO model
    dino_model = load_dino_model(device)
    
    # Load patterns data
    patterns_path = Path(args.patterns_json)
    if not patterns_path.exists():
        print(f"Error: Patterns file not found: {patterns_path}")
        return
    
    patterns = load_patterns_data(patterns_path)
    print(f"Loaded {len(patterns)} patterns from {patterns_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process each model
    all_results = []
    base_dir = Path(args.base_dir)
    
    for model_name in args.models:
        img_dir = base_dir / f"valid_{model_name}_img"
        
        if not img_dir.exists():
            print(f"\n⚠️  Warning: Directory not found: {img_dir}")
            continue
        
        model_results = process_model_directory(
            model_name, img_dir, patterns, dino_model, device
        )
        all_results.extend(model_results)
    
    # Save results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"dino_similarity_{timestamp}.csv"
        save_results_to_csv(all_results, output_file)
        
        # Compute and display statistics
        compute_statistics(all_results)
    else:
        print("\n⚠️  No results generated")


if __name__ == "__main__":
    main()
