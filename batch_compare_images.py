#!/usr/bin/env python3
"""
Batch processing script to compare crochet simulator renderings with product images.
Automatically discovers rendering images in *_img directories and matches them with
patterns in selected_crochet_patterns_3.json.

Results are saved in CSV files, separately for each model (claude, qwen, etc.).
"""

import json
import argparse
from pathlib import Path
import csv
from datetime import datetime
import traceback
import re
from typing import Dict, List, Optional

import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import cv2

# LPIPS
import lpips

# DINO
import timm

# SSIM
from skimage.metrics import structural_similarity as ssim

# CLIP
from transformers import CLIPProcessor, CLIPModel


def load_models(device):
    """Load all models needed for comparison."""
    print("Loading models...")
    
    # LPIPS model
    print("  - Loading LPIPS (VGG)...")
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    # DINO model
    print("  - Loading DINO (ViT-Small)...")
    dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    dino_model = dino_model.to(device)
    dino_model.eval()
    
    # CLIP model
    print("  - Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print(f"All models loaded on {device}\n")
    return {
        'lpips': lpips_model,
        'dino': dino_model,
        'clip': clip_model,
        'clip_processor': clip_processor,
    }


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


def preprocess_for_lpips(image, device):
    """Preprocess image for LPIPS (expects [-1, 1] range)."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)


def preprocess_for_dino(image, device):
    """Preprocess image for DINO."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def compute_lpips_similarity(image1, image2, lpips_model, device):
    """Compute LPIPS perceptual similarity."""
    img1_tensor = preprocess_for_lpips(image1, device)
    img2_tensor = preprocess_for_lpips(image2, device)
    
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor)
    
    similarity = 1 - distance.item()
    return similarity, distance.item()


def compute_dino_similarity(image1, image2, dino_model, device):
    """Compute DINO feature similarity."""
    img1_tensor = preprocess_for_dino(image1, device)
    img2_tensor = preprocess_for_dino(image2, device)
    
    with torch.no_grad():
        feat1 = dino_model.forward_features(img1_tensor)
        feat2 = dino_model.forward_features(img2_tensor)
        
        feat1 = feat1[:, 0, :]
        feat2 = feat2[:, 0, :]
        
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
    
    return similarity.item()


def compute_color_similarity(image1, image2):
    """Compute color distribution similarity."""
    img1_np = np.array(image1)
    img2_np = np.array(image2)
    
    target_size = (256, 256)
    img1_np = cv2.resize(img1_np, target_size)
    img2_np = cv2.resize(img2_np, target_size)
    
    correlations = []
    emds = []
    
    for i in range(3):
        hist1 = cv2.calcHist([img1_np], [i], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2_np], [i], None, [256], [0, 256]).flatten()
        
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        correlation = cv2.compareHist(hist1.astype(np.float32), 
                                     hist2.astype(np.float32), 
                                     cv2.HISTCMP_CORREL)
        emd = wasserstein_distance(hist1, hist2)
        
        correlations.append(correlation)
        emds.append(emd)
    
    avg_correlation = np.mean(correlations)
    avg_emd = np.mean(emds)
    emd_similarity = 1 - min(avg_emd, 1.0)
    
    return avg_correlation, emd_similarity, avg_emd


def compute_ssim_similarity(image1, image2):
    """Compute Structural Similarity Index (SSIM)."""
    img1_np = np.array(image1.resize((256, 256)))
    img2_np = np.array(image2.resize((256, 256)))
    
    ssim_scores = []
    for i in range(3):
        score = ssim(img1_np[:, :, i], img2_np[:, :, i], data_range=255)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)


def compute_clip_similarity(image1, image2, clip_model, clip_processor, device):
    """Compute CLIP image-to-image similarity."""
    inputs = clip_processor(images=[image1, image2], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarity = torch.nn.functional.cosine_similarity(
        image_features[0:1], image_features[1:2]
    )
    
    return similarity.item()


def compute_clip_text_image_similarity(image, text, clip_model, clip_processor, device):
    """
    Compute CLIP text-to-image similarity.
    
    Args:
        image: PIL Image
        text: Text string (e.g., product title or category)
        clip_model: CLIP model
        clip_processor: CLIP processor
        device: torch device
    
    Returns:
        Similarity score between 0 and 1
    """
    # Process image and text
    inputs = clip_processor(
        text=[text], 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get image and text features
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = clip_model.get_text_features(input_ids=inputs['input_ids'])
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    
    return similarity.item()


def compute_all_metrics(rendering_image, product_image, models, device):
    """
    Compute all similarity metrics for a pair of images.
    Returns a dictionary with all scores.
    """
    metrics = {}
    
    # LPIPS
    lpips_sim, lpips_dist = compute_lpips_similarity(
        rendering_image, product_image, models['lpips'], device
    )
    metrics['lpips_similarity'] = lpips_sim
    metrics['lpips_distance'] = lpips_dist
    
    # DINO
    dino_sim = compute_dino_similarity(
        rendering_image, product_image, models['dino'], device
    )
    metrics['dino_similarity'] = dino_sim
    
    # Color
    color_corr, color_emd_sim, color_emd = compute_color_similarity(
        rendering_image, product_image
    )
    metrics['color_correlation'] = color_corr
    metrics['color_emd_similarity'] = color_emd_sim
    metrics['color_emd_distance'] = color_emd
    
    # SSIM
    ssim_score = compute_ssim_similarity(rendering_image, product_image)
    metrics['ssim'] = ssim_score
    
    # CLIP
    clip_sim = compute_clip_similarity(
        rendering_image, product_image, 
        models['clip'], models['clip_processor'], device
    )
    metrics['clip_similarity'] = clip_sim
    
    # Aggregate score (weighted combination)
    aggregate = (
        0.30 * lpips_sim +
        0.30 * dino_sim +
        0.15 * color_corr +
        0.10 * ssim_score +
        0.15 * clip_sim
    )
    metrics['aggregate_score'] = aggregate
    
    return metrics


def normalize_pattern_name(name):
    """
    Normalize pattern name by keeping only alphanumeric characters and spaces.
    This helps match names with different punctuation (e.g., hyphens vs spaces).
    Example: "Two-Rectangle Sweater" -> "two rectangle sweater"
    """
    # Keep only alphanumeric characters and spaces
    normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', name)
    # Collapse multiple spaces into one
    normalized = re.sub(r'\s+', ' ', normalized)
    # Convert to lowercase and strip
    return normalized.lower().strip()


def find_pattern_in_json(patterns_list, pattern_name):
    """Find a pattern in the patterns list by name using normalized comparison."""
    normalized_search = normalize_pattern_name(pattern_name)
    
    for pattern in patterns_list:
        json_pattern_name = pattern.get("pattern_name", "")
        normalized_json = normalize_pattern_name(json_pattern_name)
        
        if normalized_json == normalized_search:
            return pattern
    
    return None


def extract_pattern_name_from_filename(filename):
    """
    Extract pattern name from filename.
    Example: "April_Flowers_Wreath_last_valid.png" -> "April Flowers Wreath"
    """
    # Remove .png extension
    name = filename.replace('.png', '')
    # Remove _last_valid suffix
    name = name.replace('_last_valid', '')
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    return name


def discover_rendering_images(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover all rendering images in *_img directories.
    Returns a dictionary mapping model names to lists of image paths.
    """
    model_images = {}
    
    # Find all *_img directories
    for img_dir in base_dir.glob('*_img'):
        if img_dir.is_dir():
            model_name = img_dir.name.replace('_img', '')
            
            # Find all *_last_valid.png files
            image_files = list(img_dir.glob('*_last_valid.png'))
            
            if image_files:
                model_images[model_name] = sorted(image_files)
                print(f"  Found {len(image_files)} images in {img_dir.name}/")
    
    return model_images

def test_two_images(image1_path: str, image2_path: str):
    """
    Standalone function to test similarity between two images.
    Loads all necessary models, computes metrics, and prints results.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
    """
    print("=" * 90)
    print("Testing Two Image Similarity")
    print("=" * 90)
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load models
    models = load_models(device)
    
    try:
        # Load images
        print("Loading images...")
        image1 = load_image_from_path(Path(image1_path))
        image2 = load_image_from_path(Path(image2_path))
        print("✓ Images loaded successfully\n")
        
        # Compute metrics
        print("Computing similarity metrics...")
        metrics = compute_all_metrics(image1, image2, models, device)
        
        # Print results
        print("\n" + "=" * 90)
        print("RESULTS")
        print("=" * 90)
        print(f"LPIPS Similarity:        {metrics['lpips_similarity']:.4f} (distance: {metrics['lpips_distance']:.4f})")
        print(f"DINO Similarity:         {metrics['dino_similarity']:.4f}")
        print(f"Color Correlation:       {metrics['color_correlation']:.4f}")
        print(f"Color EMD Similarity:    {metrics['color_emd_similarity']:.4f} (distance: {metrics['color_emd_distance']:.4f})")
        print(f"SSIM:                    {metrics['ssim']:.4f}")
        print(f"CLIP Similarity:         {metrics['clip_similarity']:.4f}")
        print(f"\n{'='*90}")
        print(f"AGGREGATE SCORE:         {metrics['aggregate_score']:.4f} ({metrics['aggregate_score']*100:.2f}%)")
        print("=" * 90)
        
        return metrics
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        traceback.print_exc()
        return None


def test_clip_text_similarity(image_path: str, text_descriptions: List[str]):
    """
    Test CLIP similarity between an image and multiple text descriptions.
    Useful for comparing a rendered image with product titles/categories.
    
    Args:
        image_path: Path to the image
        text_descriptions: List of text descriptions to compare (e.g., product titles, categories)
    """
    print("=" * 90)
    print("Testing CLIP Text-to-Image Similarity")
    print("=" * 90)
    print(f"Image: {image_path}")
    print(f"Text descriptions: {len(text_descriptions)}")
    print()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load only CLIP model (we don't need the others)
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("✓ CLIP model loaded\n")
    
    try:
        # Load image
        print("Loading image...")
        image = load_image_from_path(Path(image_path))
        print("✓ Image loaded successfully\n")
        
        # Compute similarities for each text
        print("Computing CLIP text-to-image similarities...")
        results = []
        
        for text in text_descriptions:
            similarity = compute_clip_text_image_similarity(
                image, text, clip_model, clip_processor, device
            )
            results.append({
                'text': text,
                'similarity': similarity
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Print results
        print("\n" + "=" * 90)
        print("RESULTS (sorted by similarity)")
        print("=" * 90)
        for i, result in enumerate(results, 1):
            print(f"{i}. Similarity: {result['similarity']:.4f} ({result['similarity']*100:.2f}%)")
            print(f"   Text: \"{result['text']}\"")
            print()
        print("=" * 90)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        traceback.print_exc()
        return None


def process_single_comparison(
    rendering_path: Path,
    pattern_data: dict,
    models: dict,
    device: str
) -> Optional[Dict]:
    """Process a single image comparison and return metrics."""
    try:
        # Load images
        rendering_image = load_image_from_path(rendering_path)
        product_image_url = pattern_data.get("image_link")
        
        if not product_image_url:
            print(f"    ⚠ No image_link found for pattern")
            return None
        
        product_image = load_image_from_url(product_image_url)
        
        # Compute metrics
        metrics = compute_all_metrics(rendering_image, product_image, models, device)
        
        return metrics
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return None


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save comparison results to CSV file."""
    if not results:
        print(f"  No results to save for {output_path}")
        return
    
    # Define CSV columns
    columns = [
        'pattern_name',
        'pattern_id',
        'project_type',
        'skill_level',
        'rendering_path',
        'product_url',
        'lpips_similarity',
        'lpips_distance',
        'dino_similarity',
        'color_correlation',
        'color_emd_similarity',
        'color_emd_distance',
        'ssim',
        'clip_similarity',
        'aggregate_score',
        'timestamp'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  ✓ Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare crochet renderings with product images"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory containing *_img folders and JSON file"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="selected_crochet_patterns_3.json",
        help="JSON file with pattern data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory to save CSV results"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        help="Specific models to process (e.g., claude qwen). If not specified, processes all found."
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    json_path = base_dir / args.json_file
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load patterns JSON
    print(f"Loading patterns from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        patterns_list = json.load(f)
    print(f"Loaded {len(patterns_list)} patterns\n")
    
    # Discover rendering images
    print("Discovering rendering images in *_img directories...")
    model_images = discover_rendering_images(base_dir)
    
    if not model_images:
        print("No *_img directories found with *_last_valid.png files!")
        return
    
    # Filter by specified models if provided
    if args.models:
        model_images = {k: v for k, v in model_images.items() if k in args.models}
        if not model_images:
            print(f"No matching models found for: {args.models}")
            return
    
    print(f"\nFound {len(model_images)} model(s) to process:")
    for model_name, images in model_images.items():
        print(f"  - {model_name}: {len(images)} images")
    
    # Load models
    print("\n" + "=" * 90)
    models = load_models(device)
    
    # Process each model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, image_paths in model_images.items():
        print("\n" + "=" * 90)
        print(f"Processing model: {model_name.upper()}")
        print("=" * 90)
        
        results = []
        
        for idx, rendering_path in enumerate(image_paths, 1):
            # Extract pattern name from filename
            pattern_name = extract_pattern_name_from_filename(rendering_path.name)
            
            print(f"\n[{idx}/{len(image_paths)}] {pattern_name}")
            print(f"  File: {rendering_path.name}")
            
            # Find matching pattern in JSON
            pattern_data = find_pattern_in_json(patterns_list, pattern_name)
            
            if not pattern_data:
                print(f"  ⚠ Pattern not found in JSON, skipping...")
                continue
            
            print(f"  ID: {pattern_data.get('id', 'N/A')}")
            print(f"  Type: {pattern_data.get('project_type', 'N/A')}")
            
            # Process comparison
            metrics = process_single_comparison(
                rendering_path, pattern_data, models, device
            )
            
            if metrics:
                # Add metadata to results
                result_row = {
                    'pattern_name': pattern_name,
                    'pattern_id': pattern_data.get('id', ''),
                    'project_type': pattern_data.get('project_type', ''),
                    'skill_level': pattern_data.get('skill_level', ''),
                    'rendering_path': str(rendering_path),
                    'product_url': pattern_data.get('image_link', ''),
                    'timestamp': timestamp,
                    **metrics
                }
                results.append(result_row)
                
                print(f"  ✓ Aggregate Score: {metrics['aggregate_score']:.4f} ({metrics['aggregate_score']*100:.2f}%)")
        
        # Save results for this model
        print("\n" + "-" * 90)
        output_file = output_dir / f"{model_name}_comparison_results_{timestamp}.csv"
        save_results_to_csv(results, output_file)
        
        # Print summary statistics
        if results:
            avg_aggregate = np.mean([r['aggregate_score'] for r in results])
            avg_lpips = np.mean([r['lpips_similarity'] for r in results])
            avg_dino = np.mean([r['dino_similarity'] for r in results])
            avg_clip = np.mean([r['clip_similarity'] for r in results])
            
            print(f"\n  Summary Statistics for {model_name}:")
            print(f"    Total comparisons: {len(results)}")
            print(f"    Average Aggregate Score: {avg_aggregate:.4f}")
            print(f"    Average LPIPS: {avg_lpips:.4f}")
            print(f"    Average DINO: {avg_dino:.4f}")
            print(f"    Average CLIP: {avg_clip:.4f}")
    
    print("\n" + "=" * 90)
    print("✓ Batch processing complete!")
    print(f"Results saved in: {output_dir}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
    
    # Test 1: Compare two images
    # test_two_images("/store01/nchawla/pli9/crochet_origin/hat_render.png", 
    #                 "/store01/nchawla/pli9/crochet_origin/hat_origin.png")
    
    # Test 2: Compare rendered image with text descriptions (product titles/categories)
    # test_clip_text_similarity(
    #     "/store01/nchawla/pli9/crochet_origin/hat_origin.png",
    #     [
    #         "CHUNKY BUCKET HAT",
    #         "Hat"
    #     ]
    # )
