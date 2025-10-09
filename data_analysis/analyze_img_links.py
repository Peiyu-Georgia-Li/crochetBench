#!/usr/bin/env python3
import os
import json
import glob

# Directory containing the JSON files
json_dir = "../data/crochet_pattern_by_project/"

def count_patterns_with_images():
    """Count patterns with and without image links."""
    total_patterns = 0
    patterns_with_image = 0
    patterns_without_image = 0
    files_processed = 0
    
    # Process each JSON file in the directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # Track patterns by project type
    project_stats = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                print(f"Warning: {json_file} does not contain a list of patterns")
                continue
                
            # Extract project type from filename
            filename = os.path.basename(json_file)
            project_type = os.path.splitext(filename)[0]
            
            # Initialize stats for this project type
            if project_type not in project_stats:
                project_stats[project_type] = {
                    'total': 0,
                    'with_image': 0,
                    'without_image': 0
                }
            
            # Count patterns with and without images
            for pattern in data:
                total_patterns += 1
                project_stats[project_type]['total'] += 1
                
                if 'image_link' in pattern and pattern['image_link']:
                    patterns_with_image += 1
                    project_stats[project_type]['with_image'] += 1
                else:
                    patterns_without_image += 1
                    project_stats[project_type]['without_image'] += 1
                
            files_processed += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\nProcessed {files_processed} JSON files")
    print(f"Total patterns: {total_patterns}")
    print(f"Patterns with image links: {patterns_with_image} ({patterns_with_image/total_patterns*100:.2f}%)")
    print(f"Patterns without image links: {patterns_without_image} ({patterns_without_image/total_patterns*100:.2f}%)")
    
    # Print project-specific stats
    print("\nBreakdown by project type:")
    print("-" * 60)
    print(f"{'Project Type':<30} {'Total':<10} {'With Image':<15} {'Without Image':<15}")
    print("-" * 60)
    
    for project, stats in sorted(project_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        with_image_pct = stats['with_image'] / stats['total'] * 100 if stats['total'] > 0 else 0
        without_image_pct = stats['without_image'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        print(f"{project:<30} {stats['total']:<10} {stats['with_image']:<10} ({with_image_pct:.1f}%) {stats['without_image']:<10} ({without_image_pct:.1f}%)")
    
    return {
        'total_patterns': total_patterns,
        'patterns_with_image': patterns_with_image,
        'patterns_without_image': patterns_without_image,
        'project_stats': project_stats
    }

if __name__ == "__main__":
    count_patterns_with_images()
