#!/usr/bin/env python3
"""
Create a multiple-choice dataset for crochet patterns.

For each pattern image, the model should identify the correct instructions
from 4 options, all from the same project_type.
"""

import os
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def load_json_files(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all JSON files from the specified directory and group patterns by project_type.
    
    Args:
        directory: Path to directory containing JSON files
        
    Returns:
        Dictionary mapping project_type to list of patterns
    """
    patterns_by_type = defaultdict(list)
    
    # List all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    for filename in json_files:
        try:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Add each pattern to its project_type group
            for pattern in data:
                project_type = pattern.get('project_type')
                if project_type and 'image_link' in pattern and 'instructions' in pattern:
                    patterns_by_type[project_type].append(pattern)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return patterns_by_type

def create_multiple_choice_dataset(patterns_by_type: Dict[str, List[Dict[str, Any]]],
                                  min_options_per_type: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Create multiple-choice questions for each pattern.
    
    Args:
        patterns_by_type: Dictionary mapping project_type to list of patterns
        min_options_per_type: Minimum number of patterns required for a project_type
                             to be included (need at least 4 for multiple choice)
        
    Returns:
        List of multiple-choice questions
    """
    dataset = []
    
    # Filter out project types with too few patterns
    valid_types = {
        project_type: patterns 
        for project_type, patterns in patterns_by_type.items()
        if len(patterns) >= min_options_per_type
    }
    
    # Track distribution of correct answers
    correct_distributions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    option_labels = ['A', 'B', 'C', 'D']
    
    for project_type, patterns in valid_types.items():
        for i, pattern in enumerate(patterns):
            # Create a copy of patterns without the current pattern
            other_patterns = patterns.copy()
            other_patterns.pop(i)
            
            # Select 3 random patterns from the same project type
            if len(other_patterns) >= 3:
                incorrect_options = random.sample(other_patterns, 3)
                
                # Create all options (1 correct, 3 incorrect)
                options = []
                correct_index = random.randint(0, 3)  # Randomly place correct answer
                correct_label = option_labels[correct_index]
                correct_distributions[correct_label] += 1
                
                for j in range(4):
                    label = option_labels[j]
                    if j == correct_index:
                        options.append({
                            'label': label,
                            'instructions': pattern['instructions'],
                            'is_correct': True
                        })
                    else:
                        incorrect_idx = j if j < correct_index else j - 1
                        options.append({
                            'label': label,
                            'instructions': incorrect_options[incorrect_idx]['instructions'],
                            'is_correct': False
                        })
                
                # Create the multiple-choice question
                question = {
                    'id': pattern['id'],
                    'pattern_name': pattern['pattern_name'],
                    'project_type': project_type,
                    'image_url': pattern['image_link'],
                    'options': options,
                    'correct_label': correct_label,
                    'correct_index': correct_index
                }
                
                dataset.append(question)
    
    return dataset, correct_distributions

def main():
    parser = argparse.ArgumentParser(description='Create multiple-choice dataset for crochet patterns')
    parser.add_argument('--input_dir', type=str, default='../data/crochet_pattern_by_project',
                        help='Directory containing JSON files')
    parser.add_argument('--output_file', type=str, default='../data/mc_data.json',
                        help='Output file path for the dataset')
    parser.add_argument('--min_options', type=int, default=5,
                        help='Minimum number of patterns required for a project type')
    args = parser.parse_args()
    
    # Load patterns from JSON files
    print(f"Loading patterns from {args.input_dir}...")
    patterns_by_type = load_json_files(args.input_dir)
    
    # Create dataset
    print("Creating multiple-choice dataset...")
    dataset, correct_distributions = create_multiple_choice_dataset(patterns_by_type, args.min_options)
    
    # Save dataset to JSON file
    print(f"Saving dataset to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Done! Created {len(dataset)} multiple-choice questions.")
    
    # Print statistics
    print("\nStatistics:")
    for project_type, patterns in patterns_by_type.items():
        questions = len([q for q in dataset if q['project_type'] == project_type])
        if questions > 0:
            print(f"  {project_type}: {questions} questions from {len(patterns)} patterns")
    
    # Print correct answer distribution
    total = sum(correct_distributions.values())
    if total > 0:
        print("\nCorrect answer distribution:")
        for label, count in correct_distributions.items():
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
