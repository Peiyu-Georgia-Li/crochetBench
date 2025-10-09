#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
import statistics

# Directory containing the JSON files
json_dir = "../data/crochet_pattern_by_project/"

def analyze_instruction_complexity():
    """Analyze the complexity of patterns based on instruction length and other factors."""
    instruction_lengths = []
    instruction_lengths_by_skill = {}
    instruction_lengths_by_project = {}
    
    abbreviation_counts = []
    abbreviation_counts_by_skill = {}
    
    total_patterns = 0
    patterns_with_instructions = 0
    
    # Process each JSON file in the directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
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
            if project_type not in instruction_lengths_by_project:
                instruction_lengths_by_project[project_type] = []
            
            for pattern in data:
                total_patterns += 1
                
                # Get skill level
                skill_level = pattern.get('skill_level', 'unknown').strip().lower()
                
                # Initialize skill level lists if needed
                if skill_level not in instruction_lengths_by_skill:
                    instruction_lengths_by_skill[skill_level] = []
                    abbreviation_counts_by_skill[skill_level] = []
                
                # Calculate instruction length
                if 'instructions' in pattern and pattern['instructions']:
                    patterns_with_instructions += 1
                    instruction_length = len(pattern['instructions'])
                    instruction_lengths.append(instruction_length)
                    instruction_lengths_by_skill[skill_level].append(instruction_length)
                    instruction_lengths_by_project[project_type].append(instruction_length)
                
                # Count abbreviations
                if 'abbreviations' in pattern and pattern['abbreviations']:
                    abbrev_count = len(pattern['abbreviations'])
                    abbreviation_counts.append(abbrev_count)
                    abbreviation_counts_by_skill[skill_level].append(abbrev_count)
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Overall complexity statistics
    print("\nOverall Pattern Complexity:")
    print("-" * 40)
    
    if instruction_lengths:
        percentiles = np.percentile(instruction_lengths, [25, 50, 75, 90])
        print(f"Instruction Length Statistics:")
        print(f"  Average: {statistics.mean(instruction_lengths):.1f} characters")
        print(f"  Median: {statistics.median(instruction_lengths):.1f} characters")
        print(f"  Min: {min(instruction_lengths)} characters")
        print(f"  Max: {max(instruction_lengths)} characters")
        print(f"  25th percentile: {percentiles[0]:.1f} characters")
        print(f"  75th percentile: {percentiles[2]:.1f} characters")
        print(f"  90th percentile: {percentiles[3]:.1f} characters")
    
    if abbreviation_counts:
        print(f"\nAbbreviations Count Statistics:")
        print(f"  Average: {statistics.mean(abbreviation_counts):.1f} abbreviations")
        print(f"  Median: {statistics.median(abbreviation_counts):.1f} abbreviations")
        print(f"  Min: {min(abbreviation_counts)} abbreviations")
        print(f"  Max: {max(abbreviation_counts)} abbreviations")
    
    print(f"\nTotal patterns: {total_patterns}")
    print(f"Patterns with instructions: {patterns_with_instructions} ({patterns_with_instructions/total_patterns*100:.2f}% of total)")
    
    # Complexity by skill level
    print("\nInstruction Length by Skill Level:")
    print("-" * 40)
    print(f"{'Skill Level':<20} {'Average Length':<15} {'Median Length':<15} {'Count':<10}")
    print("-" * 40)
    
    for skill, lengths in sorted(instruction_lengths_by_skill.items(), key=lambda x: statistics.mean(x[1]) if x[1] else 0, reverse=True):
        if lengths:
            avg_length = statistics.mean(lengths)
            median_length = statistics.median(lengths)
            print(f"{skill:<20} {avg_length:<15.1f} {median_length:<15.1f} {len(lengths):<10}")
    
    # Abbreviation count by skill level
    print("\nAbbreviation Count by Skill Level:")
    print("-" * 40)
    print(f"{'Skill Level':<20} {'Average Count':<15} {'Median Count':<15} {'Count':<10}")
    print("-" * 40)
    
    for skill, counts in sorted(abbreviation_counts_by_skill.items(), key=lambda x: statistics.mean(x[1]) if x[1] else 0, reverse=True):
        if counts:
            avg_count = statistics.mean(counts)
            median_count = statistics.median(counts)
            print(f"{skill:<20} {avg_count:<15.1f} {median_count:<15.1f} {len(counts):<10}")
    
    # Project types by complexity (top 10)
    print("\nTop 10 Most Complex Project Types (by instruction length):")
    print("-" * 40)
    print(f"{'Project Type':<30} {'Average Length':<15} {'Median Length':<15} {'Count':<10}")
    print("-" * 40)
    
    top_complex_projects = sorted(
        [(proj, lengths) for proj, lengths in instruction_lengths_by_project.items() if lengths],
        key=lambda x: statistics.mean(x[1]),
        reverse=True
    )[:10]
    
    for project, lengths in top_complex_projects:
        avg_length = statistics.mean(lengths)
        median_length = statistics.median(lengths)
        print(f"{project:<30} {avg_length:<15.1f} {median_length:<15.1f} {len(lengths):<10}")
    
    # Project types by simplicity (bottom 10)
    print("\nTop 10 Simplest Project Types (by instruction length):")
    print("-" * 40)
    print(f"{'Project Type':<30} {'Average Length':<15} {'Median Length':<15} {'Count':<10}")
    print("-" * 40)
    
    bottom_complex_projects = sorted(
        [(proj, lengths) for proj, lengths in instruction_lengths_by_project.items() if lengths and len(lengths) >= 10],
        key=lambda x: statistics.mean(x[1])
    )[:10]
    
    for project, lengths in bottom_complex_projects:
        avg_length = statistics.mean(lengths)
        median_length = statistics.median(lengths)
        print(f"{project:<30} {avg_length:<15.1f} {median_length:<15.1f} {len(lengths):<10}")

if __name__ == "__main__":
    analyze_instruction_complexity()
