#!/usr/bin/env python3
import os
import json
import glob
from collections import Counter

# Directory containing the JSON files
json_dir = "../data/crochet_pattern_by_project/"

def analyze_skill_levels():
    """Analyze the distribution of skill levels across patterns."""
    skill_levels = []
    skill_levels_by_project = {}
    missing_skill_level = 0
    total_patterns = 0
    
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
            if project_type not in skill_levels_by_project:
                skill_levels_by_project[project_type] = Counter()
            
            # Count skill levels
            for pattern in data:
                total_patterns += 1
                
                if 'skill_level' in pattern and pattern['skill_level']:
                    skill_level = pattern['skill_level'].strip().lower()
                    skill_levels.append(skill_level)
                    skill_levels_by_project[project_type][skill_level] += 1
                else:
                    missing_skill_level += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Overall distribution
    skill_level_counter = Counter(skill_levels)
    print("\nOverall Skill Level Distribution:")
    print("-" * 40)
    print(f"{'Skill Level':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    
    for skill_level, count in sorted(skill_level_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(skill_levels) * 100
        print(f"{skill_level:<15} {count:<10} {percentage:.2f}%")
    
    print(f"\nTotal patterns with skill level: {len(skill_levels)}")
    print(f"Missing skill level: {missing_skill_level} ({missing_skill_level/total_patterns*100:.2f}% of total)")
    
    # Project-specific distribution for the top 10 project types
    print("\nSkill Level Distribution by Top 10 Project Types:")
    top_projects = sorted(skill_levels_by_project.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]
    
    for project, counter in top_projects:
        print(f"\n{project}:")
        print("-" * 30)
        total = sum(counter.values())
        for skill, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total * 100
            print(f"{skill:<15} {count:<5} ({percentage:.1f}%)")
    
    return {
        'skill_level_distribution': skill_level_counter,
        'skill_levels_by_project': skill_levels_by_project,
        'missing_skill_level': missing_skill_level,
        'total_patterns': total_patterns
    }

if __name__ == "__main__":
    analyze_skill_levels()
