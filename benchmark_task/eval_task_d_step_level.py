#!/usr/bin/env python3
import json
import subprocess
import os
import sys
import re
from collections import defaultdict
import argparse

def extract_and_verify_dsl(model, filename):
    # File path for the JSON file with generated DSL patterns
    input_file = f'task_d_step_{model}/{filename}' #⚠️
    print(f"Loading patterns from {input_file}...")
    
    try:
        # Load the patterns
        with open(input_file, 'r') as f:
            patterns = json.load(f)
        print(f"Loaded {len(patterns)} patterns")
    except Exception as e:
        print(f"Error loading pattern file: {str(e)}")
        sys.exit(1)

    # Create a list to store the patterns with verification results
    verified_patterns = []
    
    # Create dictionaries to track statistics
    total_stats = {"processed": 0, "valid": 0, "invalid": 0}
    
    # Dictionary to track error types
    error_types = defaultdict(int)
    pattern_errors = defaultdict(list)
    
    # Define error categories and corresponding patterns for classification
    # Organized by type: DSL/compiler errors vs Runtime errors
    ERROR_CATEGORIES = {
        # DSL/Compiler errors - occur during parsing/compilation
        'Syntax error': r'syntax error|unexpected token',
        'Unbalanced brackets': r'unbalanced brackets|brackets in original|brackets in:',
        'Undefined stitch': r'stitch type not defined|stitch:',
        'Multiple references': r'multiple references defined without',
        'Label not found': r'label not found',
        'Non-adjacent labels': r'same label over non-adjacent',
        'Turning issue': r'turning can happen only',
        'Variable name conflict': r'variable name matches stitch name',
        'Multiplier issue': r'multiplier set, but no stitch',
        'Invalid abbreviation': r'invalid abbreviation',
        'Missing expected element': r'expected',
        'Invalid repeat structure': r'invalid repeat',
        
        # Runtime errors - occur during JavaScript execution
        'Runtime Error': r'cannot read properties|cannot use in',
        
        # Catch-all pattern
        'Other': r'.'
    }
    
    # Function to categorize and count error types
    def categorize_error(error_msg):
        if not error_msg:
            return "Unknown error"
            
        error_msg = error_msg.lower()
        
        # Check each category pattern against the error message
        for category, pattern in ERROR_CATEGORIES.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                # Return the first matching category (except 'Other')
                if category != 'Other' or all(not re.search(p, error_msg, re.IGNORECASE) 
                                            for c, p in ERROR_CATEGORIES.items() if c != 'Other'):
                    return category
        
        return "Other error"

    # Create directory for storing individual pattern files
    os.makedirs('claude_dsl_patterns_to_verify', exist_ok=True)
    
    # Process each pattern
    for i, pattern in enumerate(patterns):
        pattern_name = pattern.get('pattern_name', f"Pattern_{i}")
        generated_dsl = pattern.get('generated_dsl', '')
        
        if not generated_dsl:
            print(f"Warning: No generated_dsl found for pattern: {pattern_name}")
            continue
        
        # Update processed count
        total_stats["processed"] += 1
        
        # Create a safe filename
        safe_name = ''.join(c if c.isalnum() else '_' for c in pattern_name)
        file_path = f'claude_dsl_patterns_to_verify/{safe_name}.txt'
        
        # Write the pattern to a file
        with open(file_path, 'w') as f:
            f.write(generated_dsl)
        
        print(f"\nVerifying pattern: {pattern_name} ({i+1}/{len(patterns)})")
        try:
            # Try to find node executable - might be nodejs instead of node on some systems
            node_cmd = 'node'  # Default command
            try:
                # Check if node exists
                subprocess.run(['which', 'node'], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                # Try nodejs instead
                try:
                    subprocess.run(['which', 'nodejs'], capture_output=True, check=True)
                    node_cmd = 'nodejs'
                except subprocess.CalledProcessError:
                    raise FileNotFoundError("Could not find 'node' or 'nodejs' executable")
            
            # Verify the pattern using the JS script
            result = subprocess.run(
                [node_cmd, 'verify_crochet_pattern.js', file_path],
                capture_output=True,
                text=True
            )
            
            # Check if verification was successful
            try:
                verification_result = json.loads(result.stdout)
                is_valid = verification_result.get('valid', False)
                status = "VALID" if is_valid else "INVALID"
                print(f"Pattern '{pattern_name}' is {status}")
                
                # Update total statistics
                if is_valid:
                    total_stats["valid"] += 1
                else:
                    total_stats["invalid"] += 1
                
                if not is_valid and 'errors' in verification_result:
                    pattern_specific_errors = []
                    for error in verification_result['errors']:
                        print(f"  - Error: {error}")
                        error_category = categorize_error(error)
                        error_types[error_category] += 1
                        pattern_specific_errors.append({"error": error, "category": error_category})
                    
                    # Store errors for this pattern
                    if pattern_specific_errors:
                        pattern_errors[pattern_name] = pattern_specific_errors
                        
                # Add the verification result to the pattern
                pattern_data = {
                    "pattern_name": pattern_name,
                    "pattern_index": i,
                    "is_valid": is_valid,
                    "verification_result": verification_result
                }
                
                # Include any other fields from the original pattern that might be useful
                for key in ['id', 'materials', 'skill_level', 'abbreviations']:
                    if key in pattern:
                        pattern_data[key] = pattern[key]
                
                verified_patterns.append(pattern_data)
            except json.JSONDecodeError:
                print(f"Could not parse verification result: {result.stdout}")
                
                # Update total statistics - treat as invalid
                total_stats["invalid"] += 1
                
                # Track error type
                error_category = "Parse error"
                error_types[error_category] += 1
                pattern_errors[pattern_name] = [{"error": "Invalid JSON result", "category": error_category}]
                
                verified_patterns.append({
                    "pattern_name": pattern_name,
                    "pattern_index": i,
                    "is_valid": False,
                    "error": "Invalid JSON result",
                    "raw_output": result.stdout
                })
        except Exception as e:
            print(f"Error verifying pattern '{pattern_name}': {str(e)}")
            
            # Update total statistics - treat as invalid
            total_stats["invalid"] += 1
            
            # Track error type
            error_category = "Execution error"
            error_types[error_category] += 1
            pattern_errors[pattern_name] = [{"error": str(e), "category": error_category}]
            
            verified_patterns.append({
                "pattern_name": pattern_name,
                "pattern_index": i,
                "is_valid": False,
                "error": str(e)
            })
    
    # Save verification results to a JSON file
    output_file = 'claude_dsl_verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(verified_patterns, f, indent=2)
    
    # Generate statistics
    # Convert defaultdict to regular dict for JSON serialization
    error_types_dict = dict(error_types)
    
    statistics = {
        "total": total_stats,
        "error_types": {
            "overall": error_types_dict
        },
        "patterns_with_errors": len(pattern_errors)
    }
    
    # Save statistics to a JSON file
    stats_file = 'claude_dsl_verification_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # Report statistics
    print(f"\nVerification complete!")
    print(f"\nOverall Statistics:")
    print(f"-" * 50)
    print(f"Total: {total_stats['processed']} patterns processed")
    print(f"  - {total_stats['valid']} valid patterns ({total_stats['valid']/max(1, total_stats['processed'])*100:.1f}%)")
    print(f"  - {total_stats['invalid']} invalid patterns ({total_stats['invalid']/max(1, total_stats['processed'])*100:.1f}%)")
    
    # Error Type Summary
    print(f"\nError Type Summary:")
    print(f"-" * 50)
    
    # Sort error types by frequency
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    total_errors = sum(error_types.values())
    
    if total_errors > 0:
        print(f"Total errors found: {total_errors}")
        
        # Display error type distribution
        print(f"\nError Distribution:")
        for error_type, count in sorted_errors:
            percentage = (count / total_errors) * 100
            print(f"  - {error_type}: {count} occurrences ({percentage:.1f}%)")
    else:
        print("No errors found in the patterns.")

    print(f"\n- Results saved to '{output_file}'")
    print(f"- Statistics saved to '{stats_file}'")
    print(f"- Individual pattern files saved in 'claude_dsl_patterns_to_verify/' directory")

def main():
    parser = argparse.ArgumentParser(description='Verify generated DSL patterns')
    parser.add_argument('model_name', type=str, help='Model name', default='claude')
    parser.add_argument('filename', type=str, help='Filename', default='step_level_test_1_2.json')
    args = parser.parse_args()
    extract_and_verify_dsl(args.model_name, args.filename)

if __name__ == "__main__":
    main()
