# usage
# python eval_task_d_project_level.py claude

#!/usr/bin/env python3
import json
import subprocess
import os
import sys
import re
import argparse
import csv
from collections import defaultdict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Progressively verify generated DSL patterns")
parser.add_argument("model_name",  type=str, nargs='?', default="claude", 
                    help="Model name (default: claude)")
args = parser.parse_args()

# Global variable for model name
MODEL = args.model_name

def clean_pattern_dsl(dsl_text):
    """
    Clean up the pattern DSL by removing markdown formatting and other problematic elements.
    
    Args:
        dsl_text: Raw DSL text that may contain markdown or other formatting
        
    Returns:
        Clean DSL text ready for verification
    """
    # Remove markdown code block delimiters with or without language specifiers
    dsl_text = re.sub(r'^```(?:dsl|crochet)?\s*$', '', dsl_text, flags=re.MULTILINE)
    dsl_text = re.sub(r'^```\s*$', '', dsl_text, flags=re.MULTILINE)
    
    # Remove any HTML comments
    dsl_text = re.sub(r'<!--.*?-->', '', dsl_text, flags=re.DOTALL)
    
    # Remove line number indicators like '-->   1:' or '      2:'
    dsl_text = re.sub(r'^\s*(?:-->\s*)?\d+:\s*', '', dsl_text, flags=re.MULTILINE)
    
    # Remove comments like "// Back of eye (make 2)"
    dsl_text = re.sub(r'^\s*//.*$', '', dsl_text, flags=re.MULTILINE)
    
    # Remove any leading/trailing whitespace from each line and the whole text
    lines = [line.strip() for line in dsl_text.split('\n')]
    dsl_text = '\n'.join(line for line in lines if line)
    
    # If the last line contains instructions like "first clean the", remove it
    if any(phrase in dsl_text.split('\n')[-1].lower() for phrase in ['clean the', 'first clean', 'remove']):
        dsl_text = '\n'.join(dsl_text.split('\n')[:-1])
    
    return dsl_text


def categorize_error(error_msg):
    """
    Categorize error messages into types based on pattern matching.
    
    Args:
        error_msg: The error message to categorize
        
    Returns:
        String representing the error category
    """
    if not error_msg:
        return "Unknown error"
        
    error_msg = error_msg.lower()
    
    # Define error categories and corresponding patterns based on summarize_dsl_errors.py
    ERROR_CATEGORIES = {
        'Unbalanced brackets': r'unbalanced brackets|brackets in original|brackets in:',
        'Undefined stitch': r'stitch type not defined|stitch:',
        'Multiple references': r'multiple references defined without',
        'Label not found': r'label not found',
        'Non-adjacent labels': r'same label over non-adjacent',
        'Turning issue': r'turning can happen only',
        'Variable name conflict': r'variable name matches stitch name',
        'Runtime error': r'cannot read properties|cannot use \'in\'',
        'Multiplier issue': r'multiplier set, but no stitch',
        'Syntax error': r'syntax error|unexpected token',
        'Invalid abbreviation': r'invalid abbreviation',
        'Missing expected element': r'expected',
        'Invalid repeat structure': r'invalid repeat',
        'Other': r'.'  # Catch-all pattern
    }
    
    # Check each category pattern against the error message
    for category, pattern in ERROR_CATEGORIES.items():
        if re.search(pattern, error_msg, re.IGNORECASE):
            # Return the first matching category (except 'Other')
            if category != 'Other' or all(not re.search(p, error_msg, re.IGNORECASE) 
                                        for c, p in ERROR_CATEGORIES.items() if c != 'Other'):
                return category
    
    return "Other error"


def verify_dsl(file_path, include_history=False, timeout=10):
    """
    Verify a DSL pattern file using the crochet pattern verifier with history support.
    
    Args:
        file_path: Path to the file containing the DSL pattern
        include_history: Whether to include history steps in the verification
        timeout: Maximum time in seconds to wait for verification
        
    Returns:
        Dict with verification results
    """
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
        
        # Command to run - use the history-enabled verifier
        cmd = [node_cmd, 'verify_crochet_pattern_with_history.js', file_path]
        
        # If including history, set an environment variable that the verifier can use
        env = os.environ.copy()
        if include_history:
            env['INCLUDE_HISTORY'] = '1'
        
        # Run the verification with timeout
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            print(f"Verification timed out after {timeout} seconds")
            return {
                'valid': False,
                'error': f"Verification timed out after {timeout} seconds",
                'timed_out': True
            }
        
        # Parse the result
        try:
            verification_result = json.loads(result.stdout)
            return verification_result
        except json.JSONDecodeError:
            return {
                'valid': False,
                'error': "Invalid JSON result",
                'raw_output': result.stdout
            }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def progressive_verify_dsl():
    """
    Extract generated_dsl from JSON file, separate by newline, and verify progressively
    to identify at which step the error occurs.
    """
    # File path for the JSON file with generated DSL patterns
    input_file = f'task_d_project_{MODEL}/project_level_test.json'
    print(f"Loading patterns from {input_file} for model {MODEL}...")
    
    try:
        # Load the patterns
        with open(input_file, 'r') as f:
            patterns = json.load(f)
        print(f"Loaded {len(patterns)} patterns")
    except Exception as e:
        print(f"Error loading pattern file: {str(e)}")
        sys.exit(1)

    # Create directories for storing pattern files and progressive verification results
    os.makedirs(f'{MODEL}_dsl_patterns', exist_ok=True)
    os.makedirs(f'{MODEL}_dsl_progressive_results', exist_ok=True)
    
    # Results container
    progressive_results = []
    total_stats = {"processed": 0, "valid": 0, "invalid": 0}
    
    # Process each pattern
    for i, pattern in enumerate(patterns):
        pattern_name = pattern.get('pattern_name', f"Pattern_{i}")
        generated_dsl = pattern.get('generated_dsl', '')
        
        if not generated_dsl:
            print(f"Warning: No generated_dsl found for pattern: {pattern_name}")
            continue
            
        # Clean the DSL pattern - remove markdown code blocks and other problematic formatting
        print(f"\nOriginal DSL pattern:\n{'='*40}")
        print(generated_dsl[:500] + "..." if len(generated_dsl) > 500 else generated_dsl)
        
        generated_dsl = clean_pattern_dsl(generated_dsl)
        
        print(f"\nCleaned DSL pattern:\n{'='*40}")
        print(generated_dsl[:500] + "..." if len(generated_dsl) > 500 else generated_dsl)
        
        # Update processed count
        total_stats["processed"] += 1
        
        # Create a safe filename base
        safe_name = ''.join(c if c.isalnum() else '_' for c in pattern_name)
        
        print(f"\n{'='*80}")
        print(f"Processing pattern {i+1}/{len(patterns)}: {pattern_name}")
        print(f"{'='*80}")
        
        # Split the DSL into lines
        dsl_lines = generated_dsl.split('\n')
        
        # Results for this pattern
        pattern_result = {
            "pattern_name": pattern_name,
            "pattern_id": pattern.get('id', f"id_{i}"),  # Extract the actual ID from the JSON
            "pattern_index": i,
            "total_lines": len(dsl_lines),
            "progressive_validation": [],
            "first_error_at_line": None,
            "first_error_details": None,
            "last_valid_line": None,
            "is_valid_full": False
        }
        
        # First verify the complete pattern to know if it's valid overall
        full_dsl_path = f'{MODEL}_dsl_patterns/{safe_name}_full.txt'
        with open(full_dsl_path, 'w') as f:
            f.write(generated_dsl)
        
        full_result = verify_dsl(full_dsl_path)
        pattern_result["is_valid_full"] = full_result.get('valid', False)
        pattern_result["full_validation"] = full_result
        
        if pattern_result["is_valid_full"]:#verify_crochet_pattern
            print(f"Full pattern is VALID")
            total_stats["valid"] += 1
        else:
            print(f"Full pattern is INVALID")
            if 'errors' in full_result:
                for error in full_result['errors']:
                    print(f"  - Error: {error}")
            total_stats["invalid"] += 1
        
        # If the pattern is invalid, perform progressive validation
        if not pattern_result["is_valid_full"]:
            print("\nPerforming progressive validation to find where the error occurs:")
            
            # Test progressively adding lines
            last_valid_dsl = ""
            last_valid_line = 0
            current_valid = True
            
            # Variables to detect repeating errors
            last_errors = []
            repeated_error_count = 0
            max_repeated_errors = 5  # Stop after this many consecutive identical errors
            
            # Limit total number of errors processed per pattern
            total_errors_processed = 0
            max_total_errors = 50  # Maximum number of errors to process before moving on
            
            for line_num, line in enumerate(dsl_lines):
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Build current DSL - include all lines up to this point
                current_dsl = '\n'.join(dsl_lines[:line_num+1])
                
                # Create a file for this progressive check
                prog_file_path = f'{MODEL}_dsl_progressive_results/{safe_name}_line_{line_num+1}.txt'
                with open(prog_file_path, 'w') as f:
                    f.write(current_dsl)
                
                # Verify this progressive version
                validation_result = verify_dsl(prog_file_path, include_history=True)
                is_valid = validation_result.get('valid', False)
                
                # Check for repeating errors
                current_errors = validation_result.get('errors', [])
                
                # Check if we're seeing the same errors repeatedly
                if not is_valid and current_errors and current_errors == last_errors:
                    repeated_error_count += 1
                    if repeated_error_count >= max_repeated_errors:
                        print(f"\nDetected {max_repeated_errors} consecutive identical errors. Pattern appears to be stuck.")
                        print(f"Skipping remaining lines in this pattern.")
                        break
                else:
                    repeated_error_count = 0
                    last_errors = current_errors
                
                # Record the first line that causes an error
                if current_valid and not is_valid:
                    pattern_result["first_error_at_line"] = line_num + 1
                    pattern_result["last_valid_line"] = last_valid_line
                    current_valid = False
                    
                    # Store detailed error information
                    error_details = []
                    if 'errors' in validation_result:
                        error_details = validation_result['errors']
                    
                    pattern_result["first_error_details"] = {
                        "line": line,
                        "errors": error_details,
                        "context": dsl_lines[max(0, line_num-2):min(len(dsl_lines), line_num+3)] if line_num > 0 else dsl_lines[:min(5, len(dsl_lines))]
                    }
                    
                    print(f"\n! Error first occurs at line {line_num + 1}: '{line}'")
                    if error_details:
                        for error in error_details:
                            print(f"  - Error: {error}")
                    
                    # Show context around the error
                    print("\nContext around error:")
                    for ctx_idx, ctx_line in enumerate(pattern_result["first_error_details"]["context"]):
                        line_offset = max(0, line_num-2) + ctx_idx
                        pointer = "-->" if line_offset == line_num else "   "
                        print(f"{pointer} {line_offset+1:3d}: {ctx_line}")
                
                # Store the progressive validation result
                pattern_result["progressive_validation"].append({
                    "line_number": line_num + 1,
                    "line_content": line,
                    "is_valid": is_valid,
                    "validation_result": validation_result
                })
                
                # Print progress indicator
                status = "✓" if is_valid else "✗"
                print(f"Line {line_num+1:3d} {status}: {line[:60]}{'...' if len(line) > 60 else ''}")
                
                # Check if we've processed too many errors
                if not is_valid:
                    total_errors_processed += 1
                    if total_errors_processed >= max_total_errors:
                        print(f"\nProcessed maximum number of errors ({max_total_errors}) for this pattern.")
                        print("Skipping remaining lines to avoid getting stuck.")
                        break
                
                # If valid, update last valid DSL and line number
                if is_valid:
                    last_valid_dsl = current_dsl
                    last_valid_line = line_num + 1
            
            # Calculate Partial Executable Rate (PER)
            non_empty_lines = len([line for line in dsl_lines if line.strip()])
            per = last_valid_line / non_empty_lines if non_empty_lines > 0 else 0.0
            
            # Save PER and related metrics to pattern_result
            pattern_result["last_valid_line"] = last_valid_line
            pattern_result["non_empty_lines"] = non_empty_lines
            pattern_result["per"] = per
            
            print(f"\nPartial Executable Rate (PER): {per:.4f} ({last_valid_line}/{non_empty_lines} lines)")
            
            # Save the last valid DSL if it exists
            if last_valid_dsl:
                last_valid_path = f'{MODEL}_dsl_patterns/{safe_name}_last_valid.txt'
                with open(last_valid_path, 'w') as f:
                    f.write(last_valid_dsl)
                pattern_result["last_valid_file"] = last_valid_path
        
        # Add this pattern's results to the overall results
        progressive_results.append(pattern_result)
    
    # Save all results to a JSON file
    with open(f'{MODEL}_dsl_progressive_verification.json', 'w') as f:
        json.dump(progressive_results, f, indent=2)
    
    # Analyze error types and locations
    error_types = defaultdict(int)
    line_error_distribution = defaultdict(int)
    error_patterns = [p for p in progressive_results if not p["is_valid_full"]]
    pattern_error_types = {}  # Store error types for each pattern
    
    # Count error types and positions
    for pattern in error_patterns:
        pattern_name = pattern['pattern_name']
        pattern_error_types[pattern_name] = []
        
        if pattern.get("first_error_details") and pattern["first_error_details"].get("errors"):
            # Count errors by type
            for error in pattern["first_error_details"]["errors"]:
                error_type = categorize_error(error)
                error_types[error_type] += 1
                pattern_error_types[pattern_name].append(error_type)
            
            # Count errors by line position
            if pattern.get("first_error_at_line"):
                line_error_distribution[pattern["first_error_at_line"]] += 1
                
    # Add error types to results
    for pattern in progressive_results:
        if pattern['pattern_name'] in pattern_error_types:
            pattern['error_types'] = pattern_error_types[pattern['pattern_name']]
    
    # Report statistics
    print(f"\n{'='*80}")
    print(f"Verification Summary")
    print(f"{'='*80}")
    print(f"Total patterns processed: {total_stats['processed']}")
    print(f"Valid patterns: {total_stats['valid']} ({total_stats['valid']/max(1, total_stats['processed'])*100:.1f}%)")
    print(f"Invalid patterns: {total_stats['invalid']} ({total_stats['invalid']/max(1, total_stats['processed'])*100:.1f}%)")
    
    # Display PER results in a table
    print(f"\n{'='*80}")
    print("Partial Executable Rate (PER) Results")
    print(f"{'='*80}")
    print(f"{'Pattern ID':<30} {'Total Steps':>12} {'Passed':>8} {'PER':>8}")
    print(f"{'-'*30} {'-'*12} {'-'*8} {'-'*8}")
    
    # Calculate average PER
    total_per = 0.0
    pattern_count = 0
    
    # Prepare CSV data for export
    csv_rows = [['pattern_id', 'pattern_name', 'total_steps', 'passed_steps', 'per']]
    
    # Sort patterns by PER for better visualization
    sorted_patterns = sorted(progressive_results, key=lambda p: p.get('per', 0.0) if p.get('per') is not None else 0.0, reverse=True)
    
    for p in sorted_patterns:
        pattern_name = p['pattern_name']
        pattern_id = p.get('pattern_id', f"id_{p.get('pattern_index', 0)}")
        total_steps = p.get('non_empty_lines', 0)
        passed_steps = p.get('last_valid_line', 0)
        
        # Handle the case where passed_steps or total_steps might be None
        if passed_steps is None:
            passed_steps = 0
        if total_steps is None:
            total_steps = 0
            
        # For valid patterns, set passed_steps = total_steps
        if p.get('is_valid_full', False) and total_steps > 0:
            passed_steps = total_steps
            
        # Calculate PER
        per = passed_steps / total_steps if total_steps > 0 else 0.0
        
        # Display using pattern ID
        print(f"{pattern_id[:30]:<30} {total_steps:>12} {passed_steps:>8} {per:>8.4f}")
        
        # Add to CSV data
        csv_rows.append([pattern_id, pattern_name, total_steps, passed_steps, per])
        
        total_per += per
        pattern_count += 1
        
    # Export PER results to CSV
    csv_file = f'per_{MODEL}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nPER results exported to {csv_file}")
    
    # Print average PER if we have patterns
    if pattern_count > 0:
        avg_per = total_per / pattern_count
        print(f"\nAverage PER across {pattern_count} patterns: {avg_per:.4f}")
    
    # Calculate overall PER statistics for the analysis results
    per_values = [p.get('per', 0.0) for p in progressive_results if p.get('per') is not None]
    avg_per = sum(per_values) / len(per_values) if per_values else 0.0
    per_stats = {
        "average_per": avg_per,
        "pattern_count": len(per_values),
        "per_distribution": {
            "0.0-0.2": len([p for p in per_values if p < 0.2]),
            "0.2-0.4": len([p for p in per_values if 0.2 <= p < 0.4]),
            "0.4-0.6": len([p for p in per_values if 0.4 <= p < 0.6]),
            "0.6-0.8": len([p for p in per_values if 0.6 <= p < 0.8]),
            "0.8-1.0": len([p for p in per_values if p >= 0.8])
        }
    }
    
    # Display Error Type Breakdown
    if error_types:
        print(f"\n{'='*80}")
        print("Error Type Breakdown 错误类型分布")
        print(f"{'='*80}")
        print(f"{'Error Type':<30} {'Count':>10} {'Percentage':>15}")
        print(f"{'-'*30} {'-'*10} {'-'*15}")
        
        # Get total error count
        total_errors = sum(error_types.values())
        
        # Sort error types by frequency
        sorted_error_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        for error_type, count in sorted_error_types:
            percentage = (count / total_errors) * 100 if total_errors > 0 else 0
            print(f"{error_type[:30]:<30} {count:>10} {percentage:>14.1f}%")
    
    # Report on where errors occur
    if line_error_distribution:
        print(f"\nError Line Distribution:")
        print(f"{'-'*40}")
        sorted_lines = sorted(line_error_distribution.items(), key=lambda x: x[0])
        for line_num, count in sorted_lines:
            print(f"- Line {line_num}: {count} patterns")
    
    # Report on patterns with errors and where they first occur
    if error_patterns:
        print("\nPatterns with errors:")
        print(f"{'-'*40}")
        for p in error_patterns:
            line_info = f"at line {p['first_error_at_line']}" if p["first_error_at_line"] else "unknown line"
            error_msg = ""
            if p.get("first_error_details") and p["first_error_details"].get("errors"):
                error_msg = f" - {p['first_error_details']['errors'][0]}" if p['first_error_details']['errors'] else ""
            print(f"- {p['pattern_name']} (first error {line_info}){error_msg}")

    # Save the statistics to a JSON file
    analysis_results = {
        "total": total_stats,
        "per_stats": per_stats,
        "error_types": dict(error_types),
        "line_error_distribution": dict(line_error_distribution)
    }
    
    with open(f'{MODEL}_dsl_error_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\nResults saved to '{MODEL}_dsl_progressive_verification.json'")
    print(f"Pattern files saved in '{MODEL}_dsl_patterns/' and '{MODEL}_dsl_progressive_results/' directories")


if __name__ == "__main__":
    progressive_verify_dsl()
