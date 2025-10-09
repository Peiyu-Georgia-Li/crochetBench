import json
import os
import glob
import sys
def load_unique_stitches():
    """
    Load the set of unique stitches from the data_analysis/unique_stitches.txt file
    Returns a set of stitch names (lowercase, without counts)
    """
    stitch_set = set()
    try:
        with open("../data_analysis/unique_stitches.txt", "r") as f:
            for line in f:
                # Extract just the stitch name without the count in parentheses
                if "(" in line:
                    stitch = line.split("(")[0].strip().lower()
                    stitch_set.add(stitch)
                else:
                    stitch_set.add(line.strip().lower())
        return stitch_set
    except FileNotFoundError:
        print("Warning: unique_stitches.txt file not found.")
        return set()

# Load the unique stitches set once
UNIQUE_STITCHES = load_unique_stitches()

def extract_abbrev(abbrev_list):
    """
    Convert the abbreviations list into a set of canonical short forms
    e.g., "Sl st = Slip stitch" -> "sl st"
         "sc - single crochet" -> "sc"
         "sl st : slip stitch" -> "sl st"
         "[ = Begin repeat" -> "["
    
    Also categorizes abbreviations as either stitch or non-stitch based on
    whether they appear in unique_stitches.txt.
    """
    stitch_abbrev_set = set()
    non_stitch_abbrev_set = set()
    
    for entry in abbrev_list:
        # Handle multiple separator styles (=, -, :)
        if "=" in entry:
            short_form = entry.split("=")[0].strip().lower()  # get left side, lowercase
        elif " - " in entry:
            short_form = entry.split(" - ")[0].strip().lower()
        elif " : " in entry:
            short_form = entry.split(" : ")[0].strip().lower()
        elif "-" in entry and not entry.startswith("-"):  # Ensure it's not just a negative sign
            short_form = entry.split("-")[0].strip().lower()
        elif ":" in entry:
            short_form = entry.split(":")[0].strip().lower()
        else:
            # If there's no separator, treat the whole entry as the short form
            short_form = entry.strip().lower()
        
        # Check if this abbreviation is a stitch
        if short_form in UNIQUE_STITCHES:
            stitch_abbrev_set.add(short_form)
        else:
            non_stitch_abbrev_set.add(short_form)
    
    return stitch_abbrev_set, non_stitch_abbrev_set

def extract_generated_stitches(generated_str):
    """
    Convert the generated_stitches string into a set of canonical stitches
    """
    return set(s.strip().lower() for s in generated_str.split(","))

args = sys.argv
model_name = args[1]

# --- Find all JSON files in task_a_qwen/ ---
json_files = glob.glob(f"task_a_{model_name}/*.json")
# Skip files that are not pattern data files
json_files = [f for f in json_fi1les if not os.path.basename(f) in ["abbreviation_progress.json", "missing_abbreviations.json"]]
print(f"Found {len(json_files)} JSON files to process")

# --- Process each file ---
coverage_results = []

for json_file in sorted(json_files):
    category_name = os.path.basename(json_file).replace('.json', '')
    print(f"Processing {category_name}...")
    
    # Load the JSON data
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        continue
        
    # Process each item in the file
    for item in data:
        stitch_abbrev_set, non_stitch_abbrev_set = extract_abbrev(item.get("abbreviations", []))
        
        # Handle different field names for different models
        if model_name == "gpt4o":
            generated_field = "generated_instructions"
        else:
            generated_field = "generated_stitches"
            
        generated_set = extract_generated_stitches(item.get(generated_field, ""))
        
        # Calculate coverage for stitch abbreviations
        if not stitch_abbrev_set:  # avoid division by zero
            stitch_coverage = None
            stitch_matched = set()
        else:
            stitch_matched = stitch_abbrev_set.intersection(generated_set)
            stitch_coverage = len(stitch_matched) / len(stitch_abbrev_set) * 100 if stitch_abbrev_set else None
        
        # Calculate coverage for non-stitch abbreviations
        if not non_stitch_abbrev_set:  # avoid division by zero
            non_stitch_coverage = None
            non_stitch_matched = set()
        else:
            non_stitch_matched = non_stitch_abbrev_set.intersection(generated_set)
            non_stitch_coverage = len(non_stitch_matched) / len(non_stitch_abbrev_set) * 100 if non_stitch_abbrev_set else None
        
        # Calculate total coverage
        all_abbrev = stitch_abbrev_set.union(non_stitch_abbrev_set)
        if not all_abbrev:  # avoid division by zero
            total_coverage = None
            total_matched = set()
        else:
            total_matched = all_abbrev.intersection(generated_set)
            total_coverage = len(total_matched) / len(all_abbrev) * 100
        
        # Calculate F1 scores
        # For stitches
        stitch_tp = len(stitch_matched)
        stitch_fp = len(generated_set - stitch_abbrev_set)
        stitch_fn = len(stitch_abbrev_set - generated_set)
        
        stitch_precision = stitch_tp / (stitch_tp + stitch_fp) if (stitch_tp + stitch_fp) > 0 else 0
        stitch_recall = stitch_tp / (stitch_tp + stitch_fn) if (stitch_tp + stitch_fn) > 0 else 0
        stitch_f1 = 2 * (stitch_precision * stitch_recall) / (stitch_precision + stitch_recall) if (stitch_precision + stitch_recall) > 0 else 0
        
        # For non-stitches
        non_stitch_tp = len(non_stitch_matched)
        non_stitch_fp = len(generated_set - non_stitch_abbrev_set)
        non_stitch_fn = len(non_stitch_abbrev_set - generated_set)
        
        non_stitch_precision = non_stitch_tp / (non_stitch_tp + non_stitch_fp) if (non_stitch_tp + non_stitch_fp) > 0 else 0
        non_stitch_recall = non_stitch_tp / (non_stitch_tp + non_stitch_fn) if (non_stitch_tp + non_stitch_fn) > 0 else 0
        non_stitch_f1 = 2 * (non_stitch_precision * non_stitch_recall) / (non_stitch_precision + non_stitch_recall) if (non_stitch_precision + non_stitch_recall) > 0 else 0
        
        # Total F1
        total_tp = len(total_matched)
        total_fp = len(generated_set - all_abbrev)
        total_fn = len(all_abbrev - generated_set)
        
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        
        coverage_results.append({
            "category": category_name,
            "pattern_name": item.get("pattern_name", "unknown"),
            "total_coverage_percent": total_coverage,
            "stitch_coverage_percent": stitch_coverage,
            "non_stitch_coverage_percent": non_stitch_coverage,
            
            # Add F1 scores to results
            "stitch_precision": stitch_precision,
            "stitch_recall": stitch_recall,
            "stitch_f1": stitch_f1,
            
            "non_stitch_precision": non_stitch_precision,
            "non_stitch_recall": non_stitch_recall,
            "non_stitch_f1": non_stitch_f1,
            
            "total_precision": total_precision,
            "total_recall": total_recall,
            "total_f1": total_f1,
            
            "total_matched": list(total_matched),
            "stitch_matched": list(stitch_matched),
            "non_stitch_matched": list(non_stitch_matched),
            "generated": list(generated_set),
            "reference_stitches": list(stitch_abbrev_set),
            "reference_non_stitches": list(non_stitch_abbrev_set),
            "reference_all": list(all_abbrev)
        })

# --- Summary ---
total_patterns = len(coverage_results)

# Calculate overall averages
valid_total_coverages = [c["total_coverage_percent"] for c in coverage_results if c["total_coverage_percent"] is not None]
average_total_coverage = sum(valid_total_coverages) / len(valid_total_coverages) if valid_total_coverages else 0

valid_stitch_coverages = [c["stitch_coverage_percent"] for c in coverage_results if c["stitch_coverage_percent"] is not None]
average_stitch_coverage = sum(valid_stitch_coverages) / len(valid_stitch_coverages) if valid_stitch_coverages else 0

valid_non_stitch_coverages = [c["non_stitch_coverage_percent"] for c in coverage_results if c["non_stitch_coverage_percent"] is not None]
average_non_stitch_coverage = sum(valid_non_stitch_coverages) / len(valid_non_stitch_coverages) if valid_non_stitch_coverages else 0

# Calculate counts
total_stitch_abbrevs = sum(len(c["reference_stitches"]) for c in coverage_results)
total_non_stitch_abbrevs = sum(len(c["reference_non_stitches"]) for c in coverage_results)

# Calculate overall metrics (average across all patterns)
stitch_precision = sum(c["stitch_precision"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
stitch_recall = sum(c["stitch_recall"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
stitch_f1 = sum(c["stitch_f1"] for c in coverage_results) / len(coverage_results) if coverage_results else 0

non_stitch_precision = sum(c["non_stitch_precision"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
non_stitch_recall = sum(c["non_stitch_recall"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
non_stitch_f1 = sum(c["non_stitch_f1"] for c in coverage_results) / len(coverage_results) if coverage_results else 0

total_precision = sum(c["total_precision"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
total_recall = sum(c["total_recall"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
total_f1 = sum(c["total_f1"] for c in coverage_results) / len(coverage_results) if coverage_results else 0
# Summary by category
print("\n===== SUMMARY BY CATEGORY =====")
categories = sorted(set(c["category"] for c in coverage_results))

for category in categories:
    category_results = [c for c in coverage_results if c["category"] == category]
    category_patterns = len(category_results)
    
    # Calculate category averages
    cat_valid_total = [c["total_coverage_percent"] for c in category_results if c["total_coverage_percent"] is not None]
    cat_avg_total = sum(cat_valid_total) / len(cat_valid_total) if cat_valid_total else 0
    
    cat_valid_stitch = [c["stitch_coverage_percent"] for c in category_results if c["stitch_coverage_percent"] is not None]
    cat_avg_stitch = sum(cat_valid_stitch) / len(cat_valid_stitch) if cat_valid_stitch else 0
    
    cat_valid_non_stitch = [c["non_stitch_coverage_percent"] for c in category_results if c["non_stitch_coverage_percent"] is not None]
    cat_avg_non_stitch = sum(cat_valid_non_stitch) / len(cat_valid_non_stitch) if cat_valid_non_stitch else 0
    
    # Calculate category counts
    cat_stitch_abbrevs = sum(len(c["reference_stitches"]) for c in category_results)
    cat_non_stitch_abbrevs = sum(len(c["reference_non_stitches"]) for c in category_results)
    
    # Calculate category F1 scores
    cat_macro_stitch_f1 = sum(c["stitch_f1"] for c in category_results) / len(category_results) if category_results else 0
    cat_macro_non_stitch_f1 = sum(c["non_stitch_f1"] for c in category_results) / len(category_results) if category_results else 0
    cat_macro_total_f1 = sum(c["total_f1"] for c in category_results) / len(category_results) if category_results else 0
    
    print(f"\nCategory: {category}")
    print(f"  Patterns: {category_patterns}")
    print(f"  Avg total coverage: {cat_avg_total:.2f}%")
    print(f"  Avg stitch coverage: {cat_avg_stitch:.2f}%")
    print(f"  Avg non-stitch coverage: {cat_avg_non_stitch:.2f}%")
    print(f"  Stitch abbreviations: {cat_stitch_abbrevs}")
    print(f"  Non-stitch abbreviations: {cat_non_stitch_abbrevs}")
    if cat_non_stitch_abbrevs:
        print(f"  Ratio stitch:non-stitch: {cat_stitch_abbrevs / cat_non_stitch_abbrevs:.2f} : 1")
    
    # Calculate category metrics
    cat_stitch_precision = sum(c["stitch_precision"] for c in category_results) / len(category_results) if category_results else 0
    cat_stitch_recall = sum(c["stitch_recall"] for c in category_results) / len(category_results) if category_results else 0
    cat_stitch_f1 = sum(c["stitch_f1"] for c in category_results) / len(category_results) if category_results else 0
    
    cat_non_stitch_precision = sum(c["non_stitch_precision"] for c in category_results) / len(category_results) if category_results else 0
    cat_non_stitch_recall = sum(c["non_stitch_recall"] for c in category_results) / len(category_results) if category_results else 0
    cat_non_stitch_f1 = sum(c["non_stitch_f1"] for c in category_results) / len(category_results) if category_results else 0
    
    cat_total_precision = sum(c["total_precision"] for c in category_results) / len(category_results) if category_results else 0
    cat_total_recall = sum(c["total_recall"] for c in category_results) / len(category_results) if category_results else 0
    cat_total_f1 = sum(c["total_f1"] for c in category_results) / len(category_results) if category_results else 0
    
    # Print category precision scores
    print(f"  Precision:")
    print(f"    Stitch: {cat_stitch_precision:.4f}")
    print(f"    Non-stitch: {cat_non_stitch_precision:.4f}")
    print(f"    Total: {cat_total_precision:.4f}")
    
    # Print category recall scores
    print(f"  Recall:")
    print(f"    Stitch: {cat_stitch_recall:.4f}")
    print(f"    Non-stitch: {cat_non_stitch_recall:.4f}")
    print(f"    Total: {cat_total_recall:.4f}")
    
    # Print category F1 scores
    print(f"  F1 scores:")
    print(f"    Stitch: {cat_stitch_f1:.4f}")
    print(f"    Non-stitch: {cat_non_stitch_f1:.4f}")
    print(f"    Total: {cat_total_f1:.4f}")
# Print overall summary
print("===== OVERALL SUMMARY =====")
print(f"Total patterns: {total_patterns}")
print(f"Average total coverage: {average_total_coverage:.2f}%")
print(f"Average stitch coverage: {average_stitch_coverage:.2f}%")
print(f"Average non-stitch coverage: {average_non_stitch_coverage:.2f}%")
print(f"Total stitch abbreviations: {total_stitch_abbrevs}")
print(f"Total non-stitch abbreviations: {total_non_stitch_abbrevs}")
print(f"Ratio of stitch to non-stitch: {total_stitch_abbrevs / total_non_stitch_abbrevs:.2f} : 1" if total_non_stitch_abbrevs else "No non-stitch abbreviations found.")

# Print precision scores
print("\n===== PRECISION SCORES =====")
print(f"  Stitch: {stitch_precision:.4f}")
print(f"  Non-stitch: {non_stitch_precision:.4f}")
print(f"  Total: {total_precision:.4f}")

# Print recall scores
print("\n===== RECALL SCORES =====")
print(f"  Stitch: {stitch_recall:.4f}")
print(f"  Non-stitch: {non_stitch_recall:.4f}")
print(f"  Total: {total_recall:.4f}")

# Print F1 scores
print("\n===== F1 SCORES =====")
print(f"  Stitch: {stitch_f1:.4f}")
print(f"  Non-stitch: {non_stitch_f1:.4f}")
print(f"  Total: {total_f1:.4f}")



# Save results with model name
results_filename = f"stitch_coverage_results_{model_name}.json"
summary_filename = f"stitch_coverage_summary_{model_name}.txt"

print(f"\nSaving detailed results to {results_filename}")
with open(results_filename, "w") as f:
    json.dump(coverage_results, f, indent=2)
    
# Save summary to a text file
print(f"Saving summary to {summary_filename}")
with open(summary_filename, "w") as f:
    # Write category summaries
    f.write("\n===== SUMMARY BY CATEGORY =====\n")
    for category in categories:
        category_results = [c for c in coverage_results if c["category"] == category]
        category_patterns = len(category_results)
        
        # Calculate category averages
        cat_valid_total = [c["total_coverage_percent"] for c in category_results if c["total_coverage_percent"] is not None]
        cat_avg_total = sum(cat_valid_total) / len(cat_valid_total) if cat_valid_total else 0
        
        cat_valid_stitch = [c["stitch_coverage_percent"] for c in category_results if c["stitch_coverage_percent"] is not None]
        cat_avg_stitch = sum(cat_valid_stitch) / len(cat_valid_stitch) if cat_valid_stitch else 0
        
        cat_valid_non_stitch = [c["non_stitch_coverage_percent"] for c in category_results if c["non_stitch_coverage_percent"] is not None]
        cat_avg_non_stitch = sum(cat_valid_non_stitch) / len(cat_valid_non_stitch) if cat_valid_non_stitch else 0
        
        # Calculate category counts
        cat_stitch_abbrevs = sum(len(c["reference_stitches"]) for c in category_results)
        cat_non_stitch_abbrevs = sum(len(c["reference_non_stitches"]) for c in category_results)
        
        # Calculate category metrics
        cat_stitch_precision = sum(c["stitch_precision"] for c in category_results) / len(category_results) if category_results else 0
        cat_stitch_recall = sum(c["stitch_recall"] for c in category_results) / len(category_results) if category_results else 0
        cat_stitch_f1 = sum(c["stitch_f1"] for c in category_results) / len(category_results) if category_results else 0
        
        cat_non_stitch_precision = sum(c["non_stitch_precision"] for c in category_results) / len(category_results) if category_results else 0
        cat_non_stitch_recall = sum(c["non_stitch_recall"] for c in category_results) / len(category_results) if category_results else 0
        cat_non_stitch_f1 = sum(c["non_stitch_f1"] for c in category_results) / len(category_results) if category_results else 0
        
        cat_total_precision = sum(c["total_precision"] for c in category_results) / len(category_results) if category_results else 0
        cat_total_recall = sum(c["total_recall"] for c in category_results) / len(category_results) if category_results else 0
        cat_total_f1 = sum(c["total_f1"] for c in category_results) / len(category_results) if category_results else 0
        
        f.write(f"\nCategory: {category}\n")
        f.write(f"  Patterns: {category_patterns}\n")
        f.write(f"  Avg total coverage: {cat_avg_total:.2f}%\n")
        f.write(f"  Avg stitch coverage: {cat_avg_stitch:.2f}%\n")
        f.write(f"  Avg non-stitch coverage: {cat_avg_non_stitch:.2f}%\n")
        f.write(f"  Stitch abbreviations: {cat_stitch_abbrevs}\n")
        f.write(f"  Non-stitch abbreviations: {cat_non_stitch_abbrevs}\n")
        if cat_non_stitch_abbrevs:
            f.write(f"  Ratio stitch:non-stitch: {cat_stitch_abbrevs / cat_non_stitch_abbrevs:.2f} : 1\n")
        
        # Write category precision scores
        f.write(f"  Precision:\n")
        f.write(f"    Stitch: {cat_stitch_precision:.4f}\n")
        f.write(f"    Non-stitch: {cat_non_stitch_precision:.4f}\n")
        f.write(f"    Total: {cat_total_precision:.4f}\n")
        
        # Write category recall scores
        f.write(f"  Recall:\n")
        f.write(f"    Stitch: {cat_stitch_recall:.4f}\n")
        f.write(f"    Non-stitch: {cat_non_stitch_recall:.4f}\n")
        f.write(f"    Total: {cat_total_recall:.4f}\n")
        
        # Write category F1 scores
        f.write(f"  F1 scores:\n")
        f.write(f"    Stitch: {cat_stitch_f1:.4f}\n")
        f.write(f"    Non-stitch: {cat_non_stitch_f1:.4f}\n")
        f.write(f"    Total: {cat_total_f1:.4f}\n")

    # Write overall summary
    f.write("===== OVERALL SUMMARY =====\n")
    f.write(f"Total patterns: {total_patterns}\n")
    f.write(f"Average total coverage: {average_total_coverage:.2f}%\n")
    f.write(f"Average stitch coverage: {average_stitch_coverage:.2f}%\n")
    f.write(f"Average non-stitch coverage: {average_non_stitch_coverage:.2f}%\n")
    f.write(f"Total stitch abbreviations: {total_stitch_abbrevs}\n")
    f.write(f"Total non-stitch abbreviations: {total_non_stitch_abbrevs}\n")
    if total_non_stitch_abbrevs:
        f.write(f"Ratio of stitch to non-stitch: {total_stitch_abbrevs / total_non_stitch_abbrevs:.2f} : 1\n")
    else:
        f.write("No non-stitch abbreviations found.\n")
        
    # Write precision scores
    f.write("\n===== PRECISION SCORES =====\n")
    f.write(f"  Stitch: {stitch_precision:.4f}\n")
    f.write(f"  Non-stitch: {non_stitch_precision:.4f}\n")
    f.write(f"  Total: {total_precision:.4f}\n")
    
    # Write recall scores
    f.write("\n===== RECALL SCORES =====\n")
    f.write(f"  Stitch: {stitch_recall:.4f}\n")
    f.write(f"  Non-stitch: {non_stitch_recall:.4f}\n")
    f.write(f"  Total: {total_recall:.4f}\n")
    
    # Write F1 scores
    f.write("\n===== F1 SCORES =====\n")
    f.write(f"  Stitch: {stitch_f1:.4f}\n")
    f.write(f"  Non-stitch: {non_stitch_f1:.4f}\n")
    f.write(f"  Total: {total_f1:.4f}\n")
    
