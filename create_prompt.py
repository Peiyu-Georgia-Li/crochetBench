import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

JSON_DIR = "text_json"

def convert_to_prompt_json(input_file):
    """
    Convert a JSON file with crochet patterns to a prompt JSON format.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
    """
    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create the output JSON structure
    output_data = []
    
    # Process each pattern
    for i, pattern in enumerate(data):

        
        # Extract the image filename from the image_link or source
        image_filename = None
        if 'image_link' in pattern:
            image_filename = pattern['image_link']
        else:
            continue

        SYSTEM_PROMPT = (
  "You are a senior crochet pattern designer and technical editor. "
  "Task: infer a complete crochet pattern from the provided product/detail photo(s). "
  "Output JSON only with the exact keys in this fixed order: "
  "pattern_name, gauge, measurements, materials, abbreviations, instructions. "
  "Rules: 1) Output JSON only; 2) Use US terms (sc, hdc, dc, tr, sl st, ch, inc, dec, sk); "
  "3) Use inches (in; cm optional in parentheses); "
  "4) Gauge format: \"<stitches> and <rows> = 4 in (<stitch>, hook size)\" or null; "
  "5) measurements = key finished dimensions; null if unknown; "
  "6) materials = array of strings (no brands/colors); "
  "7) abbreviations = only those used, e.g., \"Sc=Single crochet\"; "
  "8) instructions = multi-line string labeled R1, R2... or Round 1, etc.; "
  "use *...* for repeats, () for groups, [] for sizes, and end each row with stitch count when applicable; "
  "9) Valid JSON only (double quotes, no trailing commas), keep key order."
)
        USER_INSTRUCTION = "Extract the full pattern as JSON."

        # Create the user prompt
        user_prompt = f"Picture 1: <img>{image_filename}</img>\n{USER_INSTRUCTION}"
        
        # Create the assistant response (JSON string)
        # Ensure the keys are in the specified order
        response_data = {
            "pattern_name": pattern.get("pattern_name", ""),
            "gauge": pattern.get("gauge", "null"),
            "measurements": pattern.get("measurements", "null"),
            "materials": clean_materials(pattern.get("materials", [])),
            "abbreviations": pattern.get("abbreviations", []),
            "instructions": pattern.get("instructions", "")
        }
        
        # Convert to JSON string
        assistant_response = json.dumps(response_data, ensure_ascii=False)
        
        # Create the prompt JSON entry
        prompt_entry = {
            "id": pattern.get("id", ""),
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {
                    "from": "user",
                    "value": user_prompt
                },
                {
                    "from": "assistant",
                    "value": assistant_response
                }
            ]
        }
        
        output_data.append(prompt_entry)

    input_file_name = input_file.split("/")[-1].split(".")[0]
    output_file = f"prompt_json/prompt_{input_file_name}.json"

    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(output_data)} patterns to prompt JSON format.")
    print(f"Output saved to {output_file}")

def clean_materials(materials):
    """
    Clean the materials list by removing brand names and colors.
    
    Args:
        materials (list): List of material strings
        
    Returns:
        list: Cleaned materials list
    """
    cleaned_materials = []
    
    for material in materials:
        # Remove specific brand names and color codes
        # This is a simple approach - you might need to enhance this based on your data
        material = material.split(':', 1)[-1] if ':' in material else material
        
        # Remove color codes and brand names in parentheses
        import re
        material = re.sub(r'\([^)]*\d+[^)]*\)', '', material)
        material = re.sub(r'#\d+', '', material)
        
        # Remove registered trademark symbols and other special characters
        material = material.replace('\u00ae', '').replace('\u2122', '').replace('\u2019', "'")
        
        # Remove extra spaces and clean up
        material = re.sub(r'\s+', ' ', material).strip()
        material = re.sub(r'\s+,', ',', material)
        
        if material:
            cleaned_materials.append(material)
    
    return cleaned_materials

def main():
    # parser = argparse.ArgumentParser(description='Convert JSON files to prompt JSON format')
    # parser.add_argument('input_file', help='Path to the input JSON file')
    
    # args = parser.parse_args()
    
    
    # Get the absolute path to the JSON directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(script_dir, JSON_DIR)
    
    # Get all JSON files
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(json_dir, f))][:1]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(convert_to_prompt_json, json_files))


if __name__ == '__main__':
    main()
