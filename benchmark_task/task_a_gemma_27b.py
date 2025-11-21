import os
import json
from tqdm import tqdm
import torch
from transformers import pipeline
from pathlib import Path
SYSTEM_PROMPT = """Analyze this crochet image and list ALL visible stitches using ONLY standard U.S. crochet abbreviations.

RULES:
1. ONLY output a comma-separated list of abbreviations (e.g., 'sc, hdc, dc, tr, ch, sl st, pop')
2. NO explanations, NO additional text, NO markdown
3. NO periods, numbers, or special characters except commas
4. If no stitches are visible, output: none
5. Valid abbreviations only: sc, hdc, dc, tr, ch, sl st, pop
6. Remove any duplicate stitches
7. Sort abbreviations alphabetically

Example outputs:
sc, hdc, dc
dc, tr, ch
sc, hdc, dc, tr, ch, sl st
none"""
# --- Pipeline setup ---
gemma_pipe = pipeline(
    task="image-text-to-text",
    model="google/gemma-3-27b-pt",
    device=0,
    dtype=torch.bfloat16
)

# --- Directories ---
json_dir = "../data/crochet_pattern_by_project/"  # input JSONs
output_dir = "./task_a_gemma/"  # output JSONs
os.makedirs(output_dir, exist_ok=True)

# --- List all JSON files ---
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

# --- Iterate over JSON files ---
for json_file in tqdm(json_files, desc="Processing JSON files"):
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # --- Iterate over entries/images ---
    for idx, entry in enumerate(tqdm(data, desc=f"{json_file} images", leave=False)):
        # Try URL first, fallback to local path
        image_source = entry.get("image_link") or entry.get("image_path")
        if not image_source:
            continue
        
        try:
            # Prepare the prompt
            prompt_text = f"""{SYSTEM_PROMPT}
            
            <start_of_image>
            List all visible stitches using only standard abbreviations, comma-separated:"""
            
            # Handle both URLs and local paths
            if not image_source.startswith("http"):
                image_source = str(Path(image_source).resolve())
            
            try:
                # Generate response from the model
                output = gemma_pipe(
                    image_source,
                    text=prompt_text,
                    max_new_tokens=50,  # Keep it short since we only need abbreviations
                    temperature=0.1,    # Lower temperature for more focused output
                    do_sample=False,    # Disable sampling for more deterministic output
                    num_beams=1,        # Faster generation
                    early_stopping=True
                )
                
                # Extract the generated text
                full_response = output[0]["generated_text"]
                
                # Clean up the response
                stitches = full_response.replace(prompt_text, "").strip()
                
                # Clean up common issues
                stitches = (stitches.split('\n')[0]  # Take only first line
                          .split(':')[-1]             # Remove any prefix before a colon
                          .strip()                    # Remove extra whitespace
                          .lower()                    # Convert to lowercase
                          .replace('.', ',')          # Replace periods with commas
                          .replace(';', ',')          # Replace semicolons with commas
                          .replace(' and ', ', ')     # Replace 'and' with comma
                          .replace(' ', '')           # Remove all spaces
                          )
                
                # Remove any non-standard characters
                valid_stitches = {'sc', 'hdc', 'dc', 'tr', 'ch', 'slst', 'pop'}
                stitches_list = [s.strip() for s in stitches.split(',') if s.strip() in valid_stitches]
                
                # Remove duplicates and sort
                stitches_list = sorted(list(set(stitches_list)))
                
                # Join back into a clean string
                clean_stitches = ', '.join(stitches_list) if stitches_list else 'none'
                
                print(f"Generated stitches: {clean_stitches}")
                entry["generated_stitches"] = clean_stitches
                
            except Exception as e:
                print(f"Error processing image {image_source}: {str(e)}")
                entry["generated_stitches"] = "error"
            
        except Exception as e:
            print(f"Error processing {json_file} [{idx}]: {e}")
            continue
    
    # --- Save updated JSON ---
    output_path = os.path.join(output_dir, json_file)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

