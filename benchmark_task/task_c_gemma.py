import os
import json
from tqdm import tqdm
import torch
from transformers import pipeline
from pathlib import Path
SYSTEM_PROMPT = """
You are a professional crochet pattern writer. 
Examine the image of the finished crochet product carefully. 
Write a complete set of crochet instructions in the standard style used in published patterns.

Requirements:
- Use standard abbreviations: sc (single crochet), hdc (half double crochet), dc (double crochet), tr (treble), ch (chain), sl st (slip stitch), rep (repeat).
- Organize the instructions row by row or round by round (e.g., "Rnd 1: ...", "Row 2: ...").
- If color changes are visible in the image, include them in the pattern.
- Keep the instructions concise and precise, as if for experienced crocheters.
- Output only the crochet pattern. Do not add any explanations, commentary, or extra text."""
# --- Pipeline setup ---
gemma_pipe = pipeline(
    task="image-text-to-text",
    model="google/gemma-3-4b-pt",
    device=0,
    dtype=torch.bfloat16
)

# --- Directories ---
json_dir = "../data/crochet_pattern_by_project/"  # input JSONs
output_dir = "./task_c_gemma/"  # output JSONs
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
            # Prepare prompt for Gemma with system instructions
            prompt_text = f"""{SYSTEM_PROMPT}
            
            <start_of_image>
            Generate step-by-step crochet instructions for this image."""
            
            # Gemma pipeline expects either URL or local path
            if not image_source.startswith("http"):
                # ensure absolute path
                image_source = Path(image_source).resolve().as_posix()
            
            # Generate caption / instructions
            output = gemma_pipe(image_source, text=prompt_text, max_new_tokens=1024)
            
            # Extract just the generated text (after the input prompt)
            full_response = output[0]["generated_text"]
            # Remove the input prompt from the response
            generated_instructions = full_response.replace(prompt_text, "").strip()
            
            print(f"Generated instructions: {generated_instructions}") 
            entry["generated_instructions"] = generated_instructions
            
        except Exception as e:
            print(f"Error processing {json_file} [{idx}]: {e}")
            continue
    
    # --- Save updated JSON ---
    output_path = os.path.join(output_dir, json_file)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

