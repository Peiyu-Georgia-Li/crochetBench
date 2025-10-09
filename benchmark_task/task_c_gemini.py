import os
import json
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

from google import genai
import argparse
# --- Gemini API Setup ---
# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate crochet instructions using Gemini API')
parser.add_argument('--model', type=str, default="gemini-2.5-flash-lite", 
                    help='Gemini model to use (default: gemini-2.5-flash-lite)')
args = parser.parse_args()
# Choose your Gemini model (Flash is faster, Pro is better quality)
model = args.model

SYSTEM_PROMPT = """
You are a professional crochet pattern writer.
Examine the image of the finished crochet product carefully.
Write a complete set of crochet instructions in the standard style used in published patterns.

Requirements:
- Use standard abbreviations: sc (single crochet), hdc (half double crochet), dc (double crochet), tr (treble), ch (chain), sl st (slip stitch), rep (repeat).
- Organize the instructions row by row or round by round (e.g., "Rnd 1: ...", "Row 2: ...").
- If color changes are visible in the image, include them in the pattern.
- Keep the instructions concise and precise, as if for experienced crocheters.
- Output only the crochet pattern. Do not add any explanations, commentary, or extra text.
"""

# --- Directories ---
json_dir = "../data/crochet_pattern_by_project/"  # Input JSON directory
output_dir = "./task_c_gemini/"
os.makedirs(output_dir, exist_ok=True)

# --- Load Image from URL ---
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        return None

# --- Process all JSON files ---
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
for json_file in tqdm(json_files, desc="Processing with Gemini"):
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # If the dictionary has a list of entries, use that
        if any(isinstance(v, list) for v in data.values()):
            for key, value in data.items():
                if isinstance(value, list):
                    entries = value
                    break
        else:
            # Otherwise treat the dictionary as a single entry
            entries = [data]
    else:
        # It's already a list
        entries = data
    
    for idx, entry in enumerate(tqdm(entries, desc=f"{json_file} images", leave=False)):
        # Try URL first, fallback to local path
        if not isinstance(entry, dict):
            print(f"Skipping non-dictionary entry at index {idx}")
            continue
            
        image_source = entry.get("image_link") or entry.get("image_path")
        if not image_source:
            print(f"No image source found for entry {idx}")
            continue
        #image_url = data.get("image_url") or data.get("image")  # Adjust key as needed
        if not image_source:
            print(f"⚠️ No image source in {idx}")
            continue

        image = load_image_from_url(image_source)
        if image is None:
            continue

        # --- Run Gemini Multimodal ---
        try:
            client=genai.Client()
            response = client.models.generate_content(
                model=model,
                contents=[SYSTEM_PROMPT, image, "Generate step-by-step crochet instructions for this image."],
            )
            output_text = response.text.strip()
        except Exception as e:
            print(f"❌ Gemini failed on {idx}: {e}")
            continue

        # --- Write output ---
        entry["generated_instructions"] = output_text
        print("output instructions:", output_text)
        out_path = os.path.join(output_dir, json_file)
        with open(out_path, "w") as f_out:
            json.dump(entries, f_out, indent=2)
