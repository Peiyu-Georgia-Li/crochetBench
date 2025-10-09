import os
import json
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import base64

import anthropic
import argparse

# --- Claude API Setup ---
# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate crochet stitch abbreviations using Claude API')
parser.add_argument('--model', type=str, default="claude-sonnet-4-20250514", 
                    help='Claude model to use (default: claude-sonnet-4-20250514)')
args = parser.parse_args()

# Choose your Claude model
model = args.model

SYSTEM_PROMPT = """
You are a crochet stitch expert. Identify which stitches appear in a crochet product image. 
Always answer using standard U.S. crochet abbreviations only (sc, hdc, dc, tr, ch, sl st, pop, etc.). 
Output only a comma-separated list of abbreviations—no explanations or extra commentary.
"""

# --- Directories ---
json_dir = "../data/crochet_pattern_by_project/"  # Input JSON directory
output_dir = "./task_a_claude/"
os.makedirs(output_dir, exist_ok=True)

# --- Initialize Claude client ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- Load Image from URL ---
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        return None

# --- Convert image to base64 ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# --- Process all JSON files ---
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
for json_file in tqdm(json_files, desc="Processing with Claude"):
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
            
        image = load_image_from_url(image_source)
        if image is None:
            continue

        # --- Run Claude Multimodal ---
        try:
            image_base64 = image_to_base64(image)
            
            message = client.messages.create(
                model=model,
                max_tokens=100,  # Reduced token count since we only need abbreviations
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": "Look at this crochet product image and list the stitches used."
                            }
                        ]
                    }
                ]
            )
            
            output_text = message.content[0].text.strip()
            
        except Exception as e:
            print(f"❌ Claude failed on {idx}: {e}")
            continue

        # --- Write output ---
        entry["generated_stitches"] = output_text
        print("identified stitches:", output_text)
    
    # --- Save updated data ---
    out_path = os.path.join(output_dir, json_file)
    with open(out_path, "w") as f_out:
        json.dump(entries, f_out, indent=2)

print(f"Finished processing all files. Results saved to {output_dir}")
