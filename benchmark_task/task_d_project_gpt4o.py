import os
import json
import base64
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import argparse

# --- GPT-4V API Setup ---
parser = argparse.ArgumentParser(description='Generate crochet DSL using GPT-4V API with few-shot learning')
parser.add_argument('--model', type=str, default="gpt-4o",
                    help='GPT model to use')
args = parser.parse_args()
model = args.model

SYSTEM_PROMPT = """
You are a professional crochet pattern writer.
Convert instructions + images into compilable CrochetPARADE DSL code.
Output only the DSL code. No explanations, commentary, or extra text.
"""

# --- Few-shot reference examples ---
FEW_SHOT_EXAMPLES = [
    {
        "image_path": "https://www.yarnspirations.com/cdn/shop/files/BRC0116-035467M.jpg",
        "instructions": """Note:  Join with sl st to first sc at end of each rnd.
        Ch 2.
        **Rnd 1:** 6 sc in 2nd ch from hook. Join. (6 sc)
        **Rnd 2:** Ch 1. 2 sc in each sc around. Join. (12 sc)
        **Rnd 3:** Ch 1. _(2 sc in next sc, 1 sc in next sc)_ repeat around. End with 1 sc. Join. (18 sc)
        **Rnd 4:** Ch 1. _(2 sc in next sc, 1 sc in each of next 2 sc)_ repeat. End with 1 sc in last 2 sc. Join. (24 sc)
        **Rnd 5:** Ch 1. Sc in each sc around. Join. (24 sc)
        **Rnd 6:** Ch 1. _(2 sc in next sc, 1 sc in each of next 3 sc)_ repeat. End with 1 sc in last 3 sc. Join. (30 sc)
        **Rnds 7–8:** Repeat Rnd 5 (sc in each sc). Join. (30 sc each round)
        **Rnd 9:** Ch 1. **Working in back loops only**: _(2 sc in next sc, 1 sc in each of next 2 sc)_ repeat. End with 1 sc in last 2 sc. Join. (40 sc)
        **Rnd 10:** Ch 1. Sc in each sc around (both loops). Join. (40 sc)
        **Rnd 11:** Ch 1. _(2 sc in next sc, 1 sc in each of next 3 sc)_ repeat. End with 1 sc in last 3 sc. Join. (50 sc)
        **Finish:** Fasten off.""",
        "dsl": """
        ch.B
        sc@B.A,5sc@B,ss@A
        ch.A,sk,6sc2inc,ss@A
        ch.A,sk,[sc2inc,sc]*6,ss@A
        ch.A,sk,[sc2inc,2sc]*6,ss@A
        ch.A,sk,24sc,ss@A
        ch.A,sk,[sc2inc,3sc]*6,ss@A
        [ch.A,sk,30sc,ss@A
        ]*2
        ch.A,sk,[scbl,scbl@[@],2scbl]*10,ss@A
        ch.A,sk,40sc,ss@A
        ch.A,sk,[sc2inc,3sc]*10,ss@A"""
    }
]

# --- API setup ---
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

GPT_API_URL = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# --- Directories ---
output_dir = "./generated_dsl_gpt4v/"
os.makedirs(output_dir, exist_ok=True)

# --- Helpers ---
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        return None

def encode_image_to_base64(image_path_or_url):
    """Convert image to base64 encoding for API submission"""
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image from {image_path_or_url}")
        image_content = response.content
    else:
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
        with open(image_path_or_url, "rb") as image_file:
            image_content = image_file.read()
    
    return base64.b64encode(image_content).decode('utf-8')

# --- Process JSON files ---
json_file = 'selected_crochet_patterns_3.json'
json_dir = '/store01/nchawla/pli9/crochet/'
json_path = os.path.join(json_dir, json_file)
with open(json_path, "r") as f:
    data = json.load(f)

# Normalize to list of entries
if isinstance(data, dict):
    if any(isinstance(v, list) for v in data.values()):
        for key, value in data.items():
            if isinstance(value, list):
                entries = value
                break
    else:
        entries = [data]
else:
    entries = data

for idx, entry in enumerate(tqdm(entries, desc=f"{json_file} entries", leave=False)):
    image_source = entry.get("image_link") or entry.get("image_path")
    if not image_source:
        continue

    try:
        # --- Build few-shot + target content ---
        message_content = []

        # Add few-shot examples as separate images + text
        for ex in FEW_SHOT_EXAMPLES:
            try:
                ex_base64 = encode_image_to_base64(ex["image_path"])
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ex_base64}"
                    }
                })
                message_content.append({
                    "type": "text",
                    "text": f"Convert the following instructions into DSL:\n{ex['instructions']}\nExpected DSL:\n{ex['dsl']}"
                })
            except Exception as e:
                print(f"Failed to process example image: {e}")
                continue

        # Add target image + instructions
        target_base64 = encode_image_to_base64(image_source)
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{target_base64}"
            }
        })
        target_instructions = entry.get("instructions", "")
        message_content.append({
            "type": "text",
            "text": f"Convert the following instructions into DSL:\n{target_instructions}"
        })

        # --- Build messages ---
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message_content
            }
        ]

        # --- Send to GPT-4V ---
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 3000
        }

        response = requests.post(GPT_API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            continue

        response_data = response.json()
        dsl_output = response_data["choices"][0]["message"]["content"].strip()
        entry["generated_dsl"] = dsl_output
        print(f"{json_file} [{idx}] DSL preview:\n{dsl_output}...\n")

    except Exception as e:
        print(f"❌ GPT-4V failed on {idx}: {e}")
        continue

# --- Save updated JSON ---
out_path = os.path.join(output_dir, json_file)
with open(out_path, "w") as f_out:
    json.dump(entries, f_out, indent=2)