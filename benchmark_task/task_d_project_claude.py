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
parser = argparse.ArgumentParser(description='Generate crochet instructions using Claude API')
parser.add_argument('--model', type=str, default="claude-sonnet-4-20250514",
                    help='Claude model to use')
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

# --- Directories ---
output_dir = "./task_d_project_claude/"
os.makedirs(output_dir, exist_ok=True)

# --- Initialize Claude client ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- Helpers ---
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        return None

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Process JSON files ---
json_file = 'project_level_test.json'
json_dir = '../data/'
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
    image = load_image_from_url(image_source)
    if image is None:
        continue

    try:
        # --- Build few-shot + target content ---
        message_content = []

        # Add few-shot examples as separate images + text
        for ex in FEW_SHOT_EXAMPLES:
            ex_image = load_image_from_url(ex["image_path"])
            if ex_image is None:
                continue
            ex_base64 = image_to_base64(ex_image)
            message_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": ex_base64}
            })
            message_content.append({
                "type": "text",
                "text": f"Convert the following instructions into DSL:\n{ex['instructions']}\nExpected DSL:\n{ex['dsl']}"
            })

        # Add target image + instructions
        target_base64 = image_to_base64(image)
        message_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": target_base64}
        })
        target_instructions = entry.get("instructions", "")
        message_content.append({
            "type": "text",
            "text": f"Convert the following instructions into DSL:\n{target_instructions}"
        })

        # --- Send to Claude ---
        response = client.messages.create(
            model=model,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message_content}]
        )

        dsl_output = response.content[0].text.strip()
        entry["generated_dsl"] = dsl_output
        print(f"{json_file} [{idx}] DSL preview:\n{dsl_output}...\n")

    except Exception as e:
        print(f"❌ Claude failed on {idx}: {e}")
        continue

# --- Save updated JSON ---
out_path = os.path.join(output_dir, json_file)
with open(out_path, "w") as f_out:
    json.dump(entries, f_out, indent=2)
