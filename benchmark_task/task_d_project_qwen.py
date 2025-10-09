import os
import json
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
import traceback
import torch
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

SYSTEM_PROMPT = """
You are a professional crochet pattern writer. 
Examine the image of the finished crochet product carefully. 
Convert instructions + image into compilable CrochetPARADE DSL code.
Output only the DSL code, no explanations or extra text.
"""

# --- Pipeline setup ---
# Check for GPU availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct").to(device)

# --- Directories ---
output_dir = "./task_d_project_qwen/"  # output JSONs
os.makedirs(output_dir, exist_ok=True)

json_file = 'project_level_test.json'
json_dir='../data/'
# --- Iterate over JSON files ---

json_path = os.path.join(json_dir, json_file)
with open(json_path, "r") as f:
    data = json.load(f)

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

for idx, entry in enumerate(tqdm(entries, desc=f"{json_file} images", leave=False)):
    image_source = entry.get("image_link") or entry.get("image_path")
    if not image_source:
        continue

    try:
        # --- Build few-shot + target messages ---
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        
        # Add few-shot examples as user+assistant pairs
        for ex in FEW_SHOT_EXAMPLES:
            ex_image_url = ex["image_path"]
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "url": ex_image_url},
                    {"type": "text", "text": f"Convert the following instructions into DSL:\n{ex['instructions']}"}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex["dsl"]}
                ]
            })
        
        # Add the target image + instructions
        target_image_url = image_source
        target_instructions = entry.get("instructions", "")
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "url": target_image_url if target_image_url.startswith("http") else "file://" + os.path.abspath(target_image_url)},
                {"type": "text", "text": f"Convert the following instructions into DSL:\n{target_instructions}"}
            ]
        })

        # Generate DSL code
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=512)
        generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        entry["generated_dsl"] = generated_text.strip()
        print(f"{json_file} [{idx}] DSL preview:\n{generated_text}...\n")
    
    except Exception as e:

        print(f"Error processing {json_file} [{idx}]: {e}")
        traceback.print_exc()
        continue

# --- Save updated JSON ---
output_path = os.path.join(output_dir, json_file)
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

