import os
import json
from tqdm import tqdm
import torch
from transformers import pipeline
from pathlib import Path
import traceback

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

def create_few_shot_prompt(examples, target_instructions):
    """Create a few-shot prompt with examples and target instructions"""
    prompt = SYSTEM_PROMPT + "\n\n"
    
    # Add few-shot examples
    for i, ex in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Instructions: {ex['instructions']}\n"
        prompt += f"DSL: {ex['dsl']}\n\n"
    
    # Add target task
    prompt += "Now convert the following image <start_of_image> and instructions into DSL:\n"
    prompt += f"Instructions: {target_instructions}\n"
    prompt += "DSL:"
    
    return prompt

# --- Pipeline setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

gemma_pipe = pipeline(
    task="image-text-to-text",
    model="google/gemma-3-27b-pt",
    device=0 if device == "cuda" else -1,
    dtype=torch.bfloat16
)

# --- Directories ---
output_dir = "./task_d_project_gemma/"
os.makedirs(output_dir, exist_ok=True)

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

for idx, entry in enumerate(tqdm(entries, desc=f"{json_file} images", leave=False)):
    image_source = entry.get("image_link") or entry.get("image_path")
    if not image_source:
        continue

    try:
        # Get target instructions
        target_instructions = entry.get("instructions", "")
        
        # Create few-shot prompt
        prompt_text = create_few_shot_prompt(FEW_SHOT_EXAMPLES, target_instructions)
        
        # Prepare image source for Gemma
        if not image_source.startswith("http"):
            # Ensure absolute path for local files
            image_source = Path(image_source).resolve().as_posix()
        
        # Generate DSL code using Gemma
        output = gemma_pipe(
            image_source, text=prompt_text, \
            max_new_tokens=1024
        )
        
        # Extract generated text (remove the input prompt)
        full_response = output[0]["generated_text"]
        generated_text = full_response.replace(prompt_text, "").strip()
        
        # Extract only the DSL part if "DSL:" appears in output
        if "DSL:" in generated_text:
            dsl_output = generated_text.split("DSL:")[-1].strip()
        else:
            dsl_output = generated_text.strip()
        
        entry["generated_dsl"] = dsl_output
        print(f"{json_file} [{idx}] DSL preview:\n{dsl_output}...\n")
        
    except Exception as e:
        print(f"Error processing {json_file} [{idx}]: {e}")
        traceback.print_exc()
        continue

# --- Save updated JSON ---
output_path = os.path.join(output_dir, json_file)
with open(output_path, "w") as f:
    json.dump(entries, f, indent=2)

print(f"Finished processing. Results saved to {output_path}")