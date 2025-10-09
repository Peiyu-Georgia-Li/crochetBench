import os
import json
import torch
from tqdm import tqdm
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import requests
from io import BytesIO
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

# --- Model setup ---
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# --- Directories ---
output_dir = "./task_d_project_dsvl/"
os.makedirs(output_dir, exist_ok=True)

# --- Helper functions ---
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        return None

def create_few_shot_conversation(examples, target_instructions, target_image_path):
    """Create a conversation with few-shot examples"""
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]
    
    # Add few-shot examples
    for ex in examples:
        # User message with example image and instructions
        conversation.append({
            "role": "User",
            "content": f"<image_placeholder>Convert the following instructions into DSL:\n{ex['instructions']}",
            "images": [ex["image_path"]]
        })
        # Assistant response with DSL
        conversation.append({
            "role": "Assistant",
            "content": ex["dsl"]
        })
    
    # Add target task
    conversation.append({
        "role": "User",
        "content": f"<image_placeholder>Convert the following instructions into DSL:\n{target_instructions}",
        "images": [target_image_path]
    })
    conversation.append({
        "role": "Assistant", 
        "content": ""
    })
    
    return conversation

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
        # Load target image
        target_image = load_image_from_url(image_source)
        if target_image is None:
            continue
            
        # Get target instructions
        target_instructions = entry.get("instructions", "")
        
        # Load few-shot example images
        example_images = []
        for ex in FEW_SHOT_EXAMPLES:
            ex_image = load_image_from_url(ex["image_path"])
            if ex_image is not None:
                example_images.append(ex_image)
        
        # Create conversation with few-shot examples
        conversation = create_few_shot_conversation(
            FEW_SHOT_EXAMPLES, 
            target_instructions, 
            image_source
        )
        
        # Prepare all images (examples + target)
        all_images = example_images + [target_image]
        
        # Process with DeepSeek VL
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=all_images,
            force_batchify=True
        ).to(vl_gpt.device)
        
        # Prepare embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate DSL code
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        # Decode the generated response
        generated_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Extract only the assistant's response (last part)
        if "Assistant:" in generated_text:
            dsl_output = generated_text.split("Assistant:")[-1].strip()
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