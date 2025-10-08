import os
import json
import torch
from tqdm import tqdm
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import requests
from io import BytesIO
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
# --- Model setup ---
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# --- Directories ---
json_dir = "/store01/nchawla/pli9/crochet/text_json"  # input JSONs
output_dir = "./generated_instructions_deepseek/"  # updated output path
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
            # Load image
            if image_source.startswith("http"):
                response = requests.get(image_source)
                if response.status_code != 200:
                    print(f"Failed to fetch {image_source}")
                    continue
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                if not os.path.exists(image_source):
                    print(f"Local image not found: {image_source}")
                    continue
                image = Image.open(image_source).convert("RGB")
            
            # Prepare conversation
            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "User",
                    "content": "<image_placeholder>Generate step-by-step crochet instructions for this image.",
                    "images": [image_source]
                },
                {"role": "Assistant", "content": ""}
            ]
            
            pil_images = [image]  # already loaded
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(vl_gpt.device)
            
            # Prepare embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            # Generate response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1028,
                do_sample=False,
                use_cache=True
            )
            
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Save generated instructions
            entry["generated_instructions"] = answer
            print(answer)
            
        except Exception as e:
            print(f"Error processing {json_file} [{idx}]: {e}")
            continue
    
    # --- Save updated JSON ---
    output_path = os.path.join(output_dir, json_file)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

