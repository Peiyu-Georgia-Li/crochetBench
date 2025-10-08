import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import traceback

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image_from_source(image_source):
    """Load an image from URL or local path with enhanced error handling"""
    if not image_source:
        raise ValueError("Empty image source provided")
        
    if isinstance(image_source, str) and image_source.strip():
        image_source = image_source.strip()
        if image_source.startswith("http"):
            try:
                response = requests.get(image_source, timeout=10)
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch image: {response.status_code}")
                image = Image.open(BytesIO(response.content)).convert("RGB")
                # Validate image has valid dimensions
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Invalid image dimensions")
                return image
            except requests.RequestException as e:
                raise Exception(f"Request error: {str(e)}")
            except Exception as e:
                raise Exception(f"Error processing image from URL: {str(e)}")
        else:
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            try:
                image = Image.open(image_source).convert("RGB")
                # Validate image has valid dimensions
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Invalid image dimensions")
                return image
            except Exception as e:
                raise Exception(f"Error loading local image: {str(e)}")
    else:
        raise ValueError(f"Invalid image source: {type(image_source)}")

def generate_crochet_instructions(image_source):
    """Generate crochet instructions for a given image using Qwen2-VL model"""
    # System prompt for crochet pattern generation
    system_prompt = """You are a professional crochet pattern writer. 
Examine the image of the finished crochet product carefully. 
Write a complete set of crochet instructions in the standard style used in published patterns.

Requirements:
- Use standard abbreviations: sc (single crochet), hdc (half double crochet), dc (double crochet), tr (treble), ch (chain), sl st (slip stitch), rep (repeat).
- Organize the instructions row by row or round by round (e.g., "Rnd 1: ...", "Row 2: ...").
- If color changes are visible in the image, include them in the pattern.
- Keep the instructions concise and precise, as if for experienced crocheters.
- Output only the crochet pattern. Do not add any explanations, commentary, or extra text."""

    # Load model and processor
    print("Loading Qwen2-VL model...")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load the image
    print(f"Loading image: {image_source}")
    image = load_image_from_source(image_source)
    
    # Prepare the conversation with system prompt and image
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Generate step-by-step crochet instructions for this image."}
            ]
        }
    ]
    
    # Apply chat template to get formatted text prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Trim the prompt tokens before decoding
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

def main():
    """Main function to process all JSON files with the Qwen2-VL model"""
    # --- Directories ---
    json_dir = "/store01/nchawla/pli9/crochet/text_json"  # input JSONs
    output_dir = "./generated_instructions_qwen/"  # output JSONs
    os.makedirs(output_dir, exist_ok=True)
    
    # --- List all JSON files ---
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files to process")
    
    # --- Iterate over JSON files ---
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Convert to list if data is a dictionary
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
        
        # --- Iterate over entries/images ---
        for idx, entry in enumerate(tqdm(entries, desc=f"{json_file} images", leave=False)):
            # Skip non-dictionary entries
            if not isinstance(entry, dict):
                print(f"Skipping non-dictionary entry at index {idx}")
                continue
                
            # Get image source, mark if none found
            image_source = None
            # Check for image_link first (URLs)
            if "image_link" in entry and entry["image_link"]:
                if isinstance(entry["image_link"], str) and entry["image_link"].strip():
                    image_source = entry["image_link"].strip()
            # Then check for local image_path as fallback
            elif "image_path" in entry and entry["image_path"]:
                if isinstance(entry["image_path"], str) and entry["image_path"].strip():
                    image_source = entry["image_path"].strip()
                
            if not image_source:
                print(f"No valid image source found for entry {idx} - skipping")
                continue
            
            try:
                print(f"Processing image: {image_source[:50]}...")
                
                # Generate instructions
                instructions = generate_crochet_instructions(image_source)
                
                # Store the result
                entry["generated_instructions"] = instructions
                print(instructions)
                print(f"Successfully generated instructions for entry {idx}")
                
            except Exception as e:
                error_msg = f"Error processing {json_file} [{idx}]: {str(e)}"
                print(error_msg)
                print(f"Error details: {traceback.format_exc()}")
                # Store the error as instructions
                entry["generated_instructions"] = error_msg
                continue
        
        # --- Save updated JSON ---
        output_path = os.path.join(output_dir, json_file)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
