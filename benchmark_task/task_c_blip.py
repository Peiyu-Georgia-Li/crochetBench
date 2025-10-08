import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import os
from tqdm import tqdm  # tqdm for progress bars
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
# Load model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")

# Directories
json_dir = "/store01/nchawla/pli9/crochet/text_json/"
output_dir = "./generated_instructions_blip/"
os.makedirs(output_dir, exist_ok=True)

# List all JSON files
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

# Progress bar over JSON files
for json_file in tqdm(json_files, desc="Processing JSON files"):
    json_path = os.path.join(json_dir, json_file)
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Progress bar over entries/images in each JSON
    for idx, entry in enumerate(tqdm(data, desc=f"{json_file} images", leave=False)):
        image_url = entry.get("image_link", None)
        if not image_url:
            continue
        
        try:
            # Load image from URL
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Prepare input with system prompt and user message
            prompt = f"""{SYSTEM_PROMPT}
            
            User: Generate step-by-step crochet instructions for this image."""
            
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to("cuda")
            
            # Generate instructions with increased token limit and better generation parameters
            # Generate instructions with parameters optimized for detailed patterns
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,    # More focused generation
                num_beams=3,           # Faster generation with reasonable quality
                no_repeat_ngram_size=2,  # Allow some repetition for patterns
                early_stopping=False,   # Let it generate full length
                min_length=100,        # Ensure minimum length
                length_penalty=2.0,    # Encourage longer outputs
                repetition_penalty=1.2  # Reduce repetition
            )
            instructions = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up and format the instructions
            instructions = instructions.strip()
            
            # Add generated instructions to the JSON entry
            entry["generated_instructions"] = instructions
            
            # Print a preview of the instructions (first 200 chars)
            print(f"Generated instructions (preview): {instructions}...")
        except Exception as e:
            print(f"Error processing {json_file} [{idx}]: {e}")
    
    # Save updated JSON with generated instructions
    output_path = os.path.join(output_dir, json_file)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
