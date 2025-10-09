import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

# --- Command line arguments ---
parser = argparse.ArgumentParser(description='Translate natural language crochet instructions to CrochetPARADE DSL using DeepSeek-VL')
parser.add_argument('--model_path', type=str, default="deepseek-ai/deepseek-vl-7b-chat",
                    help='DeepSeek model to use (default: deepseek-ai/deepseek-vl-7b-chat)')
parser.add_argument('--input_file', type=str, default="../data/step_level_test_1_2.json",
                    help='Path to the input JSON file containing crochet patterns')
parser.add_argument('--output_dir', type=str, default="./task_d_step_dsvl/",
                    help='Path to save the translated DSL outputs')
parser.add_argument('--pattern_index', type=int, default=None,
                    help='Index of a specific pattern to translate (default: translate all)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose output')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, args.input_file.split("/")[-1])

# --- DeepSeek VL model setup ---
print(f"Loading {args.model_path}...")
vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
tokenizer = vl_chat_processor.tokenizer

vl_model = MultiModalityCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vl_model = vl_model.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device).eval()

# --- Create a blank placeholder image ---
# This is a workaround to use VL model for text-only tasks
placeholder_image = Image.new('RGB', (32, 32), color='white')

# --- System prompt for the translation ---
SYSTEM_PROMPT = """
You are a crochet compiler. Translate the next instruction NL into one line of CrochetPARADE DSL.
Use consistent naming and syntax.

Important rules for translations:
1. Make sure your output ONLY contains the DSL code, nothing else.
2. Use the previous examples to understand the pattern of translation.
3. Be consistent in naming conventions with the examples.
4. Your output should be exactly one line of DSL code.
"""

# --- Load the input JSON file ---
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# --- Save the results to an output JSON file ---
def save_results(file_path, results):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {file_path}")

# --- Process a single pattern entry ---
def translate_instruction(prompt):
    try:
        # For text-only tasks with the VL model, we need to provide a placeholder image
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "User",
                "content": "<image_placeholder>Ignore this image placeholder. " + prompt,
                "images": [placeholder_image]
            },
            {"role": "Assistant", "content": ""}
        ]
        
        # Process the conversation with the placeholder image
        pil_images = [placeholder_image]
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_model.device)
        
        # Generate response
        with torch.no_grad():
            # Prepare embeddings
            inputs_embeds = vl_model.prepare_inputs_embeds(**prepare_inputs)
            
            # Generate tokens
            outputs = vl_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=100,
                temperature=0.2,
                do_sample=True,
                use_cache=True
            )
            
            # Decode the response
            response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Clean up any potential prefixes like "DSL:" that might be generated
            if "DSL:" in response:
                response = response.split("DSL:", 1)[1].strip()
            
            return response.strip()
            
    except Exception as e:
        print(f"❌ DeepSeek VL model failed: {e}")
        return None

# --- Extract last NL and its DSL from the prompt ---
def extract_nl_and_dsl(prompt):
    # Extract the last NL instruction that needs translation
    if "NL:" in prompt and "DSL:" in prompt:
        nl_parts = prompt.split("NL:")
        if len(nl_parts) > 1:
            last_part = nl_parts[-1]
            if "DSL:" in last_part:
                nl_instruction = last_part.split("DSL:")[0].strip()
                # Check if there's a DSL translation already provided
                dsl_parts = last_part.split("DSL:")
                if len(dsl_parts) > 1:
                    dsl_translation = dsl_parts[-1].strip()
                    return nl_instruction, dsl_translation
                return nl_instruction, ""
    return None, None

# --- Main function ---
def main():
    # Load the JSON file
    patterns = load_json_file(args.input_file)
    results = []
    
    # If a specific pattern index is provided, only process that one
    if args.pattern_index is not None:
        if 0 <= args.pattern_index < len(patterns):
            process_indices = [args.pattern_index]
        else:
            print(f"Error: Pattern index {args.pattern_index} is out of range (0-{len(patterns)-1})")
            return
    else:
        process_indices = range(len(patterns))
    
    # Process each pattern entry
    for pattern_idx in tqdm(process_indices, desc="Translating patterns"):
        pattern = patterns[pattern_idx]
        # Include pattern ID and more fields from the original pattern
        result = {
            "pattern_index": pattern_idx,
            "pattern_name": pattern.get("pattern_name", "Unknown"),
            "id": pattern.get("id", ""),
            "prompt": pattern.get("prompt", "")
        }
        
        if "prompt" in pattern:
            prompt = pattern["prompt"]
            nl_instruction, existing_dsl = extract_nl_and_dsl(prompt)
            
            if nl_instruction:
                if args.verbose:
                    print("\n====================================")
                    print(f"Pattern: {pattern.get('pattern_name', 'Unknown')} (Index: {pattern_idx})")
                    print(f"NL instruction: {nl_instruction}")
                
                # Store the NL instruction
                result["nl_instruction"] = nl_instruction
                
                # Check if there's already a DSL translation in the prompt
                if existing_dsl and existing_dsl != "\n":
                    if args.verbose:
                        print(f"Existing DSL translation: {existing_dsl}")
                    dsl_translation = existing_dsl
                else:
                    # Translate the NL instruction to DSL
                    dsl_translation = translate_instruction(prompt)
                    
                    if args.verbose:
                        if dsl_translation:
                            print(f"Generated DSL translation: {dsl_translation}")
                        else:
                            print("❌ Translation failed")
                
                # Add the generated DSL to the result
                if dsl_translation:
                    result["generated_dsl"] = dsl_translation
                    results.append(result)
                
                if args.verbose:
                    print("===================================\n")
                
                # Add a small delay to avoid overloading the GPU
                time.sleep(0.2)
    
    # Save the results to the output file
    if results:
        save_results(args.output_file, results)
        print(f"Saved {len(results)} translations to {args.output_file}")
    else:
        print("No translations performed.")

if __name__ == "__main__":
    main()
