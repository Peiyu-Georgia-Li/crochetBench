import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from transformers import Qwen2ForCausalLM, AutoTokenizer

# --- Command line arguments ---
parser = argparse.ArgumentParser(description='Translate natural language crochet instructions to CrochetPARADE DSL using Qwen')
parser.add_argument('--model_id', type=str, default="Qwen/Qwen2-7B-Instruct",
                    help='Qwen model to use (default: Qwen/Qwen2-7B-Instruct)')
parser.add_argument('--input_file', type=str, default="/store01/nchawla/pli9/crochet/crochet_patterns_part4.json",
                    help='Path to the input JSON file containing crochet patterns')
parser.add_argument('--output_file', type=str, default="/store01/nchawla/pli9/crochet/crochet_dsl_translations_qwen_part4.json",
                    help='Path to save the translated DSL outputs')
parser.add_argument('--pattern_index', type=int, default=None,
                    help='Index of a specific pattern to translate (default: translate all)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose output')
args = parser.parse_args()

# --- Qwen model setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading {args.model_id}...")
model = Qwen2ForCausalLM.from_pretrained(
    args.model_id, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

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
        # Prepare the conversation with system prompt and user prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template to get formatted text prompt
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.2,  # Lower temperature for more consistent outputs
                top_p=0.9
            )
        
        # Trim the prompt tokens before decoding
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        
        return response
    except Exception as e:
        print(f"❌ Qwen model failed: {e}")
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
