import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Command line arguments ---
parser = argparse.ArgumentParser(description='Translate natural language crochet instructions to CrochetPARADE DSL using Gemma')
parser.add_argument('--model_id', type=str, default="google/gemma-3-27b-pt",
                    help='Gemma model to use (default: google/gemma-3-27b-pt)')
parser.add_argument('--input_file', type=str, default="../data/step_level_test_1_2.json",
                    help='Path to the input JSON file containing crochet patterns')
parser.add_argument('--output_dir', type=str, default="./task_d_step_gemma_27b/",
                    help='Path to save the translated DSL outputs')
parser.add_argument('--pattern_index', type=int, default=None,
                    help='Index of a specific pattern to translate (default: translate all)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose output')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, args.input_file.split("/")[-1])

# --- Gemma model setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading {args.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

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
        # Extract just the NL instruction from the full prompt for clarity
        nl_instruction = None
        if "NL:" in prompt and "DSL:" in prompt:
            parts = prompt.split("NL:")
            if len(parts) > 1:
                last_part = parts[-1]
                if "DSL:" in last_part:
                    nl_instruction = last_part.split("DSL:")[0].strip()
        
        if not nl_instruction:
            nl_instruction = prompt
        
        # Create a more structured prompt with examples to encourage correct translation
        direct_prompt = f"Translate this crochet instruction to CrochetPARADE DSL:\n\nInstruction: {nl_instruction}\n\nYour DSL translation (one line only):"
        
        # Manually format the prompt for Gemma
        formatted_prompt = f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        formatted_prompt += f"<start_of_turn>user\n{direct_prompt}<end_of_turn>\n"
        formatted_prompt += "<start_of_turn>model\n"
        
        if args.verbose:
            print("Formatted prompt:\n", formatted_prompt[:200], "...")
        
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate response with stricter settings to prevent unnecessary text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Allow slightly more tokens for complete translations
                do_sample=True,     # Use sampling to improve creativity for DSL translation
                temperature=0.3,    # Low temperature for consistent but not too rigid results
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        if args.verbose:
            print("Raw output:", response)
        
        # Clean up the response to get just the DSL code
        # First try to find "DSL code:" or "DSL:" as markers
        if "DSL code:" in response:
            response = response.split("DSL code:", 1)[1].strip()
        elif "DSL:" in response:
            response = response.split("DSL:", 1)[1].strip()
            
        # Remove any explanatory text that might follow the code
        if "\n" in response:
            response = response.split("\n", 1)[0].strip()
            
        # Further cleanup to extract just the code pattern
        response = response.replace('```', '').strip()
        
        # Even if the response seems invalid, we'll still return it
        if len(response) > 150 or "example" in response.lower():
            print("Response seems invalid - too long or contains explanatory text")
        
        # Always return whatever the model generated
        return response
    except Exception as e:
        print(f"❌ Gemma model failed: {e}")
        import traceback
        traceback.print_exc()
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
                
                # Add the generated DSL to the result, even if it's invalid
                result["generated_dsl"] = dsl_translation if dsl_translation is not None else "[TRANSLATION FAILED]"
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
