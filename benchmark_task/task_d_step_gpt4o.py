import os
import json
import time
import argparse
import random
import requests
from tqdm import tqdm

# --- Command line arguments ---
parser = argparse.ArgumentParser(description='Translate natural language crochet instructions to CrochetPARADE DSL using GPT-4V')
parser.add_argument('--model_path', type=str, default="gpt-4o",
                    help='GPT-4V model to use (default: gpt-4o)')
parser.add_argument('--input_file', type=str, default="/store01/nchawla/pli9/crochet/crochet_patterns_part3.json",
                    help='Path to the input JSON file containing crochet patterns')
parser.add_argument('--output_file', type=str, default="/store01/nchawla/pli9/crochet/crochet_dsl_translations_gpt4v_part3.json",
                    help='Path to save the translated DSL outputs')
parser.add_argument('--pattern_index', type=int, default=None,
                    help='Index of a specific pattern to translate (default: translate all)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose output')
parser.add_argument('--timeout', type=int, default=30,
                    help='Timeout in seconds for API calls (default: 30)')
parser.add_argument('--max_retries', type=int, default=5,
                    help='Maximum number of retries for failed API calls (default: 5)')
parser.add_argument('--initial_backoff', type=float, default=1.0,
                    help='Initial backoff time in seconds for retry mechanism (default: 1.0)')
args = parser.parse_args()

# --- API setup ---
# Get API key from environment variable for security
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# API configuration
GPT_API_URL = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

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
def translate_instruction(prompt, max_retries=5, initial_backoff=1.0, timeout=30):
    retries = 0
    backoff = initial_backoff
    
    while retries <= max_retries:
        try:
            # Create the messages for the GPT-4V API - text only
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Prepare API payload
            payload = {
                "model": args.model_path,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.2
            }
            
            # Call API
            response = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=timeout)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                # Get retry-after header or use exponential backoff
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else backoff
                print(f"Rate limited. Waiting {wait_time} seconds before retry {retries+1}/{max_retries}...")
                time.sleep(wait_time)
                backoff = min(backoff * 2 + 0.1 * backoff * (random.random() - 0.5), 60)
                retries += 1
                continue
            
            # Handle other API errors
            if response.status_code != 200:
                if response.status_code >= 500:
                    print(f"Server error ({response.status_code}): {response.text}. Retrying after {backoff} seconds...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2 + 0.1 * backoff * (random.random() - 0.5), 60)
                    retries += 1
                    continue
                else:
                    raise Exception(f"API Error: {response.status_code} - {response.text}")
            
            # Parse the successful response
            response_data = response.json()
            result = response_data["choices"][0]["message"]["content"]
            
            # Clean up any potential prefixes like "DSL:" that might be generated
            if "DSL:" in result:
                result = result.split("DSL:", 1)[1].strip()
            
            return result.strip()
            
        except (requests.RequestException, TimeoutError) as e:
            print(f"Request error: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff = min(backoff * 2 + 0.1 * backoff * (random.random() - 0.5), 60)
            retries += 1
        except Exception as e:
            print(f"❌ GPT-4V model failed: {e}")
            if retries < max_retries:
                print(f"Unexpected error. Retrying after {backoff} seconds...")
                time.sleep(backoff)
                backoff = min(backoff * 2 + 0.1 * backoff * (random.random() - 0.5), 60)
                retries += 1
            else:
                return None
    
    print(f"❌ Max retries ({max_retries}) reached. Giving up.")
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
                    print("\n====================================\n")
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
                    dsl_translation = translate_instruction(
                        prompt,
                        max_retries=args.max_retries,
                        initial_backoff=args.initial_backoff,
                        timeout=args.timeout
                    )
                    
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
                
                # Add a small delay to avoid overloading the API
                time.sleep(0.5 + random.random())
    
    # Save the results to the output file
    if results:
        save_results(args.output_file, results)
        print(f"Saved {len(results)} translations to {args.output_file}")
    else:
        print("No translations performed.")

if __name__ == "__main__":
    main()
