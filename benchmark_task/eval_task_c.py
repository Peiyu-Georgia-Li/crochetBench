import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import CHRF
from tqdm import tqdm
import os.path as osp
current_dir = osp.dirname(osp.abspath(__file__))
json_dir = osp.join(current_dir, "generated_instructions_gpt4v")
output_dir = osp.join(current_dir, "crochet_eval_results_gpt4v")
os.makedirs(output_dir, exist_ok=True)
# ---------- Metrics ----------
def compute_metrics(data):
    results = []

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    chrf = CHRF()

    for item in tqdm(data):
        # Check if item is a dictionary
        if not isinstance(item, dict):
            print(f"Skipping non-dictionary item: {type(item).__name__}")
            continue
        if "instructions" not in item or "generated_instructions" not in item:
            print(f"Skipping item with ID: {item.get('id', 'unknown')} - missing required keys")
            continue
        # Handle both string and list formats for instructions
        ref = item["instructions"]
        if isinstance(ref, list):
            ref = " ".join(ref)
        ref = ref.strip()
        
        gen = item["generated_instructions"]
        if isinstance(gen, list):
            gen = " ".join(gen)
        gen = gen.strip()

        # Tokenize by space (for BLEU/METEOR)
        ref_tokens = ref.split()
        gen_tokens = gen.split()

        # BLEU
        bleu = sentence_bleu(
            [ref_tokens], gen_tokens, smoothing_function=SmoothingFunction().method1
        )

        # METEOR
        meteor = meteor_score([ref.split()], gen.split())

        #meteor = meteor_score([ref], gen)

        # ROUGE
        rouge_scores = rouge.score(ref, gen)

        # ChrF
        chrf_score = chrf.sentence_score(gen, [ref]).score

        results.append({
            "reference": ref,
            "generated": gen,
            "BLEU": bleu,
            "METEOR": meteor,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-2": rouge_scores["rouge2"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "ChrF": chrf_score
        })

    return results
# ---------- Load JSON ----------
# Assume structure: [{"reference": "...", "generated": "..."}, ...]
#json_dir = "generated_instructions_deepseek"
#output_dir = "crochet_eval_results_deepseek"
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
for json_file in tqdm(json_files, desc="Processing JSON files"):
    with open(os.path.join(json_dir, json_file), "r") as f:
        data = json.load(f)
    
    # Debug info
    print(f"Processing file: {json_file}")
    print(f"Data type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"First item type: {type(data[0])}")
        # If the first item is a string, try to parse it as JSON
        if isinstance(data[0], str):
            try:
                # Try to parse each string item as JSON
                data = [json.loads(item) if isinstance(item, str) else item for item in data]
                print("Converted string items to dictionaries")
            except json.JSONDecodeError:
                print("Failed to parse string items as JSON")
    
    # ---------- Run ----------
    all_results = compute_metrics(data)

    # ---------- BERTScore (batched for efficiency) ----------
    refs = []
    gens = []
    for d in data:
        if "instructions" in d and "generated_instructions" in d:
            ref = d["instructions"]
            gen = d["generated_instructions"]
            
            # Convert lists to strings
            if isinstance(ref, list):
                ref = " ".join(ref)
            if isinstance(gen, list):
                gen = " ".join(gen)
                
            refs.append(ref)
            gens.append(gen)

    if refs and gens:  # Only proceed if lists are not empty
        P, R, F1 = bert_score(gens, refs, lang="en", verbose=True)

        for i, r in enumerate(all_results):
            r["BERTScore_P"] = float(P[i])
            r["BERTScore_R"] = float(R[i])
            r["BERTScore_F1"] = float(F1[i])
    else:
        print("Warning: No valid items found for BERTScore calculation")
    # ---------- Save results ----------
    with open(os.path.join(output_dir, "crochet_eval_results_" + json_file), "w") as f:
        json.dump(all_results, f, indent=2)

print("✅ Evaluation complete. Saved to " + output_dir)
