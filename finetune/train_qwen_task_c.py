import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import traceback
import logging
from datetime import datetime
import wandb

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF


# Ensure NLTK data is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"qwen_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

SAMPLE_DIR = "./generation_samples"
EVAL_DIR = "./evaluation_detailed_logs"
OUPUT_DIR = "./evaluation_results/"
LOG_DIR = "./generation_logs"
RAW_OUTPUT_DIR = "./raw_model_outputs"
CHECKPOINT_DIR = "./qwen_checkpoints"
# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

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


class CrochetDataset(Dataset):
    """Dataset for fine-tuning Qwen2-VL on crochet patterns"""
    def __init__(self, json_file, processor, max_length=1024):  
        self.processor = processor
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each pattern with an image
        for pattern in tqdm(data, desc="Processing patterns"):
            if "image_link" not in pattern or not pattern["image_link"]:
                continue
                
            try:
                # Extract pattern information
                image_source = pattern["image_link"]
                instructions = pattern.get("instructions", "")
                pattern_name = pattern.get("pattern_name", "")
                
                # Skip if no instructions
                if not instructions:
                    continue
                    
                # Create system prompt
                system_prompt = """You are a professional crochet pattern writer. 
Examine the image of the finished crochet product carefully. 
Write a complete set of crochet instructions in the standard style used in published patterns.

Requirements:
- Use standard abbreviations: sc (single crochet), hdc (half double crochet), dc (double crochet), tr (treble), ch (chain), sl st (slip stitch), rep (repeat).
- Organize the instructions row by row or round by round (e.g., "Rnd 1: ...", "Row 2: ...").
- If color changes are visible in the image, include them in the pattern.
- Keep the instructions concise and precise, as if for experienced crocheters.
- Output only the crochet pattern. Do not add any explanations, commentary, or extra text."""
                
                # Create user prompt
                user_prompt = "Generate step-by-step crochet instructions for this image."
                
                # Create example
                self.examples.append({
                    "image_source": image_source,
                    "instructions": instructions,
                    "pattern_name": pattern_name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt
                })
                
            except Exception as e:
                logger.warning(f"Error processing pattern: {str(e)}")
                continue
                
        logger.info(f"Loaded {len(self.examples)} examples with images and instructions")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        try:
            # Load image
            image = load_image_from_source(example["image_source"])
            
            # Prepare input conversation (without assistant response)
            input_messages = [
                {
                    "role": "system",
                    "content": example["system_prompt"]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": example["user_prompt"]}
                    ]
                }
            ]
            
            # Prepare target conversation (with assistant response)
            target_messages = [
                {
                    "role": "system",
                    "content": example["system_prompt"]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": example["user_prompt"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": example["instructions"]
                }
            ]
            
            # Apply chat template to get formatted text prompts
            input_text = self.processor.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
            target_text = self.processor.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=False)
            
            # Log input and target text (truncated for readability)
            if idx % 100 == 0:  # Log only occasionally to avoid excessive output
                logger.info(f"Sample {idx} - Pattern: {example['pattern_name']}")
                logger.info(f"Input text (truncated): {input_text[:100]}...")
                logger.info(f"Target text (truncated): {example['instructions'][:100]}...")
                logger.info(f"Input text length: {len(input_text)}")
                logger.info(f"Target text length: {len(example['instructions'])}")
            
            # Process inputs with images and text
            model_inputs = self.processor(
                text=[target_text],  # Use the full conversation for input_ids
                images=[image],
                padding="max_length",
                max_length=self.max_length,
                truncation=False,  # Disable truncation to prevent image token mismatch
                return_tensors="pt"
            )
            
            # Process the input text to find where the assistant's response starts
            input_tokens = self.processor.tokenizer(input_text, add_special_tokens=False, return_tensors="pt").input_ids
            
            # Create labels: -100 for input tokens (ignored in loss), actual token ids for output tokens
            labels = model_inputs["input_ids"].clone()
            
            # Set all tokens corresponding to the input prompt to -100 (ignored in loss calculation)
            input_length = input_tokens.shape[1]
            if input_length < labels.shape[1]:
                labels[:, :input_length] = -100
            
            model_inputs["labels"] = labels
            
            # Log token lengths
            if idx % 100 == 0:  # Log only occasionally
                logger.info(f"Input tokens length: {input_length}")
                logger.info(f"Total tokens length: {labels.shape[1]}")
                if input_length >= self.max_length:
                    logger.warning(f"Input length {input_length} exceeds max_length {self.max_length}")
            
            # Remove batch dimension
            model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
            
            return model_inputs
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
            # Return a default item in case of error
            return self.__getitem__((idx + 1) % len(self))


class WandbCallback(TrainerCallback):
    """Custom callback for logging to Weights & Biases"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and wandb.run is not None:
            wandb.log(logs)


class MetricsCallback(TrainerCallback):
    """Custom callback for computing text generation metrics during training"""
    def __init__(self, eval_dataset, processor, tokenizer, eval_steps=500):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.best_metrics = {}
        # Create a directory for saving generation samples
        self.samples_dir = SAMPLE_DIR
        os.makedirs(self.samples_dir, exist_ok=True)
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Only compute metrics every eval_steps
        if state.global_step % self.eval_steps != 0:
            return
            
        logger.info(f"Computing text generation metrics at step {state.global_step}")
        
        # Get model from kwargs
        model = kwargs.get('model')
        if model is None:
            logger.warning("Model not available in callback, skipping metrics computation")
            return
            
        # Sample a subset of validation data for metrics computation
        num_samples = min(10, len(self.eval_dataset))
        indices = random.sample(range(len(self.eval_dataset)), num_samples)
        
        references = []
        predictions = []
        
        # Create a file for this evaluation step
        eval_log_file = os.path.join(self.samples_dir, f"eval_samples_step_{state.global_step}.txt")
        
        # Generate predictions
        for idx in indices:
            try:
                # Get example
                example = self.eval_dataset.examples[idx]
                
                # Load image
                image = load_image_from_source(example["image_source"])
                
                # Prepare conversation with system prompt and image
                messages = [
                    {
                        "role": "system",
                        "content": example["system_prompt"]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": example["user_prompt"]}
                        ]
                    }
                ]
                
                # Apply chat template to get formatted text prompt
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Process inputs
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    truncation=False,  # Disable truncation to prevent image token mismatch
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate response with memory-efficient settings
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,  # Reduced to save memory
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True  # Enable KV caching for efficiency
                    )
                
                # Trim the prompt tokens before decoding
                # Make sure we're accessing input_ids correctly
                if hasattr(inputs, 'input_ids'):
                    input_ids = inputs.input_ids
                else:
                    input_ids = inputs['input_ids']
                    
                trimmed = [o[len(i):] for i, o in zip(input_ids, output_ids)]
                generated_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                # Add to lists for metric calculation
                references.append(example["instructions"])
                predictions.append(generated_text)
                
                # Save input-output pairs t[o file for inspection
                with open(eval_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n===== SAMPLE {idx} - PATTERN: {example['pattern_name']} =====\n")
                    f.write(f"IMAGE SOURCE: {example['image_source']}\n")
                    f.write(f"\nINPUT PROMPT:\n{text}\n")
                    f.write(f"\nREFERENCE OUTPUT (FULL):\n{example['instructions']}\n")
                    f.write(f"\nMODEL OUTPUT (FULL):\n{generated_text}\n")
                    f.write(f"\nTOKEN LENGTHS - Input: {len(self.tokenizer.encode(text))}, Output: {len(self.tokenizer.encode(generated_text))}\n")
                    f.write("=" * 80 + "\n")
                
                # Also log to console
                logger.info(f"Generated text for sample {idx} - Pattern: {example['pattern_name']}")
                logger.info(f"MODEL OUTPUT (first 200 chars): {generated_text[:200]}...")
                logger.info(f"REFERENCE (first 200 chars): {example['instructions'][:200]}...")
                
            except Exception as e:
                logger.error(f"Error generating prediction for example {idx}: {str(e)}")
                continue
        
        # Calculate metrics
        if references and predictions:
            metrics_dict = compute_metrics(references, predictions)
            
            # Log metrics
            for metric_name, metric_value in metrics_dict.items():
                logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
                
                # Add to wandb if available
                if wandb.run is not None:
                    wandb.log({f"val_{metric_name}": metric_value}, step=state.global_step)
                
                # Track best metrics
                if metric_name not in self.best_metrics or metric_value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
                    logger.info(f"New best {metric_name.upper()}: {metric_value:.4f}")
            
            # Log summary to file
            with open(eval_log_file, 'a', encoding='utf-8') as f:
                f.write("\n\n===== METRICS SUMMARY =====\n")
                for metric_name, metric_value in metrics_dict.items():
                    f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
                    




def compute_metrics(references, predictions):
    """Compute evaluation metrics (BLEU, ROUGE, METEOR, chrF) for text generation"""
    results = {}
    
    # Initialize metrics
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    chrf = CHRF()
    
    # Lists to store individual scores
    bleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    chrf_scores = []
    
    # Process each reference-prediction pair
    for ref, pred in zip(references, predictions):
        # Ensure inputs are strings
        if isinstance(ref, list):
            ref = " ".join(ref)
        if isinstance(pred, list):
            pred = " ".join(pred)
            
        ref = ref.strip()
        pred = pred.strip()
        
        # Skip empty predictions or references
        if not ref or not pred:
            continue
        
        # Tokenize for BLEU/METEOR
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        
        # Skip if tokens are empty
        if not ref_tokens or not pred_tokens:
            continue
        
        # BLEU score
        try:
            bleu = sentence_bleu(
                [ref_tokens], pred_tokens, 
                smoothing_function=SmoothingFunction().method1
            )
            bleu_scores.append(bleu)
        except Exception as e:
            logger.warning(f"Error computing BLEU score: {e}")
        
        # METEOR score
        try:
            meteor = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(meteor)
        except Exception as e:
            logger.warning(f"Error computing METEOR score: {e}")
        
        # ROUGE scores
        try:
            rouge_score = rouge.score(ref, pred)
            rouge1_scores.append(rouge_score['rouge1'].fmeasure)
            rouge2_scores.append(rouge_score['rouge2'].fmeasure)
            rougeL_scores.append(rouge_score['rougeL'].fmeasure)
        except Exception as e:
            logger.warning(f"Error computing ROUGE scores: {e}")
        
        # chrF score
        try:
            chrf_score = chrf.sentence_score(pred, [ref]).score
            chrf_scores.append(chrf_score)
        except Exception as e:
            logger.warning(f"Error computing chrF score: {e}")
    
    # Calculate average scores
    if bleu_scores:
        results['bleu'] = sum(bleu_scores) / len(bleu_scores)
    if meteor_scores:
        results['meteor'] = sum(meteor_scores) / len(meteor_scores)
    if rouge1_scores:
        results['rouge1'] = sum(rouge1_scores) / len(rouge1_scores)
    if rouge2_scores:
        results['rouge2'] = sum(rouge2_scores) / len(rouge2_scores)
    if rougeL_scores:
        results['rougeL'] = sum(rougeL_scores) / len(rougeL_scores)
    if chrf_scores:
        results['chrf'] = sum(chrf_scores) / len(chrf_scores)
    
    return results

def evaluate_model(model, processor, test_file, num_samples=10, config=None):
    """Evaluate the model on a sample of test data with metrics"""
    logger.info(f"Evaluating model on {num_samples} samples from {test_file}")
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Select random samples
    if len(test_data) > num_samples:
        test_samples = random.sample(test_data, num_samples)
    else:
        test_samples = test_data
    
    results = []
    references = []
    predictions = []
    
    # Create a directory for detailed evaluation logs
    eval_dir = EVAL_DIR
    os.makedirs(eval_dir, exist_ok=True)
    eval_log_file = os.path.join(eval_dir, f"eval_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Log header
    with open(eval_log_file, 'w', encoding='utf-8') as f:
        f.write(f"DETAILED EVALUATION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Qwen2-VL-7B-Instruct\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write("=" * 80 + "\n\n")
    
    for sample in tqdm(test_samples, desc="Evaluating"):
        if "image_link" not in sample or not sample["image_link"]:
            continue
            
        try:
            # Load image
            image_source = sample["image_link"]
            image = load_image_from_source(image_source)
            
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
                truncation=False,  # Disable truncation to prevent image token mismatch
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response with memory-efficient settings
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.get("max_new_tokens", 256),  # Use config value or default to 256
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True  # Enable KV caching for efficiency
                )
            
            # Trim the prompt tokens before decoding
            # Make sure we're accessing input_ids correctly
            if hasattr(inputs, 'input_ids'):
                input_ids = inputs.input_ids
            else:
                input_ids = inputs['input_ids']
                
            trimmed = [o[len(i):] for i, o in zip(input_ids, output_ids)]
            generated_instructions = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Get reference instructions
            reference_instructions = sample.get("instructions", "")
            
            # Add to lists for metric calculation
            if reference_instructions:
                references.append(reference_instructions)
                predictions.append(generated_instructions)
            
            # Store individual result
            results.append({
                "pattern_name": sample.get("pattern_name", ""),
                "original_instructions": reference_instructions,
                "generated_instructions": generated_instructions,
                "image_source": image_source
            })
            
            # Log detailed input-output for this sample
            with open(eval_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n===== SAMPLE {len(results)} - PATTERN: {sample.get('pattern_name', '')} =====\n")
                f.write(f"IMAGE SOURCE: {image_source}\n")
                f.write(f"\nSYSTEM PROMPT (FULL):\n{system_prompt}\n")
                f.write(f"\nUSER PROMPT: Generate step-by-step crochet instructions for this image.\n")
                f.write(f"\nREFERENCE OUTPUT (FULL):\n{reference_instructions}\n")
                f.write(f"\nMODEL OUTPUT (FULL):\n{generated_instructions}\n")
                
                # Log token counts
                input_token_count = len(processor.tokenizer.encode(text))
                output_token_count = len(processor.tokenizer.encode(generated_instructions))
                f.write(f"\nTOKEN COUNTS - Input: {input_token_count}, Output: {output_token_count}\n")
                
                if input_token_count > config.get("max_length", 512):
                    f.write(f"WARNING: Input token count {input_token_count} exceeds max_length {config.get('max_length', 512)}\n")
                
                f.write("=" * 80 + "\n")
            
            # Also log to console
            logger.info(f"Evaluated sample {len(results)} - Pattern: {sample.get('pattern_name', '')}")
            logger.info(f"MODEL OUTPUT (first 300 chars): {generated_instructions[:300]}...")
            logger.info(f"REFERENCE (first 300 chars): {reference_instructions[:300]}...")
            logger.info(f"Token counts - Input: {input_token_count}, Output: {output_token_count}")
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {str(e)}")
            continue
    
    # Calculate metrics if we have predictions
    metrics = {}
    if references and predictions:
        logger.info(f"Computing evaluation metrics for {len(references)} samples")
        metrics = compute_metrics(references, predictions)
        
        # Add metrics to results
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")
    


    
    # Save evaluation results
    output_dir = OUPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Add metrics to the results
    evaluation_results = {
        "metrics": metrics,
        "samples": results
    }
    
    output_file = os.path.join(output_dir, f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Log summary metrics to the detailed log file
    with open(eval_log_file, 'a', encoding='utf-8') as f:
        f.write("\n\n===== FINAL METRICS SUMMARY =====\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
        f.write("\nDetailed results saved to: " + output_file + "\n")
    
    logger.info(f"Evaluation results saved to {output_file}")
    logger.info(f"Detailed evaluation log saved to {eval_log_file}")
    return evaluation_results


class TrainerWithGenerationCallback(Trainer):
    """Custom Trainer class that logs generated text during training"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = LOG_DIR
        # self.log_dir = "./generation_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_generations(self, step):
        """Generate and log text from a few examples"""
        if not hasattr(self, 'eval_dataset') or self.eval_dataset is None:
            return
            
        # Sample a few examples
        num_samples = min(2, len(self.eval_dataset))
        indices = random.sample(range(len(self.eval_dataset)), num_samples)
        
        log_file = os.path.join(self.log_dir, f"generations_step_{step}.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Generated text at step {step}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx in indices:
                try:
                    # Get example
                    example = self.eval_dataset.examples[idx]
                    
                    # Load image
                    image = load_image_from_source(example["image_source"])
                    
                    # Prepare conversation with system prompt and image
                    messages = [
                        {
                            "role": "system",
                            "content": example["system_prompt"]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": example["user_prompt"]}
                            ]
                        }
                    ]
                    
                    # Apply chat template
                    text = self.eval_dataset.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    # Process inputs
                    inputs = self.eval_dataset.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # Generate response
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                    
                    # Decode output
                    # Make sure we're accessing input_ids correctly
                    if hasattr(inputs, 'input_ids'):
                        input_ids = inputs.input_ids
                    else:
                        input_ids = inputs['input_ids']
                        
                    trimmed = [o[len(i):] for i, o in zip(input_ids, output_ids)]
                    generated_text = self.eval_dataset.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                    
                    # Write to file - FULL TEXT
                    f.write(f"Example {idx} - {example['pattern_name']}\n")
                    f.write(f"IMAGE SOURCE: {example['image_source']}\n\n")
                    f.write(f"SYSTEM PROMPT:\n{example['system_prompt']}\n\n")
                    f.write(f"USER PROMPT: {example['user_prompt']}\n\n")
                    f.write(f"REFERENCE (FULL):\n{example['instructions']}\n\n")
                    f.write(f"GENERATED (FULL):\n{generated_text}\n\n")
                    
                    # Add token counts
                    input_token_count = len(self.eval_dataset.processor.tokenizer.encode(text))
                    output_token_count = len(self.eval_dataset.processor.tokenizer.encode(generated_text))
                    f.write(f"TOKEN COUNTS - Input: {input_token_count}, Output: {output_token_count}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Log to console
                    logger.info(f"Generated text for example {idx} at step {step}")
                    logger.info(f"Pattern: {example['pattern_name']}")
                    logger.info(f"GENERATED (FULL):\n{generated_text}\n")
                    logger.info(f"REFERENCE (first 300 chars):\n{example['instructions'][:300]}...\n")
                    logger.info(f"Token counts - Input: {input_token_count}, Output: {output_token_count}")
                    logger.info("=" * 50)
                    
                except Exception as e:
                    logger.error(f"Error generating text for example {idx}: {str(e)}")
        
        logger.info(f"Generated text saved to {log_file}")
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to log generations"""
        output = super().evaluate(*args, **kwargs)
        self.log_generations(self.state.global_step)
        return output


def fine_tune_qwen2vl(train_file, val_file, output_dir, batch_size=1, num_epochs=3, learning_rate=2e-5, use_lora=True):
    def print_trainable_params(model):
        trainable = 0
        total = 0
        for _, p in model.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
        pct = 100 * trainable / total if total else 0
        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


    """Fine-tune Qwen2-VL model on crochet dataset"""
    logger.info("Starting fine-tuning of Qwen2-VL model")
    
    # Load model and processor
    logger.info("Loading Qwen2-VL model...")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Load processor first
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model with appropriate configuration
    if use_lora:
        logger.info("Using LoRA for parameter-efficient fine-tuning")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            # load_in_8bit=torch.cuda.is_available()  # Use 8-bit quantization if CUDA is available
        )
        # assert any(p.requires_grad for p in model.parameters())
        # print(loss.requires_grad)



        # Prepare model for k-bit training if using 8-bit quantization
        # if torch.cuda.is_available():
        #     model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        print_trainable_params(model)

        model.print_trainable_parameters()
        
        # Enable gradients for LoRA parameters
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
    else:
        logger.info("Using full fine-tuning")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    # logger.info("Using full fine-tuning with memory optimization")
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32,
    #     device_map="auto",
    #     low_cpu_mem_usage=True)
    
    # Enable gradient checkpointing to save memory
    # Use gradient_checkpointing_kwargs to avoid issues with LoRA
    if use_lora:
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Set pad token if not set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Create datasets
    train_dataset = CrochetDataset(train_file, processor)
    val_dataset = CrochetDataset(val_file, processor)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize wandb for tracking
    wandb.init(
        project="qwen2vl-crochet-finetune-lora",
        name=f"qwen2vl-{'lora' if use_lora else 'full'}-{datetime.now().strftime('%Y%m%d_%H%M')}",
        config={
            "model": model_id,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "use_lora": use_lora,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset)
        }
    )
    
    # Define training arguments with memory optimization
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Reduce batch size to save memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Increase gradient accumulation to compensate for smaller batch size
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,  # Keep only the last checkpoint
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=False,  # Disable fp16 to avoid gradient scaling issues
        bf16=False,  # Disable bf16 to troubleshoot gradient issues
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to="wandb",
        remove_unused_columns=False,  # Important for custom datasets
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",  # Use PyTorch's AdamW optimizer
        dataloader_num_workers=0,  # Reduce multiprocessing overhead
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Clip gradients to prevent exploding gradients
        ddp_find_unused_parameters=False  # Optimize distributed training
    )
    
    # Initialize callbacks
    callbacks = [
        WandbCallback(),
        MetricsCallback(
            eval_dataset=val_dataset,
            processor=processor,
            tokenizer=processor.tokenizer,
            eval_steps=training_args.eval_steps
        )
    ]
    
    # Initialize custom Trainer with generation logging
    trainer = TrainerWithGenerationCallback(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks
    )
    
    # Log model and dataset information
    logger.info("===== TRAINING SETUP SUMMARY =====")
    logger.info(f"Model: {model_id}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable out of {sum(p.numel() for p in model.parameters()):,} total")
    logger.info(f"Training dataset size: {len(train_dataset)} examples")
    logger.info(f"Validation dataset size: {len(val_dataset)} examples")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size} * {training_args.gradient_accumulation_steps} (gradient accumulation) = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps} effective batch size")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Max sequence length: {train_dataset.max_length}")
    logger.info(f"Using LoRA: {use_lora}")
    logger.info(f"Number of epochs: {training_args.num_train_epochs}")
    logger.info(f"Using mixed precision: {training_args.bf16 or training_args.fp16}")
    logger.info(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"Evaluation steps: {training_args.eval_steps}")
    logger.info(f"Save steps: {training_args.save_steps}")
    logger.info("==================================")
    
    # Sample and log a few examples from the training dataset
    logger.info("\n===== SAMPLE TRAINING EXAMPLES =====")
    sample_indices = random.sample(range(len(train_dataset)), min(3, len(train_dataset)))
    for i, idx in enumerate(sample_indices):
        example = train_dataset.examples[idx]
        logger.info(f"\nSample {i+1} - Pattern: {example['pattern_name']}")
        logger.info(f"Instructions (first 300 chars): {example['instructions'][:300]}...")
    logger.info("==================================")
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # Close wandb run
    wandb.finish()
    
    return model, processor

def main():
    """Main function to fine-tune Qwen2-VL model on crochet dataset"""
    # Configure logging to show more details
    logging.getLogger().setLevel(logging.INFO)
    
    # Create a directory for raw model outputs
    raw_outputs_dir = RAW_OUTPUT_DIR
    os.makedirs(raw_outputs_dir, exist_ok=True)
    
    # Add a file handler for raw outputs
    raw_outputs_file = os.path.join(raw_outputs_dir, f"raw_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    raw_outputs_handler = logging.FileHandler(raw_outputs_file)
    raw_outputs_handler.setLevel(logging.INFO)
    raw_outputs_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(raw_outputs_handler)
    
    logger.info(f"Raw model outputs will be saved to: {raw_outputs_file}")
    
    # --- Directories ---
    dataset_dir = "../split_dataset"  # Split dataset directory
    train_file = os.path.join(dataset_dir, "train.json")  # Training data
    val_file = os.path.join(dataset_dir, "val.json")  # Validation data
    test_file = os.path.join(dataset_dir, "test.json")  # Test data
    
    # Output directory for the fine-tuned model
    output_dir = f"{CHECKPOINT_DIR}_{datetime.now().strftime('%Y%m%d_%H%M')}"  
    os.makedirs(output_dir, exist_ok=True)
    
    # Training configuration with memory optimization
    config = {
        "batch_size": 1,  
        "num_epochs": 3,  # Number of training epochs
        "learning_rate": 1e-5,  # Slightly lower learning rate for stability
        "use_lora": True,  # Full fine-tuning
        "eval_samples": 274,  # Reduced evaluation samples to save memory
        "max_new_tokens": 4096  # Reduced generation length to save memory
    }
    
    # Log configuration
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Check if files exist
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
    
    # Fine-tune the model
    model, processor = fine_tune_qwen2vl(
        train_file=train_file,
        val_file=val_file,
        output_dir=output_dir,
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        use_lora=config["use_lora"]
    )
    
    # Evaluate the model on test set
    logger.info("Evaluating fine-tuned model on test set")
    evaluation_results = evaluate_model(
        model=model,
        processor=processor,
        test_file=test_file,
        num_samples=config["eval_samples"],
        config=config
    )
    
    # Log evaluation summary
    logger.info(f"Evaluation completed with {len(evaluation_results['samples'])} samples")
    
    # Log metrics
    if 'metrics' in evaluation_results and evaluation_results['metrics']:
        logger.info("Test set metrics:")
        for metric_name, metric_value in evaluation_results['metrics'].items():
            logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({f"test_{metric_name}": metric_value})
    
    logger.info(f"Model and evaluation results saved to {output_dir}")
    
    # Generate a sample prediction
    if evaluation_results['samples']:
        sample = evaluation_results['samples'][0]
        logger.info("Sample prediction:")
        logger.info(f"Pattern: {sample['pattern_name']}")
        logger.info(f"Generated instructions (excerpt): {sample['generated_instructions'][:200]}...")
        
    # Close wandb run if active
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()

