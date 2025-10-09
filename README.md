# CrochetBench 🧶

> Can vision-language models move from describing to doing in the crochet domain?
## 📋 Table of Contents

- [About](#about)
- [Benchmark Tasks](#benchmark-tasks)
- [Models Evaluated](#models-evaluated)
- [Setup](#setup)
- [Data Collection](#data-collection)
- [Data Analysis](#data-analysis)
- [Running Evaluations](#running-evaluations)
- [Project Structure](#project-structure)

---

## About

CrochetBench is a benchmark for evaluating the ability of multimodal large language models to perform fine-grained, low-level procedural reasoning in the domain of crochet. Unlike prior benchmarks that focus on high-level description or visual question answering, CrochetBench shifts the emphasis from **describing to doing**: models are required to recognize stitches, select structurally appropriate instructions, generate crochet pattern instructions, and produce compilable crochet procedures.

### Key Features

- 🧵 **Domain-Specific Language (DSL)**: Adopts the [CrochetPARADE DSL](https://codeberg.org/crochetparade/CrochetPARADE.git) as an intermediate representation
- ✅ **Structural Validation**: Enables functional evaluation through program execution
- 📊 **Comprehensive Tasks**: Covers stitch classification, instruction grounding, and natural language/image-to-DSL translation
- 🎯 **Executable Correctness**: Evaluates models on executable precision beyond surface-level similarity

CrochetBench exposes critical limitations in long-range symbolic reasoning and 3D-aware procedural synthesis, offering a rigorous framework for assessing procedural competence in multimodal models. It highlights the substantial gap between surface-level understanding and executable precision in real-world creative domains.

---

## Benchmark Tasks

The benchmark consists of four evaluation tasks that progress systematically from recognition to comprehension, generation, and ultimately executable synthesis:

| Task ID | Ability Tested | Task Description | Evaluation Metrics | Test Size |
|---------|----------------|------------------|-------------------|----------|
| **A** (CrochetBench-A) | Recognition | Stitch Recognition | F1, Precision, Recall | 6,009 |
| **B** (CrochetBench-B) | Comprehension | Instruction Selection | Accuracy | 6,003 |
| **C** (CrochetBench-C) | Generation | Instruction Generation | BLEU, ROUGE, ChrF | 6,009 |
| **D-step** (CrochetBench-Dstep) | Formalization | Instr.-to-DSL (Step) | Valid Pattern Rate | 119 |
| **D-proj** (CrochetBench-Dproj) | Formalization | Instr.-to-DSL (Project) | Valid Pattern Rate | 100 |

### Task A: Stitch Recognition

Evaluates a model's ability to detect symbolic primitives in crochet images, establishing the foundation for multimodal perception.

- **Input**: Crochet product image  
- **Output**: List of stitches used (e.g., sc, hdc, dc, ch, sl st)  
- **Metrics**: F1, Precision, Recall

### Task B: Instruction Selection

Requires models to align visual evidence with candidate textual instructions, testing multimodal grounding and fine-grained comprehension. Unlike conventional description tasks, candidates are procedural steps rather than captions, requiring reasoning about how individual steps contribute to the final product.

- **Input**: Crochet product image + Multiple choice instructions  
- **Output**: Selected instruction index  
- **Metrics**: Accuracy

### Task C: Instruction Generation

Advances from comprehension to open-ended production, challenging models to generate natural language procedural instructions that are both perceptually grounded and linguistically faithful to domain conventions.

- **Input**: Crochet product image  
- **Output**: Natural language crochet instructions  
- **Metrics**: BLEU, ROUGE, ChrF

### Task D: Instruction-to-DSL Translation

Requires models to output a compilable program in the CrochetPARADE DSL. This task has two variants:

#### D-step: Step-Level Translation

In the step-level setting, the model receives a prefix of natural language–DSL pairs and must generate the DSL line corresponding to the next natural language instruction. This setup reflects an incremental synthesis process in which correctness depends on maintaining stitch-level consistency across steps.

Since crochet patterns are inherently stateful, earlier context is critical for resolving constructs such as increases, repeats, and turning chains. To capture progression through a pattern, the dataset includes:
- **52 early examples** (steps 1–2)
- **34 mid examples** (steps 3–4)  
- **33 late examples** (steps 5–6)

**Input Format**: `Prefix (NL–DSL pairs) + Next NL instruction → Next DSL line`

**Evaluation Metrics**: 
- **Compilation Success Rate (CSR)**: Proportion of generated DSL outputs that compile successfully with the CrochetPARADE validator
- **Error Analysis**: Errors are categorized into (1) syntax structure errors, (2) stitch definition errors, (3) labeling and reference errors, and (4) structural or formatting issues

#### D-proj: Project-Level Translation

In the project-level setting, the model is provided with complete crochet instructions in natural language together with the corresponding product image, and must generate an entire CrochetPARADE program. This variant is globally self-contained but considerably more challenging than the step-level task: models must track stitch states over long horizons, resolve ambiguities in natural language, and produce code that is both syntactically valid and semantically aligned with the final design.

- **Input**: Natural language instructions + Product image  
- **Output**: Complete CrochetPARADE DSL program

**Evaluation Metrics**: 
- **Compilation Success Rate (CSR)**: All-or-nothing executability measure
- **Partial Executable Rate (PER)**: Average fraction of a program that compiles successfully before failure

Compilation-based evaluation directly measures executable faithfulness, ensuring that generated instructions are not only linguistically plausible but also structurally sound.

---

## Models Evaluated

We evaluate a diverse set of vision-language models (VLMs) spanning both open-source and closed-source families:

### Open-Source Models
- **Salesforce BLIP-2 Flan-T5 XL (3B)**: Perception-focused baseline widely adopted in image-text tasks
- **Google Gemma 3 (4B)**: Recent model trained with large-scale multimodal alignment
- **Qwen2-VL (7B)**: Advanced open-source model trained with large-scale multimodal alignment
- **DeepSeek-VL (7B)**: Larger open-source model designed for enhanced vision-language reasoning

### Closed-Source Models
- **GPT-4o**: State-of-the-art commercial vision-language model from OpenAI
- **Gemini 2.5 Flash-Lite**: Google's advanced multimodal model optimized for efficiency
- **Claude Sonnet 4**: Anthropic's state-of-the-art vision-language model

---

## Setup

### Prerequisites

- Python 3.8+
- Node.js (for CrochetPARADE DSL validation)
- GPU recommended for running open-source models

### Installation

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd crochetBench
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Clone CrochetPARADE repository** (required for Task D validation):
   ```bash
   cd benchmark_task
   git clone https://codeberg.org/crochetparade/CrochetPARADE.git
   cd ..
   ```

4. **Configure API Keys**:
   
   Copy the `.env.example` file to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your actual API keys:
   ```bash
   # Required for Claude-based evaluations
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key
   
   # Required for GPT-4o evaluations
   OPENAI_API_KEY=your_actual_openai_api_key
   
   # Required for Gemini evaluations
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```
   
   **Important**: Never commit your `.env` file to version control. It is already listed in `.gitignore`.

---

## Data Collection

### Using Pre-collected Data

The benchmark includes pre-collected test datasets:
- `data/crochet_pattern_by_project/` - Crochet patterns organized by project type for Task A and Task C
- `data/mc_data.json.zip` - Multiple choice data for Task B (must be unzipped before use)
- `data/project_level_test.json` - Project-level test data for Task D
- `data/step_level_test_*.json` - Step-level test data for Task D (organized by step position)

**To extract the Task B dataset**:
```bash
cd data
unzip mc_data.json.zip
```

### Reproducing the Data Collection Process

1. **Navigate to the data collection folder**:
   ```bash
   cd collect_data
   ```

2. **Run the CLI tool** to scrape data:
   ```bash
   # Scrape specific project type
   python cli.py --project-type "Hats" --pages 3
   ```

3. **Preprocess the collected data**:
   ```bash
   python data_preprocessing.py
   ```

4. **Add images** to the collected data:
   ```bash
   python add_image.py
   ```

5. **Create multiple-choice dataset** for Task B:
   ```bash
   python create_mc_dataset.py
   ```

---

## Data Analysis

The `data_analysis/` directory contains tools for analyzing dataset characteristics:

- `analyze_img_links.py` - Validates image link accessibility
- `analyze_pattern_complexity.py` - Computes pattern complexity metrics
- `analyze_skill_level.py` - Analyzes skill level distribution across patterns
- `unique_stitches.txt` - Reference list of unique stitch types used in Task A

**To run the analysis tools**:
```bash
python data_analysis/analyze_img_links.py
python data_analysis/analyze_pattern_complexity.py
python data_analysis/analyze_skill_level.py
```
---

## Running Evaluations

All evaluation scripts are located in the `benchmark_task/` folder.

### Task A: Stitch Recognition

1. **Generate predictions**:
   ```bash
   cd benchmark_task
   python task_a_<model_name>.py
   ```

2. **Evaluate results**:
   ```bash
   python eval_task_a.py <model_name>
   ```

**Example**:
```bash
python task_a_blip.py
python eval_task_a.py blip
```

**Available models**: `blip`, `claude`, `dsvl`, `gemini`, `gemma`, `gpt4o`, `qwen`

---

### Task B: Instruction Selection

**Run inference and evaluation** (evaluation is included in the task script):
```bash
cd benchmark_task
python task_b_<model_name>.py
```

**Example**:
```bash
python task_b_claude.py
```

**Available models**: `blip`, `claude`, `dsvl`, `gemini`, `gemma`, `gpt4o`, `qwen`

---

### Task C: Instruction Generation

1. **Generate instructions**:
   ```bash
   cd benchmark_task
   python task_c_<model_name>.py
   ```

2. **Evaluate results**:
   ```bash
   python eval_task_c.py <model_name>
   ```

**Example**:
```bash
python task_c_gemini.py
python eval_task_c.py gemini
```

**Available models**: `blip`, `claude`, `dsvl`, `gemini`, `gemma`, `gpt4o`, `qwen`

---

### Task D: Instruction-to-DSL Translation

#### Step-Level Evaluation

1. **Generate step-level DSL predictions**:
   ```bash
   cd benchmark_task
   python task_d_step_<model_name>.py
   ```

2. **Evaluate results**:
   ```bash
   python eval_task_d_step_level.py <model_name> <filename>
   ```

**Example**:
```bash
python task_d_step_gpt4o.py
python eval_task_d_step_level.py gpt4o step_level_test_1_2.json
```

**Available models**: `blip`, `claude`, `dsvl`, `gemini`, `gemma`, `gpt4o`, `qwen`

**Available test files**:
- `step_level_test_1_2.json` (early steps: 1–2)
- `step_level_test_3_4.json` (mid steps: 3–4)
- `step_level_test_5_6.json` (late steps: 5–6)

#### Project-Level Evaluation

1. **Generate project-level DSL**:
   ```bash
   cd benchmark_task
   python task_d_project_<model_name>.py
   ```

2. **Evaluate results**:
   ```bash
   python eval_task_d_project_level.py <model_name>
   ```

**Example**:
```bash
python task_d_project_claude.py
python eval_task_d_project_level.py claude
```

**Available models**: `blip`, `claude`, `dsvl`, `gemini`, `gemma`, `gpt4o`, `qwen`

---

## Project Structure

```
crochetBench/
├── benchmark_task/              # Task inference and evaluation scripts
│   ├── task_a_*.py             # Task A: Stitch recognition (by model)
│   ├── task_b_*.py             # Task B: Instruction selection (by model)
│   ├── task_c_*.py             # Task C: Instruction generation (by model)
│   ├── task_d_step_*.py        # Task D: Step-level DSL (by model)
│   ├── task_d_project_*.py     # Task D: Project-level DSL (by model)
│   ├── eval_task_a.py          # Evaluation script for Task A
│   ├── eval_task_c.py          # Evaluation script for Task C
│   ├── eval_task_d_step_level.py       # Evaluation script for Task D (step)
│   ├── eval_task_d_project_level.py    # Evaluation script for Task D (project)
│   ├── verify_crochet_pattern.js       # DSL validator wrapper
│   └── verify_crochet_pattern_with_history.js  # DSL validator with context
│
├── collect_data/                # Data collection utilities
│   ├── cli.py                  # CLI tool for scraping patterns
│   ├── data_scraping.py        # Web scraping functions
│   ├── data_preprocessing.py   # Data preprocessing functions
│   ├── add_image.py            # Image download utility
│   └── create_mc_dataset.py    # Multiple choice dataset builder
│
├── data/                        # Benchmark datasets
│   ├── crochet_pattern_by_project/  # Raw patterns organized by project type
│   ├── mc_data.json.zip        # Multiple choice data (Task B)
│   ├── project_level_test.json # Project-level test set (Task D)
│   ├── step_level_test_1_2.json    # Early steps test set
│   ├── step_level_test_3_4.json    # Mid steps test set
│   └── step_level_test_5_6.json    # Late steps test set
│
├── data_analysis/               # Analysis and statistics tools
│   ├── analyze_img_links.py    # Image link validation
│   ├── analyze_pattern_complexity.py  # Pattern complexity analysis
│   ├── analyze_skill_level.py  # Skill level distribution
│   └── unique_stitches.txt     # List of valid stitch types
│
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore patterns
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```


