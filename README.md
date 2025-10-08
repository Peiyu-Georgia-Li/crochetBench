# CrochetBench 🧶

> A benchmark for evaluating multimodal large language models on fine-grained procedural reasoning in crochet

## 📋 Table of Contents

- [About](#about)
- [Benchmark Tasks](#benchmark-tasks)
- [Models Evaluated](#models-evaluated)
- [Setup](#setup)
- [Data Collection](#data-collection)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)

## About

CrochetBench is a benchmark for evaluating the ability of multimodal large language models to perform fine-grained, low-level procedural reasoning in the domain of crochet. Unlike prior benchmarks that focus on high-level description or visual question answering, CrochetBench shifts the emphasis from **describing to doing**: models are required to recognize stitches, select structurally appropriate instructions, and generate compilable crochet procedures.

### Key Features

- **Domain-Specific Language (DSL)**: Adopts the [CrochetPARADE DSL](https://codeberg.org/crochetparade/CrochetPARADE.git) as intermediate representation
- **Structural Validation**: Enables functional evaluation via execution
- **Comprehensive Tasks**: Covers stitch classification, instruction grounding, and natural language/image-to-DSL translation
- **Executable Correctness**: Evaluates models on executable precision beyond surface-level similarity

CrochetBench exposes limitations in long-range symbolic reasoning and 3D-aware procedural synthesis, offering a new lens for assessing procedural competence in multimodal models and highlighting the gap between surface-level understanding and executable precision in real-world creative domains.

## Benchmark Tasks

The benchmark consists of four evaluation tasks that progress systematically from recognition to comprehension, generation, and ultimately executable synthesis:

### Task A: Stitch Recognition
Evaluates a model's ability to detect symbolic primitives in crochet images, establishing the foundation for multimodal perception.

### Task B: Instruction Selection
Requires models to align visual evidence with candidate textual instructions, testing multimodal grounding and fine-grained comprehension. Unlike conventional description tasks, candidates are procedural steps rather than captions, requiring reasoning about how local steps contribute to the final product.

### Task C: Instruction Generation
Advances from comprehension to open-ended production, challenging models to generate natural-language procedural instructions that are both perceptually grounded and linguistically faithful to domain conventions. Emphasizes lexical and symbolic fidelity.

### Task D: Instruction-to-DSL Translation
Requires models to output a compilable program in the CrochetPARADE DSL:
- **Step-level variant**: Tests local semantic grounding
- **Project-level variant**: Demands global structural consistency across the entire pattern

Compilation-based evaluation directly measures executable faithfulness, ensuring generated instructions are not only linguistically plausible but also structurally sound.

## Models Evaluated

We evaluate a diverse set of vision-language models (VLMs) spanning both open-source and closed-source families:

### Open-Source Models
- **Salesforce BLIP-2 Flan-T5 XL (3B)**: Perception-focused baseline widely used in image-text tasks
- **Google Gemma 3 (4B)**: Recent model trained with large-scale multimodal alignment
- **Qwen2-VL (7B)**: Recent model trained with large-scale multimodal alignment
- **DeepSeek-VL (7B)**: Larger open-source model designed for stronger vision-language reasoning

### Closed-Source Models
- **GPT-4o**: State-of-the-art commercial VLM
- **Gemini 2.5 Flash-Lite**: Google's advanced multimodal model
- **Claude Sonnet 4**: Anthropic's state-of-the-art VLM

## Setup

### Prerequisites

1. **Clone CrochetPARADE repository**:
   ```bash
   git clone https://codeberg.org/crochetparade/CrochetPARADE.git
   ```

2. **Install dependencies** (if applicable):
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   Copy the `.env.example` file to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your actual API keys:
   ```bash
   # Required for Claude-based tasks (Task A, B, C, D with Claude)
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key
   
   # Required for GPT-4 tasks
   OPENAI_API_KEY=your_actual_openai_api_key
   ```
   
   **Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

## Data Collection

To collect crochet data from YarnInspiration:

1. Navigate to the `collect_data` folder
2. Run the CLI tool to scrape data:
   ```bash
   python cli.py
   ```
   
   Or specify project type and pages:
   ```bash
   python cli.py --project-type "Hats" --pages 3
   ```

3. Add images to the collected data:
   ```bash
   python add_image.py
   ```

## Usage

### Running Evaluations

Evaluation scripts are located in the `benchmark_task` folder. Example evaluations:

- **Task A (Stitch Recognition)**:
  ```bash
  python benchmark_task/task_a_blip.py
  python benchmark_task/task_a_claude.py
  ```

- **Task D (DSL Translation)**:
  ```bash
  # Step-level evaluation
  python eval_task_d_step_level.py
  
  # Project-level evaluation
  python eval_task_d_project_level.py
  ```

### Verifying Crochet Patterns

Use the verification scripts to validate DSL outputs:

```bash
node verify_crochet_pattern.js
node verify_crochet_pattern_with_history.js
```

## Project Structure

```
crochetBench/
├── benchmark_task/          # Evaluation scripts for all tasks
│   ├── eval_task_a.py
│   ├── eval_task_c.py
│   ├── task_a_blip.py
│   ├── task_a_claude.py
│   └── ...
├── collect_data/            # Data collection utilities
│   ├── cli.py
│   ├── add_image.py
│   └── data_scraping.py
├── data/                    # Benchmark datasets
│   ├── crochet_pattern_by_project/
│   ├── mc_data.json
│   ├── project_level_test.json
│   └── step_level_test_1_2.json
├── data_analysis/           # Analysis tools
│   └── analyze_img_links.py
├── eval_task_d_project_level.py
├── eval_task_d_step_level.py
├── verify_crochet_pattern.js
└── verify_crochet_pattern_with_history.js
```

## Citation

If you use CrochetBench in your research, please cite:

```bibtex
@article{crochetbench2024,
  title={CrochetBench: A Benchmark for Procedural Reasoning in Multimodal Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

**Note**: This benchmark is designed to assess procedural competence in multimodal models through the lens of executable precision in the crochet domain.