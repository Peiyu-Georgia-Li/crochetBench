<div align="center">

# CrochetBench 🧶

### Can Vision-Language Models Move from Describing to Doing in the Crochet Domain?

[![ACL 2026](https://img.shields.io/badge/ACL%202026-Long%20Paper-%23B31B1B?style=for-the-badge)](https://aclanthology.org/)
[![Paper](https://img.shields.io/badge/Paper-📎-lightgrey?style=for-the-badge)](https://aclanthology.org/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-CrochetBench-green?style=for-the-badge)](https://huggingface.co/datasets/peiyugeorgia/CrochetBench)
[![Model](https://img.shields.io/badge/🤗%20Model-Qwen2VL--TaskC-purple?style=for-the-badge)](https://huggingface.co/peiyugeorgia/qwen2vl-crochetbench-task-c-finetuned)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue?style=for-the-badge)](LICENSE)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-orange?style=for-the-badge)](LICENSE)

<img src="https://img.shields.io/github/last-commit/Peiyu-Georgia-Li/crochetBench?style=flat-square&color=5D6D7E" />
<img src="https://img.shields.io/github/stars/Peiyu-Georgia-Li/crochetBench?style=flat-square&color=5D6D7E" />
<img src="https://img.shields.io/github/languages/top/Peiyu-Georgia-Li/crochetBench?style=flat-square&color=5D6D7E" />

**[English](#english) | [中文](#chinese)**

</div>

---

## 🗞️ Updates & News

- **[2026]** 🎉 CrochetBench has been accepted at **ACL 2026 (Long Paper)**!

<details>
<summary>Click to expand more</summary>

- **[2026]** Paper, dataset, and evaluation code released publicly.
- **[2025]** Data collection and benchmark construction completed.
- **[2025]** CrochetPARADE DSL adopted as the executable intermediate representation for Task D.

</details>

---

## 👂 TL;DR

- **CrochetBench** is a multimodal benchmark that shifts the evaluation paradigm from *describing* to *doing* in the crochet domain.
- It features **4 tasks** (recognition → comprehension → generation → DSL formalization) evaluated on **7 VLMs**.
- Task D uniquely tests **executable correctness** via compilation against the CrochetPARADE DSL validator.

---

<a id="english"></a>

# 🌐 CrochetBench — English

## 📋 Table of Contents

- [About](#about)
- [Citation](#citation)
- [License](#license)
- [Benchmark Tasks](#benchmark-tasks)
- [Models Evaluated](#models-evaluated)
- [Setup](#setup)
- [Data Collection](#data-collection)
- [Data Analysis](#data-analysis)
- [Fine-tuning](#fine-tuning)
- [Running Evaluations](#running-evaluations)
- [Project Structure](#project-structure)

---

## 🙋 About

CrochetBench is a benchmark for evaluating the ability of multimodal large language models to perform fine-grained, low-level procedural reasoning in the domain of crochet. Unlike prior benchmarks that focus on high-level description or visual question answering, CrochetBench shifts the emphasis from **describing to doing**: models are required to recognize stitches, select structurally appropriate instructions, generate crochet pattern instructions, and produce compilable crochet procedures.

### Key Features

- 🧵 **Domain-Specific Language (DSL)**: Adopts the [CrochetPARADE DSL](https://codeberg.org/crochetparade/CrochetPARADE.git) as an intermediate representation
- ✅ **Structural Validation**: Enables functional evaluation through program execution
- 📊 **Comprehensive Tasks**: Covers stitch classification, instruction grounding, and natural language / image-to-DSL translation
- 🎯 **Executable Correctness**: Evaluates models on executable precision beyond surface-level similarity

CrochetBench exposes critical limitations in long-range symbolic reasoning and 3D-aware procedural synthesis, offering a rigorous framework for assessing procedural competence in multimodal models.

---

## ⚖️ License

This repository uses a dual license:

| Component | License |
|-----------|---------|
| Code (`benchmark_task/`, `collect_data/`, `data_analysis/`, `finetune/`, `split_dataset/`) | [MIT License](LICENSE) |
| Dataset (`data/`) | [CC BY-NC 4.0](LICENSE) — research & personal use only, no commercial use |

### Data Attribution

The dataset is derived from crochet patterns published on [Yarnspirations](https://www.yarnspirations.com/) (Bernat, Patons, Lily, Caron, Red Heart, Sugar Bush, Phentex). Yarnspirations has granted permission for academic research and personal use. Reproduction for resale is **not** permitted.

<details>
<summary>View copyright permission from Yarnspirations</summary>

![copyright_permission_email](https://github.com/user-attachments/assets/f684bc93-c652-42a8-8e28-c952e2e406e4)

</details>

---

## 🛎️ Benchmark Tasks

The benchmark progresses from recognition → comprehension → generation → executable synthesis:

| Task | Ability | Description | Evaluation Metrics |
|------|---------|-------------|---------|
| **A** | Recognition | Stitch Recognition | F1, Precision, Recall |
| **B** | Comprehension | Instruction Selection | Accuracy |
| **C** | Generation | Instruction Generation | BLEU, ROUGE, ChrF |
| **D-step** | Formalization | Instruction-to-DSL (Step) | Valid Pattern Rate |
| **D-proj** | Formalization | Instruction-to-DSL (Project) | Valid Pattern Rate, Dino Similarity |

### Task A: Stitch Recognition

- **Input**: Crochet product image
- **Output**: List of stitches (e.g., sc, hdc, dc, ch, sl st)
- **Metrics**: F1, Precision, Recall

### Task B: Instruction Selection

- **Input**: Crochet product image + multiple-choice instructions
- **Output**: Selected instruction index
- **Metrics**: Accuracy

### Task C: Instruction Generation

- **Input**: Crochet product image
- **Output**: Natural language crochet instructions
- **Metrics**: BLEU, ROUGE, ChrF

### Task D: Instruction-to-DSL Translation

#### D-step: Step-Level Translation

The model receives NL–DSL prefix pairs and must generate the next DSL line.

Dataset split by step position:
- **52 early examples** (steps 1–2)
- **34 mid examples** (steps 3–4)
- **33 late examples** (steps 5–6)

**Metrics**: Valid Pattern Rate.

#### D-proj: Project-Level Translation

The model generates a **complete** CrochetPARADE program from full NL instructions + image.

**Metrics**:
- **Valid Pattern Rate** — fraction of generated programs that pass DSL validation and execute successfully
- **Dino Similarity** — visual similarity between rendered outputs and reference crochet images

---

## 🤖 Models Evaluated

### Open-Source
| Model | Size | Notes |
|-------|------|-------|
| BLIP-2 Flan-T5 XL | 3B | Perception-focused baseline |
| Gemma 3 | 4B | Large-scale multimodal alignment |
| Qwen2-VL | 7B | Advanced open-source VLM |
| DeepSeek-VL | 7B | Enhanced vision-language reasoning |

### Closed-Source
| Model | Provider |
|-------|----------|
| GPT-4o | OpenAI |
| Gemini 2.5 Flash-Lite | Google |
| Claude Sonnet 4 | Anthropic |

---

## 🧹 Setup

### Prerequisites

- Python 3.8+
- Node.js (for CrochetPARADE DSL validation)
- GPU recommended for open-source models

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd crochetBench

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Clone CrochetPARADE (required for Task D)
cd benchmark_task
git clone https://codeberg.org/crochetparade/CrochetPARADE.git
cd ..

# 4. Set up API keys
cp .env.example .env
# Edit .env and fill in your keys:
#   ANTHROPIC_API_KEY=...
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...
```

> **Security**: Never commit your `.env` file — it is already listed in `.gitignore`.

---

## 📦 Data Collection

### Using Pre-collected Data

| File | Description |
|------|-------------|
| `data/crochet_pattern_by_project/` | Raw patterns by project type (Task A & C) |
| `data/mc_data.json.zip` | Multiple-choice data for Task B |
| `data/project_level_test.json` | Project-level test set (Task D) |
| `data/step_level_test_*.json` | Step-level test sets (Task D) |

```bash
# Unzip Task B data
cd data && unzip mc_data.json.zip
```

### Reproducing Data Collection

```bash
cd collect_data
python cli.py --project-type "Hats" --pages 3   # collect
python data_preprocessing.py                     # preprocess
python add_image.py                              # download images
python create_mc_dataset.py                      # build Task B dataset
```

---

## 📊 Data Analysis

```bash
python data_analysis/analyze_img_links.py           # validate image links
python data_analysis/analyze_pattern_complexity.py  # complexity metrics
python data_analysis/analyze_skill_level.py         # skill level distribution
```

---

## 🔧 Fine-tuning

The `finetune/` directory contains a **full fine-tuning** (no LoRA) experiment for **Qwen2-VL-7B-Instruct** on Task C (instruction generation). Data splits are in `split_dataset/`.

The dataset splits and fine-tuned model are available on 🤗 Hugging Face:

| Resource | Link |
|----------|------|
| Dataset (train/val/test splits) | **[peiyugeorgia/CrochetBench](https://huggingface.co/datasets/peiyugeorgia/CrochetBench)** |
| Fine-tuned model (Task C) | **[peiyugeorgia/qwen2vl-crochetbench-task-c-finetuned](https://huggingface.co/peiyugeorgia/qwen2vl-crochetbench-task-c-finetuned)** |

| Split | File | Size |
|-------|------|------|
| Train | `split_dataset/train.json` | 4,859 examples |
| Val | `split_dataset/val.json` | 274 examples |
| Test | `split_dataset/test.json` | 877 examples |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Fine-tuning | Full (all 8.29B parameters trainable) |
| Learning rate | `1e-5` |
| Epochs | `3` |
| Batch size | `1` (× 8 grad. accum. = effective batch 8) |
| Max sequence length | `1024` |
| Mixed precision | BF16 |
| Gradient checkpointing | ✓ |
| Eval / save steps | `500` |
| Best checkpoint | Step 1500 |

```bash
# 1. In finetune/train_qwen_task_c.py, update the config dict in main():
#      "use_lora": False,       # full fine-tuning
#      "learning_rate": 1e-5,
#      "num_epochs": 3,
#      "batch_size": 1,
#    and set bf16=True in the TrainingArguments block.

# 2. Install dependencies
pip install transformers accelerate wandb qwen-vl-utils

# 3. Run training (single GPU)
cd finetune
CUDA_VISIBLE_DEVICES=0 python train_qwen_task_c.py
```

> **Hardware:** Full fine-tuning of 8.3B parameters requires ≥ 80 GB VRAM (e.g., A100-80GB). For smaller GPUs use `"use_lora": True`. Uses 🤗 Transformers `Trainer` with W&B logging.

---

## 🙌 Running Evaluations

All scripts are in `benchmark_task/`. Run `cd benchmark_task` first.

### Task A — Stitch Recognition

```bash
python task_a_<model>.py          # generate predictions
python eval_task_a.py <model>     # evaluate
# e.g.: python task_a_claude.py && python eval_task_a.py claude
```

### Task B — Instruction Selection

```bash
python task_b_<model>.py          # inference + evaluation combined
```

### Task C — Instruction Generation

```bash
python task_c_<model>.py          # generate instructions
python eval_task_c.py <model>     # evaluate
```

### Task D — Step-Level

```bash
python task_d_step_<model>.py
python eval_task_d_step_level.py <model> <test_file>
# e.g.: python eval_task_d_step_level.py gpt4o step_level_test_1_2.json
```

Test files: `step_level_test_1_2.json` · `step_level_test_3_4.json` · `step_level_test_5_6.json`

### Task D — Project-Level

```bash
python task_d_project_<model>.py
python eval_task_d_project_level.py <model>
```

**Available models**: `blip` · `claude` · `dsvl` · `gemini` · `gemma` · `gpt4o` · `qwen`

---

## 📁 Project Structure

```
crochetBench/
├── benchmark_task/              # Inference and evaluation scripts
│   ├── utils.py                #   Shared image & data utilities
│   ├── task_a_*.py             #   Task A (by model)
│   ├── task_b_*.py             #   Task B (by model)
│   ├── task_c_*.py             #   Task C (by model)
│   ├── task_d_step_*.py        #   Task D step-level (by model)
│   ├── task_d_project_*.py     #   Task D project-level (by model)
│   ├── eval_task_a.py
│   ├── eval_task_c.py
│   ├── eval_task_d_step_level.py
│   ├── eval_task_d_project_level.py
│   ├── verify_crochet_pattern.js
│   └── verify_crochet_pattern_with_history.js
├── collect_data/               # Data collection pipeline
├── data/                       # Benchmark datasets
├── data_analysis/              # Dataset statistics tools
├── finetune/                   # Qwen2-VL LoRA fine-tuning (Task C)
├── split_dataset/              # Train / val / test splits
├── .env.example
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

---

<a id="chinese"></a>

# 🌐 CrochetBench — 中文文档

## 📋 目录

- [项目介绍](#项目介绍)
- [引用](#引用)
- [许可证](#许可证)
- [评测任务](#评测任务)
- [评测模型](#评测模型)
- [环境配置](#环境配置)
- [数据收集](#数据收集)
- [数据分析](#数据分析)
- [模型微调](#模型微调)
- [运行评测](#运行评测)
- [项目结构](#项目结构)

---

## 🙋 项目介绍

CrochetBench 是一个评测多模态大语言模型在钩针编织领域执行细粒度过程性推理能力的基准测试集。与以往侧重于高层描述或视觉问答的基准不同，CrochetBench 将重点从**「描述」转向「执行」**：模型需要识别针法、选择结构正确的操作说明、生成钩针花样文本，并输出可编译的钩针程序。

### 核心特性

- 🧵 **领域专用语言（DSL）**：采用 [CrochetPARADE DSL](https://codeberg.org/crochetparade/CrochetPARADE.git) 作为中间表示
- ✅ **结构化验证**：通过程序执行实现功能性评测
- 📊 **全面的任务体系**：涵盖针法分类、说明定位、自然语言 / 图像到 DSL 的转换
- 🎯 **可执行正确性**：超越表面相似度，评测模型输出的可执行精确度

CrochetBench 揭示了当前模型在长程符号推理和三维感知过程合成方面的核心瓶颈，为评估多模态模型的过程性能力提供了严格的框架。

---

## 📝 引用

如果您在研究中使用了 CrochetBench，请引用：

```bibtex
@inproceedings{peiyuli2026,
  title     = {CrochetBench: Can Vision-Language Models Move from Describing to
               Doing in Crochet Domain?},
  author    = {Li, Peiyu and Huang, Xiaobao and Hua, Ting and Chawla, Nitesh V.},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  note      = {To appear}
}
```

---

## ⚖️ 许可证

| 组件 | 许可证 |
|------|--------|
| 代码（`benchmark_task/`、`collect_data/`、`data_analysis/`、`finetune/`、`split_dataset/`） | [MIT License](LICENSE) |
| 数据集（`data/`） | [CC BY-NC 4.0](LICENSE) — 仅限学术研究和个人使用，禁止商业用途 |

### 数据版权说明

数据集来源于 [Yarnspirations](https://www.yarnspirations.com/) 发布的钩针花样（品牌：Bernat、Patons、Lily、Caron、Red Heart、Sugar Bush、Phentex）。Yarnspirations 已授权用于学术研究和个人使用，**禁止转售**。

<details>
<summary>查看 Yarnspirations 版权授权邮件</summary>

![copyright_permission_email](https://github.com/user-attachments/assets/f684bc93-c652-42a8-8e28-c952e2e406e4)

</details>

---

## 🛎️ 评测任务

四个任务从识别到理解、生成，最终到可执行合成，层层递进：

| 任务 | 测试能力 | 任务描述 | 评测指标 |
|------|---------|---------|---------|
| **A** | 识别 | 针法识别 | F1、精确率、召回率 |
| **B** | 理解 | 说明选择 | 准确率 |
| **C** | 生成 | 说明生成 | BLEU、ROUGE、ChrF |
| **D-step** | 形式化 | 说明到 DSL（步骤级） | Valid Pattern Rate |
| **D-proj** | 形式化 | 说明到 DSL（项目级） | Valid Pattern Rate、Dino Similarity |

### 任务 A：针法识别

- **输入**：钩针成品图像
- **输出**：所用针法列表（如 sc、hdc、dc、ch、sl st）
- **指标**：F1、精确率、召回率

### 任务 B：说明选择

- **输入**：钩针成品图像 + 多选说明
- **输出**：所选说明的索引
- **指标**：准确率

### 任务 C：说明生成

- **输入**：钩针成品图像
- **输出**：自然语言钩针说明
- **指标**：BLEU、ROUGE、ChrF

### 任务 D：说明到 DSL 转换

#### D-step：步骤级转换

模型接收 NL–DSL 前缀对，需生成下一条 DSL 行，按步骤位置划分：

- **52 条早期样本**（步骤 1–2）
- **34 条中期样本**（步骤 3–4）
- **33 条后期样本**（步骤 5–6）

**指标**：Valid Pattern Rate（有效模式率）。

#### D-proj：项目级转换

模型根据完整自然语言说明和成品图像，生成**完整的** CrochetPARADE 程序。

**指标**：
- **Valid Pattern Rate** — 生成程序通过 DSL 校验并成功执行的比例
- **Dino Similarity** — 渲染结果与参考钩针图像的视觉相似度

---

## 🤖 评测模型

### 开源模型

| 模型 | 参数量 | 说明 |
|------|--------|------|
| BLIP-2 Flan-T5 XL | 3B | 图文任务感知基线 |
| Gemma 3 | 4B | 大规模多模态对齐 |
| Qwen2-VL | 7B | 先进开源 VLM |
| DeepSeek-VL | 7B | 增强视觉语言推理 |

### 闭源模型

| 模型 | 提供商 |
|------|--------|
| GPT-4o | OpenAI |
| Gemini 2.5 Flash-Lite | Google |
| Claude Sonnet 4 | Anthropic |

---

## 🧹 环境配置

### 前置要求

- Python 3.8+
- Node.js（用于 CrochetPARADE DSL 验证）
- 推荐使用 GPU 运行开源模型

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd crochetBench

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 克隆 CrochetPARADE（任务 D 验证所需）
cd benchmark_task
git clone https://codeberg.org/crochetparade/CrochetPARADE.git
cd ..

# 4. 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入以下密钥：
#   ANTHROPIC_API_KEY=...
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...
```

> **注意**：请勿将 `.env` 文件提交到版本控制，它已被加入 `.gitignore`。

---

## 📦 数据收集

### 使用预收集数据

| 文件 | 说明 |
|------|------|
| `data/crochet_pattern_by_project/` | 按项目类型整理的花样（任务 A & C） |
| `data/mc_data.json.zip` | 任务 B 多选数据（使用前解压） |
| `data/project_level_test.json` | 任务 D 项目级测试集 |
| `data/step_level_test_*.json` | 任务 D 步骤级测试集 |

```bash
cd data && unzip mc_data.json.zip
```

### 复现数据收集

```bash
cd collect_data
python cli.py --project-type "Hats" --pages 3   # 采集数据
python data_preprocessing.py                     # 数据预处理
python add_image.py                              # 下载图片
python create_mc_dataset.py                      # 构建任务 B 数据集
```

---

## 📊 数据分析

```bash
python data_analysis/analyze_img_links.py           # 验证图片链接
python data_analysis/analyze_pattern_complexity.py  # 花样复杂度
python data_analysis/analyze_skill_level.py         # 技能等级分布
```

---

## 🔧 模型微调

`finetune/` 目录包含针对任务 C（钩针说明生成）的 **Qwen2-VL-7B-Instruct 全量微调**（无 LoRA）实验，数据集划分位于 `split_dataset/`。

数据集划分及微调模型已上传至 🤗 Hugging Face：

| 资源 | 链接 |
|------|------|
| 数据集（train/val/test 划分） | **[peiyugeorgia/CrochetBench](https://huggingface.co/datasets/peiyugeorgia/CrochetBench)** |
| 微调模型（任务 C） | **[peiyugeorgia/qwen2vl-crochetbench-task-c-finetuned](https://huggingface.co/peiyugeorgia/qwen2vl-crochetbench-task-c-finetuned)** |

| 划分 | 文件 | 规模 |
|------|------|------|
| 训练集 | `split_dataset/train.json` | 4,859 条 |
| 验证集 | `split_dataset/val.json` | 274 条 |
| 测试集 | `split_dataset/test.json` | 877 条 |

**训练配置：**

| 参数 | 值 |
|------|-----|
| 基础模型 | `Qwen/Qwen2-VL-7B-Instruct` |
| 微调方式 | 全量微调（8.29B 参数全部可训练） |
| 学习率 | `1e-5` |
| 训练轮数 | `3` |
| Batch size | `1`（× 8 梯度累积 = 有效 batch 8） |
| 最大序列长度 | `1024` |
| 混合精度 | BF16 |
| 梯度检查点 | ✓ |
| 评估 / 保存步数 | `500` |
| 最佳检查点 | Step 1500 |

```bash
# 1. 在 finetune/train_qwen_task_c.py 的 main() 中修改配置：
#      "use_lora": False,       # 全量微调
#      "learning_rate": 1e-5,
#      "num_epochs": 3,
#      "batch_size": 1,
#    并在 TrainingArguments 中设置 bf16=True。

# 2. 安装依赖
pip install transformers accelerate wandb qwen-vl-utils

# 3. 启动训练（单卡）
cd finetune
CUDA_VISIBLE_DEVICES=0 python train_qwen_task_c.py
```

> **硬件要求：** 8.3B 参数全量微调需要 ≥ 80 GB 显存（如 A100-80GB）。显存不足时可改用 `"use_lora": True`。使用 🤗 Transformers `Trainer`，日志记录到 W&B。

---

## 🙌 运行评测

所有脚本位于 `benchmark_task/`，请先 `cd benchmark_task`。

### 任务 A — 针法识别

```bash
python task_a_<model>.py          # 生成预测
python eval_task_a.py <model>     # 计算指标
# 示例：python task_a_claude.py && python eval_task_a.py claude
```

### 任务 B — 说明选择

```bash
python task_b_<model>.py          # 推理 + 评测一体
```

### 任务 C — 说明生成

```bash
python task_c_<model>.py          # 生成说明
python eval_task_c.py <model>     # 计算指标
```

### 任务 D — 步骤级

```bash
python task_d_step_<model>.py
python eval_task_d_step_level.py <model> <test_file>
# 示例：python eval_task_d_step_level.py gpt4o step_level_test_1_2.json
```

测试文件：`step_level_test_1_2.json`（早期）· `step_level_test_3_4.json`（中期）· `step_level_test_5_6.json`（后期）

### 任务 D — 项目级

```bash
python task_d_project_<model>.py
python eval_task_d_project_level.py <model>
```

**可用模型名**：`blip` · `claude` · `dsvl` · `gemini` · `gemma` · `gpt4o` · `qwen`

---

## 📁 项目结构

```
crochetBench/
├── benchmark_task/              # 任务推理与评测脚本
│   ├── utils.py                #   共享图像与数据工具函数
│   ├── task_a_*.py             #   任务 A（按模型）
│   ├── task_b_*.py             #   任务 B（按模型）
│   ├── task_c_*.py             #   任务 C（按模型）
│   ├── task_d_step_*.py        #   任务 D 步骤级（按模型）
│   ├── task_d_project_*.py     #   任务 D 项目级（按模型）
│   ├── eval_task_a.py
│   ├── eval_task_c.py
│   ├── eval_task_d_step_level.py
│   ├── eval_task_d_project_level.py
│   ├── verify_crochet_pattern.js
│   └── verify_crochet_pattern_with_history.js
├── collect_data/               # 数据收集工具
├── data/                       # 基准数据集
├── data_analysis/              # 数据集统计工具
├── finetune/                   # Qwen2-VL LoRA 微调（任务 C）
├── split_dataset/              # 训练 / 验证 / 测试集划分
├── .env.example
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

---

## 📝 Citation / 引用

If you use CrochetBench in your research, please cite / 如果您在研究中使用了 CrochetBench，请引用：

```bibtex
@inproceedings{peiyuli2026,
  title     = {CrochetBench: Can Vision-Language Models Move from Describing to
               Doing in Crochet Domain?},
  author    = {Li, Peiyu and Huang, Xiaobao and Hua, Ting and Chawla, Nitesh V.},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  note      = {To appear}
}
```
