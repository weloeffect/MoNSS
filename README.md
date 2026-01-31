# MoNSSpecialists: Specialist Models for Multi-Hop Fact Verification over Knowledge Graphs

**A neuro-symbolic pipeline for fact verification using trained specialist LMs and comparison with general-purpose LLMs.**

---

## Abstract

This project investigates **specialist small language models (SLMs)** for **multi-hop fact verification** over knowledge-graph-backed claims. We train two specialist modules: (1) **SLM1** — natural language to SPARQL (Text2SPARQL), and (2) **SLM2** — knowledge-graph triples to natural language (SPARQL2Text). These are combined in a **MoNSS** (Modular Neuro-Symbolic Specialists) pipeline and evaluated on a fact-verification benchmark with explicit **single-hop** and **two-hop** reasoning splits. We compare the specialist pipeline against **vanilla** (zero-shot Llama-3.2-3B) and **general-purpose LLM** (Phi-3) baselines. Results show that the specialist pipeline substantially outperforms the vanilla baseline (80.5% vs 51.7% overall accuracy) and is competitive with a larger LLM, with a clear advantage of general LLMs on two-hop reasoning (82.3% vs 75.7%).

---

## 1. Introduction

Fact verification over knowledge graphs (KGs) often requires **multi-hop reasoning**: linking entities across several relations to validate or refute a claim. Large language models (LLMs) can perform this task in a closed-box way, but they are costly and may hallucinate. An alternative is a **neuro-symbolic** design: use the KG explicitly (e.g., via SPARQL) and employ **specialist** models for (a) translating questions into queries and (b) interpreting query results into natural language, with verification decided by symbolic execution over the KG.

This repository implements and evaluates:

- **MoNSS**: A pipeline built from specialist SLMs (Text2SPARQL + SPARQL2Text) for fact verification.
- **Baselines**: Zero-shot **Llama-3.2-3B (Vanilla)** and **Phi-3** for direct claim verification.

The benchmark distinguishes **single-hop** and **two-hop** reasoning to analyze where specialist vs general models excel.

---

## 2. Methodology

### 2.1 Task

Given a **natural language claim** (e.g., *“The successor to Gaston Flosse was called Nuihau Laurey”*), the system must predict **True** or **False** with respect to a reference knowledge graph (DBpedia-style). Claims may require:

- **Single-hop**: one relation (e.g., successor of X).
- **Two-hop**: chaining two relations (e.g., X → relation → Y → relation → Z).

### 2.2 Models and Pipelines

| Setting | Description |
|--------|-------------|
| **MoNSS** | Specialist pipeline: trained Text2SPARQL (SLM1) and SPARQL2Text (SLM2) models used in a KG-grounded verification pipeline. |
| **Vanilla** | Llama-3.2-3B-Instruct with zero-shot prompt: *“Is the following claim factually correct? Answer only True or False.”* |
| **LLM** | Microsoft Phi-3-medium-4k-instruct with the same verification prompt (batch inference). |

### 2.3 Specialist Modules

- **SLM1 (Text2SPARQL)**  
  - **Base**: Meta-Llama-3-8B.  
  - **Training**: LoRA/QLoRA on `slm1_train.jsonl` (claim → SPARQL ASK query, DBpedia).  
  - **Input**: Natural language question + optional schema context.  
  - **Output**: SPARQL query.

- **SLM2 (SPARQL2Text)**  
  - **Base**: Qwen2.5-1.5B (training) / configurable for inference.  
  - **Training**: QLoRA on `slm2_train.jsonl` (KG triples → natural language sentence).  
  - **Input**: Triples (single-hop or two-hop with `INTERMEDIATE`).  
  - **Output**: Natural language statement.

---

## 3. Data

- **Training**
  - `data/train/slm1_train.jsonl`: claim → SPARQL ASK (instruction, input, output).
  - `data/train/slm2_train.jsonl`: KG triples → natural language (instruction, input list of triples, output).
- **Evaluation**
  - A fact-verification test set (JSONL) with fields: `input_question`, `label` (true/false), `hop_count` (1 or 2). Test data is used under `data/test/` (see Reproduction).

---

## 4. Experimental Results

Benchmark results on the fact-verification test set (~3.5k examples):

| Model / Pipeline | Overall Acc. | Single-Hop Acc. | Two-Hop Acc. |
|------------------|-------------|-----------------|---------------|
| **MoNSS** (specialist) | **80.5%** | **86.6%** | **75.7%** |
| Vanilla (Llama-3.2-3B) | 51.7% | 48.9% | 53.9% |
| LLM (Phi-3) | **86.3%** | **91.5%** | **82.3%** |

- The **specialist pipeline (MoNSS)** strongly outperforms the **vanilla** baseline (+28.8 pp overall), showing the benefit of explicit KG use and specialist modules.
- The **general-purpose LLM (Phi-3)** achieves the highest accuracy overall and on both single-hop and two-hop, suggesting that two-hop reasoning remains challenging for the current specialist pipeline and that scale/instruction tuning help for this benchmark.

Detailed result files (for reviewers):

- `benchmark/MoNSS_results.txt`
- `benchmark/Vanilla_SLM_results.txt`
- `benchmark/LLM_results.txt`

---

## 5. Project Structure

```
MoNSSpecialists/
├── README.md
├── main.py                    # (Legacy/utility scripts; see code.)
├── model/
│   ├── Vanilla/               # Llama-3.2-3B zero-shot verification
│   │   └── inference.py
│   ├── LLM/                   # Phi-3: Text2SPARQL, SPARQL2Text, verify
│   │   └── inference.py
│   └── Sparql2Text/           # SLM2 training & inference (Qwen2.5-1.5B / LoRA)
│       ├── main.py            # Training
│       └── inference.py
├── benchmark/
│   ├── slm1_inference.py      # SLM1 (Text2SPARQL) inference for evaluation
│   ├── slm2_output.jsonl     # SLM2 benchmark outputs
│   ├── MoNSS_results.txt      # Specialist pipeline results
│   ├── Vanilla_SLM_results.txt
│   └── LLM_results.txt
├── data/
│   └── train/
│       ├── slm1_train.jsonl
│       └── slm2_train.jsonl
└── scripts/
    ├── build_nodes.py         # (Data prep placeholder)
    ├── build_relationships.py
    ├── push2HF_slm1.py        # Upload SLM1 adapters to Hugging Face
    └── push2HF_slm2.py        # Upload SLM2 adapters to Hugging Face
```

---

## 6. Reproduction

### 6.1 Environment

- Python 3.x, PyTorch, `transformers`, `peft` (LoRA/QLoRA), `datasets`, `trl` (for training).
- GPU recommended for training and inference.

### 6.2 Training Specialist Models

- **SLM1**: Train with the script that uses `data/train/slm1_train.jsonl` and Llama-3-8B (see `benchmark/slm1_inference.py` for expected adapter path `outputs/Llama-3-8B-text2sparql-qlora`).
- **SLM2**: From project root, run training in `model/Sparql2Text/main.py` (reads `slm2_train.jsonl`; output dir `outputs/Qwen2.5-1.5B-SPARQL2TEXT` or equivalent).

### 6.3 Running Benchmarks

- **Vanilla**:  
  `python model/Vanilla/inference.py --input <test.jsonl> --output results/vanilla_results.jsonl`

- **LLM (Phi-3)**:  
  `python model/LLM/inference.py --task verify --input <test.jsonl> --output results/llm_results.jsonl`

- **SLM1**:  
  Place test JSONL under `benchmark/` as expected by `benchmark/slm1_inference.py` and run it to generate SPARQL predictions.

Test JSONL must include `input_question` (or `claim`/`input`), `label`, and `hop_count` for accuracy and single-hop/two-hop breakdowns.

### 6.4 Hugging Face

- SLM1 adapters: `scripts/push2HF_slm1.py` (target repo, e.g. `weloSai/Llama-3.2-3B-TEXT2SPARQL`).
- SLM2 adapters: `scripts/push2HF_slm2.py` (e.g. `weloSai/Qwen2.5-1.5B-SPARQL2TEXT`).

---

## 7. Citation and Contact

If you use this code or the reported results in your research, please cite this repository. For questions or reviewer feedback, please open an issue or contact the authors.

---

*MoNSSpecialists — Modular Neuro-Symbolic Specialists for multi-hop fact verification over knowledge graphs.*
