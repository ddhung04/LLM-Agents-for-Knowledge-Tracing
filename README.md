# LLM Agents for Knowledge Tracing

A multi-agent system using Llama-3.1-8B for personalized knowledge tracing in tutor-student dialogues. Built on top of [Scarlatos et al. (LAK 2025)](https://arxiv.org/abs/2409.16490), extended with Error Analysis and Feedback Generation agents.

---

## Overview

Traditional knowledge tracing (KT) methods work on multiple-choice data and cannot handle free-text dialogue. This project builds a 3-agent pipeline that operates directly on tutor-student conversations:

| Agent | Role | Output |
|---|---|---|
| **Mastery Estimator** | Predicts P(mastery) per knowledge component via LLM logit reading | `[0, 1]` per KC |
| **Error Analyzer** | Classifies error type when student answers incorrectly | JSON with type + explanation |
| **Feedback Generator** | Produces personalized pedagogical feedback | JSON with feedback + scaffolding |

A rule-based **Adaptive Planner** orchestrates the three agents sequentially per dialogue turn.

---

## Results

| Agent | Metric | Result |
|---|---|---|
| Mastery (zero-shot) | AUC — CoMTA | 0.562 |
| Mastery (zero-shot) | AUC — MathDial | 0.568 |
| Mastery (fine-tuned) | AUC — CoMTA | 0.525 |
| Error Analyzer | Agreement with annotation | 89.1% |
| Feedback Generator | Overall quality (LLM eval, /5) | 3.96 |
| Pipeline | End-to-end success rate | 100% |

Zero-shot LLMKT outperforms DKT, AKT, and SAINT without any training data.

---

## Setup

```bash
git clone https://github.com/ddhung04/LLM-Agents-for-Knowledge-Tracing.git
cd LLM-Agents-for-Knowledge-Tracing
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Set environment variable:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**Model:** Download `meta-llama/Llama-3.1-8B-Instruct` from Hugging Face and update the model path in `dialogue_kt/models/lm.py`.

---

## Data

Annotated datasets are included in `data/annotated/` and can be used directly:

| File | Description |
|---|---|
| `comta_atc.csv` | CoMTA dialogues with KC + correctness labels |
| `mathdial_test_atc.csv` | MathDial test set with KC + correctness labels |
| `kc_dict_*.json` | Knowledge component dictionaries |

Data is subject to original licenses. See `data/annotated/COMTA_LICENSE.txt` for CoMTA. MathDial is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

---

## Usage

### 1. Verify setup
```bash
python test_zeroshot.py
```

### 2. Run Mastery Estimator (zero-shot)
```bash
python run_zeroshot_eval.py          # CoMTA fold 1
python run_mathdial_zeroshot.py      # MathDial subset
```

### 3. Run Error Analyzer
```bash
python run_error_analysis.py
python evaluate_errors.py
```

### 4. Run full pipeline (all 3 agents)
```bash
python run_feedback_pipeline.py
python evaluate_feedback.py
```

### 5. Fine-tune Mastery Estimator
```bash
python run_mini_finetune.py          # 4-bit QLoRA on CoMTA
python eval_finetune.py
```

### 6. Visualize mastery curves
```bash
python visualize_mastery.py
```

Results are saved to `results/`.

---

## Project Structure

```
├── dialogue_kt/
│   ├── models/lm.py          # Llama-3.1-8B with 4-bit QLoRA (modified)
│   ├── data_loading.py
│   ├── kt_data_loading.py
│   ├── prompting.py
│   └── utils.py
├── error_analyzer.py          # Agent 2: error classification
├── feedback_generator.py      # Agent 3: personalized feedback
├── run_*.py                   # Experiment scripts
├── eval_*.py                  # Evaluation scripts
├── data/annotated/            # Datasets
├── results/                   # Metrics and figures
└── saved_models/              # LoRA adapter checkpoints (not tracked)
```

---

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{scarlatos2024exploringknowledgetracingtutorstudent,
  title={Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs},
  author={Alexander Scarlatos and Ryan S. Baker and Andrew Lan},
  year={2025},
  booktitle={Proceedings of the 15th Learning Analytics and Knowledge Conference, LAK 2025},
  publisher={ACM},
}
```

---

## Acknowledgements

Built on [dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt) by Scarlatos et al. Extended with multi-agent architecture, 4-bit QLoRA adaptation for consumer GPUs, and new Error Analysis and Feedback Generation components as part of a graduation thesis at Vietnam National University, Hanoi — University of Engineering and Technology.