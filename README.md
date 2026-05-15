# LLM Agent for Knowledge Tracing in Tutor-Student Dialogues

> **Khóa luận tốt nghiệp** | Trường Đại học Công nghệ — Khoa Trí tuệ Nhân tạo | 2026

Hệ thống multi-agent dựa trên Large Language Model (LLM) cho Knowledge Tracing trong dialogue tutor-học sinh. Mở rộng từ [paper LLMKT (Scarlatos et al., LAK 2025)](https://arxiv.org/abs/2503.11733) với 3 agents chuyên biệt và pipeline end-to-end.

##  Đặc điểm nổi bật

-  **3 LLM Agents** — Mastery Estimator, Error Analyzer, Feedback Generator
-  **Zero-shot + Fine-tuned** — Hỗ trợ cả 2 chế độ trên Llama-3.1-8B
-  **Đánh giá đa metric** — AUC, F1, Agreement rate, Quality scores
-  **End-to-end pipeline** — Demo trên 30 cases với 100% success rate

##  Kết quả thực nghiệm

| Component | Metric | Kết quả |
|---|---|---|
| Mastery Estimator (zero-shot) | AUC CoMTA | **0.562** |
| Mastery Estimator (zero-shot) | AUC MathDial | 0.568 |
| Mastery Estimator (fine-tuned) | AUC CoMTA per-KC | 0.525 |
| Error Analyzer | Agreement rate | **89.1%** |
| Feedback Generator | Quality (1-5) | **3.96/5.0** |
| End-to-end Pipeline | Success rate | **100%** (30/30) |

Zero-shot LLMKT vượt các baseline cổ điển: DKT (0.532), AKT (0.514), SAINT (0.478) — **không cần training**.

##  Kiến trúc hệ thống

```
                    ┌──────────────────┐
                    │   Học sinh       │
                    └────────┬─────────┘
                             │ dialogue
                             ▼
                ┌────────────────────────┐
                │  Pipeline Orchestrator │
                └──┬──────────┬──────────┘
                   │          │          │
        ┌──────────┘          │          └──────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Mastery    │    │    Error     │    │   Feedback   │
│   Estimator  │───▶│   Analyzer   │───▶│  Generator   │
│              │    │              │    │              │
│ Llama-8B 4bit│    │ Llama-8B 4bit│    │ Llama-8B 4bit│
│ logit T/F    │    │ JSON output  │    │ Pedagogical  │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                  ┌─────────────────────┐
                  │   Memory Layer      │
                  │ Online + Offline    │
                  │ Profile Store       │
                  └─────────────────────┘
```

##  Bắt đầu nhanh

### Yêu cầu

- Python 3.10+
- CUDA 12.x
- HuggingFace account với quyền truy cập Llama-3.1-8B-Instruct

### Cài đặt

```bash
# Clone repo
git clone https://github.com/ddhung04/llm-agent-knowledge-tracing.git
cd llm-agent-knowledge-tracing

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate    # Linux/Mac

# Cài dependencies
pip install -r requirements.txt

# Login HuggingFace để download model
huggingface-cli login
```

### Verify setup

```bash
python test_zeroshot.py
```

Kết quả mong đợi: `TẤT CẢ TESTS PASSED`.

##  Cấu trúc dự án

```
.
├── data/annotated/              # Dataset (CoMTA + MathDial annotated)
│   ├── comta_atc.csv
│   └── mathdial_test_atc.csv
│
├── dialogue_kt/                 # Core code (từ paper LLMKT, sửa lm.py)
│   ├── models/lm.py             #  Modified: 4-bit QLoRA
│   ├── prompting.py             # KT prompts
│   └── ...
│
├── results/                     # Kết quả thực nghiệm
│   ├── *.json                   # Metrics + predictions
│   └── fig_*.png                # Visualizations
│
├── saved_models/                # LoRA adapters sau fine-tune
│
├── # Mastery Estimator (Agent 1)
├── test_zeroshot.py
├── run_zeroshot_eval.py
├── visualize_mastery.py
├── run_mathdial_zeroshot.py
├── run_mini_finetune.py
├── eval_finetune.py
│
├── # Error Analyzer (Agent 2) ★ New
├── error_analyzer.py
├── run_error_analysis.py
├── evaluate_errors.py
│
├── # Feedback Generator (Agent 3) ★ New
├── feedback_generator.py
├── run_feedback_pipeline.py     # End-to-end pipeline
├── evaluate_feedback.py
│
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md            # Báo cáo chi tiết
```

## 🔧 Sử dụng

### 1. Mastery Estimator — Zero-shot trên CoMTA

```bash
python run_zeroshot_eval.py
python visualize_mastery.py
```

Output: `results/comta_zeroshot_fold1_metrics.json` + 3 hình PNG.

### 2. Mastery Estimator — Zero-shot trên MathDial

```bash
python run_mathdial_zeroshot.py
```

### 3. Mastery Estimator — Fine-tune

```bash
python mini_finetune.py    
python eval_finetune.py
```

### 4. Error Analyzer

```bash
python run_error_analysis.py      # Tạo predictions
# → Mở results/error_annotation_sheet.csv, điền ground truth
python evaluate_errors.py         # Tính metrics
```

### 5. End-to-end Pipeline (3 agents)

```bash
python run_feedback_pipeline.py   # Chạy cả 3 agents trên 30 cases
python evaluate_feedback.py
```

Output: `results/pipeline_demo.json` — 3 case studies cho appendix.

##  Các Agent

### Agent 1: Mastery Estimator

Dự đoán xác suất nắm vững Knowledge Component (KC) tại mỗi turn.

**Input:** dialogue history + KC text
**Output:** `P(mastery) ∈ [0, 1]`

**Kỹ thuật:**
- Đọc logit của 2 tokens `True`/`False` → softmax
- Không thêm classification head
- Multi-KC aggregation: mean-arithmetic (Compensatory model)

### Agent 2: Error Analyzer

Phân loại lỗi học sinh thành 5 categories: `conceptual` / `procedural` / `calculation` / `careless` / `other`.

**Input:** dialogue + incorrect answer + KCs
**Output:** JSON với `error_type`, `explanation`, `severity`, `suggestion`

### Agent 3: Feedback Generator

Sinh feedback cá nhân hóa dựa trên mastery + error analysis.

**Input:** dialogue + mastery vector + error analysis
**Output:** JSON với `feedback_text`, `scaffolding_question`, `pedagogical_strategy`

**Pedagogical principles:**
- Scaffolding (hướng dẫn từng bước)
- Socratic Method (câu hỏi gợi mở)
- Mastery-based adaptation (low/medium/high)

##  Datasets

| Dataset | Dialogues | Avg turns | Multi-KC % | Domain |
|---|---|---|---|---|
| CoMTA | 153 | 6-10 | 58.75% | 4 môn toán |
| MathDial | ~1500 | 3-5 | 82.89% | Word problems |

Cả 2 dataset đã được GPT-4o annotate trước với Common Core standards (3 tầng: Domain → Cluster → Standard) và correctness labels. **Không cần re-annotate** — dùng file CSV có sẵn.

##  Đóng góp nghiên cứu

Mở rộng từ [LLMKT (Scarlatos et al., LAK 2025)](https://arxiv.org/abs/2503.11733) với:

| Khía cạnh | Paper gốc | Đề tài này |
|---|---|---|
| Số agents | 1 | **3 + rule-based planner** |
| Error Analyzer | ❌ | ✅ taxonomy 5 lớp |
| Feedback Generator | ❌ | ✅ pedagogical principles |
| Pipeline orchestration | ❌ | ✅ end-to-end |

##  Tech Stack

- **LLM:** [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- **Framework:** PyTorch 2.6, Transformers 4.45
- **Quantization:** BitsAndBytes (4-bit NF4)
- **Fine-tuning:** PEFT (LoRA)
- **Data:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib

## Acknowledgments

- [Scarlatos et al. (2025)](https://arxiv.org/abs/2503.11733) — Paper gốc LLMKT
- [umass-ml4ed/dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt) — Codebase nền tảng
- [Chu et al. (2025)](https://arxiv.org/abs/2503.11733) — Survey LLM Agents in Education
- Đặng Văn Khải (2025) — Khóa luận tham khảo về Generative AI trong giáo dục

## License

MIT License — xem [LICENSE](LICENSE)

## Contact

Khóa luận tốt nghiệp 2026 — Trường Đại học Công nghệ, Khoa Trí tuệ Nhân tạo

---

**Note:** Đây là dự án nghiên cứu khóa luận, KHÔNG phải production system. Đánh giá dùng dataset học thuật (CoMTA, MathDial), không trên học sinh thật.