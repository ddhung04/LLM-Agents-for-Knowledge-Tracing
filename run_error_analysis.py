"""
Run Error Analyzer on all incorrect turns in CoMTA fold 1.
Also generates annotation CSV for human evaluation.

python run_error_analysis.py

Output:
    results/error_analysis_fold1.json    — full results
    results/error_annotation_sheet.csv   — CSV for human annotation
    results/error_summary.json           — summary statistics
"""
import sys, json, time, torch
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm

sys.path.insert(0, '.')

from dialogue_kt.models.lm import get_model
from dialogue_kt.kt_data_loading import apply_annotations
from error_analyzer import run_error_analysis, ERROR_TYPES


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
FOLD = 1
RESULTS_DIR = "results"
OUTPUT_FILE = f"{RESULTS_DIR}/error_analysis_fold1.json"
ANNOTATION_CSV = f"{RESULTS_DIR}/error_annotation_sheet.csv"
SUMMARY_FILE = f"{RESULTS_DIR}/error_summary.json"


def load_test_fold(fold):
    df = pd.read_csv(
        "data/annotated/comta_atc.csv",
        converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}
    )
    df = df.sample(frac=1, random_state=221)
    split_point = int(len(df) * ((fold - 1) / 5))
    df = pd.concat([df[split_point:], df[:split_point]])
    test_df = df[int(len(df) * .8):]
    return test_df


def collect_incorrect_turns(test_df):
    """Collect all incorrect turns from test set for error analysis."""
    incorrect_turns = []
    for idx, sample in test_df.iterrows():
        if "error" in sample["annotation"]:
            continue
        dialogue_anno = apply_annotations(sample)
        if not dialogue_anno:
            continue
        for turn in dialogue_anno:
            if turn["turn"] == 0:
                continue
            if turn["correct"] is False:  # explicitly False, not None
                incorrect_turns.append({
                    "dialogue_idx": int(idx),
                    "turn_idx": turn["turn"],
                    "teacher": turn.get("teacher", ""),
                    "student": turn.get("student", ""),
                    "kcs": turn.get("kcs", []),
                    "sample": sample,
                    "dialogue_anno": dialogue_anno,
                    "turn_data": turn,
                })
    return incorrect_turns


def main():
    print("=" * 70)
    print("ERROR ANALYZER — CoMTA Fold 1")
    print("=" * 70)

    # 1. Load model (same Llama-8B as Mastery Estimator)
    print("\n[1/4] Loading model...")
    model, tokenizer = get_model(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        test=True, model_name=None, quantize=True
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 2. Load data + collect incorrect turns
    print(f"\n[2/4] Loading CoMTA fold {FOLD}...")
    test_df = load_test_fold(FOLD)
    incorrect = collect_incorrect_turns(test_df)
    print(f"  Test dialogues: {len(test_df)}")
    print(f"  Incorrect turns found: {len(incorrect)}")

    if len(incorrect) == 0:
        print("ERROR: No incorrect turns found!")
        return

    # 3. Run Error Analyzer on each incorrect turn
    print(f"\n[3/4] Running Error Analyzer...")
    results = []
    start = time.time()

    for item in tqdm(incorrect, desc="Analyzing errors"):
        result = run_error_analysis(
            model, tokenizer,
            item["sample"], item["dialogue_anno"],
            item["turn_idx"], item["turn_data"]
        )
        result["dialogue_idx"] = item["dialogue_idx"]
        result["turn_idx"] = item["turn_idx"]
        result["teacher_text"] = item["teacher"]
        result["student_text"] = item["student"]
        result["kcs"] = item["kcs"]
        results.append(result)

    elapsed = time.time() - start
    print(f"  Done in {elapsed/60:.1f} min")

    # 4. Save results + generate annotation CSV
    print(f"\n[4/4] Saving results...")

    # Save full JSON
    results_serializable = []
    for r in results:
        r_copy = {k: v for k, v in r.items()
                  if k not in ("sample", "dialogue_anno", "turn_data")}
        results_serializable.append(r_copy)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"  Results: {OUTPUT_FILE}")

    # Generate annotation CSV for human evaluation
    anno_rows = []
    for i, r in enumerate(results_serializable):
        anno_rows.append({
            "id": i + 1,
            "dialogue_idx": r["dialogue_idx"],
            "turn": r["turn_idx"],
            "teacher_question": r["teacher_text"][:200],
            "student_answer": r["student_text"][:200],
            "kcs": "; ".join(r["kcs"])[:200] if r.get("kcs") else "",
            "llm_error_type": r.get("error_type", ""),
            "llm_explanation": r.get("explanation", "")[:200],
            "human_error_type": "",   # ← BẠN ĐIỀN VÀO ĐÂY
            "agree_with_llm": "",     # ← yes/no/partial
            "notes": "",             # ← ghi chú nếu cần
        })

    anno_df = pd.DataFrame(anno_rows)
    anno_df.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8-sig")
    print(f"  Annotation CSV: {ANNOTATION_CSV}")

    # Summary statistics
    success_count = sum(1 for r in results_serializable if r.get("success"))
    type_dist = {}
    for r in results_serializable:
        t = r.get("error_type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1

    summary = {
        "total_incorrect_turns": len(results_serializable),
        "successful_parses": success_count,
        "failed_parses": len(results_serializable) - success_count,
        "error_type_distribution": type_dist,
        "elapsed_minutes": round(elapsed / 60, 2),
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {SUMMARY_FILE}")

    # Print summary table
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total incorrect turns analyzed: {len(results_serializable)}")
    print(f"Successfully parsed: {success_count}/{len(results_serializable)}")
    print(f"\nError type distribution:")
    for etype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        pct = count / len(results_serializable) * 100
        bar = "█" * int(pct / 2)
        print(f"  {etype:<15} {count:>3} ({pct:5.1f}%) {bar}")

    print(f"\nTime elapsed: {elapsed/60:.1f} min")
    print(f"\nNEXT STEPS:")
    print(f"  1. Mở file: {ANNOTATION_CSV}")
    print(f"  2. Điền cột 'human_error_type' cho mỗi dòng")
    print(f"     (chọn: conceptual / procedural / calculation / careless / other)")
    print(f"  3. Điền cột 'agree_with_llm' (yes / no / partial)")
    print(f"  4. Chạy: python evaluate_errors.py")


if __name__ == "__main__":
    main()