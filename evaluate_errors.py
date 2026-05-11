"""
Evaluate Error Analyzer sau khi bạn đã annotate ground truth.

Prerequisites:
    - Đã chạy run_error_analysis.py
    - Đã điền cột 'human_error_type' và 'agree_with_llm' trong error_annotation_sheet.csv

Usage:
    python evaluate_errors.py

Output:
    results/error_eval_metrics.json
    In ra terminal: F1, accuracy, confusion matrix, agreement rate
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

ANNOTATION_CSV = "results/error_annotation_sheet.csv"
OUTPUT_FILE = "results/error_eval_metrics.json"
ERROR_TYPES = ["conceptual", "procedural", "calculation", "careless", "other"]


def main():
    print("=" * 70)
    print("ERROR ANALYZER EVALUATION")
    print("=" * 70)

    # Load annotated CSV
    df = pd.read_csv(ANNOTATION_CSV, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {ANNOTATION_CSV}")

    # Filter rows with human annotation
    annotated = df[df["human_error_type"].notna() & (df["human_error_type"] != "")]
    print(f"Annotated rows: {len(annotated)}/{len(df)}")

    if len(annotated) == 0:
        print("\nERROR: Chưa có annotation! Mở file CSV và điền cột 'human_error_type'.")
        return

    # Normalize
    y_true = annotated["human_error_type"].str.strip().str.lower().tolist()
    y_pred = annotated["llm_error_type"].str.strip().str.lower().tolist()

    # Validate
    invalid_true = [y for y in y_true if y not in ERROR_TYPES]
    invalid_pred = [y for y in y_pred if y not in ERROR_TYPES]
    if invalid_true:
        print(f"WARNING: Invalid human labels: {set(invalid_true)}")
    if invalid_pred:
        print(f"WARNING: Invalid LLM labels: {set(invalid_pred)}")

    # --- Metrics ---
    print("\n" + "-" * 50)
    print("CLASSIFICATION METRICS")
    print("-" * 50)

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy     : {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1 (macro)   : {f1_mac:.4f}")
    print(f"F1 (weighted): {f1_wt:.4f}")

    # Per-class report
    print("\nPer-class classification report:")
    present_labels = sorted(set(y_true + y_pred))
    print(classification_report(y_true, y_pred, labels=present_labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    print("Confusion Matrix:")
    print(f"{'':>15} " + " ".join(f"{l:>12}" for l in present_labels) + "  ← Predicted")
    for i, label in enumerate(present_labels):
        row = " ".join(f"{cm[i][j]:>12}" for j in range(len(present_labels)))
        print(f"{label:>15} {row}")
    print(f"{'↑ Actual':>15}")

    # --- Agreement Rate ---
    print("\n" + "-" * 50)
    print("AGREEMENT ANALYSIS")
    print("-" * 50)

    if "agree_with_llm" in annotated.columns:
        agree_col = annotated["agree_with_llm"].str.strip().str.lower()
        agree_filled = agree_col[agree_col.isin(["yes", "no", "partial"])]
        if len(agree_filled) > 0:
            agree_counts = agree_filled.value_counts()
            total = len(agree_filled)
            for val in ["yes", "partial", "no"]:
                count = agree_counts.get(val, 0)
                pct = count / total * 100
                print(f"  {val:<10}: {count:>3} ({pct:.1f}%)")
            agreement_rate = agree_counts.get("yes", 0) / total
            print(f"\n  Full agreement rate: {agreement_rate:.4f} ({agreement_rate*100:.1f}%)")
            partial_rate = (agree_counts.get("yes", 0) + agree_counts.get("partial", 0)) / total
            print(f"  Partial+ agreement: {partial_rate:.4f} ({partial_rate*100:.1f}%)")
        else:
            print("  Column 'agree_with_llm' empty — skipping")
            agreement_rate = None
            partial_rate = None
    else:
        print("  Column 'agree_with_llm' not found — skipping")
        agreement_rate = None
        partial_rate = None

    # --- Error type distribution comparison ---
    print("\n" + "-" * 50)
    print("DISTRIBUTION COMPARISON")
    print("-" * 50)
    print(f"{'Type':<15} {'Human':>8} {'LLM':>8}")
    print("-" * 35)
    for et in ERROR_TYPES:
        h_count = y_true.count(et)
        l_count = y_pred.count(et)
        print(f"{et:<15} {h_count:>8} {l_count:>8}")

    # Save metrics
    metrics = {
        "n_annotated": len(annotated),
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wt),
        "agreement_rate": float(agreement_rate) if agreement_rate else None,
        "partial_agreement_rate": float(partial_rate) if partial_rate else None,
        "human_distribution": {et: y_true.count(et) for et in ERROR_TYPES},
        "llm_distribution": {et: y_pred.count(et) for et in ERROR_TYPES},
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {OUTPUT_FILE}")

    # --- Summary for thesis ---
    print("\n" + "=" * 70)
    print("TABLE FOR CHAPTER 4 — Error Analysis Results")
    print("=" * 70)
    print(f"Error Analyzer Configuration:")
    print(f"  Model: Llama-3.1-8B-Instruct (4-bit, zero-shot)")
    print(f"  Taxonomy: 5 classes (conceptual/procedural/calculation/careless/other)")
    print(f"  Test set: CoMTA fold 1, n={len(annotated)} incorrect turns")
    print(f"\n  Accuracy     : {acc*100:.1f}%")
    print(f"  F1 (macro)   : {f1_mac*100:.1f}%")
    if agreement_rate:
        print(f"  Agreement    : {agreement_rate*100:.1f}%")


if __name__ == "__main__":
    main()