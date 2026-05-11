"""
Run zero-shot LLMKT on MathDial test set.
Match paper LLMKT setup for direct comparison.

Usage:
    python run_mathdial_zeroshot.py
"""
import sys, json, time, torch
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, '.')

from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, get_true_false_tokens
from dialogue_kt.kt_data_loading import apply_annotations


# Config
RESULTS_DIR = "results"
PRED_FILE = f"{RESULTS_DIR}/mathdial_zeroshot.json"
METRICS_FILE = f"{RESULTS_DIR}/mathdial_zeroshot_metrics.json"

# Limit để chạy nhanh (paper dùng full)
# MathDial test có ~700+ dialogues sau filter typical_cutoff=1
MAX_DIALOGUES = 100   # subset cho thời gian. Đủ để có AUC tin cậy.


class Args:
    dataset = "mathdial"
    typical_cutoff = 1
    tag_src = "atc"
    prompt_inc_labels = False


def load_mathdial_test():
    """Load MathDial test split with typical_cutoff filter."""
    test_df = pd.read_csv(
        "data/annotated/mathdial_test_atc.csv",
        converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}
    )

    def pass_filter(row):
        md = row["meta_data"]
        return (md.get("self_typical_confusion", 0) >= 1 and
                md.get("self_typical_interactions", 0) >= 1)

    test_df = test_df[test_df.apply(pass_filter, axis=1)]
    print(f"MathDial test (after typical_cutoff=1 filter): {len(test_df)} dialogues")

    if len(test_df) > MAX_DIALOGUES:
        test_df = test_df.sample(n=MAX_DIALOGUES, random_state=221)
        print(f"Subsampled to {MAX_DIALOGUES} for time")

    return test_df


@torch.no_grad()
def predict(model, tokenizer, sample, dialogue_anno, turn_idx, kc, true_id, false_id, args):
    system = kt_system_prompt(args)
    user = kt_user_prompt(sample, dialogue_anno, turn_idx, kc, args)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model(inputs)
    logits = outputs.logits[0, -1, :]
    t = logits[true_id].item()
    f = logits[false_id].item()
    return torch.softmax(torch.tensor([t, f]), dim=0)[0].item()


def main():
    args = Args()

    print("=" * 70)
    print("MathDial Zero-shot Evaluation")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = get_model(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        test=True, model_name=None, quantize=True
    )
    true_id, false_id = get_true_false_tokens(tokenizer)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load data
    print("\n[2/3] Loading MathDial test set...")
    test_df = load_mathdial_test()

    # Inference
    print("\n[3/3] Running inference...")
    predictions = []
    skipped = 0
    start = time.time()

    for idx, sample in tqdm(test_df.iterrows(), total=len(test_df), desc="Dialogues"):
        if "error" in sample["annotation"]:
            skipped += 1
            continue
        dialogue_anno = apply_annotations(sample)
        if not dialogue_anno:
            skipped += 1
            continue

        for turn in dialogue_anno:
            if turn["turn"] == 0:
                continue
            if turn["correct"] is None:
                continue
            if not turn.get("kcs"):
                continue

            label = 1 if turn["correct"] else 0
            kc_probs = []
            for kc in turn["kcs"]:
                p = predict(model, tokenizer, sample, dialogue_anno,
                            turn["turn"], kc, true_id, false_id, args)
                kc_probs.append(p)

            corr_prob = float(np.mean(kc_probs))
            predictions.append({
                "dialogue_idx": int(idx),
                "turn": turn["turn"],
                "label": label,
                "pred_prob": corr_prob,
                "n_kcs": len(turn["kcs"]),
            })

    elapsed = time.time() - start
    print(f"\n  Done in {elapsed/60:.1f} min, {len(predictions)} predictions")

    # Metrics
    labels = [p["label"] for p in predictions]
    probs = [p["pred_prob"] for p in predictions]
    preds = [1 if p > 0.5 else 0 for p in probs]

    # Final turn metrics
    last_data = {}
    for p in predictions:
        last_data[p["dialogue_idx"]] = p
    final_labels = [v["label"] for v in last_data.values()]
    final_probs = [v["pred_prob"] for v in last_data.values()]
    final_preds = [1 if p > 0.5 else 0 for p in final_probs]

    metrics = {
        "overall": {
            "n": len(labels),
            "auc": float(roc_auc_score(labels, probs)),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "f1_pos": float(f1_score(labels, preds, pos_label=1)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "pos_ratio": float(np.mean(labels)),
        },
        "final_turn": {
            "n": len(final_labels),
            "auc": float(roc_auc_score(final_labels, final_probs)) if len(set(final_labels)) > 1 else None,
            "accuracy": float(accuracy_score(final_labels, final_preds)),
        },
        "config": {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "quantize": "4-bit NF4",
            "approach": "zero-shot",
            "n_dialogues_used": len(test_df),
            "elapsed_minutes": round(elapsed / 60, 2),
        }
    }

    # Save
    with open(PRED_FILE, "w") as f:
        json.dump(predictions, f, indent=2)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("MathDial RESULTS")
    print("=" * 70)
    print(f"Predictions: {PRED_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"\nOverall (n={metrics['overall']['n']}):")
    print(f"  AUC       : {metrics['overall']['auc']:.4f}")
    print(f"  Accuracy  : {metrics['overall']['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['overall']['f1_macro']:.4f}")
    print(f"  Pos ratio : {metrics['overall']['pos_ratio']:.4f}")
    print(f"\nFinal turn (n={metrics['final_turn']['n']}):")
    print(f"  AUC       : {metrics['final_turn']['auc']}")
    print(f"  Accuracy  : {metrics['final_turn']['accuracy']:.4f}")
    print("\nCompare with paper LLMKT (full fine-tune): MathDial AUC ~76.71")


if __name__ == "__main__":
    main()