"""
Run zero-shot LLMKT evaluation on CoMTA fold 1.
Saves predictions + computes AUC, Accuracy, F1.

    python run_zeroshot_eval.py
"""
import sys, json, time, torch, pandas as pd, numpy as np
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, '.')

from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, get_true_false_tokens
from dialogue_kt.kt_data_loading import apply_annotations


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
class Args:
    dataset = "comta"
    prompt_inc_labels = False
    split_by_subject = False
    typical_cutoff = 1
    tag_src = "atc"


FOLD = 1                        # fold 1 = first 20% as test set
RESULTS_DIR = "results"
PREDICTIONS_FILE = f"{RESULTS_DIR}/comta_zeroshot_fold{FOLD}.json"
METRICS_FILE = f"{RESULTS_DIR}/comta_zeroshot_fold{FOLD}_metrics.json"
KC_RESULTS_FILE = f"{RESULTS_DIR}/comta_zeroshot_fold{FOLD}_kcs.json"


# -------------------------------------------------------------------
# Load data — same split logic as paper LLMKT
# -------------------------------------------------------------------
def load_test_fold(args, fold):
    """Replicate load_annotated_data logic for CoMTA fold."""
    df = pd.read_csv(
        "data/annotated/comta_atc.csv",
        converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}
    )
    df = df.sample(frac=1, random_state=221)
    split_point = int(len(df) * ((fold - 1) / 5))
    df = pd.concat([df[split_point:], df[:split_point]])
    test_df = df[int(len(df) * .8):]
    return test_df


# -------------------------------------------------------------------
# Inference for one (dialogue, turn, KC) triplet
# -------------------------------------------------------------------
@torch.no_grad()
def predict_mastery(model, tokenizer, sample, dialogue_anno, turn_idx, kc, true_id, false_id, args):
    system = kt_system_prompt(args)
    user = kt_user_prompt(sample, dialogue_anno, turn_idx, kc, args)

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model(inputs)
    logits = outputs.logits[0, -1, :]
    t_logit = logits[true_id].item()
    f_logit = logits[false_id].item()

    # Softmax giữa 2 logit
    prob = torch.softmax(torch.tensor([t_logit, f_logit]), dim=0)[0].item()
    return prob


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = Args()

    print("=" * 70)
    print("Zero-shot LLMKT Evaluation — CoMTA Fold 1")
    print("=" * 70)

    # 1. Load model
    print("\n[1/4] Loading Llama-3.1-8B-Instruct (4-bit)...")
    model, tokenizer = get_model(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        test=True, model_name=None, quantize=True
    )
    true_id, false_id = get_true_false_tokens(tokenizer)
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 2. Load test data
    print(f"\n[2/4] Loading CoMTA fold {FOLD} test set...")
    test_df = load_test_fold(args, FOLD)
    print(f"  Test dialogues: {len(test_df)}")

    # 3. Run inference
    print(f"\n[3/4] Running inference...")
    all_predictions = []          # list of (prob, label) for AUC
    per_dialogue_kcs = {}         # dialogue_idx -> list of {kc: prob} per turn
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

        per_turn_kcs = []

        for turn in dialogue_anno:
            # Skip turn 0 (no question being asked)
            if turn["turn"] == 0:
                continue
            # Skip turns with no correctness label or 'na' label
            if turn["correct"] is None:
                continue
            # Skip turns without KCs
            if not turn.get("kcs"):
                continue

            label = 1 if turn["correct"] else 0
            kc_probs = {}

            for kc in turn["kcs"]:
                prob = predict_mastery(
                    model, tokenizer, sample, dialogue_anno,
                    turn["turn"], kc, true_id, false_id, args
                )
                kc_probs[kc] = prob

            # Aggregate (mean-arithmetic) → correctness probability
            corr_prob = float(np.mean(list(kc_probs.values())))
            all_predictions.append({
                "dialogue_idx": int(idx),
                "turn": turn["turn"],
                "label": label,
                "pred_prob": corr_prob,
                "kc_probs": kc_probs,
            })
            per_turn_kcs.append({"turn": turn["turn"], "kcs": kc_probs})

        per_dialogue_kcs[int(idx)] = per_turn_kcs

    elapsed = time.time() - start
    print(f"  Inference done in {elapsed/60:.1f} min")
    print(f"  Predictions: {len(all_predictions)}, skipped dialogues: {skipped}")

    # 4. Compute metrics
    print(f"\n[4/4] Computing metrics...")
    labels = [p["label"] for p in all_predictions]
    probs = [p["pred_prob"] for p in all_predictions]
    preds = [1 if p > 0.5 else 0 for p in probs]

    # Final-turn metrics (last labeled turn of each dialogue)
    last_turn_data = {}
    for p in all_predictions:
        last_turn_data[p["dialogue_idx"]] = p
    final_labels = [v["label"] for v in last_turn_data.values()]
    final_probs = [v["pred_prob"] for v in last_turn_data.values()]
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
            "fold": FOLD,
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "quantize": "4-bit NF4",
            "aggregation": "mean-ar",
            "approach": "zero-shot",
            "elapsed_minutes": round(elapsed / 60, 2),
        }
    }

    # Save
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(all_predictions, f, indent=2)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(KC_RESULTS_FILE, "w") as f:
        json.dump({str(k): v for k, v in per_dialogue_kcs.items()}, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Predictions saved : {PREDICTIONS_FILE}")
    print(f"Metrics saved     : {METRICS_FILE}")
    print(f"KC results saved  : {KC_RESULTS_FILE}")
    print(f"\nOverall metrics (n={metrics['overall']['n']}):")
    print(f"  AUC       : {metrics['overall']['auc']:.4f}")
    print(f"  Accuracy  : {metrics['overall']['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['overall']['f1_macro']:.4f}")
    print(f"  F1 (pos)  : {metrics['overall']['f1_pos']:.4f}")
    print(f"  Pos ratio : {metrics['overall']['pos_ratio']:.4f}")
    print(f"\nFinal turn metrics (n={metrics['final_turn']['n']}):")
    print(f"  AUC       : {metrics['final_turn']['auc']}")
    print(f"  Accuracy  : {metrics['final_turn']['accuracy']:.4f}")
    print("\nCompare with paper LLMKT (full fine-tune): AUC ~65.79 on CoMTA")


if __name__ == "__main__":
    main()