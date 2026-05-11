"""
Evaluate fine-tuned LLMKT trên test set.
Tách riêng để tránh OOM/đơ trong run_mini_finetune.py.
"""
import sys, json, gc, torch
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, '.')

from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, get_true_false_tokens
from dialogue_kt.kt_data_loading import apply_annotations
from dialogue_kt.utils import initialize_seeds

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = "comta_lmkt_4bit_mini_fold1"
MAX_SEQ_LEN = 1500
FOLD = 1


class Args:
    dataset = "comta"
    tag_src = "atc"
    split_by_subject = False
    typical_cutoff = 1
    prompt_inc_labels = False


def load_test_fold(fold):
    df = pd.read_csv(
        "data/annotated/comta_atc.csv",
        converters={c: literal_eval for c in ["dialogue", "meta_data", "annotation"]}
    )
    df = df.sample(frac=1, random_state=221)
    sp = int(len(df) * ((fold - 1) / 5))
    df = pd.concat([df[sp:], df[:sp]])
    return df[int(len(df) * .8):]


def build_samples(data, tokenizer, args):
    samples = []
    system = kt_system_prompt(args)
    for _, row in data.iterrows():
        if "error" in row["annotation"]:
            continue
        dialogue = apply_annotations(row)
        if not dialogue:
            continue
        for turn in dialogue:
            if turn["turn"] == 0 or turn["correct"] is None or not turn.get("kcs"):
                continue
            label = 1.0 if turn["correct"] else 0.0
            for kc in turn["kcs"]:
                user = kt_user_prompt(row, dialogue, turn["turn"], kc, args)
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
                ids = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt"
                )[0]
                if len(ids) > MAX_SEQ_LEN:
                    continue
                samples.append({"input_ids": ids, "label": label,
                                "turn": turn["turn"], "dialogue_idx": int(_)})
    return samples


@torch.no_grad()
def eval_model(model, tokenizer, samples, true_id, false_id, label):
    model.eval()
    probs, labels = [], []
    skipped = 0
    for s in tqdm(samples, desc=f"Eval {label}"):
        try:
            inputs = s["input_ids"].unsqueeze(0).to(model.device)
            outputs = model(inputs)
            logits = outputs.logits[0, -1, :]
            t = logits[true_id].item()
            f = logits[false_id].item()
            prob = torch.softmax(torch.tensor([t, f]), dim=0)[0].item()
            probs.append(prob)
            labels.append(s["label"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            skipped += 1
            continue

    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
    preds = [1 if p > 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"  {label}: n={len(probs)}, skipped={skipped}, AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    return {"n": len(probs), "auc": float(auc), "accuracy": float(acc),
            "f1_macro": float(f1), "skipped": skipped,
            "probs": probs, "labels": labels}


def main():
    initialize_seeds(221)
    args = Args()

    print("Loading FINE-TUNED model...")
    model_ft, tokenizer = get_model(
        base_model_name=BASE_MODEL,
        test=True,
        model_name=MODEL_NAME,
        quantize=True,
    )
    true_id, false_id = get_true_false_tokens(tokenizer)

    print("\nLoading test set...")
    test_df = load_test_fold(FOLD)
    test_samples = build_samples(test_df, tokenizer, args)
    print(f"  Test samples: {len(test_samples)}")

    print("\nEvaluating fine-tuned model...")
    ft_results = eval_model(model_ft, tokenizer, test_samples, true_id, false_id, "fine-tuned")

    # Cleanup before loading zero-shot baseline
    del model_ft
    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading ZERO-SHOT model for comparison...")
    model_zs, _ = get_model(
        base_model_name=BASE_MODEL,
        test=True,
        model_name=None,
        quantize=True,
    )

    print("\nEvaluating zero-shot model on SAME test set...")
    zs_results = eval_model(model_zs, tokenizer, test_samples, true_id, false_id, "zero-shot")

    # Compare
    results = {
        "fine_tuned": {k: v for k, v in ft_results.items() if k not in ("probs", "labels")},
        "zero_shot": {k: v for k, v in zs_results.items() if k not in ("probs", "labels")},
        "improvement": {
            "auc_delta": ft_results["auc"] - zs_results["auc"],
            "accuracy_delta": ft_results["accuracy"] - zs_results["accuracy"],
        }
    }

    with open("results/mini_finetune_test_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS — Fine-tune vs Zero-shot")
    print("=" * 70)
    print(f"{'Metric':<15} {'Zero-shot':>12} {'Fine-tuned':>12} {'Δ':>10}")
    print("-" * 50)
    for m in ["auc", "accuracy", "f1_macro"]:
        zs = zs_results[m]
        ft = ft_results[m]
        d = ft - zs
        sign = "+" if d > 0 else ""
        print(f"{m:<15} {zs:>12.4f} {ft:>12.4f} {sign}{d:>9.4f}")
    print("\nSaved: results/mini_finetune_test_eval.json")


if __name__ == "__main__":
    main()