"""
Fine-tune LLMKT trên CoMTA fold 1 với 4-bit QLoRA.

Usage:
    python run_finetune.py

Output:
    saved_models/comta_lmkt_4bit_fold1/   — adapter
    results/finetune_log.json              — training log
    results/finetune_eval.json             — test metrics
"""
import sys
import os
import json
import time
import gc
import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, '.')

from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, get_true_false_tokens
from dialogue_kt.kt_data_loading import apply_annotations
from dialogue_kt.utils import initialize_seeds, get_checkpoint_path

# Config — Tối ưu 12GB
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FOLD = 1
EPOCHS = 3
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
GRAD_ACCUM = 32
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
EVAL_EVERY_N_STEPS = 50
MODEL_NAME = f"comta_lmkt_4bit_fold{FOLD}"

RESULTS_DIR = "results"
LOG_FILE = f"{RESULTS_DIR}/finetune_log.json"
EVAL_FILE = f"{RESULTS_DIR}/finetune_eval.json"


class Args:
    dataset = "comta"
    tag_src = "atc"
    split_by_subject = False
    typical_cutoff = 1
    prompt_inc_labels = False
    base_model = BASE_MODEL
    quantize = True


def load_data(fold):
    df = pd.read_csv(
        "data/annotated/comta_atc.csv",
        converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}
    )
    df = df.sample(frac=1, random_state=221)
    split_point = int(len(df) * ((fold - 1) / 5))
    df = pd.concat([df[split_point:], df[:split_point]])
    train_df = df[:int(len(df) * .65)]
    val_df = df[int(len(df) * .65):int(len(df) * .8)]
    test_df = df[int(len(df) * .8):]
    return train_df, val_df, test_df


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
            if turn["turn"] == 0:
                continue
            if turn["correct"] is None:
                continue
            if not turn.get("kcs"):
                continue

            label = 1.0 if turn["correct"] else 0.0
            for kc in turn["kcs"]:
                user = kt_user_prompt(row, dialogue, turn["turn"], kc, args)
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": user}]
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )[0]
                if len(input_ids) > 2500:
                    continue
                samples.append({
                    "input_ids": input_ids,
                    "label": label,
                })
    return samples


@torch.no_grad()
def evaluate(model, tokenizer, samples, true_id, false_id, max_n=None):
    model.eval()
    probs, labels = [], []
    eval_samples = samples[:max_n] if max_n else samples

    for s in eval_samples:
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
            continue

    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
    preds = [1 if p > 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    return auc, acc, len(probs)


def main():
    initialize_seeds(221)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    print("=" * 70)
    print("FINE-TUNE LLMKT — CoMTA Fold 1")
    print(f"Epochs: {EPOCHS}, LR: {LR}, LoRA r: {LORA_R}, Grad accum: {GRAD_ACCUM}")
    print("=" * 70)

    print("\n[1/5] Loading model in TRAINING mode (4-bit QLoRA)...")
    model, tokenizer = get_model(
        base_model_name=BASE_MODEL,
        test=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        quantize=True,
        use_gradient_checkpointing=True,
    )
    true_id, false_id = get_true_false_tokens(tokenizer)
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    args = Args()

    print("\n[2/5] Preparing data...")
    train_df, val_df, test_df = load_data(FOLD)
    train_samples = build_samples(train_df, tokenizer, args)
    val_samples = build_samples(val_df, tokenizer, args)
    test_samples = build_samples(test_df, tokenizer, args)
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    print("\n[3/5] Setting up optimizer...")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    print("\n[4/5] Training...")
    log = {"train_loss": [], "val_metrics": [], "best_val_auc": 0.0}
    global_step = 0
    accum_loss = 0.0
    start = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        np.random.shuffle(train_samples)
        model.train()
        pbar = tqdm(train_samples, desc=f"Ep {epoch+1}")

        for step, s in enumerate(pbar):
            try:
                inputs = s["input_ids"].unsqueeze(0).to(model.device)
                label = torch.tensor([s["label"]], dtype=torch.float32).to(model.device)

                outputs = model(inputs)
                logits = outputs.logits[0, -1, :]

                tf_logits = torch.stack([logits[true_id], logits[false_id]])
                prob_true = torch.softmax(tf_logits, dim=0)[0]
                eps = 1e-7
                prob_clamped = torch.clamp(prob_true, eps, 1 - eps)
                loss = -(label * torch.log(prob_clamped) + (1 - label) * torch.log(1 - prob_clamped))
                loss = loss.squeeze() / GRAD_ACCUM

                loss.backward()
                accum_loss += loss.item() * GRAD_ACCUM

                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
                    avg_loss = accum_loss / GRAD_ACCUM
                    log["train_loss"].append(avg_loss)
                    accum_loss = 0.0
                    global_step += 1
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

                    if global_step % EVAL_EVERY_N_STEPS == 0:
                        val_auc, val_acc, _ = evaluate(model, tokenizer, val_samples, true_id, false_id)
                        log["val_metrics"].append({"step": global_step, "auc": val_auc, "acc": val_acc})
                        print(f"\n  [Step {global_step}] val_auc={val_auc:.4f}")
                        if val_auc > log["best_val_auc"]:
                            log["best_val_auc"] = val_auc
                            print(f"  -> Saving best (val_auc={val_auc:.4f})")
                            model.save_pretrained(get_checkpoint_path(MODEL_NAME))
                        model.train()

                        with open(LOG_FILE, "w") as f:
                            json.dump(log, f, indent=2)

            except torch.cuda.OutOfMemoryError:
                print(f"\n  OOM, skipping (len={len(s['input_ids'])})")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                accum_loss = 0.0
                continue

        # End of epoch
        print(f"\n  End epoch {epoch+1}, eval val...")
        val_auc, val_acc, _ = evaluate(model, tokenizer, val_samples, true_id, false_id)
        log["val_metrics"].append({"step": global_step, "auc": val_auc, "acc": val_acc, "epoch_end": True})
        print(f"  Epoch {epoch+1} val_auc={val_auc:.4f}")
        if val_auc > log["best_val_auc"]:
            log["best_val_auc"] = val_auc
            model.save_pretrained(get_checkpoint_path(MODEL_NAME))
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    elapsed = time.time() - start
    print(f"\n  Training done in {elapsed/3600:.2f} hours")

    # Final test eval
    print("\n[5/5] Final test evaluation (loading best adapter)...")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    model, tokenizer = get_model(
        base_model_name=BASE_MODEL,
        test=True,
        model_name=MODEL_NAME,
        quantize=True,
    )
    true_id, false_id = get_true_false_tokens(tokenizer)
    test_auc, test_acc, n_test = evaluate(model, tokenizer, test_samples, true_id, false_id)

    final_metrics = {
        "test_auc": test_auc,
        "test_accuracy": test_acc,
        "test_n": n_test,
        "best_val_auc": log["best_val_auc"],
        "epochs_trained": EPOCHS,
        "training_hours": round(elapsed / 3600, 2),
        "config": {
            "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "grad_accum": GRAD_ACCUM, "quantize": "4-bit NF4",
        }
    }
    with open(EVAL_FILE, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("FINE-TUNE RESULTS")
    print("=" * 70)
    print(f"Test AUC      : {test_auc:.4f}  (zero-shot was 0.562)")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Best Val AUC  : {log['best_val_auc']:.4f}")
    print(f"Training hours: {elapsed/3600:.2f}")
    print(f"\nPaper LLMKT (full fine-tune): CoMTA AUC = 0.658")


if __name__ == "__main__":
    main()