"""
MINI Fine-tune LLMKT

Khác biệt so với run_finetune.py gốc:
- Subsample training: chỉ 200 samples thay vì 773
- Max seq length: 1500 thay vì 2500
- Epochs: 2 thay vì 3
- Eval mỗi 25 steps (sớm hơn để bắt được best model)

Thời gian dự kiến: 4-6 giờ.

Usage:
    python run_mini_finetune.py
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

# Config — MINI version
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FOLD = 1
EPOCHS = 2
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
GRAD_ACCUM = 16              # giảm xuống 16 (effective batch=16)
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
EVAL_EVERY_N_STEPS = 25
MAX_TRAIN_SAMPLES = 200      # chỉ dùng 200 samples!
MAX_SEQ_LEN = 1500           # giảm từ 2500
MODEL_NAME = f"comta_lmkt_4bit_mini_fold{FOLD}"

RESULTS_DIR = "results"
LOG_FILE = f"{RESULTS_DIR}/mini_finetune_log.json"
EVAL_FILE = f"{RESULTS_DIR}/mini_finetune_eval.json"


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


def build_samples(data, tokenizer, args, max_seq_len=MAX_SEQ_LEN):
    samples = []
    skipped_long = 0
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
                if len(input_ids) > max_seq_len:
                    skipped_long += 1
                    continue
                samples.append({
                    "input_ids": input_ids,
                    "label": label,
                })
    return samples, skipped_long


@torch.no_grad()
def evaluate(model, tokenizer, samples, true_id, false_id):
    model.eval()
    probs, labels = [], []
    for s in samples:
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
    print("MINI FINE-TUNE LLMKT — CoMTA Fold 1")
    print(f"  Train samples: max {MAX_TRAIN_SAMPLES}, max seq len: {MAX_SEQ_LEN}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, LoRA r: {LORA_R}")
    print(f"  Estimated: ~5-6 hours")
    print("=" * 70)

    print("\n[1/5] Loading model in TRAINING mode...")
    model, tokenizer = get_model(
        base_model_name=BASE_MODEL,
        test=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        quantize=True,
        use_gradient_checkpointing=True,
    )
    true_id, false_id = get_true_false_tokens(tokenizer)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    args = Args()

    print("\n[2/5] Preparing data...")
    train_df, val_df, test_df = load_data(FOLD)
    train_all, n_long_train = build_samples(train_df, tokenizer, args)
    val_samples, _ = build_samples(val_df, tokenizer, args)
    test_samples, _ = build_samples(test_df, tokenizer, args)

    # Sort by length and take medium-length samples for stable training
    train_all.sort(key=lambda s: len(s["input_ids"]))
    if len(train_all) > MAX_TRAIN_SAMPLES:
        # Take samples from middle of distribution (avoid both very short and very long)
        start = (len(train_all) - MAX_TRAIN_SAMPLES) // 2
        train_samples = train_all[start:start + MAX_TRAIN_SAMPLES]
    else:
        train_samples = train_all

    avg_len = np.mean([len(s["input_ids"]) for s in train_samples])
    print(f"  Train: {len(train_samples)} (filtered from {len(train_all)}), avg len: {avg_len:.0f}")
    print(f"  Skipped {n_long_train} samples > {MAX_SEQ_LEN} tokens")
    print(f"  Val: {len(val_samples)}, Test: {len(test_samples)}")

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

    # Initial val eval (zero-shot baseline)
    print("\n  Initial val eval (zero-shot baseline)...")
    val_auc, val_acc, _ = evaluate(model, tokenizer, val_samples, true_id, false_id)
    print(f"  Step 0: val_auc={val_auc:.4f}, val_acc={val_acc:.4f}")
    log["val_metrics"].append({"step": 0, "auc": val_auc, "acc": val_acc, "is_baseline": True})
    log["best_val_auc"] = val_auc

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
                    pbar.set_postfix({"loss": f"{avg_loss:.3f}", "step": global_step})

                    if global_step % EVAL_EVERY_N_STEPS == 0:
                        val_auc, val_acc, _ = evaluate(model, tokenizer, val_samples, true_id, false_id)
                        log["val_metrics"].append({"step": global_step, "auc": val_auc, "acc": val_acc})
                        print(f"\n  [Step {global_step}] val_auc={val_auc:.4f}")
                        if val_auc > log["best_val_auc"]:
                            log["best_val_auc"] = val_auc
                            print(f"  -> Best! Saving (val_auc={val_auc:.4f})")
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

        # End of epoch eval
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

    print("\n[5/5] Test eval (loading best adapter)...")
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
        "training_hours": round(elapsed / 3600, 2),
        "config": {
            "lr": LR, "lora_r": LORA_R, "epochs": EPOCHS,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "quantize": "4-bit NF4",
        }
    }
    with open(EVAL_FILE, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("MINI FINE-TUNE RESULTS")
    print("=" * 70)
    print(f"Test AUC      : {test_auc:.4f}  (zero-shot was 0.562)")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Best Val AUC  : {log['best_val_auc']:.4f}")
    print(f"Training time : {elapsed/3600:.2f} hours")
    print(f"\nPaper LLMKT (full): CoMTA AUC = 0.658")


if __name__ == "__main__":
    main()