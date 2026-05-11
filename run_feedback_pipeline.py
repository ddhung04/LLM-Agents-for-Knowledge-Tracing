"""
Run full pipeline: Mastery → Error → Feedback on CoMTA fold 1.
This is the END-TO-END DEMO of the multi-agent system.

Usage:
    python run_feedback_pipeline.py

Output:
    results/feedback_results.json       — full results
    results/feedback_eval_sheet.csv     — CSV for self-evaluation
    results/pipeline_demo.json          — 3 best cases for thesis appendix
"""
import sys, json, time, torch
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm

sys.path.insert(0, '.')

from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, get_true_false_tokens
from dialogue_kt.kt_data_loading import apply_annotations
from error_analyzer import run_error_analysis
from feedback_generator import run_feedback_generation


FOLD = 1
RESULTS_DIR = "results"
MAX_CASES = 30  # Limit to 30 for time


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


@torch.no_grad()
def predict_mastery_for_turn(model, tokenizer, sample, dialogue_anno, turn_idx, kcs, true_id, false_id):
    """Get mastery vector for a specific turn."""
    class Args:
        dataset = "comta"
        prompt_inc_labels = False

    mastery = {}
    for kc in kcs:
        system = kt_system_prompt(Args())
        user = kt_user_prompt(sample, dialogue_anno, turn_idx, kc, Args())
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model(inputs)
        logits = outputs.logits[0, -1, :]
        t = logits[true_id].item()
        f = logits[false_id].item()
        prob = torch.softmax(torch.tensor([t, f]), dim=0)[0].item()
        mastery[kc] = round(prob, 4)
    return mastery


def collect_cases(test_df):
    """Collect cases with incorrect turns for full pipeline."""
    cases = []
    for idx, sample in test_df.iterrows():
        if "error" in sample["annotation"]:
            continue
        dialogue_anno = apply_annotations(sample)
        if not dialogue_anno:
            continue
        for turn in dialogue_anno:
            if turn["turn"] == 0:
                continue
            if turn["correct"] is False and turn.get("kcs"):
                cases.append({
                    "dialogue_idx": int(idx),
                    "turn_idx": turn["turn"],
                    "turn_data": turn,
                    "sample": sample,
                    "dialogue_anno": dialogue_anno,
                })
                if len(cases) >= MAX_CASES:
                    return cases
    return cases


def main():
    print("=" * 70)
    print("FULL PIPELINE: Mastery → Error → Feedback")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = get_model(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        test=True, model_name=None, quantize=True
    )
    true_id, false_id = get_true_false_tokens(tokenizer)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 2. Load data
    print(f"\n[2/5] Loading CoMTA fold {FOLD}...")
    test_df = load_test_fold(FOLD)
    cases = collect_cases(test_df)
    print(f"  Cases to process: {len(cases)}")

    # 3. Run full pipeline
    print(f"\n[3/5] Running pipeline on {len(cases)} cases...")
    results = []
    start = time.time()

    for case in tqdm(cases, desc="Pipeline"):
        sample = case["sample"]
        dialogue_anno = case["dialogue_anno"]
        turn_idx = case["turn_idx"]
        turn_data = case["turn_data"]

        # Agent 1: Mastery Estimator
        mastery = predict_mastery_for_turn(
            model, tokenizer, sample, dialogue_anno,
            turn_idx, turn_data["kcs"], true_id, false_id
        )

        # Agent 2: Error Analyzer
        error = run_error_analysis(
            model, tokenizer, sample, dialogue_anno,
            turn_idx, turn_data
        )

        # Agent 3: Feedback Generator
        feedback = run_feedback_generation(
            model, tokenizer, sample, dialogue_anno,
            turn_idx, mastery, error
        )

        result = {
            "dialogue_idx": case["dialogue_idx"],
            "turn_idx": turn_idx,
            "teacher": turn_data.get("teacher", ""),
            "student": turn_data.get("student", ""),
            "ground_truth_correct": turn_data["correct"],
            "kcs": turn_data["kcs"],
            "mastery_vector": mastery,
            "avg_mastery": round(float(np.mean(list(mastery.values()))), 4),
            "error_type": error.get("error_type", ""),
            "error_explanation": error.get("explanation", ""),
            "feedback_text": feedback.get("feedback_text", ""),
            "scaffolding_question": feedback.get("scaffolding_question", ""),
            "mastery_adaptation": feedback.get("mastery_adaptation", ""),
            "pedagogical_strategy": feedback.get("pedagogical_strategy", ""),
            "next_step_hint": feedback.get("next_step_hint", ""),
            "feedback_success": feedback.get("success", False),
        }
        results.append(result)

    elapsed = time.time() - start
    print(f"  Done in {elapsed/60:.1f} min")

    # 4. Save results
    print(f"\n[4/5] Saving results...")

    with open(f"{RESULTS_DIR}/feedback_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Create evaluation CSV
    eval_rows = []
    for i, r in enumerate(results):
        eval_rows.append({
            "id": i + 1,
            "dialogue_idx": r["dialogue_idx"],
            "turn": r["turn_idx"],
            "teacher_question": r["teacher"][:150],
            "student_answer": r["student"][:150],
            "avg_mastery": r["avg_mastery"],
            "error_type": r["error_type"],
            "feedback_text": r["feedback_text"][:300],
            "scaffolding_question": r["scaffolding_question"][:200],
            "correctness_score": "",     # ← 1-5: feedback có đúng toán học không
            "relevance_score": "",       # ← 1-5: feedback có đúng vào lỗi không
            "clarity_score": "",         # ← 1-5: feedback có dễ hiểu không
            "notes": "",
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(f"{RESULTS_DIR}/feedback_eval_sheet.csv", index=False, encoding="utf-8-sig")

    # Select 3 best cases for thesis appendix
    successful = [r for r in results if r["feedback_success"]]
    demo_cases = successful[:3] if len(successful) >= 3 else successful
    with open(f"{RESULTS_DIR}/pipeline_demo.json", "w", encoding="utf-8") as f:
        json.dump(demo_cases, f, indent=2, ensure_ascii=False)

    print(f"  Results: {RESULTS_DIR}/feedback_results.json")
    print(f"  Eval CSV: {RESULTS_DIR}/feedback_eval_sheet.csv")
    print(f"  Demo cases: {RESULTS_DIR}/pipeline_demo.json")

    # 5. Summary
    print(f"\n[5/5] Summary...")
    fb_success = sum(1 for r in results if r["feedback_success"])
    strategies = {}
    adaptations = {}
    for r in results:
        s = r.get("pedagogical_strategy", "unknown")
        a = r.get("mastery_adaptation", "unknown")
        strategies[s] = strategies.get(s, 0) + 1
        adaptations[a] = adaptations.get(a, 0) + 1

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total cases: {len(results)}")
    print(f"Feedback generated successfully: {fb_success}/{len(results)}")
    print(f"Time: {elapsed/60:.1f} min ({elapsed/len(results):.1f}s per case)")

    print(f"\nPedagogical strategy distribution:")
    for s, c in sorted(strategies.items(), key=lambda x: -x[1]):
        print(f"  {s:<25} {c:>3} ({c/len(results)*100:.0f}%)")

    print(f"\nMastery adaptation distribution:")
    for a, c in sorted(adaptations.items(), key=lambda x: -x[1]):
        print(f"  {a:<25} {c:>3} ({c/len(results)*100:.0f}%)")

    avg_mastery_all = np.mean([r["avg_mastery"] for r in results])
    print(f"\nAvg mastery across all cases: {avg_mastery_all:.3f}")

    print(f"\n{'='*70}")
    print("SAMPLE OUTPUT (Case 1):")
    print("="*70)
    if results:
        r = results[0]
        print(f"Teacher: {r['teacher'][:100]}")
        print(f"Student: {r['student'][:100]}")
        print(f"Mastery: {r['avg_mastery']}")
        print(f"Error: {r['error_type']} — {r['error_explanation'][:100]}")
        print(f"Feedback: {r['feedback_text'][:200]}")
        print(f"Question: {r['scaffolding_question'][:150]}")

    print(f"\nNEXT STEPS:")
    print(f"  1. Mở: {RESULTS_DIR}/feedback_eval_sheet.csv")
    print(f"  2. Chấm 3 cột: correctness_score, relevance_score, clarity_score (1-5)")
    print(f"  3. Chạy: python evaluate_feedback.py")


if __name__ == "__main__":
    main()