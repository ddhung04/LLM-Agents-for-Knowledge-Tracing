"""
Test script — verify zero-shot Mastery Estimator.
Chạy từ D:\thesis\dialogue-kt:
    python test_zeroshot.py
"""
import sys, torch, pandas as pd
from ast import literal_eval
sys.path.insert(0, '.')

def test_model_loading():
    print("\n" + "="*60)
    print("TEST 1: Load Llama-3.1-8B trong 4-bit")
    print("="*60)
    from dialogue_kt.models.lm import get_model
    model, tokenizer = get_model(
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        test=True,
        model_name=None,
        quantize=True,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM used : {vram:.2f} GB")
    print(f"VRAM peak : {vram_peak:.2f} GB")
    print("TEST 1 PASSED" if vram_peak < 11.5 else "WARNING: VRAM cao, theo dõi khi inference")
    return model, tokenizer

def test_tokens(tokenizer):
    print("\n" + "="*60)
    print("TEST 2: True/False token IDs")
    print("="*60)
    from dialogue_kt.prompting import get_true_false_tokens
    true_id, false_id = get_true_false_tokens(tokenizer)
    print(f"True  token: ID={true_id}, text='{tokenizer.decode([true_id])}'")
    print(f"False token: ID={false_id}, text='{tokenizer.decode([false_id])}'")
    assert tokenizer.decode([true_id]).strip() == "True", "True token sai!"
    assert tokenizer.decode([false_id]).strip() == "False", "False token sai!"
    print("TEST 2 PASSED")
    return true_id, false_id

def test_inference(model, tokenizer, true_id, false_id):
    print("\n" + "="*60)
    print("TEST 3: Inference trên 1 dialogue từ CoMTA")
    print("="*60)
    from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt
    from dialogue_kt.kt_data_loading import apply_annotations

    # Load data
    df = pd.read_csv(
        "data/annotated/comta_atc.csv",
        converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}
    )
    print(f"Loaded {len(df)} dialogues")

    # Lấy dialogue đầu tiên có annotation hợp lệ
    sample = None
    for _, row in df.iterrows():
        if "error" not in row["annotation"]:
            sample = row
            break

    dialogue_anno = apply_annotations(sample)
    print(f"Dialogue: {len(dialogue_anno)} turns")

    # Lấy turn đầu tiên có KC
    target = next((t for t in dialogue_anno if t.get("kcs")), None)
    if not target:
        print("Không tìm thấy turn có KC")
        return

    kc = target["kcs"][0]
    turn_idx = target["turn"]
    print(f"Turn {turn_idx} | KC: {kc[:60]}...")
    print(f"Ground truth: {target['correct']}")

    # Build prompt
    class Args:
        dataset = "comta"
        prompt_inc_labels = False

    system = kt_system_prompt(Args())
    user = kt_user_prompt(sample, dialogue_anno, turn_idx, kc, Args())

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    print(f"Input tokens: {inputs.shape[1]}")

    with torch.no_grad():
        outputs = model(inputs)

    logits = outputs.logits[0, -1, :]
    t_logit = logits[true_id].item()
    f_logit = logits[false_id].item()
    prob = torch.softmax(torch.tensor([t_logit, f_logit]), dim=0)[0].item()

    print(f"\nKết quả:")
    print(f"  P(mastery=True)  = {prob:.4f}")
    print(f"  Predicted        = {prob > 0.5}")
    print(f"  Ground truth     = {target['correct']}")
    correct = (prob > 0.5) == target['correct']
    print(f"  Match            = {correct}")
    print("TEST 3 PASSED")
    return prob

def main():
    print("="*60)
    print("ZERO-SHOT MASTERY ESTIMATOR — VERIFICATION")
    print("="*60)
    try:
        model, tokenizer = test_model_loading()
        true_id, false_id = test_tokens(tokenizer)
        test_inference(model, tokenizer, true_id, false_id)
        print("\n" + "="*60)
        print("TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()