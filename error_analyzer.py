"""
Error Analyzer Agent — Phân tích lỗi trong câu trả lời sai của học sinh.

Là Agent thứ 2 trong hệ thống multi-agent KT.
Input: dialogue context + student's incorrect answer + KCs
Output: error_type + explanation + affected_concepts
"""

import torch
import json
import re
from typing import Dict, List, Optional

# Error taxonomy — 5 lớp
ERROR_TYPES = {
    "conceptual": "Student misunderstands a core mathematical concept or definition (e.g., confusing area with perimeter, wrong formula).",
    "procedural": "Student knows the concept but applies wrong steps or wrong order of operations (e.g., adding before multiplying).",
    "calculation": "Student understands the method and follows correct steps but makes a numerical computation error (e.g., 5*7=36).",
    "careless": "Student makes an unintentional mistake like dropping a sign, copying a number wrong, or misreading the problem.",
    "other": "Error does not clearly fit the above categories (e.g., off-topic response, incomplete answer, guessing).",
}

ERROR_ANALYZER_SYSTEM_PROMPT = """You are an experienced math teacher analyzing student errors. Given a dialogue between a teacher and student, you must analyze WHY the student's response at a specific turn is incorrect.

Classify the error into exactly ONE of these categories:

1. **conceptual** — Student misunderstands a core mathematical concept or definition (e.g., confusing area with perimeter, using wrong formula).
2. **procedural** — Student knows the concept but applies wrong steps or wrong order of operations (e.g., solving equation steps in wrong order).
3. **calculation** — Student follows correct method but makes a numerical computation error (e.g., arithmetic mistake like 5×7=36).
4. **careless** — Student makes an unintentional slip: dropping a sign, copying a number wrong, misreading the problem.
5. **other** — Error does not fit the above categories (off-topic, incomplete, guessing).

You MUST respond in the following JSON format and nothing else:
```json
{
    "error_type": "conceptual|procedural|calculation|careless|other",
    "explanation": "Brief explanation of what went wrong (1-2 sentences)",
    "affected_concepts": ["list of KCs affected by this error"],
    "severity": "low|medium|high",
    "suggestion": "Brief suggestion for the student (1 sentence)"
}
```"""

ERROR_ANALYZER_USER_TEMPLATE = """Below is a dialogue between a teacher and student about math. The student's response at Turn {turn_idx} is INCORRECT.

{dialogue_text}

The student's incorrect response at Turn {turn_idx}: "{student_response}"

Knowledge Components being assessed at this turn:
{kc_list}

Analyze the error and respond with JSON only."""


def build_error_prompt(sample: dict, dialogue_anno: List[dict], turn_idx: int, turn_data: dict) -> tuple:
    """
    Build system + user prompt for Error Analyzer.
    Returns (system_prompt, user_prompt).
    """
    from dialogue_kt.prompting import get_dialogue_text

    # Get dialogue text up to and including the target turn
    dialogue_text = get_dialogue_text(dialogue_anno, turn_idx=None)

    # Student response at the incorrect turn
    student_response = turn_data.get("student", "")

    # KC list
    kcs = turn_data.get("kcs", [])
    kc_list = "\n".join([f"- {kc}" for kc in kcs]) if kcs else "- None specified"

    user_prompt = ERROR_ANALYZER_USER_TEMPLATE.format(
        turn_idx=turn_idx,
        dialogue_text=dialogue_text,
        student_response=student_response,
        kc_list=kc_list,
    )

    return ERROR_ANALYZER_SYSTEM_PROMPT, user_prompt


def parse_error_response(response_text: str) -> Optional[Dict]:
    """
    Parse JSON response from LLM. Handles common formatting issues.
    Similar to extract_result() in annotate.py.
    """
    # Try to extract JSON from markdown code blocks
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    match = re.search(r'\{[^{}]*"error_type"[^{}]*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try entire response as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    return None


def run_error_analysis(model, tokenizer, sample, dialogue_anno, turn_idx, turn_data) -> Dict:
    """
    Run Error Analyzer on a single incorrect turn.
    Returns parsed result dict or error dict.
    """
    system, user = build_error_prompt(sample, dialogue_anno, turn_idx, turn_data)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Generate response (not just logits — need full text for error analysis)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=300,
            temperature=0.1,       # low temp for consistency
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Parse
    parsed = parse_error_response(response)

    if parsed and "error_type" in parsed:
        # Validate error_type
        valid_types = list(ERROR_TYPES.keys())
        if parsed["error_type"] not in valid_types:
            parsed["error_type"] = "other"
        return {
            "success": True,
            "error_type": parsed["error_type"],
            "explanation": parsed.get("explanation", ""),
            "affected_concepts": parsed.get("affected_concepts", []),
            "severity": parsed.get("severity", "medium"),
            "suggestion": parsed.get("suggestion", ""),
            "raw_response": response,
        }
    else:
        return {
            "success": False,
            "error_type": "other",
            "explanation": "Failed to parse LLM response",
            "raw_response": response,
        }