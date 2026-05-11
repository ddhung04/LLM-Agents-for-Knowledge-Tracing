"""
Feedback Generator Agent — Sinh phản hồi cá nhân hóa cho học sinh.

Agent thứ 3 trong hệ thống multi-agent KT.
Input: dialogue context + mastery vector + error analysis
Output: personalized feedback với scaffolding

"""

import torch
import json
import re
from typing import Dict, List, Optional


FEEDBACK_SYSTEM_PROMPT = """You are an experienced, encouraging math tutor. Given a dialogue between a teacher and student, along with information about the student's current knowledge state and error analysis, generate personalized feedback.

You have access to:
1. The dialogue history
2. The student's mastery level for relevant Knowledge Components (0.0 = no mastery, 1.0 = full mastery)
3. Error analysis of the student's incorrect response (if applicable)

Your feedback MUST follow these pedagogical principles:
- Use SCAFFOLDING: Guide the student step-by-step, don't give the answer directly
- Use SOCRATIC METHOD: Ask probing questions to help the student discover the answer
- ADAPT to mastery level:
  * Low mastery (< 0.4): Use simpler language, provide more scaffolding, break into smaller steps
  * Medium mastery (0.4-0.7): Give hints, ask guiding questions
  * High mastery (> 0.7): Challenge with deeper questions, encourage self-reflection
- Be ENCOURAGING: Acknowledge what the student did right before addressing errors
- Be SPECIFIC: Reference the exact part of the student's response that needs attention

Respond in the following JSON format:
```json
{
    "feedback_text": "Your personalized feedback to the student (2-4 sentences)",
    "scaffolding_question": "A guiding question to help the student think deeper (1 sentence)",
    "mastery_adaptation": "low|medium|high",
    "pedagogical_strategy": "scaffolding|socratic|direct_instruction|encouragement",
    "next_step_hint": "Brief hint about what the student should focus on next (1 sentence)"
}
```"""

FEEDBACK_USER_TEMPLATE = """## Dialogue Context
{dialogue_text}

## Student's Current Knowledge State
{mastery_info}

## Error Analysis (for the most recent incorrect turn)
{error_info}

## Task
Generate personalized feedback for this student based on their knowledge state and error analysis. Respond with JSON only."""


def build_feedback_prompt(
    sample: dict,
    dialogue_anno: List[dict],
    mastery_vector: Dict[str, float],
    error_result: Optional[Dict],
    target_turn_idx: int,
) -> tuple:
    """Build system + user prompt for Feedback Generator."""
    from dialogue_kt.prompting import get_dialogue_text

    # Dialogue text
    dialogue_text = get_dialogue_text(dialogue_anno, turn_idx=None)

    # Mastery info
    if mastery_vector:
        mastery_lines = []
        for kc, prob in sorted(mastery_vector.items(), key=lambda x: x[1]):
            level = "LOW" if prob < 0.4 else "MEDIUM" if prob < 0.7 else "HIGH"
            mastery_lines.append(f"- {kc[:80]}: {prob:.2f} ({level})")
        mastery_info = "\n".join(mastery_lines)
    else:
        mastery_info = "No mastery data available."

    # Error info
    if error_result and error_result.get("success"):
        error_info = (
            f"Error type: {error_result.get('error_type', 'unknown')}\n"
            f"Explanation: {error_result.get('explanation', 'N/A')}\n"
            f"Severity: {error_result.get('severity', 'medium')}\n"
            f"Affected concepts: {', '.join(error_result.get('affected_concepts', []))}"
        )
    else:
        error_info = "No error analysis available (student may have answered correctly)."

    user_prompt = FEEDBACK_USER_TEMPLATE.format(
        dialogue_text=dialogue_text,
        mastery_info=mastery_info,
        error_info=error_info,
    )

    return FEEDBACK_SYSTEM_PROMPT, user_prompt


def parse_feedback_response(response_text: str) -> Optional[Dict]:
    """Parse JSON from LLM response."""
    # Try markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    match = re.search(r'\{[^{}]*"feedback_text"[^{}]*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Entire response
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: treat entire response as feedback_text
    if len(response_text.strip()) > 10:
        return {
            "feedback_text": response_text.strip()[:500],
            "scaffolding_question": "",
            "mastery_adaptation": "medium",
            "pedagogical_strategy": "scaffolding",
            "next_step_hint": "",
        }
    return None


def run_feedback_generation(
    model, tokenizer,
    sample, dialogue_anno,
    target_turn_idx: int,
    mastery_vector: Dict[str, float],
    error_result: Optional[Dict],
) -> Dict:
    """Run Feedback Generator on a single turn."""
    system, user = build_feedback_prompt(
        sample, dialogue_anno, mastery_vector, error_result, target_turn_idx
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed = parse_feedback_response(response)

    if parsed and "feedback_text" in parsed:
        return {
            "success": True,
            **parsed,
            "raw_response": response,
        }
    else:
        return {
            "success": False,
            "feedback_text": response[:500] if response else "",
            "raw_response": response,
        }