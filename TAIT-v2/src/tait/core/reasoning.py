"""Lightweight reasoning/CoT serialization helpers for TAIT datasets."""

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"


def format_example(prompt, response, reasoning="", enable_reasoning=True):
    """Serialize one instruction example for causal next-token training."""
    prompt = str(prompt).strip()
    response = str(response).strip()
    reasoning = str(reasoning or "").strip()
    if enable_reasoning and reasoning:
        target = f"{THINK_START}{reasoning}{THINK_END}{ANSWER_START}{response}{ANSWER_END}"
    elif enable_reasoning:
        target = f"{ANSWER_START}{response}{ANSWER_END}"
    else:
        target = response
    return prompt, target


def extract_generation(text):
    """Return reasoning and answer sections from generated text."""
    reasoning = ""
    answer = text
    if THINK_START in text:
        part = text.split(THINK_START, 1)[1]
        if THINK_END in part:
            reasoning = part.split(THINK_END, 1)[0].strip()
            answer = part.split(THINK_END, 1)[1]
        else:
            answer = part
    if ANSWER_START in answer:
        answer = answer.split(ANSWER_START, 1)[1]
    if ANSWER_END in answer:
        answer = answer.split(ANSWER_END, 1)[0]
    return reasoning.strip(), answer.strip()
