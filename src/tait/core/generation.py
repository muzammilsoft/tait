"""Text generation helpers."""
from .tokenizer import SEP, END
from .reasoning import extract_generation


def generate_reply(model, tokenizer, prompt, show_reasoning=False, **kwargs):
    full = model.generate(tokenizer, prompt, **kwargs)
    if SEP in full:
        full = full.split(SEP, 1)[1]
    if END in full:
        full = full.split(END, 1)[0]
    reasoning, answer = extract_generation(full)
    if show_reasoning and reasoning:
        return f"Thinking: {reasoning}\n\nAnswer: {answer}".strip()
    return answer.strip()
