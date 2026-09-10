"""Text generation helpers."""
from .tokenizer import SEP, END


def generate_reply(model, tokenizer, prompt, **kwargs):
    full = model.generate(tokenizer, prompt, **kwargs)
    if SEP in full:
        full = full.split(SEP, 1)[1]
    if END in full:
        full = full.split(END, 1)[0]
    return full.strip()
