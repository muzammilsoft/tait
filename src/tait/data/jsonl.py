"""JSONL dataset source."""
import json

ALIASES_PROMPT = ("prompt", "question", "input")
ALIASES_RESPONSE = ("response", "answer", "output")


def load_jsonl(path):
    rows = []
    invalid = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            prompt = next((str(obj.get(k) or "") for k in ALIASES_PROMPT if obj.get(k)), "")
            response = next((str(obj.get(k) or "") for k in ALIASES_RESPONSE if obj.get(k)), "")
            if prompt and response:
                rows.append({"prompt": prompt, "response": response})
            else:
                invalid += 1
    return rows, invalid
