# Models

The current TAIT model is a small from-scratch `MiniGPT` implemented with NumPy. It is designed for experimentation on Termux and constrained Linux devices rather than matching large commercial LLMs.

## Reasoning model format

TAIT 2.1 can train examples in this form:

```text
prompt <SEP> <think>compact reasoning</think><answer>final answer</answer> <END>
```

At chat time, TAIT extracts the answer section by default. Use `--show-reasoning` to display the generated reasoning trace.

## Compatibility

The `.npz` format continues to store model weights, BPE merge data, special token IDs, and block size. Loading remains compatible with existing TAIT v2 models because the model file format itself is unchanged.
