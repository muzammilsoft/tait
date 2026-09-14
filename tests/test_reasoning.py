import numpy as np
from tait.core.reasoning import format_example, extract_generation
from tait.cli.commands import _batch_from_indices


def test_reasoning_format_and_extract():
    prompt, target = format_example("Q", "A", "R", True)
    assert prompt == "Q"
    assert target == "<think>R</think><answer>A</answer>"
    reasoning, answer = extract_generation(target)
    assert reasoning == "R"
    assert answer == "A"


def test_response_only_mask_starts_after_sep():
    sep, pad = 9, 99
    seqs = [[1, 2, sep, 3, 4, 5]]
    x, y, valid, target = _batch_from_indices(seqs, [0], pad, 8, sep_id=sep, response_only=True)
    assert valid.all()
    assert target.tolist() == [[False, False, True, True, True]]
    assert np.array_equal(y[0], np.array([2, sep, 3, 4, 5]))
