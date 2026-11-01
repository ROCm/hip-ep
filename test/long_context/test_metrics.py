#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Small CPU-only checks; the long model campaign is never auto-discovered."""

import numpy as np
import pytest

from metrics import compare_logits, summarize
from fixture import make_fixture


def test_shift_invariance_and_stability():
    a = np.array([10000.0, 9999.0, -10000.0])
    row = compare_logits(a, a + 700, 1)
    assert row["kl_reference_actual"] == pytest.approx(0, abs=1e-12)
    assert row["js_divergence"] == pytest.approx(0, abs=1e-12)
    assert row["nll_delta"] == pytest.approx(0, abs=1e-12)
    assert row["max_abs"] == 700


def test_wrong_confident_prediction_and_margin():
    row = compare_logits([3, 1, 0], [0, 1, 3], 0)
    assert row["nll_delta"] == pytest.approx(3)
    assert row["reference_top_margin"] == 2
    assert not row["top1_match"]
    assert row["kl_reference_actual"] > 0
    assert 0 < row["js_divergence"] <= np.log(2)
    assert summarize([row])["worst_position"] == 0


@pytest.mark.parametrize("actual", [[0, float("nan")], [0], [0, float("inf")]])
def test_invalid_logits(actual):
    with pytest.raises(ValueError):
        compare_logits([0, 1], actual, 0)


def test_fixture_keeps_questions_and_exact_budget():
    class Tokenizer:
        def encode(self, text):
            return list(text.encode())

        def decode(self, tokens):
            return bytes(tokens).decode()

    fixture = make_fixture(Tokenizer())
    assert len(fixture["turns"]) == 8
    for turn in fixture["turns"]:
        assert len(turn["tokens"]) == 4096
        assert turn["facts"] in turn["text"] and turn["question"] in turn["text"]
        assert turn["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
