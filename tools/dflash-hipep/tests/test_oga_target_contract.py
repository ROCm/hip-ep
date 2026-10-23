"""Unit tests for the OGA adapter operations that replace PyTorch target calls.

These tests use scripted OGA generators and ORT sessions. They verify the
adapter's API calls, tensor ordering, and state transitions without loading the
35B model. ``validate_oga_target_integration.py`` provides the corresponding
real-model HIPEP integration coverage.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

HIPEP_STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIPEP_STUDY_ROOT))

from oga_target_native import (
    DFLASH_LOGITS,
    OgaDFlashTarget,
)

TEST_HIDDEN_OUTPUT_NAMES = tuple(
    f"/model/layers.{layer_id + 1}/input_layernorm/output_3"
    for layer_id in (1, 6, 11, 16, 22, 27, 32, 37)
)


def _adapter() -> OgaDFlashTarget:
    """Construct an adapter without loading OGA, ONNX Runtime, or model files."""
    target = object.__new__(OgaDFlashTarget)
    target.hidden_size = 2048
    target.vocab_size = 248320
    target.hidden_output_names = TEST_HIDDEN_OUTPUT_NAMES
    target.final_hidden_output = "/model/layers.40/final_norm_layernorm/output_0"
    target.generator = None
    target.max_length = 0
    target.processed_length = 0
    target.partial_rollback_count = 0
    target._expected_next_token = None
    target.rollback_correction_mismatch_count = 0
    return target


class _RecordingSession:
    def __init__(self, result: np.ndarray) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def run(
        self,
        output_names: list[str],
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.calls.append((output_names, feeds))
        return [self.result]


class _PrefillGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.logits = np.asarray([[0.0, 1.0, 4.0, 2.0]], dtype=np.float32)
        self.pending = 2
        self.hidden = {
            name: np.full((1, 3, 2048), index, dtype=np.float16)
            for index, name in enumerate(TEST_HIDDEN_OUTPUT_NAMES)
        }

    def append_tokens(self, tokens: np.ndarray) -> None:
        self.calls.append(("append_tokens", tokens.copy()))

    def get_logits(self) -> np.ndarray:
        self.calls.append(("get_logits", None))
        return self.logits

    def generate_next_token(self) -> None:
        self.calls.append(("generate_next_token", None))

    def get_next_tokens(self) -> np.ndarray:
        self.calls.append(("get_next_tokens", None))
        return np.asarray([self.pending], dtype=np.int32)

    def get_output(self, name: str) -> np.ndarray:
        self.calls.append(("get_output", name))
        return self.hidden[name]


class _SnapshotRewindGenerator:
    """Script native snapshot, contamination, rewind, and accepted-row append."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.phase = "prefilled"
        self.sequence_length = 0

    def snapshot_state(self) -> None:
        self.calls.append(("snapshot_state", None))

    def rewind_to(self, length: int) -> None:
        self.calls.append(("rewind_to", length))
        self.phase = "rewound"

    def append_tokens(self, tokens: np.ndarray) -> None:
        self.calls.append(("append_tokens", tokens.copy()))
        self.sequence_length = int(tokens.size)
        self.phase = "accepted-append" if self.phase == "rewound" else "verification"

    def get_output(self, name: str) -> np.ndarray:
        self.calls.append(("get_output", name))
        if name == DFLASH_LOGITS:
            logits = np.zeros(
                (1, self.sequence_length, 248320),
                dtype=np.float16,
            )
            predictions = [11, 12, 13][: self.sequence_length]
            for row, prediction in enumerate(predictions):
                logits[0, row, prediction] = 1
            return logits

        tap_index = TEST_HIDDEN_OUTPUT_NAMES.index(name)
        phase_offset = 30 if self.phase == "accepted-append" else 10
        return np.full(
            (1, self.sequence_length, 2048),
            tap_index + phase_offset,
            dtype=np.float16,
        )


class OgaTargetContractTests(unittest.TestCase):
    def test_prefill_matches_pytorch_target_forward_contract(self) -> None:
        """Prefill returns next-token logits and selected target features."""
        target = _adapter()
        generator = _PrefillGenerator()
        target._new_generator = lambda: generator  # type: ignore[method-assign]
        input_ids = np.asarray([[4, 5, 6]], dtype=np.int64)

        output = target.prefill(input_ids, max_new_tokens=7, block_size=4)

        np.testing.assert_array_equal(output.logits, generator.logits[:, None, :])
        self.assertEqual(output.target_hidden.shape, (1, 3, 8 * 2048))
        for index in range(8):
            start = index * 2048
            np.testing.assert_array_equal(
                output.target_hidden[:, :, start : start + 2048],
                generator.hidden[TEST_HIDDEN_OUTPUT_NAMES[index]],
            )
        np.testing.assert_array_equal(
            generator.calls[0][1],
            np.asarray([4, 5, 6], dtype=np.int32),
        )
        self.assertEqual(
            [call[0] for call in generator.calls[:2]],
            ["append_tokens", "get_logits"],
        )
        self.assertEqual(target.max_length, 3 + 7 + 4 + 2)
        self.assertEqual(target.processed_length, 3)
        self.assertEqual(target.partial_rollback_count, 0)

    def test_embed_matches_target_embed_tokens_contract(self) -> None:
        """Embedding uses IDs plus the required empty image-feature input."""
        target = _adapter()
        expected = np.ones((1, 2, 2048), dtype=np.float16)
        target.embedding_session = _RecordingSession(expected)

        actual = target.embed(np.asarray([[5, 9]], dtype=np.int32))

        np.testing.assert_array_equal(actual, expected)
        output_names, feeds = target.embedding_session.calls[0]
        self.assertEqual(output_names, ["inputs_embeds"])
        self.assertEqual(feeds["input_ids"].dtype, np.int64)
        self.assertEqual(feeds["image_features"].shape, (0, 2048))
        self.assertEqual(feeds["image_features"].dtype, np.float16)

    def test_lm_head_matches_target_lm_head_contract(self) -> None:
        """LM-head forwards draft hidden states to the standalone ONNX head."""
        target = _adapter()
        expected = np.ones((1, 2, 4), dtype=np.float16)
        target.head_session = _RecordingSession(expected)
        hidden = np.ones((1, 2, 2048), dtype=np.float32)

        actual = target.lm_head(hidden)

        np.testing.assert_array_equal(actual, expected)
        output_names, feeds = target.head_session.calls[0]
        self.assertEqual(output_names, ["logits"])
        self.assertEqual(feeds["hidden_states"].dtype, np.float16)
        self.assertEqual(feeds["hidden_states"].shape, hidden.shape)

    def test_snapshot_rewind_restores_and_appends_only_accepted_rows(self) -> None:
        """Patched OGA snapshots before verify and rewinds a partial rejection."""
        target = _adapter()
        generator = _SnapshotRewindGenerator()
        target.generator = generator
        target.processed_length = 20
        target._expected_next_token = 10

        output = target.verify(np.asarray([[10, 21, 22]], dtype=np.int64))

        self.assertEqual(
            [call[0] for call in generator.calls[:2]],
            ["snapshot_state", "append_tokens"],
        )
        np.testing.assert_array_equal(
            generator.calls[1][1],
            np.asarray([10, 21, 22], dtype=np.int32),
        )
        self.assertEqual(np.argmax(output.logits[0], axis=-1).tolist(), [11, 12, 13])

        committed = target.commit_verified_prefix(
            output,
            committed_count=2,
            expected_bonus=12,
            committed_prefix_ids=np.asarray(
                [[*range(20), 10, 21]],
                dtype=np.int64,
            ),
        )

        self.assertEqual(committed, 12)
        self.assertEqual(target.processed_length, 22)
        self.assertEqual(target.partial_rollback_count, 1)
        self.assertIn(("rewind_to", 20), generator.calls)
        restored_append = [
            call for call in generator.calls if call[0] == "append_tokens"
        ][-1]
        np.testing.assert_array_equal(
            restored_append[1],
            np.asarray([10, 21], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            output.target_hidden[:, :2, :2048],
            np.full((1, 2, 2048), 30, dtype=np.float16),
        )


if __name__ == "__main__":
    unittest.main()
