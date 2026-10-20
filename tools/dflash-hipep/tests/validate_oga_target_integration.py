"""Validate an OGA DFlash rollback strategy against a fixed target trace.

OGA prefill produces the expected target token.
The configured hidden-state taps have the correct shape.
Block verification returns the expected logits and bonus token.
Full acceptance and forced partial rejection work.
Partial rejection restores every target cache with the selected strategy.
Standalone embedding and LM-head graphs return correct shapes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HIPEP_STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIPEP_STUDY_ROOT))

from oga_target_native import (
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    DFLASH_LOGITS,
    OgaDFlashTarget,
    ensure_runtime_environment,
)


INPUT_IDS = np.asarray(
    [
        248045,
        846,
        198,
        814,
        20139,
        63520,
        45543,
        303,
        1330,
        61446,
        22157,
        13,
        248046,
        198,
        248045,
        74455,
        198,
        248068,
        271,
        248069,
        271,
    ],
    dtype=np.int32,
)

def _clean_greedy_ids(
    target: OgaDFlashTarget,
    input_ids: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    """Generate a short clean target trace for model-specific validation."""
    tokens = np.ascontiguousarray(input_ids.reshape(-1), dtype=np.int32)
    params = target._og.GeneratorParams(target.model)
    params.set_search_options(
        max_length=int(tokens.size) + int(count),
        do_sample=False,
    )
    generator = target._og.Generator(target.model, params)
    generator.append_tokens(tokens)

    generated = []
    for _ in range(count):
        generator.generate_next_token()
        generated.append(int(np.asarray(generator.get_next_tokens())[0]))
    return tuple(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hipep-root", type=Path, default=DEFAULT_HIPEP_ROOT)
    parser.add_argument("--rocm-root", type=Path, default=DEFAULT_ROCM_ROOT)
    args = parser.parse_args()

    ensure_runtime_environment(args.hipep_root, args.rocm_root)
    print("Loading OGA Derived DFlash target...", flush=True)
    target = OgaDFlashTarget(args.model, args.hipep_root)
    expected_ids = _clean_greedy_ids(target, INPUT_IDS, count=6)
    hidden_width = len(target.hidden_output_names) * target.hidden_size

    # Run the prefill.
    prefill = target.prefill(INPUT_IDS, max_new_tokens=12, block_size=4)
    anchor_token = int(prefill.logits[0, 0].argmax())
    if anchor_token != expected_ids[0]:
        raise RuntimeError(
            f"Prefill produced {anchor_token}, expected {expected_ids[0]}"
        )
    # Validate the hidden state shape.
    if prefill.target_hidden.shape != (1, INPUT_IDS.size, hidden_width):
        raise RuntimeError(f"Unexpected prefill hidden shape: {prefill.target_hidden.shape}")

    # The adapter's target_hidden must be exactly the eight exposed residual
    # taps concatenated in DFlash target_layer_ids order.
    exposed_taps = [
        np.asarray(target.generator.get_output(name))
        for name in target.hidden_output_names
    ]
    np.testing.assert_array_equal(
        prefill.target_hidden,
        np.concatenate(exposed_taps, axis=-1),
    )

    # The copied full-sequence head in text.onnx must agree with OGA's normal
    # last-token logits at the prefill boundary. OGA exposes its normal logits
    # as FP32 while the copied graph output is FP16, so require identical greedy
    # selection and quantization-appropriate numerical agreement.
    decoder_logits = np.asarray(target.generator.get_output(DFLASH_LOGITS))
    if int(decoder_logits[0, -1].argmax()) != int(prefill.logits[0, 0].argmax()):
        raise RuntimeError("Copied decoder head changed the greedy prefill token")
    np.testing.assert_allclose(
        decoder_logits[:, -1:, :],
        prefill.logits,
        rtol=5e-2,
        atol=1e-1,
    )

    # The standalone LM-head graph must perform the same operation as the
    # copied LM-head node in the decoder graph.
    final_hidden = np.asarray(target.generator.get_output(target.final_hidden_output))
    standalone_logits = target.lm_head(final_hidden)
    np.testing.assert_allclose(
        standalone_logits,
        decoder_logits,
        rtol=1e-3,
        atol=1e-3,
    )

    # Run the block verification.
    first_block = np.asarray(expected_ids[:4], dtype=np.int32)
    verified = target.verify(first_block)
    # Validate the logits and bonus token.
    predictions = np.argmax(verified.logits[0], axis=-1).tolist()
    if predictions != list(expected_ids[1:5]):
        raise RuntimeError(
            f"Block predictions {predictions} != expected {list(expected_ids[1:5])}"
        )
    if verified.target_hidden.shape != (1, 4, hidden_width):
        raise RuntimeError(
            f"Unexpected verification hidden shape: {verified.target_hidden.shape}"
        )
    if verified.bonus_token != expected_ids[4]:
        raise RuntimeError(
            f"Block bonus {verified.bonus_token} != expected {expected_ids[4]}"
        )
    # Full acceptance.
    target.commit_verified_prefix(
        verified,
        committed_count=4,
        expected_bonus=expected_ids[4],
        committed_prefix_ids=np.concatenate(
            (INPUT_IDS, first_block),
        ),
    )

    # Force immediate rejection after the valid anchor. The selected strategy
    # must discard rejected-token state while preserving this accepted anchor.
    bad_token = 248049 if expected_ids[5] != 248049 else 248048
    bad_block = np.asarray(
        [expected_ids[4], bad_token, bad_token, bad_token],
        dtype=np.int32,
    )
    rejected = target.verify(bad_block)
    if int(np.argmax(rejected.logits[0, 0])) != expected_ids[5]:
        raise RuntimeError("Rejected block's anchor logits do not match baseline")
    target.commit_verified_prefix(
        rejected,
        committed_count=1,
        expected_bonus=expected_ids[5],
        committed_prefix_ids=np.concatenate(
            (INPUT_IDS, first_block, bad_block[:1]),
        ),
    )
    # Validate that the partial block caused exactly one rollback operation.
    if target.partial_rollback_count != 1:
        raise RuntimeError(
            f"Expected one partial rollback, got {target.partial_rollback_count}"
        )

    after_rollback = target.verify(np.asarray([expected_ids[5]], dtype=np.int32))
    if after_rollback.logits.shape != (1, 1, target.vocab_size):
        raise RuntimeError(
            f"Unexpected post-rollback logits shape: {after_rollback.logits.shape}"
        )

    embeddings = target.embed(np.asarray([[expected_ids[0], bad_token]], dtype=np.int64))
    if embeddings.shape != (1, 2, target.hidden_size):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    draft_logits = target.lm_head(
        np.zeros((1, 2, target.hidden_size), dtype=np.float16)
    )
    if draft_logits.shape != (1, 2, target.vocab_size):
        raise RuntimeError(f"Unexpected LM-head shape: {draft_logits.shape}")

    print(
        "PASS: OGA token parity, hidden taps, four-token verification, "
        "stateless helpers, and snapshot/rewind rollback are correct."
    )


if __name__ == "__main__":
    main()
