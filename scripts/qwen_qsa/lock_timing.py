#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Freeze common Qwen timing streams before native selector integration."""

import json
import random
from pathlib import Path

from verify_handoff import SOURCE, WORK, sha


def main():
    output = WORK / "timing-v1"
    assert not output.exists(), "Preserve the existing frozen timing protocol"
    output.mkdir()
    verified = json.loads((WORK / "handoff-verification.json").read_text())
    path = SOURCE / "data/qwen-selector-v1/test.json"
    assert sha(path) == verified["data_sha256"]["test.json"]
    documents = json.loads(path.read_text())
    assert len(documents) == 64 and all(len(d["input_ids"]) == 4096 for d in documents)
    order = list(range(64))
    random.Random(20261125).shuffle(order)
    streams = []
    for batch_index in range(8):
        rotation = batch_index * 7
        selected = [order[(rotation + i) % 64] for i in range(17)]
        tokens = [
            token for index in selected for token in documents[index]["input_ids"]
        ][:65600]
        assert len(tokens) == 65600 and min(tokens) >= 0 and max(tokens) < 248320
        streams.append(
            {
                "batch_index": batch_index,
                "source_document_indices": selected,
                "input_ids": tokens,
            }
        )
    path = output / "streams.json"
    path.write_text(json.dumps(streams) + "\n", encoding="utf-8")
    manifest = {
        "state": "locked",
        "seed": 20261125,
        "model_revision": verified["revision"],
        "source_test_sha256": verified["data_sha256"]["test.json"],
        "streams_sha256": sha(path),
        "streams": 8,
        "tokens_per_stream": 65600,
        "batch_candidates": [1, 4, 8],
        "prefix_candidates": [4096, 16384, 32768, 65536],
        "decode_calls": 64,
        "decode_input": "Common teacher-forced continuation for both variants",
        "timers": "Independent synchronized full-model prefill and decode wall timers",
        "sampling": "Outside primary phase timers; no per-step argmax in headline decode",
        "warmup_pairs": 2,
        "measured_pairs": 6,
        "order": "Alternating Original/Changed within paired trials",
        "scope": "Concatenated locked 4K test windows for timing; already exposed to source engineering. Not natural long-document quality or a blind holdout.",
        "support": "Candidates only; no shape or memory support is established by this manifest",
        "locked_before_native_qsa_integration": True,
        "training_updates": 0,
        "script_sha256": sha(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
