#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify OGA promotes the selected recurrent-state window slot exactly.

The probe processes one verification block with a windowed target, crops to an
accepted position inside that block, and checks every exposed recurrent/conv
state tensor. The selected source slot must be copied bit-exactly to the
committed final slot while every other slot remains unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HIPEP_STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIPEP_STUDY_ROOT))

from oga_target_native import (  # noqa: E402
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    ensure_runtime_environment,
)
from probe_oga_native_snapshot_restore import (  # noqa: E402
    _clean_greedy_ids,
    _load_decoder,
    _recurrent_output_names,
    _states,
)
from validate_oga_target_integration import INPUT_IDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hipep-root", type=Path, default=DEFAULT_HIPEP_ROOT)
    parser.add_argument("--rocm-root", type=Path, default=DEFAULT_ROCM_ROOT)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--accepted-position", type=int, default=2)
    args = parser.parse_args()

    if args.block_size < 2:
        parser.error("--block-size must be at least 2")
    if not 0 <= args.accepted_position < args.block_size - 1:
        parser.error("--accepted-position must select a partial prefix")

    ensure_runtime_environment(args.hipep_root, args.rocm_root)
    os.environ.setdefault("ORTGENAI_AMDGPU_TRACE_STATE_SLOT_COPY", "1")

    import onnxruntime_genai as og

    if not hasattr(og.Generator, "crop_to_accepted"):
        raise RuntimeError(
            f"Installed OGA {getattr(og, '__version__', 'unknown')} has no "
            "Generator.crop_to_accepted()"
        )

    model = _load_decoder(og, args.model)
    state_names = _recurrent_output_names(args.model / "text.onnx")
    conv_names = [name for name in state_names if name.endswith(".conv_state")]
    recurrent_names = [
        name for name in state_names if name.endswith(".recurrent_state")
    ]
    if not conv_names or len(conv_names) != len(recurrent_names):
        raise RuntimeError(
            "Expected matching convolution and recurrent state outputs, found "
            f"{len(conv_names)} convolution and {len(recurrent_names)} recurrent"
        )

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=int(INPUT_IDS.size) + args.block_size + 8,
        do_sample=False,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(INPUT_IDS)

    block = np.asarray(
        _clean_greedy_ids(og, model, args.block_size),
        dtype=np.int32,
    )
    generator.append_tokens(block)
    before = _states(generator, state_names)

    windows = {state.shape[0] for state in before.values()}
    if len(windows) != 1:
        raise RuntimeError(f"Inconsistent state windows: {sorted(windows)}")
    state_window = windows.pop()
    if state_window <= args.block_size:
        raise RuntimeError(
            f"state_window={state_window} cannot crop block_size={args.block_size}"
        )

    source_slot = args.accepted_position + state_window - args.block_size
    destination_slot = state_window - 1
    accepted_length = int(INPUT_IDS.size) + args.accepted_position + 1
    generator.crop_to_accepted(accepted_length, args.accepted_position)
    after = _states(generator, state_names)

    destination_mismatches: list[str] = []
    collateral_mismatches: list[str] = []
    for name in state_names:
        if not np.array_equal(
            after[name][destination_slot],
            before[name][source_slot],
        ):
            destination_mismatches.append(name)
        for slot in range(state_window - 1):
            if not np.array_equal(after[name][slot], before[name][slot]):
                collateral_mismatches.append(f"{name}[{slot}]")

    print(
        f"Promoted slot {source_slot} -> {destination_slot} for "
        f"{len(state_names)} tensors"
    )
    print(
        "Exact promoted destinations: "
        f"{len(state_names) - len(destination_mismatches)}/{len(state_names)}"
    )
    print(
        "Unchanged non-destination slots: "
        f"{len(state_names) * (state_window - 1) - len(collateral_mismatches)}/"
        f"{len(state_names) * (state_window - 1)}"
    )
    if destination_mismatches or collateral_mismatches:
        raise RuntimeError(
            "FAIL: window crop changed the wrong state bytes "
            f"(destinations={len(destination_mismatches)}, "
            f"collateral={len(collateral_mismatches)})"
        )

    print("PASS: OGA promoted all recurrent-state slots bit-exactly.")


if __name__ == "__main__":
    main()
