#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Compare one wide windowed forward with sequential single-token forwards.

This diagnostic does not test slot copying. It checks whether the state and
hidden taps produced by a multi-token verification are equivalent to the
corresponding states from repeated single-token execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HIPEP_STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIPEP_STUDY_ROOT))

from oga_target_native import (  # noqa: E402
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    OgaDFlashTarget,
    ensure_runtime_environment,
)
from probe_oga_native_snapshot_restore import (  # noqa: E402
    _clean_greedy_ids,
    _recurrent_output_names,
)
from validate_oga_target_integration import INPUT_IDS  # noqa: E402


def _states(target: OgaDFlashTarget, names: list[str]) -> dict[str, np.ndarray]:
    return {
        name: np.array(target.generator.get_output(name), copy=True) for name in names
    }


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hipep-root", type=Path, default=DEFAULT_HIPEP_ROOT)
    parser.add_argument("--rocm-root", type=Path, default=DEFAULT_ROCM_ROOT)
    parser.add_argument("--block-size", type=int, default=7)
    args = parser.parse_args()
    if args.block_size < 2:
        parser.error("--block-size must be at least 2")

    ensure_runtime_environment(args.hipep_root, args.rocm_root)

    wide = OgaDFlashTarget(args.model, args.hipep_root)
    sequential = OgaDFlashTarget(args.model, args.hipep_root)
    if wide.state_window <= args.block_size:
        raise RuntimeError(
            f"state_window={wide.state_window} must exceed block_size={args.block_size}"
        )

    block = np.asarray(
        _clean_greedy_ids(wide._og, wide.model, count=args.block_size),
        dtype=np.int32,
    )
    wide.prefill(
        INPUT_IDS, max_new_tokens=args.block_size + 8, block_size=args.block_size
    )
    wide_output = wide.verify(block)
    wide_states = _states(wide, _recurrent_output_names(args.model / "text.onnx"))

    sequential.prefill(
        INPUT_IDS,
        max_new_tokens=args.block_size + 8,
        block_size=1,
    )
    sequential_hidden: list[np.ndarray] = []
    sequential_states: list[dict[str, np.ndarray]] = []
    for index, token in enumerate(block):
        output = sequential.verify(np.asarray([token], dtype=np.int32))
        sequential.commit_verified_prefix(
            output,
            committed_count=1,
            expected_bonus=int(output.logits[0, 0].argmax()),
            committed_prefix_ids=np.concatenate(
                (INPUT_IDS, block[: index + 1]),
            ),
        )
        sequential_hidden.append(np.array(output.target_hidden[:, 0, :], copy=True))
        sequential_states.append(
            _states(sequential, _recurrent_output_names(args.model / "text.onnx"))
        )

    hidden_deltas = [
        _max_abs(wide_output.target_hidden[:, index, :], hidden)
        for index, hidden in enumerate(sequential_hidden)
    ]
    print(
        "hidden taps: "
        f"max_abs={max(hidden_deltas):.6g}, "
        f"mean_max_abs={np.mean(hidden_deltas):.6g}"
    )

    state_deltas: list[float] = []
    state_exact = 0
    state_total = 0
    per_position: list[tuple[int, int, float]] = []
    per_kind: dict[str, list[float]] = {"conv": [], "recurrent": []}
    exact_by_kind: dict[str, int] = {"conv": 0, "recurrent": 0}
    total_by_kind: dict[str, int] = {"conv": 0, "recurrent": 0}
    for index in range(args.block_size):
        source_slot = index + wide.state_window - args.block_size
        position_exact = 0
        position_deltas: list[float] = []
        for name, wide_state in wide_states.items():
            wide_slot = wide_state[source_slot]
            sequential_slot = sequential_states[index][name][-1]
            delta = _max_abs(wide_slot, sequential_slot)
            state_deltas.append(delta)
            position_deltas.append(delta)
            kind = "conv" if name.endswith(".conv_state") else "recurrent"
            per_kind[kind].append(delta)
            exact = int(np.array_equal(wide_slot, sequential_slot))
            position_exact += exact
            state_exact += exact
            state_total += 1
            exact_by_kind[kind] += exact
            total_by_kind[kind] += 1
        per_position.append((position_exact, len(wide_states), max(position_deltas)))
    for index, (exact, total, delta) in enumerate(per_position):
        print(f"state position {index}: exact={exact}/{total}, max_abs={delta:.6g}")
    for kind, deltas in per_kind.items():
        print(
            f"{kind} states: exact={exact_by_kind[kind]}/{total_by_kind[kind]}, "
            f"max_abs={max(deltas):.6g}, mean_max_abs={np.mean(deltas):.6g}"
        )
    print(
        "recurrent/conv states: "
        f"exact={state_exact}/{state_total}, "
        f"max_abs={max(state_deltas):.6g}, "
        f"mean_max_abs={np.mean(state_deltas):.6g}"
    )

    if max(hidden_deltas) == 0 and max(state_deltas) == 0:
        print("PASS: wide and sequential windowed execution are bit-exact.")
    else:
        print(
            "DIAGNOSTIC: wide execution differs from sequential execution; "
            "inspect the state-window recurrent kernels and hidden taps."
        )


if __name__ == "__main__":
    main()
