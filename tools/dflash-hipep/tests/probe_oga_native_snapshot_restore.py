"""Probe upstream OGA's native Qwen3.6 recurrent-state snapshot and restore.

This test does not try to replace graph inputs from Python. It exercises only
the upstream Generator contract:

1. process a clean prompt plus anchor token;
2. snapshot OGA's internally managed recurrent/conv state;
3. contaminate it with rejected tokens;
4. rewind to the snapshotted sequence length;
5. process the correct suffix and compare against a clean generator.

The second ``append_tokens()`` also proves that the installed OGA build permits
continuous decoding on the AMDGPU-backed target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx

HIPEP_STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIPEP_STUDY_ROOT))

from oga_target_native import (  # noqa: E402
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    DFLASH_LOGITS,
    ensure_runtime_environment,
)
from validate_oga_target_integration import INPUT_IDS  # noqa: E402


def _clean_greedy_ids(og, model, count: int) -> tuple[int, ...]:
    """Generate a clean greedy target trace for model-specific probes."""
    tokens = np.ascontiguousarray(INPUT_IDS.reshape(-1), dtype=np.int32)
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=int(tokens.size) + int(count),
        do_sample=False,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(tokens)

    generated = []
    for _ in range(count):
        generator.generate_next_token()
        generated.append(int(np.asarray(generator.get_next_tokens())[0]))
    return tuple(generated)


def _recurrent_output_names(model_path: Path) -> list[str]:
    """Return every exposed convolution and recurrent-state graph output."""
    graph = onnx.load(model_path, load_external_data=False).graph
    names = sorted(
        value.name
        for value in graph.output
        if value.name.endswith((".conv_state", ".recurrent_state"))
    )
    if not names:
        raise RuntimeError("The target graph exposes no recurrent-state outputs")
    return names


def _load_decoder(og, model_dir: Path):
    """Load only the stateful decoder; helper ONNX sessions are irrelevant here."""
    config = og.Config(str(model_dir))
    config.overlay(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "session_options": {
                            "disable_cpu_ep_fallback": "1",
                        }
                    }
                }
            }
        )
    )
    config.clear_providers()
    config.append_provider("AMDGPU")
    config.set_provider_option("AMDGPU", "profile", "hip")
    model = og.Model(config)
    if str(model.device_type).lower() == "cpu":
        raise RuntimeError("OGA reported CPU despite the AMDGPU override")
    return model


def _generator_at_snapshot_boundary(
    og,
    model,
    max_length: int,
    expected_ids: tuple[int, ...],
):
    """Process the prompt and first target token in one aligned OGA append."""
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, do_sample=False)
    generator = og.Generator(model, params)
    initial_tokens = np.concatenate(
        (INPUT_IDS, np.asarray([expected_ids[0]], dtype=np.int32))
    )
    generator.append_tokens(initial_tokens)
    return generator, int(initial_tokens.size)


def _states(generator, state_names: list[str]) -> dict[str, np.ndarray]:
    """Copy managed recurrent state before another execution overwrites it."""
    return {
        name: np.asarray(generator.get_output(name)).copy()
        for name in state_names
    }


def _outputs(generator, state_names: list[str]):
    """Copy logits and managed state before another graph execution overwrites them."""
    logits = np.asarray(generator.get_output(DFLASH_LOGITS)).copy()
    return logits, _states(generator, state_names)


def _compare(
    logits: np.ndarray,
    states: dict[str, np.ndarray],
    reference_logits: np.ndarray,
    reference_states: dict[str, np.ndarray],
) -> tuple[float, bool, list[str]]:
    """Return exact parity diagnostics against an uncontaminated execution."""
    logit_error = float(
        np.max(np.abs(logits.astype(np.float32) - reference_logits))
    )
    predictions_match = np.array_equal(
        np.argmax(logits, axis=-1),
        np.argmax(reference_logits, axis=-1),
    )
    state_mismatches = [
        name
        for name in states
        if not np.array_equal(states[name], reference_states[name])
    ]
    return logit_error, predictions_match, state_mismatches


def _run_repeated_cycles(
    og,
    model,
    max_length: int,
    state_names: list[str],
    expected_ids: tuple[int, ...],
    *,
    cycles: int,
    tokens_per_cycle: int,
) -> None:
    """Verify that one generator can overwrite and restore its snapshot repeatedly."""
    available = np.asarray(expected_ids[1:], dtype=np.int32)
    required = cycles * tokens_per_cycle
    if required > available.size:
        raise ValueError(
            f"Repeated probe needs {required} committed tokens, but the fixed "
            f"trace provides only {available.size}"
        )

    reference, _ = _generator_at_snapshot_boundary(
        og, model, max_length, expected_ids
    )
    restored, boundary = _generator_at_snapshot_boundary(
        og, model, max_length, expected_ids
    )

    for cycle in range(cycles):
        start = cycle * tokens_per_cycle
        committed = np.ascontiguousarray(
            available[start : start + tokens_per_cycle]
        )
        rejected = np.full(committed.shape, 248049, dtype=np.int32)

        reference.append_tokens(committed)
        reference_logits, reference_states = _outputs(reference, state_names)

        boundary_states = _states(restored, state_names)
        restored.snapshot_state()
        restored.append_tokens(rejected)
        restored.rewind_to(boundary)
        rewound_states = _states(restored, state_names)
        boundary_mismatches = [
            name
            for name in state_names
            if not np.array_equal(rewound_states[name], boundary_states[name])
        ]
        if boundary_mismatches:
            raise RuntimeError(
                f"FAIL: cycle {cycle + 1} restored only "
                f"{len(state_names) - len(boundary_mismatches)}/{len(state_names)} "
                "state tensors at its rewind boundary"
            )

        restored.append_tokens(committed)
        restored_logits, restored_states = _outputs(restored, state_names)
        error, predictions_match, state_mismatches = _compare(
            restored_logits,
            restored_states,
            reference_logits,
            reference_states,
        )
        print(
            f"Repeated cycle {cycle + 1}/{cycles}: "
            f"boundary={len(state_names)}/{len(state_names)}, "
            f"states={len(state_names) - len(state_mismatches)}/{len(state_names)}, "
            f"logit_error={error:.6f}, predictions={predictions_match}"
        )
        if error != 0.0 or not predictions_match or state_mismatches:
            raise RuntimeError(
                f"FAIL: repeated snapshot cycle {cycle + 1} diverged from clean execution"
            )

        boundary += int(committed.size)

    del reference, restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hipep-root", type=Path, default=DEFAULT_HIPEP_ROOT)
    parser.add_argument("--rocm-root", type=Path, default=DEFAULT_ROCM_ROOT)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--tokens-per-cycle", type=int, default=3)
    args = parser.parse_args()
    if args.cycles < 1 or args.tokens_per_cycle < 1:
        parser.error("--cycles and --tokens-per-cycle must both be positive")

    ensure_runtime_environment(args.hipep_root, args.rocm_root)

    import onnxruntime_genai as og

    if not hasattr(og.Generator, "snapshot_state"):
        raise RuntimeError(
            f"Installed OGA {getattr(og, '__version__', 'unknown')} has no "
            "Generator.snapshot_state()"
        )

    print(
        f"Loading OGA {getattr(og, '__version__', 'unknown')} target...",
        flush=True,
    )
    model = _load_decoder(og, args.model)
    max_length = int(INPUT_IDS.size) + max(
        32,
        1 + args.cycles * args.tokens_per_cycle,
    )
    expected_ids = _clean_greedy_ids(
        og,
        model,
        count=max(4, 1 + args.cycles * args.tokens_per_cycle),
    )
    state_names = _recurrent_output_names(args.model / "text.onnx")
    print(f"Found {len(state_names) // 2} recurrent layers ({len(state_names)} tensors).")

    correct_suffix = np.asarray(expected_ids[1:4], dtype=np.int32)
    rejected_suffix = np.full(correct_suffix.shape, 248049, dtype=np.int32)

    # Authoritative execution with no rejected-token contamination.
    reference, _ = _generator_at_snapshot_boundary(
        og, model, max_length, expected_ids
    )
    reference.append_tokens(correct_suffix)
    reference_logits, reference_states = _outputs(reference, state_names)
    del reference

    # Establish that two uncontaminated generators are exactly repeatable.
    clean_repeat, _ = _generator_at_snapshot_boundary(
        og, model, max_length, expected_ids
    )
    clean_repeat.append_tokens(correct_suffix)
    clean_logits, clean_states = _outputs(clean_repeat, state_names)
    del clean_repeat
    clean_error, clean_predictions_match, clean_state_mismatches = _compare(
        clean_logits,
        clean_states,
        reference_logits,
        reference_states,
    )
    print(f"Clean-repeat maximum logit error:      {clean_error:.6f}")
    print(f"Clean-repeat predictions match:        {clean_predictions_match}")
    print(
        "Clean-repeat exact recurrent tensors: "
        f"{len(state_names) - len(clean_state_mismatches)}/{len(state_names)}"
    )

    # Exercise native snapshot/restore around a deliberately rejected block.
    restored, boundary = _generator_at_snapshot_boundary(
        og, model, max_length, expected_ids
    )
    boundary_states = _states(restored, state_names)
    restored.snapshot_state()
    restored.append_tokens(rejected_suffix)
    contaminated_states = _states(restored, state_names)
    restored.rewind_to(boundary)
    rewound_states = _states(restored, state_names)
    contaminated_count = sum(
        not np.array_equal(contaminated_states[name], boundary_states[name])
        for name in state_names
    )
    boundary_restore_mismatches = [
        name
        for name in state_names
        if not np.array_equal(rewound_states[name], boundary_states[name])
    ]
    print(f"States changed by rejected block:       {contaminated_count}/{len(state_names)}")
    print(
        "States restored at rewind boundary:    "
        f"{len(state_names) - len(boundary_restore_mismatches)}/{len(state_names)}"
    )
    restored.append_tokens(correct_suffix)
    restored_logits, restored_states = _outputs(restored, state_names)
    del restored

    logit_error, predictions_match, state_mismatches = _compare(
        restored_logits,
        restored_states,
        reference_logits,
        reference_states,
    )

    print(f"Maximum restored-vs-clean logit error: {logit_error:.6f}")
    print(f"Greedy suffix predictions match:       {predictions_match}")
    print(
        "Exactly matching recurrent tensors:   "
        f"{len(state_names) - len(state_mismatches)}/{len(state_names)}"
    )

    # Diagnose whether AMDGPU's asynchronous device copies merely need an
    # externally visible synchronization point after snapshot_state().
    synchronized, sync_boundary = _generator_at_snapshot_boundary(
        og, model, max_length, expected_ids
    )
    synchronized.snapshot_state()
    _outputs(synchronized, state_names)
    synchronized.append_tokens(rejected_suffix)
    synchronized.rewind_to(sync_boundary)
    synchronized.append_tokens(correct_suffix)
    sync_logits, sync_states = _outputs(synchronized, state_names)
    del synchronized
    sync_error, sync_predictions_match, sync_state_mismatches = _compare(
        sync_logits,
        sync_states,
        reference_logits,
        reference_states,
    )
    print(f"Post-read-sync maximum logit error:    {sync_error:.6f}")
    print(f"Post-read-sync predictions match:      {sync_predictions_match}")
    print(
        "Post-read-sync exact state tensors:    "
        f"{len(state_names) - len(sync_state_mismatches)}/{len(state_names)}"
    )

    if clean_error != 0.0 or not clean_predictions_match or clean_state_mismatches:
        raise RuntimeError(
            "INCONCLUSIVE: clean target executions are not exactly repeatable "
            f"({len(clean_state_mismatches)}/{len(state_names)} recurrent outputs differ)."
        )

    if logit_error != 0.0 or not predictions_match or state_mismatches:
        raise RuntimeError(
            "FAIL: native rewind did not reproduce clean target execution "
            f"({len(state_mismatches)}/{len(state_names)} recurrent outputs differ)."
        )

    _run_repeated_cycles(
        og,
        model,
        max_length,
        state_names,
        expected_ids,
        cycles=args.cycles,
        tokens_per_cycle=args.tokens_per_cycle,
    )

    print(
        "PASS: upstream OGA restored exact logits and all managed recurrent "
        "state after single and repeated rejected-block contamination."
    )


if __name__ == "__main__":
    main()
