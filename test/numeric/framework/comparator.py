#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tolerance-based comparison of inference outputs against reference values."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sizeof_fmt(num_bytes: int) -> str:
    """Human-readable byte count (B / KB / MB / GB / TB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def tensor_desc(arr: np.ndarray) -> str:
    """One-line tensor summary: shape, dtype, size."""
    return f"{arr.dtype} {list(arr.shape)} ({sizeof_fmt(arr.nbytes)})"


@dataclass
class CompareResult:
    passed: bool
    details: list[str] = field(default_factory=list)

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [status] + self.details
        return "\n  ".join(lines)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compare_outputs(
    actual: list[np.ndarray],
    expected: list[np.ndarray],
    atol: float = 1e-5,
    rtol: float = 1e-3,
    cos_threshold: float = 0.999,
) -> CompareResult:
    """Compare actual vs expected with absolute/relative tolerance and cosine.

    Both ``np.allclose`` and cosine similarity must pass.
    Raises ``AssertionError`` on any mismatch.
    """
    print(
        f"[Compare] Thresholds: atol={atol}, rtol={rtol}, cos_threshold={cos_threshold}"
    )

    if len(actual) != len(expected):
        msg = f"Output count mismatch: {len(actual)} vs {len(expected)}"
        raise AssertionError(msg)

    details = []
    all_passed = True

    for i, (act, exp) in enumerate(zip(actual, expected)):
        act = act.astype(exp.dtype) if act.dtype != exp.dtype else act
        if act.shape != exp.shape:
            details.append(f"Output {i}: shape mismatch {act.shape} vs {exp.shape}")
            all_passed = False
            continue

        # Dynamic-output ops (Range, NonZero, ConstantOfShape with dynamic
        # shape input) legitimately produce zero-element tensors when their
        # runtime length resolves to 0 -- treat that as a structural match
        # without invoking min/max/mean/cosine, which all raise on empty.
        if act.size == 0:
            details.append(f"Output {i}: empty (shape match, no values to compare)")
            continue

        act_f = act.flatten().astype(np.float64)
        exp_f = exp.flatten().astype(np.float64)
        diff = np.abs(act_f - exp_f)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        close = np.allclose(act, exp, atol=atol, rtol=rtol)
        cos_sim = _cosine_similarity(act, exp)

        details.append(
            f"Output {i}: actual  stats: "
            f"min={float(act.min()):.6g}, max={float(act.max()):.6g}, "
            f"mean={act_f.mean():.6g}"
        )
        details.append(
            f"Output {i}: expect  stats: "
            f"min={float(exp.min()):.6g}, max={float(exp.max()):.6g}, "
            f"mean={exp_f.mean():.6g}"
        )
        details.append(
            f"Output {i}: diff    stats: "
            f"max={max_diff:.6e}, mean={mean_diff:.6e}, cosine={cos_sim:.6f}"
        )

        if not close:
            details.append(f"Output {i}: allclose FAILED (atol={atol}, rtol={rtol})")
            all_passed = False
        elif cos_sim < cos_threshold:
            details.append(
                f"Output {i}: cosine FAILED ({cos_sim:.6f} < {cos_threshold})"
            )
            all_passed = False
        else:
            details.append(f"Output {i}: OK")

    result = CompareResult(passed=all_passed, details=details)
    print(f"[Compare] {result}")
    if not all_passed:
        raise AssertionError(str(result))
    return result


def print_output_summary(
    actual: list[np.ndarray],
    expected: list[np.ndarray],
    tag: str = "[ModelRunner]",
) -> None:
    """Print a per-output summary comparing *actual* and *expected* tensors."""
    print(f"{tag} Outputs:")
    for i, (act, exp) in enumerate(zip(actual, expected)):
        # Both empty -> trivial match. Shapes differ or one is empty ->
        # broadcasting / reductions in the diff line below blow up, so
        # short-circuit with a structural-only summary. The full diff is
        # reported by compare_outputs() against actual/expected separately.
        if act.shape != exp.shape or act.size == 0:
            print(
                f"{tag}   [{i}] actual  {tensor_desc(act)}"
                f"  |  expect  {tensor_desc(exp)}"
                f"  |  (shape/empty -- diff suppressed)"
            )
            continue
        diff = float(np.abs(act.astype(np.float64) - exp.astype(np.float64)).max())
        print(
            f"{tag}   [{i}] actual  {tensor_desc(act)}"
            f"  |  expect  {tensor_desc(exp)}"
            f"  |  max_diff={diff:.3e}"
        )
