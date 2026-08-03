#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Guard centralized ONNX-to-HIP shape/destination conversion helpers."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path


POSITIONAL_ALLOWED = {
    "GqaConversion.cpp": 1,
    "HipGqaBuilder.cpp": 1,
    "OnnxAttentionConversion.cpp": 1,
    "OnnxRotaryEmbeddingConversion.cpp": 1,
    "OnnxToHipUtils.cpp": 1,
    "QMoEConversion.cpp": 1,
    "RotaryEmbeddingConversion.cpp": 1,
}

REDUCTIONS = {
    "ReduceL2Conversion.cpp": "ReduceL2Op",
    "ReduceMaxConversion.cpp": "ReduceMaxOp",
    "ReduceMeanConversion.cpp": "ReduceMeanOp",
    "ReduceMinConversion.cpp": "ReduceMinOp",
    "ReduceProdConversion.cpp": "ReduceProdOp",
    "ReduceSumConversion.cpp": "ReduceSumOp",
}


def _check_source_root(source_root: Path) -> list[str]:
    conversion_dir = source_root / "lib/Conversion/OnnxToHip"
    failures: list[str] = []
    positional = re.compile(r"\bcreateEmptyTensor\s*\(")
    seen_allowed: set[str] = set()
    for path in sorted(conversion_dir.rglob("*.cpp")):
        relative_path = path.relative_to(conversion_dir).as_posix()
        actual = len(positional.findall(path.read_text()))
        expected = POSITIONAL_ALLOWED.get(relative_path, 0)
        if relative_path in POSITIONAL_ALLOWED:
            seen_allowed.add(relative_path)
        if actual != expected:
            failures.append(
                f"{relative_path}: expected {expected} positional "
                f"createEmptyTensor occurrence(s), found {actual}; use the "
                "operation's shared shape helper"
            )
    for relative_path in sorted(set(POSITIONAL_ALLOWED) - seen_allowed):
        failures.append(
            f"{relative_path}: expected "
            f"{POSITIONAL_ALLOWED[relative_path]} positional createEmptyTensor "
            "occurrence(s), found 0"
        )

    reduction_forbidden = (
        "resolveReductionAxes",
        "createReductionEmptyTensor",
        "materializeReductionAxes",
    )
    for filename, hip_op in REDUCTIONS.items():
        text = (conversion_dir / filename).read_text()
        expected = f"OnnxReductionToHip<mlir::hip::{hip_op}>"
        if text.count(expected) != 1:
            failures.append(f"{filename}: expected exactly one {expected}")
        for token in reduction_forbidden:
            if token in text:
                failures.append(
                    f"{filename}: duplicated reduction skeleton token {token}"
                )
    return failures


def _run_self_test() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        source_root = Path(temp_dir)
        conversion_dir = source_root / "lib/Conversion/OnnxToHip"
        conversion_dir.mkdir(parents=True)
        for relative_path, expected in POSITIONAL_ALLOWED.items():
            path = conversion_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("createEmptyTensor(builder, loc);\n" * expected)
        for filename, hip_op in REDUCTIONS.items():
            (conversion_dir / filename).write_text(
                f"OnnxReductionToHip<mlir::hip::{hip_op}>\n"
            )
        if failures := _check_source_root(source_root):
            raise AssertionError(f"valid fixture failed: {failures}")

        nested = conversion_dir / "Shared/Builder.cpp"
        nested.parent.mkdir()
        nested.write_text("createEmptyTensor(builder, loc);\n")
        failures = _check_source_root(source_root)
        if not any("Shared/Builder.cpp" in failure for failure in failures):
            raise AssertionError("recursive shared-builder fixture unexpectedly passed")
        nested.unlink()

        allowed = conversion_dir / "GqaConversion.cpp"
        allowed.write_text(allowed.read_text() + "createEmptyTensor(builder, loc);\n")
        failures = _check_source_root(source_root)
        if not any(
            "expected 1" in failure and "found 2" in failure for failure in failures
        ):
            raise AssertionError(
                "allowlist occurrence-count fixture unexpectedly passed"
            )

    print("verified ONNX-to-HIP dedup guard fixtures")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    if args.source_root is None:
        parser.error("the following argument is required: --source-root")

    failures = _check_source_root(args.source_root)

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("verified positional destination allowlist and six shared reductions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
