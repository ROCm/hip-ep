#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Guard repository-convention ONNX-to-HIP destination construction patterns.

This token checker intentionally recognizes only the C++ forms used in-tree; it
is fail-closed for those reviewed forms and is not a general C++ parser.
"""

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
    "QMoEConversion.cpp": 1,
    "RotaryEmbeddingConversion.cpp": 1,
}

DIRECT_EMPTY_REVIEWED = {
    "AttentionConversion.cpp": 2,
    "CompressConversion.cpp": 3,
    "ConcatConversion.cpp": 1,
    "ConstantOfShapeConversion.cpp": 1,
    "ExpandConversion.cpp": 1,
    "IfOutline.cpp": 1,
    "NonZeroConversion.cpp": 2,
    "OnnxAttentionConversion.cpp": 3,
    "OnnxToHipUtils.cpp": 2,
    "OneHotConversion.cpp": 1,
    "PadConversion.cpp": 2,
    "RangeConversion.cpp": 2,
    "SizeConversion.cpp": 1,
    "SliceConversion.cpp": 1,
    "TopKConversion.cpp": 1,
}

REDUCTIONS = {
    "ReduceL2Conversion.cpp": "ReduceL2Op",
    "ReduceMaxConversion.cpp": "ReduceMaxOp",
    "ReduceMeanConversion.cpp": "ReduceMeanOp",
    "ReduceMinConversion.cpp": "ReduceMinOp",
    "ReduceProdConversion.cpp": "ReduceProdOp",
    "ReduceSumConversion.cpp": "ReduceSumOp",
}

HIP_CONSTANT_TOKENS = {
    "arith::ConstantOp": "inline arithmetic constants",
    "hip::ConstantOp": "HIP constant carriers",
}
ONNX_CONSTANT_TOKENS = {
    '"onnx.Constant"': "pre-carrier generic ONNX constants",
    "matchHipCompileTimeConstantTensor(value)": "the HIP-neutral payload matcher",
}

REMOVED_STAMP_TOKENS = {
    "populatePadShapeFoldPatterns": "Pad stamp path",
    "populateSliceShapeFoldPatterns": "Slice stamp path",
    "populateTileShapeFoldPatterns": "Tile stamp path",
    "hipdnn.pad_amounts": "Pad stamp attribute",
    "hipdnn.slice_starts": "Slice stamp attribute",
    "hipdnn.tile_repeats": "Tile stamp attribute",
}


CPP_TOKEN = re.compile(
    r"""
    //[^\n]* | /\*.*?\*/ |
    "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' |
    :: | -> | == | != | <= | >= | && | \|\| |
    [A-Za-z_]\w* | \d+ | \S
    """,
    re.DOTALL | re.VERBOSE,
)


def _cpp_tokens(text: str) -> list[str]:
    return [
        token
        for token in CPP_TOKEN.findall(text)
        if not token.startswith(("//", "/*", '"', "'"))
    ]


def _matches_qualified_empty_type(tokens: list[str], start: int) -> int | None:
    if tokens[start : start + 1] == ["::"]:
        if start > 0 and re.fullmatch(r"[A-Za-z_]\w*", tokens[start - 1]):
            return None
        start += 1
    if tokens[start : start + 2] == ["mlir", "::"]:
        start += 2
    if tokens[start : start + 3] != ["tensor", "::", "EmptyOp"]:
        return None
    return start + 3


def _collect_empty_type_aliases(tokens: list[str]) -> set[str]:
    aliases: set[str] = set()
    for index, token in enumerate(tokens):
        if token == "using" and re.fullmatch(
            r"[A-Za-z_]\w*", tokens[index + 1] if index + 1 < len(tokens) else ""
        ):
            alias = tokens[index + 1]
            if tokens[index + 2 : index + 3] != ["="]:
                continue
            type_end = _matches_qualified_empty_type(tokens, index + 3)
            if type_end is not None and tokens[type_end : type_end + 1] == [";"]:
                aliases.add(alias)
        elif token == "typedef":
            type_end = _matches_qualified_empty_type(tokens, index + 1)
            if type_end is None or type_end + 1 >= len(tokens):
                continue
            alias = tokens[type_end]
            if re.fullmatch(r"[A-Za-z_]\w*", alias) and tokens[
                type_end + 1 : type_end + 2
            ] == [";"]:
                aliases.add(alias)
    return aliases


def _count_direct_empty_calls(text: str) -> int:
    tokens = _cpp_tokens(text)
    aliases = _collect_empty_type_aliases(tokens)
    count = 0
    for index in range(len(tokens)):
        nested_mlir_qualification = (
            tokens[index] == "tensor"
            and index >= 2
            and tokens[index - 2 : index] == ["mlir", "::"]
        )
        if not nested_mlir_qualification:
            type_end = _matches_qualified_empty_type(tokens, index)
            if type_end is not None and tokens[type_end : type_end + 3] == [
                "::",
                "create",
                "(",
            ]:
                count += 1
                continue

        if tokens[index] != "create" or index == 0:
            continue
        receiver = index - 1
        if tokens[receiver] == "template":
            receiver -= 1
        if receiver < 0 or tokens[receiver] not in {".", "->"}:
            continue
        if tokens[index + 1 : index + 2] != ["<"]:
            continue
        type_end = _matches_qualified_empty_type(tokens, index + 2)
        direct_type = type_end is not None and tokens[type_end : type_end + 2] == [
            ">",
            "(",
        ]
        alias_type = (
            tokens[index + 2 : index + 3]
            and tokens[index + 2] in aliases
            and tokens[index + 3 : index + 5] == [">", "("]
        )
        if direct_type or alias_type:
            count += 1
    return count


def _check_source_root(source_root: Path) -> list[str]:
    conversion_dir = source_root / "lib/Conversion/OnnxToHip"
    failures: list[str] = []
    positional = re.compile(r"\bcreateEmptyTensor\s*\(")
    seen_allowed: set[str] = set()
    seen_direct_reviewed: set[str] = set()
    for path in sorted(conversion_dir.rglob("*.cpp")):
        relative_path = path.relative_to(conversion_dir).as_posix()
        text = path.read_text()
        actual = len(positional.findall(text))
        expected = POSITIONAL_ALLOWED.get(relative_path, 0)
        if relative_path in POSITIONAL_ALLOWED:
            seen_allowed.add(relative_path)
        if actual != expected:
            failures.append(
                f"{relative_path}: expected {expected} positional "
                f"createEmptyTensor occurrence(s), found {actual}; use the "
                "operation's shared shape helper"
            )
        direct_actual = _count_direct_empty_calls(text)
        direct_expected = DIRECT_EMPTY_REVIEWED.get(relative_path, 0)
        if relative_path in DIRECT_EMPTY_REVIEWED:
            seen_direct_reviewed.add(relative_path)
        if direct_actual != direct_expected:
            failures.append(
                f"{relative_path}: expected {direct_expected} reviewed direct "
                f"tensor::EmptyOp::create occurrence(s), found {direct_actual}; "
                "use an exact shared shape helper or update the reviewed count"
            )
    for relative_path in sorted(set(POSITIONAL_ALLOWED) - seen_allowed):
        failures.append(
            f"{relative_path}: expected "
            f"{POSITIONAL_ALLOWED[relative_path]} positional createEmptyTensor "
            "occurrence(s), found 0"
        )
    for relative_path in sorted(set(DIRECT_EMPTY_REVIEWED) - seen_direct_reviewed):
        failures.append(
            f"{relative_path}: expected "
            f"{DIRECT_EMPTY_REVIEWED[relative_path]} reviewed direct "
            "tensor::EmptyOp::create occurrence(s), found 0"
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

    utils_path = conversion_dir / "OnnxToHipUtils.cpp"
    utils_text = utils_path.read_text()
    for token, purpose in ONNX_CONSTANT_TOKENS.items():
        if token not in utils_text:
            failures.append(
                f"OnnxToHipUtils.cpp: missing {purpose} recognition token {token}"
            )

    hip_utils_path = source_root / "lib/Conversion/HipConversionUtils.cpp"
    hip_utils_text = hip_utils_path.read_text()
    for token, purpose in HIP_CONSTANT_TOKENS.items():
        if token not in hip_utils_text:
            failures.append(
                f"HipConversionUtils.cpp: missing {purpose} recognition token {token}"
            )
    direct_hip_utils = _count_direct_empty_calls(hip_utils_text)
    if direct_hip_utils != 1:
        failures.append(
            "HipConversionUtils.cpp: expected one reviewed direct "
            "tensor::EmptyOp::create occurrence, found "
            f"{direct_hip_utils}"
        )

    all_conversion_text = "\n".join(
        path.read_text()
        for path in sorted(conversion_dir.rglob("*"))
        if path.is_file() and path.suffix in {".cpp", ".h"}
    )
    for token, purpose in REMOVED_STAMP_TOKENS.items():
        if token in all_conversion_text:
            failures.append(
                f"obsolete {purpose} {token} found; inspect dense hip.constant "
                "carriers directly instead"
            )
    return failures


def _run_self_test() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        source_root = Path(temp_dir)
        conversion_dir = source_root / "lib/Conversion/OnnxToHip"
        conversion_dir.mkdir(parents=True)
        hip_utils = source_root / "lib/Conversion/HipConversionUtils.cpp"

        def append(relative_path: str, text: str) -> None:
            path = conversion_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(text)

        for relative_path, expected in POSITIONAL_ALLOWED.items():
            append(relative_path, "createEmptyTensor(builder, loc);\n" * expected)
        for relative_path, expected in DIRECT_EMPTY_REVIEWED.items():
            append(
                relative_path,
                "tensor::EmptyOp::create(builder, loc, shape, type);\n" * expected,
            )
        for filename, hip_op in REDUCTIONS.items():
            append(filename, f"OnnxReductionToHip<mlir::hip::{hip_op}>\n")
        append("OnnxToHipUtils.cpp", "\n".join(ONNX_CONSTANT_TOKENS) + "\n")
        hip_utils.write_text(
            "\n".join(HIP_CONSTANT_TOKENS)
            + "\nmlir::tensor::EmptyOp::create(builder, loc, shape, type);\n"
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

        nested_direct = conversion_dir / "Shared/DirectBuilder.cpp"
        nested_direct.write_text(
            "mlir::tensor::EmptyOp::create(builder, loc, shape, type);\n"
        )
        failures = _check_source_root(source_root)
        if not any("Shared/DirectBuilder.cpp" in failure for failure in failures):
            raise AssertionError("recursive direct-builder fixture unexpectedly passed")
        nested_direct.unlink()

        adversarial_variants = (
            "tensor :: EmptyOp :: create (builder, loc, shape, type);\n",
            "mlir::tensor :: EmptyOp::create\n(builder, loc, shape, type);\n",
            "builder.create<tensor::EmptyOp>(loc, shape, type);\n",
            "rewriter . create < mlir :: tensor :: EmptyOp > (loc, shape, type);\n",
            "builder->template create<mlir::tensor::EmptyOp>(loc, shape, type);\n",
        )
        for index, variant in enumerate(adversarial_variants):
            adversarial = conversion_dir / f"AdversarialEmpty{index}.cpp"
            adversarial.write_text(variant)
            failures = _check_source_root(source_root)
            if not any(adversarial.name in failure for failure in failures):
                raise AssertionError(
                    f"direct empty variant {index} unexpectedly passed"
                )
            adversarial.unlink()

        alias_variants = (
            "using Empty = mlir::tensor::EmptyOp;\n"
            "void build() { builder.create<Empty>(loc, shape, type); }\n",
            "void build() {\n"
            "  using LocalEmpty = tensor :: EmptyOp;\n"
            "  rewriter . create < LocalEmpty > (loc, shape, type);\n"
            "}\n",
            "typedef tensor::EmptyOp EmptyAlias;\n"
            "void build() { builder.create<EmptyAlias>(loc, shape, type); }\n",
            "void build() {\n"
            "  typedef mlir :: tensor :: EmptyOp LocalAlias;\n"
            "  builder->template create<LocalAlias>(loc, shape, type);\n"
            "}\n",
        )
        for index, variant in enumerate(alias_variants):
            adversarial = conversion_dir / f"AdversarialEmptyAlias{index}.cpp"
            adversarial.write_text(variant)
            failures = _check_source_root(source_root)
            if not any(adversarial.name in failure for failure in failures):
                raise AssertionError(
                    f"direct empty alias variant {index} unexpectedly passed"
                )
            adversarial.unlink()

        ignored = conversion_dir / "IgnoredEmptySpelling.cpp"
        ignored.write_text(
            "// tensor::EmptyOp::create(builder, loc, shape, type);\n"
            'const char *text = "builder.create<tensor::EmptyOp>()";\n'
        )
        if failures := _check_source_root(source_root):
            raise AssertionError(
                f"comment/string direct empty spellings failed: {failures}"
            )
        ignored.unlink()

        allowed = conversion_dir / "GqaConversion.cpp"
        allowed.write_text(allowed.read_text() + "createEmptyTensor(builder, loc);\n")
        failures = _check_source_root(source_root)
        if not any(
            "expected 1" in failure and "found 2" in failure for failure in failures
        ):
            raise AssertionError(
                "allowlist occurrence-count fixture unexpectedly passed"
            )
        allowed.write_text("createEmptyTensor(builder, loc);\n")

        direct_allowed = conversion_dir / "SliceConversion.cpp"
        direct_allowed.write_text(
            direct_allowed.read_text()
            + "tensor::EmptyOp::create(builder, loc, shape, type);\n"
        )
        failures = _check_source_root(source_root)
        if not any(
            "expected 1 reviewed direct" in failure and "found 2" in failure
            for failure in failures
        ):
            raise AssertionError(
                "direct allowlist occurrence-count fixture unexpectedly passed"
            )
        direct_allowed.write_text(
            "tensor::EmptyOp::create(builder, loc, shape, type);\n"
        )

        original_hip_utils = hip_utils.read_text()
        hip_utils.write_text(original_hip_utils.replace("hip::ConstantOp\n", ""))
        failures = _check_source_root(source_root)
        if not any("HIP constant carriers" in failure for failure in failures):
            raise AssertionError(
                "missing carrier extraction fixture unexpectedly passed"
            )
        hip_utils.write_text(original_hip_utils)

        obsolete = conversion_dir / "ObsoleteStamp.cpp"
        obsolete.write_text("populateSliceShapeFoldPatterns(patterns, ctx);\n")
        failures = _check_source_root(source_root)
        if not any("Slice stamp path" in failure for failure in failures):
            raise AssertionError("obsolete stamp fixture unexpectedly passed")

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
    print(
        "verified positional/direct destinations, six shared reductions, "
        "carrier extraction, and removed stamp paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
