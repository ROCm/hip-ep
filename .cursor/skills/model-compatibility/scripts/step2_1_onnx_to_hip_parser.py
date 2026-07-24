#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Step 2-1 Final v5: OnnxToHip Conversion Parser - Fixed SkipSimplifiedLayerNormalization
Extract precise mappings (onnx_op, onnx_domain, hip_op, class_name, file_name).
- SimplifiedLayerNormalization: domain is `onnx` (onnx.Custom, no com.microsoft
  constraint).
- SkipSimplifiedLayerNormalization: domain is `com.microsoft`,
  hip.skip_rms_norm (not accidentally overridden by RmsNormOp).
- mlir::hip::FooOp -> hip mnemonic: read the TableGen mnemonic from
  HipOps.td instead of guessing from CamelCase.
- Deduplicate Unsqueeze mappings.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from hip_td_mnemonics import (  # noqa: E402
    default_hip_ops_td_path,
    load_cpp_op_class_to_mnemonic_from_td,
)


class OnnxToHipParser:
    """OnnxToHip parser; dynamically extracts mappings from cpp sources."""

    # onnx_op name -> C++ struct name. C++ struct names are abbreviated,
    # so we cannot derive them via regex from the onnx_op name alone.
    _STRUCT_NAME_BY_ONNX_OP = {
        "SimplifiedLayerNormalization": "SimplifiedLayerNormToHip",
        "SkipSimplifiedLayerNormalization": "SkipSimplifiedLayerNormToHip",
    }

    def __init__(self, conversion_dir: str, hip_ops_td: Optional[str] = None):
        self.conversion_dir = Path(conversion_dir)
        self._hip_td_path: Optional[Path] = None
        self._cpp_class_to_mnemonic: Dict[str, str] = {}

        td = self._resolve_hip_ops_td(hip_ops_td)
        if td is not None:
            self._hip_td_path = td
            self._cpp_class_to_mnemonic = load_cpp_op_class_to_mnemonic_from_td(td)
        else:
            print(
                "Warning: HipOps.td not found; pass hip_ops_td or place repo at "
                "<conversion_dir>/../.. with include/hip/Dialect/IR/HipOps.td. "
                "mlir::hip::*Op -> hip mnemonic will be unavailable (non-custom paths may be wrong)."
            )

    def _resolve_hip_ops_td(self, hip_ops_td: Optional[str]) -> Optional[Path]:
        if hip_ops_td:
            p = Path(hip_ops_td)
            return p if p.is_file() else None
        candidate = default_hip_ops_td_path(self.conversion_dir)
        return candidate if candidate.is_file() else None

    def parse(self) -> Dict:
        """Parse every OnnxToHip conversion file."""
        result = {"mappings": [], "statistics": {}}

        # Discover all OnnxToHip conversion files.
        conversion_files = sorted(self.conversion_dir.glob("OnnxToHip/*.cpp"))

        print(f"Found {len(conversion_files)} conversion files\n")

        for file_path in conversion_files:
            print(f"Parsing: {file_path.name}")
            mappings = self._parse_conversion_file(file_path)
            result["mappings"].extend(mappings)

            # Echo this file's mappings.
            for mapping in mappings:
                class_name = mapping.get("class_name", "")
                print(
                    f"  {mapping['onnx_op']:30} ({mapping['onnx_domain']:15}) -> {mapping['hip_op']:30} [{class_name}]"
                )

        # Deduplicate: keep the first mapping per (onnx_op, onnx_domain, hip_op).
        unique_mappings = {}
        for mapping in result["mappings"]:
            key = (mapping["onnx_op"], mapping["onnx_domain"], mapping["hip_op"])
            if key not in unique_mappings:
                unique_mappings[key] = mapping

        result["mappings"] = list(unique_mappings.values())

        # Compute summary statistics.
        result["statistics"] = {
            "total_mappings": len(result["mappings"]),
            "total_conversion_files": len(conversion_files),
            "unique_onnx_ops": len(set(m["onnx_op"] for m in result["mappings"])),
            "unique_domains": len(set(m["onnx_domain"] for m in result["mappings"])),
            "unique_hip_ops": len(set(m["hip_op"] for m in result["mappings"])),
        }

        return result

    def _parse_conversion_file(self, file_path: Path) -> List[Dict]:
        """Parse a single conversion file and return all mappings."""
        mappings = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_name = file_path.name

            # Extract mappings via multiple modes.
            # Mode 1: doc-comment mappings  /// onnx.OpName -> hip.opname  or  hip.lib.opname.
            comment_patterns = [
                (r"///\s*onnx\.(\w+)\s*->\s*hip\.([^\s]+)", "standard"),
                (r"///\s*onnx\.Custom\((\w+)\)\s*->\s*hip\.([^\s]+)", "custom"),
            ]

            for pattern, op_type in comment_patterns:
                matches = re.findall(pattern, content)
                for onnx_op, hip_op in matches:
                    # Extract the HIP class name corresponding to this op.
                    class_name = self._extract_hip_class_name_for_op(content, onnx_op)
                    # Determine domain.
                    domain = self._determine_domain_from_code(content, onnx_op)

                    # The hip name in the onnx.Custom doc comment is authoritative;
                    # do not let the first RmsNormOp::create in the file override it
                    # (otherwise Skip would be mis-labeled as hip.rmsnorm).
                    derived_hip_op = self._derive_hip_op_from_class_name(class_name)
                    if derived_hip_op is not None:
                        hip_op = derived_hip_op

                    mapping = self._create_mapping(
                        onnx_op, hip_op, file_path, op_type, class_name, domain
                    )
                    if mapping:
                        mappings.append(mapping)

            # Mode 2: mappings discovered via RewritePattern("onnx.<Op>", ...).
            # Each occurrence is scoped to its OWN struct (a single onnx_op
            # may legitimately have multiple registrations in one file, e.g.
            # ConstantOfShapeFold + ConstantOfShapeDynamic both register
            # "onnx.ConstantOfShape" -- one folds at compile time and emits
            # no hip op, the other lowers to hip.where). End-of-parse dedup
            # by (onnx_op, domain, hip_op) collapses duplicates safely; the
            # fold variant simply yields no hip_op and gets skipped.
            for rp_match in re.finditer(r'RewritePattern\("onnx\.(\w+)"', content):
                onnx_op = rp_match.group(1)
                if onnx_op == "Custom":
                    continue

                # Restrict to the struct body containing THIS specific
                # RewritePattern position, not the file's first one.
                scoped = (
                    self._scope_to_rewrite_pattern_body(
                        content, onnx_op, start_pos=rp_match.start()
                    )
                    or content
                )

                hip_op = self._extract_hip_op_from_code(scoped)
                class_name = self._extract_hip_class_name(scoped, onnx_op)
                domain = self._determine_domain_from_code(scoped, onnx_op)
                if hip_op:
                    mapping = self._create_mapping(
                        onnx_op, hip_op, file_path, "standard", class_name, domain
                    )
                    if mapping:
                        mappings.append(mapping)
                else:
                    # No hip.* op produced; check whether the struct lowers
                    # to standard MLIR dialect ops (tensor.* / arith.* /
                    # memref.*). Such conversions are handled by MLIR's
                    # standard bufferization + lowering passes and need no
                    # HIP runtime wrapper, so we emit a synthetic
                    # `tensor.<X>` mapping. Downstream classification
                    # already treats any `tensor.*` hip_op as
                    # `COMPILE_TIME_TENSOR_OP`.
                    td_op = self._extract_std_dialect_op(scoped)
                    if td_op:
                        mapping = self._create_mapping(
                            onnx_op,
                            f"tensor.{td_op}",
                            file_path,
                            "compile_time",
                            "StdMlirLowering",
                            domain,
                        )
                        if mapping:
                            mappings.append(mapping)

            # Mode 3: onnx.Custom ops dispatched via function_name == "<Op>".
            custom_patterns = re.findall(
                r'function_name["\']?\s*==\s*["\'](\w+)["\'].*?hip\.([^"\';\s]+)',
                content,
                re.DOTALL,
            )
            for custom_op, hip_op in custom_patterns:
                # Determine domain dynamically.
                domain = self._determine_domain_from_code(content, custom_op)
                class_name = self._extract_hip_class_name_for_op(content, custom_op)
                mapping = {
                    "onnx_op": custom_op,
                    "onnx_domain": domain,
                    "hip_op": f"hip.{hip_op}",
                    "class_name": class_name,
                    "file_name": file_name,
                    "type": "custom",
                }
                if mapping not in mappings:
                    mappings.append(mapping)

            # Mode 3b: infer the Custom op name from the file name. Only triggered
            # when Mode 3 (function_name) found nothing in this file.
            if 'RewritePattern("onnx.Custom"' in content and not custom_patterns:
                inferred_op = self._infer_custom_op_from_filename(file_name)
                if inferred_op:
                    # Avoid producing a duplicate mapping for this op.
                    if not any(m["onnx_op"] == inferred_op for m in mappings):
                        hip_op = self._extract_hip_op_from_code(content)
                        class_name = self._extract_hip_class_name(content, inferred_op)
                        domain = self._determine_domain_from_code(content, inferred_op)
                        if hip_op:
                            derived_hip_op = self._derive_hip_op_from_class_name(
                                class_name
                            )
                            if derived_hip_op is not None:
                                hip_op = derived_hip_op
                            mapping = {
                                "onnx_op": inferred_op,
                                "onnx_domain": domain,
                                "hip_op": f"hip.{hip_op}",
                                "class_name": class_name,
                                "file_name": file_name,
                                "type": "custom",
                            }
                            if mapping not in mappings:
                                mappings.append(mapping)

            # Mode 4: tensor.* ops -- dynamic extraction from /// doc comments.
            tensor_ops = self._extract_tensor_ops_dynamic(content, file_path)
            mappings.extend(tensor_ops)

        except Exception as e:
            print(f"  Error parsing {file_path.name}: {e}")

        return mappings

    def _create_mapping(
        self,
        onnx_op: str,
        hip_op: str,
        file_path: Path,
        op_type: str = "standard",
        class_name: str = None,
        domain: str = None,
    ) -> Dict:
        """Construct a mapping dictionary."""
        if domain is None:
            domain = self._determine_domain_from_code("", onnx_op)

        mapping = {
            "onnx_op": onnx_op,
            "onnx_domain": domain,
            "hip_op": f"hip.{hip_op}"
            if not hip_op.startswith("hip.") and not hip_op.startswith("tensor.")
            else hip_op,
            "class_name": class_name or "",
            "file_name": file_path.name,
            "type": op_type,
        }
        return mapping

    def _determine_domain_from_code(self, content: str, onnx_op: str) -> str:
        """Determine the ONNX op domain from source code.

        Callers pass one of two kinds of `content`:
        1) Scoped to a single struct's matchAndRewrite body (from Mode 2).
        2) The entire .cpp file text (from Mode 1 / Mode 3).

        A naive whole-file heuristic is wrong on mixed-domain files:
        e.g. NormConversion.cpp contains both SkipSimplifiedLayerNormToHip
        (requires com.microsoft) and LayerNormToHip (onnx domain). A check
        like "file contains the substring 'com.microsoft'" would mis-label
        onnx.LayerNormalization as com.microsoft.
        """
        # Hard-coded whitelist for known ops (preserves historical behavior).
        if onnx_op == "SimplifiedLayerNormalization":
            return "onnx"
        if onnx_op == "SkipSimplifiedLayerNormalization":
            return "com.microsoft"
        if onnx_op.startswith("Skip"):
            return "com.microsoft"

        # Strong signal: an explicit `... != "com.microsoft"` check appears
        # within scope (or anywhere in the file when called with whole-file content).
        if re.search(
            r'domain_name["\']?\s*\)?\s*[!=]=\s*"com\.microsoft"',
            content,
        ):
            return "com.microsoft"

        # Weak signal: the literal "com.microsoft" string co-occurs with the
        # onnx_op literal anywhere in `content`. This is the main signal for
        # Mode 3 (Custom + function_name): MS custom-op files typically have
        # `getAttrOfType<...>("domain_name")` followed several lines later by
        # `!= "com.microsoft"`. The whitespace/intervening chars defeat the
        # strict regex, but the substring co-occurrence is reliable.
        # Mode 2 is safe here: it passes only one struct's scoped body, and a
        # non-MS struct's body does not contain the "com.microsoft" literal.
        if "com.microsoft" in content and onnx_op in content:
            return "com.microsoft"

        return "onnx"

    def _scope_to_rewrite_pattern_body(
        self, content: str, onnx_op: str, start_pos: int = 0
    ) -> Optional[str]:
        """Scope to the matchAndRewrite body of the struct that registers
        `RewritePattern("onnx.<onnx_op>", ...)`. Returns None when not found;
        the caller then falls back to the whole-file content.

        `start_pos` is a hint: locate the struct whose RewritePattern occurs
        at or AFTER this offset. Defaults to 0 (find the first one). The
        caller passes the actual RewritePattern match start when multiple
        structs register the same onnx_op in one file (e.g.
        ConstantOfShapeFold + ConstantOfShapeDynamic) so each gets scoped to
        its own body.

        Steps:
        1) Locate the struct that contains the RewritePattern("onnx.<op>", ...)
           ctor at or after start_pos.
        2) Slice out that struct's body. Prefer the out-of-class
           `StructName::matchAndRewrite { ... }` form; otherwise extract the
           inline struct body via balanced-brace traversal.
        """
        ctor_pat = re.compile(
            r'struct\s+(\w+)\b[^{}]*?\{[^{}]*?RewritePattern\s*\(\s*"onnx\.'
            + re.escape(onnx_op)
            + r'"',
            re.DOTALL,
        )
        # Iterate matches and pick the one whose ctor RewritePattern position
        # is >= start_pos. This lets the caller disambiguate multiple structs
        # registering the same onnx_op (Fold + Dynamic, etc.).
        m = None
        for cand in ctor_pat.finditer(content):
            if cand.end() >= start_pos:
                m = cand
                break
        if not m:
            return None
        struct_name = m.group(1)
        struct_start = m.start()

        # Pattern A: out-of-class definition, e.g.
        #     mlir::LogicalResult LayerNormToHip::matchAndRewrite(...) { ... }
        # (NormConversion.cpp style). Critical: `LayerNormToHip` is a suffix of
        # `SimplifiedLayerNormToHip`, so we MUST use the negative lookbehind
        # (?<!\w) to force a word-boundary start; otherwise the regex matches
        # the longer sibling struct's matchAndRewrite and slices the wrong body.
        body_match = re.search(
            r"(?<!\w)"
            + re.escape(struct_name)
            + r"::matchAndRewrite"
            + r"[\s\S]*?"
            + r"(?="
            + r"\n(?!"
            + re.escape(struct_name)
            + r"\b)\w+::matchAndRewrite"
            + r"|\n[\w:]*populate\w*"
            + r"|\Z)",
            content,
        )
        if body_match:
            return content[struct_start : body_match.end()]

        # Pattern B: matchAndRewrite is defined INLINE inside the struct
        # (SliceConversion.cpp style). Walk balanced braces from the struct's
        # opening `{` to find the matching closing `}`, then return the
        # full struct body. Truncate-by-window is unsafe here because inline
        # bodies routinely exceed any fixed window (Slice is >4 KB).
        brace_pos = content.find("{", struct_start)
        if brace_pos >= 0:
            depth = 0
            i = brace_pos
            n = len(content)
            while i < n:
                ch = content[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        # Include the closing brace and optional trailing `;`.
                        end = i + 1
                        if end < n and content[end] == ";":
                            end += 1
                        return content[struct_start:end]
                i += 1
            # Unbalanced braces (malformed source): fall through to window.

        # Last-resort window when nothing else matches.
        return content[struct_start : struct_start + 4000]

    def _extract_hip_class_name(self, content: str, onnx_op: str) -> str:
        """Extract the HIP C++ class name from source code."""
        # Try the mlir::hip::XxxOp::create pattern first.
        hip_classes = re.findall(r"mlir::hip::(\w+Op)::create", content)
        if hip_classes:
            return hip_classes[0]

        # Try the MiopenXxxOp pattern.
        miopen_classes = re.findall(r"(Miopen\w+Op)", content)
        if miopen_classes:
            return miopen_classes[0]

        # Try the generic XxxOp pattern (e.g. AddOp, MulOp).
        op_classes = re.findall(r"(\w+Op)(?:\s*::|::create)", content)
        if op_classes:
            return op_classes[0]

        # Fallback: derive the CamelCase class name from
        # OperationState(loc, "hip.snake_case"). Patterns like LayerNormToHip
        # that do not use *Op::create rely on this fallback. We accept both:
        #   mlir::OperationState(loc, "hip.xxx")              -- expression form
        #   mlir::OperationState state(loc, "hip.xxx");        -- declaration form
        hip_ops = re.findall(
            r'OperationState(?:\s+\w+)?\s*\(\s*[^,]+,\s*"hip\.([^"]+)"', content
        )
        if hip_ops:
            return "".join(w.capitalize() for w in hip_ops[0].split("_")) + "Op"

        return ""

    def _extract_hip_class_name_for_op(self, content: str, onnx_op: str) -> str:
        """Extract the HIP C++ class name for a specific onnx_op."""
        # Mode 0: known onnx_op -> C++ struct abbreviation (handles cases where
        # Skip... names do not match the struct via the generic regex).
        struct_hint = self._STRUCT_NAME_BY_ONNX_OP.get(onnx_op)
        if struct_hint:
            pattern = rf"struct\s+{re.escape(struct_hint)}\b.*?(?=struct\s+\w+ToHip|void\s+mlir::hip::populate|\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                struct_content = match.group(0)
                hip_classes = re.findall(r"mlir::hip::(\w+Op)::create", struct_content)
                if hip_classes:
                    return hip_classes[0]
                hip_ops = re.findall(
                    r'OperationState(?:\s+\w+)?\s*\(\s*[^,]+,\s*"hip\.([^"]+)"',
                    struct_content,
                )
                if hip_ops:
                    hip_op = hip_ops[0]
                    return (
                        "".join(word.capitalize() for word in hip_op.split("_")) + "Op"
                    )

        # Mode 1: locate the struct containing this op.
        # For SkipSimplifiedLayerNormalization look for SkipSimplifiedLayerNormToHip.
        pattern = rf"struct\s+(\w*{re.escape(onnx_op)}\w*ToHip).*?(?=struct\s+\w+ToHip|void\s+mlir::hip::populate|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            struct_name = match.group(1)

            # Then find the matchAndRewrite definition for that struct
            # (the create call usually lives there).
            fn_pat = (
                rf"{re.escape(struct_name)}::matchAndRewrite"
                rf"[\s\S]*?(?=\n\w+::matchAndRewrite|\nvoid\s+mlir::hip::populate|\Z)"
            )
            fn_match = re.search(fn_pat, content, re.DOTALL)
            if fn_match:
                fn_content = fn_match.group(0)
                hip_classes = re.findall(r"mlir::hip::(\w+Op)::create", fn_content)
                if hip_classes:
                    return hip_classes[0]
                hip_ops = re.findall(
                    r'OperationState(?:\s+\w+)?\s*\(\s*[^,]+,\s*"hip\.([^"]+)"',
                    fn_content,
                )
                if hip_ops:
                    hip_op = hip_ops[0]
                    class_name = (
                        "".join(word.capitalize() for word in hip_op.split("_")) + "Op"
                    )
                    return class_name

        # Fall back to the generic extractor when scoped lookups fail.
        return self._extract_hip_class_name(content, onnx_op)

    def _extract_std_dialect_op(self, content: str) -> Optional[str]:
        """Return a representative standard-MLIR-dialect op name produced by
        a conversion that does NOT call any hip.* op. Used to classify
        compile-time-foldable conversions (e.g. ConstantOfShape -> tensor.splat,
        Reshape -> tensor.collapse_shape, Shape -> tensor.from_elements).

        Picks the LAST `[mlir::](tensor|arith|memref)::<X>Op::create` call in
        the scoped struct body -- typically the one closest to
        `rewriter.replaceOp(...)`, i.e. the "output" op of the lowering.

        The `mlir::` qualifier is optional because conversion files often
        write bare `arith::ConstantOp::create` etc. when `using namespace
        mlir;` or `namespace mlir::hip { ... }` is in scope (ShapeConversion.cpp).

        Returns the op name in snake_case (SplatOp -> splat,
        FromElementsOp -> from_elements); caller prefixes with `tensor.`
        to trigger COMPILE_TIME_TENSOR_OP downstream.
        """
        matches = re.findall(
            r"(?:mlir::)?(?:tensor|arith|memref)::(\w+)Op::create",
            content,
        )
        if not matches:
            return None
        cls = matches[-1]
        # CamelCase -> snake_case (handles ConstantIndexOp, SplatOp, etc.)
        snake = re.sub(r"(?<!^)([A-Z])", r"_\1", cls).lower()
        return snake

    def _extract_hip_op_from_code(self, content: str) -> str:
        """Extract the HIP op mnemonic from source code."""
        # Prefer derivation from the HIP class name (more accurate).
        hip_classes = re.findall(r"mlir::hip::(\w+Op)::create", content)
        if hip_classes:
            class_name = hip_classes[0]
            from_td = self._derive_hip_op_from_class_name(class_name)
            if from_td is not None:
                return from_td
            op_name = class_name.replace("Op", "").lower()
            return op_name

        # Next, parse OperationState (accepts the OperationState state(loc, "hip.xxx") declaration form).
        hip_ops = re.findall(
            r'OperationState(?:\s+\w+)?\s*\(\s*[^,]+,\s*"hip\.([^"]+)"', content
        )
        if hip_ops:
            return hip_ops[0]

        # Last resort: pick from any inline comment / string mentioning hip.xxx.
        hip_ops = re.findall(r"hip\.(\w+(?:\.\w+)?)", content)
        if hip_ops:
            return hip_ops[0]

        return None

    def _derive_hip_op_from_class_name(self, class_name: str) -> Optional[str]:
        """Look up the hip.* mnemonic for a mlir::hip::XxxOp C++ class name
        from the HipOps.td TableGen mapping."""
        if not class_name:
            return None
        mnemonic = self._cpp_class_to_mnemonic.get(class_name)
        if mnemonic is not None:
            return mnemonic
        return None

    def _infer_custom_op_from_filename(self, file_name: str) -> str:
        """Infer a Custom op name from the conversion file name."""
        base_name = file_name.replace("Conversion.cpp", "")

        # Skip files that are already handled by Mode 3 via function_name.
        # E.g. GqaConversion.cpp is already picked up via function_name="GroupQueryAttention".
        if base_name in ["Gqa", "Norm"]:
            return None

        # Otherwise return the base name as-is and let downstream code resolve it.
        # We deliberately avoid hardcoded name mappings.
        return base_name if base_name else None

    def _extract_tensor_ops_dynamic(self, content: str, file_path: Path) -> List[Dict]:
        """Extract onnx -> tensor.* mappings from doc-comment lines.

        Older implementations used `onnx\\.Op.*?tensor\\.(...)` with DOTALL,
        which silently spanned function boundaries inside one file: e.g. the
        second `onnx.Unsqueeze` line (the RewritePattern reference) followed
        by `.*?` would extend down to a later `/// onnx.Squeeze -> tensor.collapse_shape`
        comment, mis-labeling Unsqueeze as collapse_shape. Plain string
        literals containing onnx.Constant also triggered false matches.
        Therefore we only honor tensor.xxx that appears on the SAME line as
        a `/// onnx.X -> ...` doc comment.
        """
        mappings: List[Dict] = []
        line_pat = re.compile(
            r"^\s*///\s*onnx\.(\w+)\s*->\s*(.+?)\s*$",
        )
        tensor_pat = re.compile(r"tensor\.(\w+(?:\.\w+)*)")

        for raw_line in content.splitlines():
            m = line_pat.match(raw_line)
            if not m:
                continue
            onnx_op, rhs = m.group(1), m.group(2)
            if "tensor." not in rhs:
                continue
            seen: Set[str] = set()
            for tm in tensor_pat.finditer(rhs):
                tensor_op = tm.group(1)
                if tensor_op in seen:
                    continue
                seen.add(tensor_op)
                mapping = {
                    "onnx_op": onnx_op,
                    "onnx_domain": "onnx",
                    "hip_op": f"tensor.{tensor_op}",
                    "class_name": "",
                    "file_name": file_path.name,
                    "type": "tensor",
                }
                if mapping not in mappings:
                    mappings.append(mapping)

        return mappings


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python step2_1_onnx_to_hip_parser.py <conversion_dir> [output_dir] [HipOps.td]"
        )
        sys.exit(1)

    conversion_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(conversion_dir).parent)
    hip_td = sys.argv[3] if len(sys.argv) > 3 else None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Parsing OnnxToHip conversions (v5 - HipOps.td mnemonics + Norm fixes)...")
    print(f"Conversion directory: {conversion_dir}")
    print(f"Output directory: {output_dir}\n")

    parser = OnnxToHipParser(conversion_dir, hip_ops_td=hip_td)
    if parser._hip_td_path:
        n = len(parser._cpp_class_to_mnemonic)
        print(f"HipOps.td: {parser._hip_td_path} ({n} C++ op classes -> mnemonic)\n")
    result = parser.parse()

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"  Total mappings: {result['statistics']['total_mappings']}")
    print(f"  Unique ONNX ops: {result['statistics']['unique_onnx_ops']}")
    print(f"  Unique domains: {result['statistics']['unique_domains']}")
    print(f"  Unique Hip ops: {result['statistics']['unique_hip_ops']}")
    print(f"{'=' * 80}\n")

    json_path = Path(output_dir) / "step2_1_onnx_to_hip_mappings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Mappings saved: {json_path}")

    md_path = Path(output_dir) / "step2_1_onnx_to_hip_mappings.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(result))
    print(f"[OK] Markdown report saved: {md_path}")


def generate_markdown_report(result: Dict) -> str:
    """Render the ONNX->HIP mapping table as a Markdown report."""
    report = []

    report.append(
        "# ONNX to Hip Conversion Mappings (v5 - Fixed SkipSimplifiedLayerNormalization & Unsqueeze)\n\n"
    )

    report.append("## Summary\n\n")
    stats = result["statistics"]
    report.append(f"- **Total Mappings**: {stats['total_mappings']}\n")
    report.append(f"- **Unique ONNX Operations**: {stats['unique_onnx_ops']}\n")
    report.append(f"- **Unique Domains**: {stats['unique_domains']}\n")
    report.append(f"- **Unique Hip Operations**: {stats['unique_hip_ops']}\n\n")

    report.append("## Mappings by Domain\n\n")

    domains = {}
    for mapping in result["mappings"]:
        domain = mapping["onnx_domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(mapping)

    for domain in sorted(domains.keys()):
        report.append(f"### Domain: {domain}\n\n")
        report.append("| ONNX Op | Hip Op | Class Name | File | Type |\n")
        report.append("|---|---|---|---|---|\n")

        for mapping in sorted(domains[domain], key=lambda x: x["onnx_op"]):
            onnx_op = mapping["onnx_op"]
            hip_op = mapping["hip_op"]
            class_name = mapping.get("class_name", "")
            file_name = mapping["file_name"]
            op_type = mapping["type"]
            report.append(
                f"| {onnx_op} | {hip_op} | {class_name} | {file_name} | {op_type} |\n"
            )

        report.append("\n")

    return "".join(report)


if __name__ == "__main__":
    main()
