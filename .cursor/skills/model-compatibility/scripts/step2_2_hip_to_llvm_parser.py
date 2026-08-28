#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Step 2-2: HipToLLVM Lowering Parser (v5 - Handle Nested Brackets)
Extract hip_op -> runtime_func -> file_name mappings.
- Reads runtime symbol constants from HipToLLVMUtils.h.
- hip.* names are aligned with HipOps.td TableGen mnemonics
  (consistent with step2_1).
- Correctly handles nested template angle brackets.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from hip_td_mnemonics import (  # noqa: E402
    default_hip_ops_td_path,
    load_cpp_op_class_to_mnemonic_from_td,
)


class HipToLLVMParser:
    """HipToLLVM parser; reads runtime-symbol constants from source."""

    def __init__(self, conversion_dir: str, hip_ops_td: Optional[str] = None):
        self.conversion_dir = Path(conversion_dir)
        self.constants = {}
        self._cpp_class_to_mnemonic: Dict[str, str] = {}
        self._load_constants()
        td = (
            Path(hip_ops_td)
            if hip_ops_td
            else default_hip_ops_td_path(self.conversion_dir)
        )
        if td.is_file():
            self._cpp_class_to_mnemonic = load_cpp_op_class_to_mnemonic_from_td(td)
        else:
            print(
                f"Warning: HipOps.td not found at {td}; hip op names fall back to CamelCase heuristic."
            )

    def _hip_op_from_pattern_op_class(self, op_class: str) -> str:
        """ConvertOpToLLVMPattern<ReduceSumOp> / memref::AllocOp -> hip.reduce_sum / hip.alloc."""
        op_short = op_class.split("::")[-1] if "::" in op_class else op_class
        mnemonic = self._cpp_class_to_mnemonic.get(op_short)
        if mnemonic is not None:
            return f"hip.{mnemonic}"
        if op_short.endswith("Op"):
            hip_op_name = op_short[: -len("Op")].lower()
        else:
            hip_op_name = op_short.lower()
        return f"hip.{hip_op_name}"

    def _load_constants(self):
        """Load symbolic-name constants declared in HipToLLVMUtils.h."""
        utils_header = self.conversion_dir / "HipToLLVM" / "HipToLLVMUtils.h"

        if not utils_header.exists():
            print(f"Warning: {utils_header} not found")
            return

        try:
            with open(utils_header, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            pattern = r'inline\s+constexpr\s+const\s+char\s+\*(\w+)\s*=\s*"([^"]+)"'
            matches = re.findall(pattern, content)

            for const_name, const_value in matches:
                self.constants[const_name] = const_value

            print(f"Loaded {len(self.constants)} constants from HipToLLVMUtils.h\n")

        except Exception as e:
            print(f"Error loading constants: {e}")

    def _extract_template_args(self, text: str, start_pos: int) -> Tuple[str, int]:
        """Extract template arguments starting at start_pos, honoring nested
        angle brackets. Returns (extracted_content, end_position)."""
        depth = 0
        i = start_pos
        result = []

        while i < len(text):
            char = text[i]
            if char == "<":
                depth += 1
                result.append(char)
            elif char == ">":
                if depth == 0:
                    return "".join(result), i
                depth -= 1
                result.append(char)
            else:
                result.append(char)
            i += 1

        return "".join(result), i

    def parse(self) -> Dict:
        """Parse every lowering file."""
        result = {"mappings": [], "statistics": {}}

        lowering_files = sorted(self.conversion_dir.glob("HipToLLVM/*.cpp"))

        print(f"Found {len(lowering_files)} lowering files\n")

        for file_path in lowering_files:
            print(f"Parsing: {file_path.name}")
            mappings = self._parse_lowering_file(file_path)
            result["mappings"].extend(mappings)

            for mapping in mappings:
                runtime_func = mapping.get("runtime_func", "")
                print(f"  {mapping['hip_op']:30} -> {runtime_func:40}")

        # Deduplicate.
        unique_mappings = {}
        for mapping in result["mappings"]:
            key = (mapping["hip_op"], mapping.get("runtime_func", ""))
            if key not in unique_mappings:
                unique_mappings[key] = mapping

        result["mappings"] = list(unique_mappings.values())

        result["statistics"] = {
            "total_mappings": len(result["mappings"]),
            "total_lowering_files": len(lowering_files),
            "unique_hip_ops": len(set(m["hip_op"] for m in result["mappings"])),
            "unique_runtime_funcs": len(
                set(m.get("runtime_func", "") for m in result["mappings"])
            ),
        }

        return result

    def _parse_lowering_file(self, file_path: Path) -> List[Dict]:
        """Parse a single lowering file and return all mappings."""
        mappings = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_name = file_path.name

            # Mode 1: extract hip_op + runtime function from struct definitions.
            struct_pattern = r"struct\s+(\w+OpLowering)\s*:\s*public\s+ConvertOpToLLVMPattern<(\w+Op)>"
            struct_matches = re.findall(struct_pattern, content)

            for struct_name, op_class in struct_matches:
                hip_op = self._hip_op_from_pattern_op_class(op_class)

                struct_content_pattern = (
                    rf"{struct_name}.*?(?=struct\s+\w+|void\s+mlir::hip::populate|\Z)"
                )
                struct_content_match = re.search(
                    struct_content_pattern, content, re.DOTALL
                )

                if struct_content_match:
                    struct_content = struct_content_match.group(0)

                    func_pattern = r"LLVM::lookupOrCreateFn\s*\(\s*[^,]+,\s*[^,]+,\s*([kA-Za-z_]\w+)"
                    func_matches = re.findall(func_pattern, struct_content)

                    for const_name in func_matches:
                        if const_name in self.constants:
                            runtime_func = self.constants[const_name]
                            mapping = {
                                "hip_op": hip_op,
                                "runtime_func": runtime_func,
                                "file_name": file_name,
                                "type": "struct",
                                "const_name": const_name,
                            }
                            if mapping not in mappings:
                                mappings.append(mapping)

                    template_pattern = (
                        r"lowerMiopenActivation\s*<\s*\w+Op\s*,\s*\w+\s*>\s*\("
                    )
                    if re.search(template_pattern, struct_content):
                        template_func_pattern = r"lowerMiopenActivation\s*\([^)]*\).*?LLVM::lookupOrCreateFn\s*\(\s*[^,]+,\s*[^,]+,\s*([kA-Za-z_]\w+)"
                        template_func_match = re.search(
                            template_func_pattern, content, re.DOTALL
                        )
                        if template_func_match:
                            const_name = template_func_match.group(1)
                            if const_name in self.constants:
                                runtime_func = self.constants[const_name]
                                mapping = {
                                    "hip_op": hip_op,
                                    "runtime_func": runtime_func,
                                    "file_name": file_name,
                                    "type": "template",
                                    "const_name": const_name,
                                }
                                if mapping not in mappings:
                                    mappings.append(mapping)

            # Mode 2: extract template args from patterns.add<...> (nested brackets safe).
            patterns_add_pattern = r"patterns\.add<"
            for match in re.finditer(patterns_add_pattern, content):
                start_pos = match.end()
                template_args_str, end_pos = self._extract_template_args(
                    content, start_pos
                )

                # Split multiple template instances, taking nested brackets into account.
                template_instances = self._parse_template_args(template_args_str)

                for template_name, template_args in template_instances:
                    # Handle ElementwiseOpLowering<OpType, TensorOpEnum>.
                    if template_name == "ElementwiseOpLowering":
                        args = [arg.strip() for arg in template_args.split(",")]
                        if len(args) >= 1:
                            op_type = args[0]
                            hip_op = self._hip_op_from_pattern_op_class(op_type)

                            if "kWrapMiopenOpTensor" in self.constants:
                                runtime_func = self.constants["kWrapMiopenOpTensor"]
                                mapping = {
                                    "hip_op": hip_op,
                                    "runtime_func": runtime_func,
                                    "file_name": file_name,
                                    "type": "patterns_add",
                                    "const_name": "kWrapMiopenOpTensor",
                                }
                                if mapping not in mappings:
                                    mappings.append(mapping)

                    # Handle MiopenBinaryOpLowering<OpType>.
                    elif template_name == "MiopenBinaryOpLowering":
                        args = [arg.strip() for arg in template_args.split(",")]
                        if len(args) >= 1:
                            op_type = args[0]
                            hip_op = self._hip_op_from_pattern_op_class(op_type)

                            insert_pattern = rf"patterns\.insert<{template_name}<{re.escape(op_type)}>\s*>\s*\(\s*[^,]+,\s*([kA-Za-z_]\w+)"
                            insert_match = re.search(insert_pattern, content)
                            if insert_match:
                                const_name = insert_match.group(1)
                                if const_name in self.constants:
                                    runtime_func = self.constants[const_name]
                                    mapping = {
                                        "hip_op": hip_op,
                                        "runtime_func": runtime_func,
                                        "file_name": file_name,
                                        "type": "patterns_insert",
                                        "const_name": const_name,
                                    }
                                    if mapping not in mappings:
                                        mappings.append(mapping)

            # Mode 2b: direct match for patterns.insert<MiopenBinaryOpLowering<OpType>>(..., kConst).
            insert_direct_pat = (
                r"patterns\.insert<\s*MiopenBinaryOpLowering<\s*(\w+Op)\s*>\s*>"
                r"\s*\(\s*[^,]+,\s*([kA-Za-z_]\w+)"
            )
            for op_type, const_name in re.findall(insert_direct_pat, content):
                hip_op = self._hip_op_from_pattern_op_class(op_type)
                if const_name in self.constants:
                    runtime_func = self.constants[const_name]
                    mapping = {
                        "hip_op": hip_op,
                        "runtime_func": runtime_func,
                        "file_name": file_name,
                        "type": "patterns_insert_direct",
                        "const_name": const_name,
                    }
                    if mapping not in mappings:
                        mappings.append(mapping)

            # Mode 3: extract hip_op -> runtime_func from inline comments.
            comment_pattern = r"//\s*hip\.(\w+)\(.*?\)\s*->\s*(\w+)\("
            matches = re.findall(comment_pattern, content, re.DOTALL)
            for hip_op_name, runtime_func in matches:
                hip_op = f"hip.{hip_op_name}"
                mapping = {
                    "hip_op": hip_op,
                    "runtime_func": runtime_func,
                    "file_name": file_name,
                    "type": "comment",
                }
                if not any(m["hip_op"] == hip_op for m in mappings):
                    if mapping not in mappings:
                        mappings.append(mapping)

        except Exception as e:
            print(f"  Error parsing {file_path.name}: {e}")

        return mappings

    def _parse_template_args(self, args_str: str) -> List[Tuple[str, str]]:
        """Parse a template-argument string and return
        [(template_name, template_args), ...].

        Example input:
            "ElementwiseOpLowering<MulOp, kTensorOpMul>, ElementwiseOpLowering<AddOp, kTensorOpAdd>, SubOpLowering"
        Returns:
            [("ElementwiseOpLowering", "MulOp, kTensorOpMul"),
             ("ElementwiseOpLowering", "AddOp, kTensorOpAdd"),
             ("SubOpLowering", "")]
        """
        result = []
        i = 0

        while i < len(args_str):
            # Skip whitespace and commas.
            while i < len(args_str) and args_str[i] in " \t\n,":
                i += 1

            if i >= len(args_str):
                break

            # Extract the template name.
            name_start = i
            while i < len(args_str) and (args_str[i].isalnum() or args_str[i] == "_"):
                i += 1

            template_name = args_str[name_start:i].strip()

            # Check whether template args follow.
            if i < len(args_str) and args_str[i] == "<":
                i += 1  # Skip the opening <.
                template_args, end_pos = self._extract_template_args(args_str, i)
                i = end_pos + 1  # Skip the closing >.
                result.append((template_name, template_args))
            else:
                # No template args attached.
                result.append((template_name, ""))

        return result


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python step2_2_hip_to_llvm_parser.py <conversion_dir> [output_dir] [HipOps.td]"
        )
        sys.exit(1)

    conversion_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(conversion_dir).parent)
    hip_td = sys.argv[3] if len(sys.argv) > 3 else None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Parsing HipToLLVM lowerings (v5 - HipOps.td mnemonics + nested brackets)...")
    print(f"Conversion directory: {conversion_dir}")
    print(f"Output directory: {output_dir}\n")

    parser = HipToLLVMParser(conversion_dir, hip_ops_td=hip_td)
    result = parser.parse()

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"  Total mappings: {result['statistics']['total_mappings']}")
    print(f"  Unique Hip ops: {result['statistics']['unique_hip_ops']}")
    print(f"  Unique Runtime funcs: {result['statistics']['unique_runtime_funcs']}")
    print(f"{'=' * 80}\n")

    json_path = Path(output_dir) / "step2_2_hip_to_llvm_mappings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Mappings saved: {json_path}")

    md_path = Path(output_dir) / "step2_2_hip_to_llvm_mappings.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(result))
    print(f"[OK] Markdown report saved: {md_path}")


def generate_markdown_report(result: Dict) -> str:
    """Render the HIP -> runtime mapping table as a Markdown report."""
    report = []

    report.append("# Hip to LLVM Lowering Mappings (v5 - Handle Nested Brackets)\n\n")

    report.append("## Summary\n\n")
    stats = result["statistics"]
    report.append(f"- **Total Mappings**: {stats['total_mappings']}\n")
    report.append(f"- **Unique Hip Operations**: {stats['unique_hip_ops']}\n")
    report.append(
        f"- **Unique Runtime Functions**: {stats['unique_runtime_funcs']}\n\n"
    )

    report.append("## Mappings by Hip Operation\n\n")

    hip_ops = {}
    for mapping in result["mappings"]:
        hip_op = mapping["hip_op"]
        if hip_op not in hip_ops:
            hip_ops[hip_op] = []
        hip_ops[hip_op].append(mapping)

    report.append("| Hip Op | Runtime Function | File | Type | Const Name |\n")
    report.append("|---|---|---|---|---|\n")

    for hip_op in sorted(hip_ops.keys()):
        for mapping in sorted(hip_ops[hip_op], key=lambda x: x.get("runtime_func", "")):
            runtime_func = mapping.get("runtime_func", "")
            file_name = mapping["file_name"]
            op_type = mapping["type"]
            const_name = mapping.get("const_name", "")
            report.append(
                f"| {hip_op} | {runtime_func} | {file_name} | {op_type} | {const_name} |\n"
            )

    report.append("\n")

    return "".join(report)


if __name__ == "__main__":
    main()
