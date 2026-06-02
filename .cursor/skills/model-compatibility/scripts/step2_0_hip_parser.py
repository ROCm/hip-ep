#!/usr/bin/env python3
"""
Hip Dialect Parser - Step 2 (Improved v8)
Improved inputs/outputs disambiguation: by name, any operand whose name
contains "output" is treated as an output.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import sys

class HipDialectParser:
    """Parse the TableGen-format HipOps.td file."""
    
    def __init__(self, td_file_path: str):
        self.td_file_path = td_file_path
        with open(td_file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
    
    def _extract_balanced_braces(self, start_pos: int) -> Optional[str]:
        """Extract the content between balanced braces starting at start_pos."""
        if start_pos >= len(self.content) or self.content[start_pos] != '{':
            return None
        
        depth = 0
        i = start_pos
        
        while i < len(self.content):
            char = self.content[i]
            
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return self.content[start_pos + 1:i]
            
            i += 1
        
        return None
    
    def parse(self) -> Dict:
        """Parse the TD file and extract every Hip operator definition."""
        ops_info = {}
        
        op_pattern = r'def\s+(Hip_\w+Op)\s*:\s*([^{]+)\s*\{'
        
        for match in re.finditer(op_pattern, self.content, re.DOTALL):
            op_class_name = match.group(1)
            op_base = match.group(2).strip()
            
            start_pos = match.end() - 1
            op_body = self._extract_balanced_braces(start_pos)
            
            if op_body is None:
                continue
            
            mnemonic = self._extract_mnemonic(op_base)
            
            op_info = {
                'class_name': op_class_name,
                'mnemonic': mnemonic,
                'base_class': op_base,
                'summary': self._extract_summary(op_body),
                'description': self._extract_description(op_body),
                'inputs': self._extract_inputs(op_body),
                'outputs': self._extract_outputs(op_body),
                'attributes': self._extract_attributes(op_body),
                'has_custom_format': 'hasCustomAssemblyFormat' in op_body,
            }
            
            ops_info[mnemonic] = op_info
        
        return ops_info
    
    def _extract_mnemonic(self, base_class: str) -> str:
        """Extract the mnemonic string from a base class declaration."""
        match = re.search(r'Hip_(?:Dps)?Op<"([^"]+)"', base_class)
        if match:
            return f"hip.{match.group(1)}"
        return "unknown"
    
    def _extract_summary(self, body: str) -> str:
        """Extract the let summary = "..." field."""
        match = re.search(r'let\s+summary\s*=\s*"([^"]*)"', body)
        return match.group(1) if match else ''
    
    def _extract_description(self, body: str) -> str:
        """Extract the let description = [{ ... }] field."""
        match = re.search(r'let\s+description\s*=\s*\[\{(.*?)\}\]', body, re.DOTALL)
        if match:
            desc = match.group(1).strip()
            desc = re.sub(r'\s+', ' ', desc)
            return desc[:300]
        return ''
    
    def _is_output_operand(self, name: str) -> bool:
        """Return True if the operand name indicates an output (contains "output")."""
        name_lower = name.lower()
        if 'output' in name_lower:
            return True
        return False
    
    def _extract_inputs(self, body: str) -> Dict[str, Dict]:
        """Extract input operands from a combination of assemblyFormat and
        arguments, preserving the original definition order."""
        from collections import OrderedDict
        inputs = OrderedDict()
        
        # First extract type + optionality info from the arguments block.
        args_match = re.search(
            r'let\s+arguments\s*=\s*\(\s*ins(.*?)\s*\)\s*;',
            body,
            re.DOTALL
        )
        
        args_info = {}
        if args_match:
            args_body = args_match.group(1)
            params = self._split_parameters(args_body)
            
            for param in params:
                param = param.strip()
                if not param or param.startswith('//'):
                    continue
                
                # Skip ctx and attribute parameters.
                if 'Hip_ContextType' in param or 'Attr' in param:
                    continue
                
                is_optional = 'Optional' in param
                is_variadic = 'Variadic' in param
                
                # Improved regex: first capture $name, then back-extract the type.
                name_match = re.search(r'\$([A-Za-z_]\w*)', param)
                if not name_match:
                    continue
                param_name = name_match.group(1)
                
                # Extract the type token (between < > or right before the : prefix).
                type_match = re.search(r'([A-Za-z_]\w*)(?:<[^<>]*>)?(?:\s*:\s*\$|>)', param)
                param_type = type_match.group(1) if type_match else 'unknown'
                
                type_match = None  # reset for downstream compatibility
                if param_name:
                    type_match = True  # mark as matched
                
                if type_match:
                    # Skip operands that the name heuristic classifies as outputs.
                    if self._is_output_operand(param_name):
                        continue
                    
                    args_info[param_name] = {
                        'type': param_type,
                        'required': not is_optional,
                        'variadic': is_variadic
                    }
        
        # Then extract order + optionality from assemblyFormat (preserving order).
        assembly_inputs = self._extract_from_assembly_format(body, 'ins')
        
        if assembly_inputs:
            # Merge: use the assemblyFormat order/optionality with the arguments type.
            for name, info in assembly_inputs.items():
                # Skip operands classified as outputs by name.
                if self._is_output_operand(name):
                    continue
                
                if name in args_info:
                    inputs[name] = {
                        'type': args_info[name]['type'],
                        'required': info['required'],
                        'variadic': args_info[name]['variadic']
                    }
                else:
                    inputs[name] = info
        else:
            # No assemblyFormat: fall back to arguments info directly.
            inputs = OrderedDict(args_info)
        
        return inputs
    
    def _extract_outputs(self, body: str) -> Dict[str, Dict]:
        """Extract output operands from a combination of assemblyFormat and
        results, preserving the original definition order."""
        from collections import OrderedDict
        outputs = OrderedDict()
        
        # First extract type + optionality info from the results block.
        results_match = re.search(
            r'let\s+results\s*=\s*\(\s*outs\s+(.*?)\s*\)\s*;',
            body,
            re.DOTALL
        )
        
        results_info = {}
        if results_match:
            results_body = results_match.group(1)
            params = self._split_parameters(results_body)
            
            for param in params:
                param = param.strip()
                if not param or param.startswith('//'):
                    continue
                
                is_variadic = 'Variadic' in param
                is_optional = 'Optional' in param
                
                type_match = re.search(
                    r'(?:Optional<|Variadic<)?([A-Za-z_]\w*(?:<[^>]+>)?)\)?:?\$?([A-Za-z_]\w*)',
                    param
                )
                
                if type_match:
                    param_type = type_match.group(1)
                    param_name = type_match.group(2)
                    
                    results_info[param_name] = {
                        'type': param_type,
                        'required': not is_optional,
                        'variadic': is_variadic
                    }
        
        # Then extract order + optionality from assemblyFormat (preserving order).
        assembly_outputs = self._extract_from_assembly_format(body, 'outs')
        
        if assembly_outputs:
            # Merge: use the assemblyFormat order/optionality with the results type.
            for name, info in assembly_outputs.items():
                if name in results_info:
                    outputs[name] = {
                        'type': results_info[name]['type'],
                        'required': info['required'],
                        'variadic': results_info[name]['variadic']
                    }
                else:
                    outputs[name] = info
        else:
            # No assemblyFormat: fall back to results info directly.
            outputs = OrderedDict(results_info)
        
        # Also extract output operands declared inside the arguments block
        # (some DPS-style ops put output buffers as ins).
        args_match = re.search(
            r'let\s+arguments\s*=\s*\(\s*ins(.*?)\s*\)\s*;',
            body,
            re.DOTALL
        )
        
        if args_match:
            args_body = args_match.group(1)
            params = self._split_parameters(args_body)
            
            for param in params:
                param = param.strip()
                if not param or param.startswith('//'):
                    continue
                
                # Skip ctx and attribute parameters.
                if 'Hip_ContextType' in param or 'Attr' in param:
                    continue
                
                # Only consider operands classified as outputs by name.
                if not self._is_output_operand(param):
                    continue
                
                is_optional = 'Optional' in param
                is_variadic = 'Variadic' in param
                
                # Improved regex: first capture $name, then back-extract the type.
                name_match = re.search(r'\$([A-Za-z_]\w*)', param)
                if not name_match:
                    continue
                param_name = name_match.group(1)
                
                # Extract the type token (between < > or right before the : prefix).
                type_match = re.search(r'([A-Za-z_]\w*)(?:<[^<>]*>)?(?:\s*:\s*\$|>)', param)
                param_type = type_match.group(1) if type_match else 'unknown'
                
                # Skip Variadic outputs (not concrete output buffers).
                if is_variadic:
                    continue
                
                if param_name not in outputs:
                    outputs[param_name] = {
                        'type': param_type,
                        'required': not is_optional,
                        'variadic': is_variadic
                    }
        
        return outputs
    
    def _extract_from_assembly_format(self, body: str, section: str) -> Optional[Dict[str, Dict]]:
        """Extract operands and optionality from the ins(...) or outs(...)
        section of assemblyFormat, preserving the original order."""
        from collections import OrderedDict
        
        # Locate the assemblyFormat block.
        fmt_match = re.search(
            r'let\s+assemblyFormat\s*=\s*\[\{(.*?)\}\]',
            body,
            re.DOTALL
        )
        
        if not fmt_match:
            return None
        
        fmt_body = fmt_match.group(1)
        
        # Find the ins(...) or outs(...) sub-block.
        if section == 'ins':
            pattern = r'`ins`\s*`\(`(.*?)`\)`'
        else:  # outs
            pattern = r'`outs`\s*`\(`(.*?)`\)`'
        
        section_match = re.search(pattern, fmt_body, re.DOTALL)
        if not section_match:
            return None
        
        section_content = section_match.group(1)
        
        # Parse operand names and optionality from the section content.
        # Example: $query ( `,` $key^ )? ( `,` $value^ )?
        operands = OrderedDict()
        
        # Strategy: find all $name references, then mark those that appear
        # inside an ( ... )? group as optional.
        optional_vars = set()
        
        optional_pattern = r'\(\s*[^()]*\$([a-zA-Z_]\w*)[^()]*\s*\)\s*\?'
        for match in re.finditer(optional_pattern, section_content):
            optional_vars.add(match.group(1))
        
        # Now collect all $name references in original order.
        for var_match in re.finditer(r'\$([a-zA-Z_]\w*)', section_content):
            var_name = var_match.group(1)
            
            # Skip names already seen (preserve first-occurrence order).
            if var_name in operands:
                continue
            
            is_optional = var_name in optional_vars
            
            operands[var_name] = {
                'type': 'unknown',  # assemblyFormat carries no type info
                'required': not is_optional,
                'variadic': False
            }
        
        return operands if operands else None
    
    def _extract_attributes(self, body: str) -> Dict[str, Dict]:
        """Extract attributes in original definition order.

        Required vs optional rule:
        - If the attribute carries a default value (via DefaultValuedAttr or
          similar), it is optional.
        - Otherwise it is required.
        """
        from collections import OrderedDict
        attributes = OrderedDict()
        
        # Locate the full arguments block (covers both ins and outs).
        args_match = re.search(
            r'let\s+arguments\s*=\s*\((.*?)\)\s*;',
            body,
            re.DOTALL
        )
        
        if args_match:
            args_body = args_match.group(1)
            params = self._split_parameters(args_body)
            
            for param in params:
                param = param.strip()
                if not param or param.startswith('//'):
                    continue
                
                if 'Attr' not in param:
                    continue
                
                # Check for a default value.
                has_default_value = 'DefaultValuedAttr' in param
                
                # Improved regex: handles DefaultValuedAttr<Type, "default">:$name.
                # First try to match the DefaultValuedAttr<...>:$name shape.
                attr_match = re.search(
                    r'(?:DefaultValuedAttr<)?([A-Za-z_]\w*Attr)(?:<[^>]+>)?[,\s]*:?\$?([A-Za-z_]\w*)',
                    param
                )
                
                # Fall back to a looser match if the strict one fails.
                if not attr_match:
                    # Find any $name attribute name.
                    name_match = re.search(r'\$([A-Za-z_]\w*)', param)
                    if name_match:
                        attr_name = name_match.group(1)
                        # Find the attribute type (between < >).
                        type_match = re.search(r'<([A-Za-z_]\w*Attr)', param)
                        attr_type = type_match.group(1) if type_match else 'UnknownAttr'
                        
                        default_value = None
                        if has_default_value:
                            # Extract the value from DefaultValuedAttr<Type, "value">.
                            default_match = re.search(r'DefaultValuedAttr<[^,]+,\s*"([^"]*)"', param)
                            if default_match:
                                default_value = default_match.group(1)
                        
                        # Optional iff a default value is present.
                        is_optional = has_default_value
                        
                        attributes[attr_name] = {
                            'type': attr_type,
                            'optional': is_optional,
                            'has_default': has_default_value,
                            'default_value': default_value
                        }
                else:
                    attr_type = attr_match.group(1)
                    attr_name = attr_match.group(2)
                    
                    if attr_name:
                        default_value = None
                        if has_default_value:
                            default_match = re.search(r'DefaultValuedAttr<[^,]+,\s*"([^"]*)"', param)
                            if default_match:
                                default_value = default_match.group(1)
                        
                        # Optional iff a default value is present.
                        is_optional = has_default_value
                        
                        attributes[attr_name] = {
                            'type': attr_type,
                            'optional': is_optional,
                            'has_default': has_default_value,
                            'default_value': default_value
                        }
        
        return attributes
    
    def _split_parameters(self, text: str) -> List[str]:
        """Split parameters by top-level commas, respecting nested < > and
        ignoring // line comments."""
        # First strip line comments.
        lines = []
        for line in text.split('\n'):
            # Drop trailing // comment.
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        text = '\n'.join(lines)
        
        params = []
        current = []
        depth = 0
        
        for char in text:
            if char == '<':
                depth += 1
                current.append(char)
            elif char == '>':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                params.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            params.append(''.join(current))
        
        return params


def generate_markdown_report(ops_info: Dict, td_file_path: str) -> str:
    """Render the parsed HIP op inventory as a Markdown report."""
    report = []
    report.append(f"# Hip Dialect Operators Analysis (v8)\n\n")
    report.append(f"**Source**: {td_file_path}\n\n")
    report.append(f"**Note**: Operands with 'output' in their name are classified as outputs\n\n")
    
    report.append(f"## Overview\n\n")
    report.append(f"- **Total Hip Operators**: {len(ops_info)}\n")
    report.append(f"- **Analysis Date**: {__import__('datetime').datetime.now().isoformat()}\n\n")
    
    report.append("## Operator Summary\n\n")
    report.append("| Op Name | Inputs | Outputs | Attributes |\n")
    report.append("|---------|--------|---------|------------|\n")
    
    sorted_ops = sorted(ops_info.items())
    
    for op_name, info in sorted_ops:
        # Format inputs in original order.
        input_list = []
        for param_name in info['inputs'].keys():
            param_info = info['inputs'][param_name]
            if not param_info['required']:
                input_list.append(f"{param_name} (O)")
            else:
                input_list.append(param_name)
        input_str = ', '.join(input_list) if input_list else '-'
        
        # Format outputs in original order.
        output_list = []
        for param_name in info['outputs'].keys():
            param_info = info['outputs'][param_name]
            if not param_info['required']:
                output_list.append(f"{param_name} (O)")
            else:
                output_list.append(param_name)
        output_str = ', '.join(output_list) if output_list else '-'
        
        # Format attributes in original order; mark optional with (O).
        attr_list = []
        for attr_name, attr_info in info['attributes'].items():
            if attr_info['optional']:
                attr_list.append(f"{attr_name} (O)")
            else:
                attr_list.append(attr_name)
        attr_names = ', '.join(attr_list) if attr_list else '-'
        
        summary = info['summary']
        
        # Row layout: Op Name | Inputs | Outputs | Attributes.
        report.append(f"| {op_name} | {input_str} | {output_str} | {attr_names} |\n")
    
    report.append("\n## Operator Details\n\n")
    
    for op_name, info in sorted_ops:
        # Format inputs in original order.
        input_list = []
        for param_name in info['inputs'].keys():
            param_info = info['inputs'][param_name]
            if not param_info['required']:
                input_list.append(f"{param_name} (O)")
            else:
                input_list.append(param_name)
        input_str = ', '.join(input_list) if input_list else '-'
        
        # Format outputs in original order.
        output_list = []
        for param_name in info['outputs'].keys():
            param_info = info['outputs'][param_name]
            if not param_info['required']:
                output_list.append(f"{param_name} (O)")
            else:
                output_list.append(param_name)
        output_str = ', '.join(output_list) if output_list else '-'
        
        # Format attributes (mark optional with (O)).
        attr_list = []
        for attr_name in sorted(info['attributes'].keys()):
            attr_info = info['attributes'][attr_name]
            if attr_info['optional']:
                attr_list.append(f"{attr_name} (O)")
            else:
                attr_list.append(attr_name)
        attr_names = ', '.join(attr_list) if attr_list else '-'
        
        summary = info['summary']
        
        # Render details as a bullet list.
        report.append(f"### {op_name}\n\n")
        report.append(f"**Summary**: {summary}\n\n")
        
        if input_str != '-':
            report.append(f"**Inputs**: {input_str}\n\n")
        
        if output_str != '-':
            report.append(f"**Outputs**: {output_str}\n\n")
        
        if attr_names != '-':
            report.append(f"**Attributes**: {attr_names}\n\n")
        
        report.append("---\n\n")
    
    report.append("\n## Detailed Operator Information\n\n")
    
    for op_name, info in sorted_ops:
        report.append(f"### {op_name}\n\n")
        
        if info['summary']:
            report.append(f"**Summary**: {info['summary']}\n\n")
        
        if info['description']:
            report.append(f"**Description**: {info['description']}\n\n")
        
        # Inputs (preserving original order).
        if info['inputs']:
            report.append("**Inputs**:\n\n")
            report.append("| Name | Type | Required |\n")
            report.append("|------|------|----------|\n")
            
            for param_name, param_info in info['inputs'].items():
                required = "yes" if param_info['required'] else "no"
                report.append(f"| {param_name} | `{param_info['type']}` | {required} |\n")
            report.append("\n")
        
        # Outputs (preserving original order).
        if info['outputs']:
            report.append("**Outputs**:\n\n")
            report.append("| Name | Type | Required |\n")
            report.append("|------|------|----------|\n")
            
            for param_name, param_info in info['outputs'].items():
                required = "yes" if param_info['required'] else "no"
                report.append(f"| {param_name} | `{param_info['type']}` | {required} |\n")
            report.append("\n")
        
        # Attributes (preserving original definition order).
        if info['attributes']:
            report.append("**Attributes**:\n\n")
            report.append("| Name | Type | Optional | Default Value |\n")
            report.append("|------|------|----------|----------------|\n")
            
            for attr_name, attr_info in info['attributes'].items():
                optional = "yes" if attr_info['optional'] else "no"
                default_val = attr_info['default_value'] if attr_info['default_value'] else "-"
                report.append(f"| {attr_name} | `{attr_info['type']}` | {optional} | {default_val} |\n")
            report.append("\n")
    
    return ''.join(report)


def main():
    if len(sys.argv) < 2:
        print("=" * 80)
        print("ERROR: Hip Dialect .td file path is required!")
        print("=" * 80)
        print("\nUsage: python step2_hip_parser.py <td_file_path> [output_dir]")
        print("\nExample:")
        print("  python step2_hip_parser.py /path/to/HIP.td ./output")
        print("\nPlease provide the path to the Hip Dialect .td file.")
        print("Common locations:")
        print("  - MLIR repository: mlir/lib/Dialect/HIP/IR/HIP.td")
        print("  - Or search for HIP.td in your MLIR installation")
        print("=" * 80)
        sys.exit(1)
    
    td_file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(td_file_path).parent)
    
    print(f"Parsing Hip Dialect file: {td_file_path}")
    print(f"Output directory: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    parser = HipDialectParser(td_file_path)
    ops_info = parser.parse()
    
    print(f"Found {len(ops_info)} Hip operators")
    
    # Aggregate statistics.
    total_inputs = sum(len(op['inputs']) for op in ops_info.values())
    total_outputs = sum(len(op['outputs']) for op in ops_info.values())
    total_attrs = sum(len(op['attributes']) for op in ops_info.values())
    
    print(f"  - Total inputs: {total_inputs}")
    print(f"  - Total outputs: {total_outputs}")
    print(f"  - Total attributes: {total_attrs}")
    
    # Render the Markdown report.
    markdown_report = generate_markdown_report(ops_info, td_file_path)
    
    # Save the Markdown report.
    md_path = Path(output_dir) / "step2_hip_ops.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"[OK] Markdown report saved: {md_path}")
    
    # Save the JSON data.
    json_path = Path(output_dir) / "step2_hip_ops.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ops_info, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON data saved: {json_path}")
    
    print("\nParsing complete!")


if __name__ == '__main__':
    main()
