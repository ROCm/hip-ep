#!/usr/bin/env python3
"""
Wrapper around xxd.py that adds a size variable using sizeof.

Usage: xxd-with-size.py --var VAR_NAME --output OUTPUT_FILE INPUT_FILE
"""
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="xxd with size variable")
    parser.add_argument("input_file", help="Input file to embed")
    parser.add_argument("--var", required=True, help="Variable name (e.g., petite_boot_data)")
    parser.add_argument("--output", required=True, help="Output C++ file")
    parser.add_argument("--column", type=int, default=16, help="Bytes per line")
    args = parser.parse_args()

    # Call original xxd.py
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xxd_script = os.path.join(script_dir, "xxd.py")

    subprocess.run([
        sys.executable, xxd_script,
        args.input_file,
        "--var", args.var,
        "--output", args.output,
        "--column", str(args.column)
    ], check=True)

    # Read generated file
    with open(args.output, 'r') as f:
        content = f.read()

    # Derive size variable name
    size_var = args.var.replace('_data', '_size')

    # Replace "static const" with "extern \"C\" const" and add size variable
    content = content.replace('static const unsigned char', 'extern "C" const unsigned char')

    # Add header and size variable
    final_content = '#include <cstddef>\n\n' + content
    final_content += f'\nextern "C" const size_t {size_var} = sizeof({args.var}) - 1;\n'

    # Write back
    with open(args.output, 'w') as f:
        f.write(final_content)

if __name__ == '__main__':
    main()
