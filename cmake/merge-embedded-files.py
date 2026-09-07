#!/usr/bin/env python3
"""
Merge multiple xxd-generated C arrays into a single file with size variables.

Usage:
  merge-embedded-files.py output.cpp \\
    cpp_file1:source_file1:size_var1 \\
    cpp_file2:source_file2:size_var2 ...

Example:
  merge-embedded-files.py ChezBootEmbedded.cpp \\
    ChezBootPetite.cpp:petite.boot:petite_boot_size \\
    ChezBootScheme.cpp:scheme.boot:scheme_boot_size
"""

import sys
import os

def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    output_file = sys.argv[1]
    inputs = sys.argv[2:]

    merged = '#include <cstddef>\n'

    for input_spec in inputs:
        parts = input_spec.split(':')
        if len(parts) != 3:
            print(f"Error: Invalid input spec '{input_spec}', expected format: cpp_file:source_file:size_var", file=sys.stderr)
            sys.exit(1)

        cpp_file, source_file, size_var = parts

        with open(cpp_file, 'r') as f:
            content = f.read()

        content = content.replace('static const unsigned char', 'extern "C" const unsigned char')
        merged += content

        file_size = os.path.getsize(source_file)
        merged += f'\nextern "C" const size_t {size_var} = {file_size};\n\n'

    with open(output_file, 'w') as f:
        f.write(merged)

if __name__ == '__main__':
    main()
