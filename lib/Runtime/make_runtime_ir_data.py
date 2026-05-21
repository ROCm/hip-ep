# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Converts runtime.bc to runtime_ir_data.cpp.
#
# Equivalent to the CMake two-step pipeline in lib/Target/LLVM/CMakeLists.txt:
#   1. xxd.py --var runtime_bc_data  (generates static array)
#   2. Replace `static const unsigned char` → `extern "C" const unsigned char`
#   3. Append `extern "C" const size_t runtime_bc_data_size = <N>;`
#
# Usage: make_runtime_ir_data.py <input.bc> <output.cpp> [--column N]

import argparse
import os
import sys


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input bitcode file (.bc)")
    parser.add_argument("output", help="Output C++ source file")
    parser.add_argument("--column", type=int, default=16,
                        help="Bytes per line in hex dump (default: 16)")
    args = parser.parse_args(args=argv)

    with open(args.input, "rb") as f:
        data = f.read()

    bc_size = len(data)

    with open(args.output, "w") as out:
        out.write('extern "C" const unsigned char runtime_bc_data[] = {\n')
        col = args.column
        for i in range(0, len(data), col):
            chunk = data[i:i + col]
            hex_chunk = ",".join(f"0x{b:02x}" for b in chunk)
            ascii_chunk = "".join(
                chr(b) if 32 <= b < 127 and b not in (ord("*"), ord("/"), ord("\\")) else "."
                for b in chunk
            )
            out.write(f"/*{i:08x} */  {hex_chunk:<{col * 3}}, /* {ascii_chunk} */\n")
        out.write("0x00};\n")
        out.write(f'\nextern "C" const size_t runtime_bc_data_size = {bc_size};\n')


if __name__ == "__main__":
    main(sys.argv[1:])
