#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Emit a C header embedding a compiled HSACO as a byte array plus a `_size`
# constant. Used by lib/Runtime/CMakeLists.txt to turn the ck_dsl-generated
# `.hip` kernels (compiled to a gfx-specific HSACO via `hipcc --genco`) into
# the `kCkDsl...Hsaco[]` / `kCkDsl...Hsaco_size` symbols the ck_dsl runtime
# shims include. The HSACO itself is a build artifact and is never committed;
# only the ck_dsl-generated `.hip` source lives in the tree.
#
import argparse
import sys


def main(argv):
    p = argparse.ArgumentParser(description="Embed an HSACO as a C byte array.")
    p.add_argument("file", help="input .hsaco file")
    p.add_argument("--var", required=True, help="C variable base name")
    p.add_argument("--output", required=True, help="output header path")
    p.add_argument("--column", type=int, default=12)
    args = p.parse_args(argv)

    with open(args.file, "rb") as f:
        data = f.read()

    with open(args.output, "w") as out:
        out.write("/*\n")
        out.write(" * Copyright (C) 2026 Advanced Micro Devices, Inc. "
                  "All rights reserved.\n")
        out.write(" * Licensed under the MIT License.\n")
        out.write(" */\n")
        out.write("// AUTO-GENERATED at build time from a ck_dsl-generated .hip "
                  "kernel.\n// Do not edit and do not commit; regenerate via the "
                  "EP build.\n")
        out.write("#pragma once\n\n")
        out.write(f"static const unsigned char {args.var}[] = {{\n")
        for i in range(0, len(data), args.column):
            chunk = data[i : i + args.column]
            out.write("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        out.write("};\n")
        out.write(f"static const unsigned long {args.var}_size = {len(data)};\n")


if __name__ == "__main__":
    main(sys.argv[1:])
