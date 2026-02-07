#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import argparse


def main(argv):
    parser = argparse.ArgumentParser(description="Hex dump utility.")
    parser.add_argument("file", type=str, help="File to hex dump")
    parser.add_argument(
        "--column",
        type=int,
        default=16,
        help="Number of bytes per line in the hex dump (default: 32)",
    )
    parser.add_argument("--var", type=str, required=True, help="Variable name to use in the output")
    parser.add_argument("--output", type=str, required=True, help="output file name")
    args = parser.parse_args(args=argv)

    with open(args.file, "rb") as f:
        data = f.read()
    with open(args.output, "w") as output_fh:
        print(f"static const unsigned char {args.var}[] = {{", file=output_fh)
        for i in range(0, len(data), args.column):
            chunk = data[i : i + args.column]
            hex_chunk = ",".join(f"0x{byte:02x}" for byte in chunk)
            ascii_chunk = "".join(
                chr(byte) if 32 <= byte < 127 and byte not in [ord("*"), ord("/"), ord("\\")] else "." for byte in chunk
            )
            print(
                f"/*{i:08x} */  {hex_chunk:<{args.column * 3}}, /* {ascii_chunk} */",
                file=output_fh,
            )
        print("0x00};", file=output_fh)


if __name__ == "__main__":
    main(sys.argv[1:])
