#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import zlib
import sys
import os
import glob
from pathlib import Path


def compress(filename):
    file = open(filename, "rb")
    binary_data = file.read()
    origin_size = len(binary_data)
    compressed_data = zlib.compress(binary_data)
    compressed_str = ""
    for byte in compressed_data:
        compressed_str += str(byte) + ","
    compressed_str = compressed_str[:-1]
    return compressed_str, len(compressed_data), origin_size


def generate_map(xclbin_root_path):
    data_str = ""
    map_str = "static std::unordered_map<std::string, CompressionInfo> xclbin_map = {"
    file_list = []
    # probably bundleing xclbin is disabled.
    if xclbin_root_path != "":
        print(f"-- search xclbin in {xclbin_root_path}")
        file_list = glob.glob(
            str(Path(xclbin_root_path) / "**" / "*.xclbin"), recursive=True
        )
    else:
        print("-- no xclbin path is not specified, please set -DVAIP_XCLBIN_PATH=<>")

    for file in file_list:
        if file:
            print(f"-- add xclbin file {file}")
            compressed_str, compressed_size, origin_size = compress(file)
            variable_name = "_" + os.path.basename(file).split(".xclbin")[0].replace(
                ".", "_"
            )

            data_str += f"static const uint8_t {variable_name}[] = "
            data_str += "{" + compressed_str + "};\n"

            basename = '"' + os.path.basename(file) + '"'
            compressed_arr = "{" + compressed_str + "}"
            info = (
                "CompressionInfo("
                + variable_name
                + ", "
                + str(compressed_size)
                + ", "
                + str(origin_size)
                + ")"
            )
            map_str += "{" + basename + "," + info + "},"
    map_str += "};"
    return data_str + map_str


def main():
    h_inc = sys.argv[1]
    xclbin_root_path = ""
    if len(sys.argv) > 2:
        xclbin_root_path = sys.argv[2]
    with open(h_inc, "w") as f:
        f.write(generate_map(xclbin_root_path))


if __name__ == "__main__":
    main()
