#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import zlib
import sys
import os
import glob
from pathlib import Path


def to_compressed_byte(filename):
    file = open(filename, "rb")
    binary_data = file.read()
    origin_size = len(binary_data)
    compressed_data = zlib.compress(binary_data)
    ret_str = ""
    for byte in compressed_data:
        ret_str += str(byte) + ","
    ret_str = ret_str[:-1]
    return ret_str, len(compressed_data), origin_size


def to_byte(filename):
    file = open(filename, "rb")
    binary_data = file.read()
    ret_str = ""
    for byte in binary_data:
        ret_str += str(byte) + ","
    ret_str = ret_str[:-1]
    # 0 for uncompressed
    return ret_str, 0, binary_data


def generate_map(binary_root_path):
    data_str = ""
    map_str = "static std::unordered_map<std::string, CompressionInfo> binary_map = {"
    file_list = []
    # probably bundleing xclbin is disabled.
    if binary_root_path != "":
        print(f"-- search binary files in {binary_root_path}")
        # there is also some python scripts and md files.
        # treat it as a white list for now.
        file_list = glob.glob(
            str(Path(binary_root_path) / "**" / "*.*bin"), recursive=True
        )
    else:
        print("-- no binary path is not specified, please set -DVAIP_XCLBIN_PATH=<>")

    for file in file_list:
        if os.path.isfile(file):
            print(f"-- add binary file {file}")
            suffix = Path(file).suffix
            variable_name = "_" + os.path.basename(file).split(suffix)[0].replace(
                ".", "_"
            )
            # to do, need change at create tar from mem
            # byte_str, compressed_size, origin_size = (
            #     to_compressed_byte(file) if suffix == ".xclbin" else to_byte(file)
            # )
            byte_str, compressed_size, origin_size = to_compressed_byte(file)
            data_str += f"static const uint8_t {variable_name}[] = "
            data_str += "{" + byte_str + "};\n"

            basename = '"' + os.path.basename(file) + '"'
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
    binary_root_path = ""
    if len(sys.argv) > 2:
        binary_root_path = sys.argv[2]
    with open(h_inc, "w") as f:
        f.write(generate_map(binary_root_path))


if __name__ == "__main__":
    main()
