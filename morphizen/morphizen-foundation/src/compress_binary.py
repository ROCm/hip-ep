#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import zlib
import sys
import os
import ast
from pathlib import Path


def to_compressed_byte(filename):
    file = open(filename, "rb")
    binary_data = file.read()
    origin_size = len(binary_data)
    compressed_data = zlib.compress(binary_data)
    ret_str = ",".join(f"0x{byte:02x}" for byte in compressed_data)

    return ret_str, len(compressed_data), origin_size


def to_byte(filename):
    file = open(filename, "rb")
    binary_data = file.read()
    origin_size = len(binary_data)
    ret_str = ",".join(f"0x{byte:02x}" for byte in binary_data)
    # 0 for uncompressed
    return ret_str, 0, origin_size


def get_compression_info(size, support_compression):
    if not support_compression:
        return ""
    else:
        return str(size) + ", "


def generate_map(embedded_resource_file, support_compression):
    meta_info_list = []
    if os.path.isfile(embedded_resource_file):
        print(
            f"-- load meta info from {embedded_resource_file} with {support_compression}"
        )
        f = open(embedded_resource_file, "r", encoding="utf-8")
        meta_info_list = ast.literal_eval(f.read())
    else:
        print(
            "-- embbed resource file not found, please set -DMORPHIZEN_EMBEDDED_RESOURCE_PATH=<>"
        )
    data_str = ""
    map_str = "static std::unordered_map<std::string, CompressionInfo> binary_map = {"

    for meta_info in meta_info_list:
        normalized_rel_path = (
            meta_info["name"].replace("/", os.sep).replace("\\", os.sep)
        )
        filename = os.path.basename(normalized_rel_path)
        compression = meta_info["compression"] and support_compression
        print(f"-- add binary file {filename} with compression = {compression}")
        suffix = Path(filename).suffix
        variable_name = "_" + filename.split(suffix)[0].replace(".", "_")
        path = Path(embedded_resource_file) / ".." / normalized_rel_path
        path = os.path.normpath(path)
        if not os.path.exists(path):
            print(f"-- {filename} does not exist")
            continue
        byte_str = ""
        compressed_size = 0
        origin_size = 0
        byte_str, compressed_size, origin_size = (
            to_compressed_byte(path) if compression else to_byte(path)
        )

        data_str += f"static const uint8_t {variable_name}[] = "
        data_str += "{" + byte_str + "};\n"

        basename = '"' + filename + '"'
        info = (
            "CompressionInfo("
            + variable_name
            + ", "
            + get_compression_info(compressed_size, support_compression)
            + str(origin_size)
            + ")"
        )
        map_str += "{" + basename + "," + info + "},"
    map_str += "};"
    return data_str + map_str


def main():
    support_compression = sys.argv[1] == "1"
    h_inc = sys.argv[2]
    embedded_resource_file = ""
    if len(sys.argv) > 3:
        embedded_resource_file = sys.argv[3]
    with open(h_inc, "w") as f:
        f.write(generate_map(embedded_resource_file, support_compression))


if __name__ == "__main__":
    main()
