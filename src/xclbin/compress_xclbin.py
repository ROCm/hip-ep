#
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
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


def generate_map(path):
    data_str = ""
    map_str = "static std::unordered_map<std::string, CompressionInfo> xclbin_map = {"
    file_list = []
    if path != "":
        file_list = glob.glob(str(Path(path) / "**" / "*.xclbin"), recursive=True)
    for file in file_list:
        if file:
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
    file_list = ""
    if len(sys.argv) > 1:
        file_list = sys.argv[1]
    print(generate_map(file_list))


if __name__ == "__main__":
    main()
