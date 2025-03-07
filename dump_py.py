#
# Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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
import sys


def read_py_file(file):
    ret = []
    with open(file, "r") as file:
        for line in file.readlines():
            for c in line:
                if c == "#":
                    break
                ret.append(ord(c))
    return ret


def main():
    py_file = sys.argv[1]
    py_content = read_py_file(py_file)

    output_str = ""
    for b in py_content:
        output_str += str(b)
        output_str += ","
    output_str = output_str[:-1]

    print(output_str)


if __name__ == "__main__":
    main()
