##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
import sys
import re


def get_symbol(file):
    lines = open(file, "r").readlines()
    symbol_set = set()
    pattern = r"UNDEF.+External.+protobuf@google.+"
    for line in lines:
        match = re.findall(pattern, line)
        if len(match) == 0:
            continue
        match = match[0]
        sep_idx = match.find("|")
        if "Weak" in match[:sep_idx]:
            continue
        match = match[sep_idx:]
        symbol = match.split()[1]
        if "DoNotUse" in symbol:
            continue
        symbol_set.add(symbol)
    ret = ""
    for symbol in symbol_set:
        ret += symbol
        ret += "\n"
    return ret


def replace_template(template_file, output_file, symbol):
    file = open(template_file, "r")
    template = file.read()
    result_str = template.replace("@ALL_DUMPED_SYMBOLS@", symbol)

    with open(output_file, "w") as file:
        file.write(result_str)


def main():
    dumped_symbol = sys.argv[1]
    symbol = get_symbol(dumped_symbol)
    template = sys.argv[2]
    output_file = sys.argv[3]
    replace_template(template, output_file, symbol)


if __name__ == "__main__":
    main()
