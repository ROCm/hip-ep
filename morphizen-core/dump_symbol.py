#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
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
    symbol_set = sorted(symbol_set)
    counter = 100
    for symbol in symbol_set:
        ret += symbol + f" @{counter}"
        ret += "\n"
        counter += 1
    return ret


def replace_template(template_file, output_file, symbol):
    file = open(template_file, "r")
    template = file.read()
    result_str = template.replace("@ALL_DUMPED_SYMBOLS@", symbol)

    with open(output_file, "w") as file:
        file.write(result_str)


def main():
    print(f"args = {sys.argv}")
    dumped_symbol = sys.argv[1]
    symbol = get_symbol(dumped_symbol)
    template = sys.argv[2]
    output_file = sys.argv[3]
    print(f"{sys.argv}")
    replace_template(template, output_file, symbol)


if __name__ == "__main__":
    main()
