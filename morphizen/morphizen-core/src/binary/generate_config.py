#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys


def get_escape_json_str(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        content = content.encode()
        ret = "{"
        for char in content:
            ret += str(char)
            ret += ","
        ret = ret[:-1]
        ret += "}"
        return ret


# takes root directory of morphizen repo as argument
def main():
    print(get_escape_json_str(sys.argv[1]))


if __name__ == "__main__":
    main()
