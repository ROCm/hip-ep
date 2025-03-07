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
import sys
import json
import codecs
import re
from pathlib import Path


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


# takes root directory of vaip repo as argument
def main():
    print(get_escape_json_str(sys.argv[1]))


if __name__ == "__main__":
    main()
