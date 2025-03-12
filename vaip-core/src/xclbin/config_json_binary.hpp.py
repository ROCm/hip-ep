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
from pathlib import Path
import sys
import os
import json
import glob
from pathlib import Path
import xxd

def get_trimmed_config(config):
    default_target = config["target"]
    new_targets = []
    default_target_passes = {}
    new_passes = []
    for t in config["targets"]:
        if t["name"] == default_target:
            new_targets.append(t)
            for p in t["pass"]:
                default_target_passes[p] = True
    for p in config["passes"]:
        if p["name"] in default_target_passes:
            new_passes.append(p)

    config["targets"] = new_targets
    config["passes"] = new_passes
    if "mepTable" in config:
        del config["mepTable"]


def get_file_version_info(file):
    name = file.split(".xclbin")[0]
    ext = ".xclbin"
    version = name.split("_")[-1]
    # -1 for _
    unversioned_name = name[0 : len(name) - len(version) - 1] + ext
    return unversioned_name, int(version)


def get_versioned_name(xclbin_dict, file):
    if file in xclbin_dict:
        return xclbin_dict[file]
    else:
        return file


def get_xclbin_dict(xclbin_path):
    xclbin_dict = {}
    file_list = []
    if xclbin_path != "":
        file_list = glob.glob(
            str(Path(xclbin_path) / "**" / "*.xclbin"), recursive=True
        )
    for file in file_list:
        file_name = os.path.basename(Path(file))
        unversioned_name, version = get_file_version_info(file_name)
        if unversioned_name not in xclbin_dict:
            xclbin_dict[unversioned_name] = file_name
        else:
            _, last_version = get_file_version_info(xclbin_dict[unversioned_name])
            if version > last_version:
                xclbin_dict[unversioned_name] = file_name
    return xclbin_dict


def traverse_replace(config, xclbin_dict):
    if isinstance(config, dict):
        for key, value in config.items():
            traverse_replace(config[key], xclbin_dict)
            if "xclbin" == key:
                config["xclbin"] = get_versioned_name(xclbin_dict, config["xclbin"])
    elif isinstance(config, list):
        for index, item in enumerate(config):
            traverse_replace(item, xclbin_dict)


def replace_xclbin(config, xclbin_path):
    xclbin_dict = get_xclbin_dict(xclbin_path)
    traverse_replace(config, xclbin_dict)


def get_escape_json_str(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        content = content.encode()
        ret = "{"
        c = 0
        for char in content:
            ret += str(char)
            ret += ","
            c += 1
            if c % 32 == 0:
                ret += "\n"
        ret = ret[:-1]
        ret += "}"
        return ret

def main():
    path = sys.argv[1]
    is_trim_config = sys.argv[2] == "ON"
    xclbin_path = sys.argv[3]
    # open file
    f = open(path, "r", encoding="utf-8")
    config = json.load(f)

    if is_trim_config:
        get_trimmed_config(config)
    if len(xclbin_path) > 0:
        replace_xclbin(config, xclbin_path)
    # output file
    with open("vaip_config.json", "w") as f:
        json.dump(config, f, indent=4)
    FINGERPRINT_CONFIG = get_escape_json_str("vaip_config.json")
    xxd.main(["--output", "config_json_binary.hpp", "--column", "16", "--var" , "config", "vaip_config.json"])
   


if __name__ == "__main__":
    main()
