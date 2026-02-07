#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import argparse
import json
from tools import xxd


def get_trimmed_config(config):
    if "target" not in config:
        return
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


def main(args):
    morphizen_config_path = args.in_morphizen_config_json
    is_trim_config = args.is_trim_config.upper() in ["ON", "TRUE", "YES"]
    enable_default_config = args.enable_default_config.upper() in ["ON", "TRUE"]
    # open file
    f = open(morphizen_config_path, "r", encoding="utf-8")
    config = json.load(f)

    if is_trim_config:
        get_trimmed_config(config)
    # output file
    with open(args.out_morphizen_config_json, "w") as f:
        if enable_default_config:
            json.dump(config, f, indent=4)
        else:
            json.dump({}, f, indent=4)

    xxd.main(
        [
            "--output",
            args.morphizen_config_binary_hpp,
            "--column",
            "16",
            "--var",
            "config",
            args.out_morphizen_config_json,
        ]
    )
    with open(args.morphizen_config_binary_hpp, "a") as f:
        f.write(f"static bool with_default_morphizen_config = {1 if enable_default_config else 0};\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate morphizen_config_binary.hpp and morphizen_config.json")
    parser.add_argument(
        "--morphizen_config_binary_hpp",
        type=str,
        help="morphizen_config_binary.hpp save",
    )
    parser.add_argument("--out_morphizen_config_json", type=str, help="morphizen_config.json to save")
    parser.add_argument("--in_morphizen_config_json", type=str, help="morphizen_config.json to read")
    parser.add_argument(
        "--is_trim_config",
        type=str,
        choices=["ON", "OFF"],
        default="ON",
        help="Whether to trim the config (default: ON)",
    )
    parser.add_argument(
        "--enable_default_config",
        type=str,
        choices=["ON", "OFF"],
        default="ON",
        help="Whether to enable default morphizen config (default: ON)",
    )
    args = parser.parse_args(args=sys.argv[1:])
    main(args)
