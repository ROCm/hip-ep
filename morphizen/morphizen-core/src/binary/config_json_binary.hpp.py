#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import json
import xxd


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


def main():
    path = sys.argv[1]
    is_trim_config = sys.argv[2] == "ON"
    enable_default_config = sys.argv[3].upper() in ["ON", "TRUE", "YES"]
    # open file
    f = open(path, "r", encoding="utf-8")
    config = json.load(f)

    if is_trim_config:
        get_trimmed_config(config)
    # output file
    with open("morphizen_config.json", "w") as f:
        if enable_default_config:
            json.dump(config, f, indent=4)
        else:
            json.dump({}, f, indent=4)

    xxd.main(
        [
            "--output",
            "config_json_binary.hpp",
            "--column",
            "16",
            "--var",
            "config",
            "morphizen_config.json",
        ]
    )
    with open("config_json_binary.hpp", "a") as f:
        f.write(
            f"static bool with_default_morphizen_config = {1 if enable_default_config else 0};\n"
        )


if __name__ == "__main__":
    main()
