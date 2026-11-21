#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check a target's resolved dependencies in the CMake File API codemodel."""

import json
import sys
from pathlib import Path


def check(reply, config, consumer, forbidden):
    index = max(reply.glob("index-*.json"))
    model = json.loads(index.read_text())["reply"]["codemodel-v2"]["jsonFile"]
    configs = json.loads((reply / model).read_text())["configurations"]
    targets = next(c["targets"] for c in configs if c["name"] == config)
    by_id = {t["id"]: t for t in targets}
    root = next(t["id"] for t in targets if t["name"] == consumer)
    pending = [(root, [consumer])]
    seen = set()
    while pending:
        target_id, path = pending.pop()
        if target_id in seen:
            continue
        seen.add(target_id)
        target = json.loads((reply / by_id[target_id]["jsonFile"]).read_text())
        for dep in target.get("dependencies", []):
            name = by_id[dep["id"]]["name"]
            if name in forbidden:
                raise SystemExit("Unexpected dependency: " + " -> ".join([*path, name]))
            pending.append((dep["id"], [*path, name]))


if __name__ == "__main__":
    check(Path(sys.argv[1]), sys.argv[2], sys.argv[3], set(sys.argv[4:]))
