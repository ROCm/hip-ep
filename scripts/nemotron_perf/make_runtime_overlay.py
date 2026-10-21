#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Create a new runtime stage without changing its input installations."""

import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install = args.install.resolve(strict=True)
    ci = root / ".nemotron-perf/ci-main-04f1c0365"
    llvm = Path("/scratch/fhanuman/RCOmEP/llvm24-install")
    output = args.output.resolve()
    links = {
        str(p.relative_to(ci)): p
        for kind in ("bin", "lib")
        for p in (ci / kind).iterdir()
    }
    for kind in ("bin", "lib"):
        for p in (install / kind).iterdir():
            if p.is_file() and (kind == "bin" or ".so" in p.name):
                links[f"{kind}/{p.name}"] = p
    links["lib/libamdgpu-ep.so"] = install / "lib/libhipgpu.so"
    for name in ("clang", "clang++", "ld.lld", "lld"):
        links[f"bin/{name}"] = llvm / "bin" / name
    links["lib/clang"] = llvm / "lib/clang"
    for target in links.values():
        if not target.exists():
            raise FileNotFoundError(target)
    output.mkdir(parents=True, exist_ok=False)
    (output / "bin").mkdir()
    (output / "lib").mkdir()
    for name, target in links.items():
        (output / name).symlink_to(target)
    (output / "overlay.json").write_text(
        json.dumps(
            {
                "install": str(install),
                "ort_oga_package": str(ci),
                "llvm": str(llvm),
                "links": {name: str(target) for name, target in links.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
