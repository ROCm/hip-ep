#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run vlm_benchmark.py the way CI does.

CI reads the prompt file, strips it and hands it to vlm_benchmark.py's
--prompt inside Python. It never goes on a command line: a 16K prompt is past
Windows' 32767-character limit, and vlm_benchmark.py has no --prompt_file.

--bin loads onnxruntime.dll (and so the EP) from the build under test before
onnxruntime-genai can pick up the copy in its own package directory.

usage: vlm_driver.py [--bin DIR] --prompt-file FILE VLM_BENCHMARK_PY [args...]
"""

import argparse
import ctypes
import os
import pathlib
import runpy
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", help="directory holding the EP and onnxruntime DLLs")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("script", help="vlm_benchmark.py")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="passed through")
    a = ap.parse_args()

    if a.bin:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(a.bin)
        os.environ["PATH"] = a.bin + os.pathsep + os.environ.get("PATH", "")
        ort = os.path.join(a.bin, "onnxruntime.dll")
        if os.path.exists(ort):
            ctypes.CDLL(ort)

    prompt = pathlib.Path(a.prompt_file).read_text(encoding="utf-8").strip()
    sys.argv = [a.script, *a.rest, "--prompt", prompt]
    runpy.run_path(a.script, run_name="__main__")


if __name__ == "__main__":
    main()
