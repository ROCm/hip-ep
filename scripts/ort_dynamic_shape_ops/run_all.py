#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run all data-dependent-shape op verifications (Range, ConstantOfShape,
NonZero) against the ONNX Runtime CPU EP and print a closing summary.

    python run_all.py

Each op is also runnable on its own, e.g. `python verify_nonzero.py`.
"""

import onnxruntime as ort

import common
import verify_constantofshape
import verify_nonzero
import verify_range


def main() -> None:
    print(f"onnxruntime {ort.__version__}  providers={ort.get_available_providers()}")
    print("All sessions below use providers=['CPUExecutionProvider'].\n")

    verify_range.main()
    verify_constantofshape.main()
    verify_nonzero.main()

    print(common.SEP)
    print("SUMMARY")
    print(common.SEP)
    print(
        "For Range / ConstantOfShape / NonZero (data-dependent output shapes):\n"
        "  [1] session.run            -> ORT allocates the output; you pass nothing.\n"
        "  [2] IOBinding + device     -> ORT allocates; you state only the device,\n"
        "                                never the size. This is the path to use\n"
        "                                when the size isn't known up front.\n"
        "  [3] IOBinding + pre-bound  -> see per-op output above for whether ORT\n"
        "                                reuses an exactly-sized caller buffer and\n"
        "                                how it reacts to a wrong-sized one.\n"
        "\n"
        "Bottom line: you do NOT need to pre-allocate the output for these ops.\n"
        "ONNX Runtime sizes and allocates the result from the runtime-computed\n"
        "shape. Pre-allocation is only an option (via IOBinding) when you can\n"
        "supply a correctly-sized buffer; otherwise let ORT own it.\n"
        "\n"
        "Each op section also includes a 'dynamic-shape input' block: a single\n"
        "session built with symbolic input dims is fed several concrete shapes\n"
        "in turn, confirming the input shape is bound -- and the data-dependent\n"
        "output allocated -- per run, with no recompile. That block also re-tests\n"
        "pre-allocation under shape change: ONE caller buffer (sized for the first\n"
        "run) is reused only for the run it matches and is REJECTED for the rest --\n"
        "so a fixed pre-allocated output cannot serve these ops once the shape moves.\n"
        "Practical rule: let ORT allocate, or re-request the exact shape each run."
    )


if __name__ == "__main__":
    main()
