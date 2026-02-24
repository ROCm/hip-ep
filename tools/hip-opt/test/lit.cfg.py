#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

# ruff: noqa: F821 -- `config` is injected by LLVM's lit runner at runtime.
import lit.formats
import os

config.name = "hip-opt"
config.test_format = lit.formats.ShTest(preamble_commands=[])
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)

hip_opt = os.path.join(os.path.dirname(__file__), "..", "build", "hip-opt")
config.substitutions.append(("%hip-opt", hip_opt))

llvm_build_bin = os.environ.get(
    "LLVM_BUILD_BIN",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "..",
        "..",
        "OnnxMLIR",
        "llvm-project",
        "build",
        "bin",
    ),
)
filecheck = os.path.join(llvm_build_bin, "FileCheck")
config.substitutions.append(("%FileCheck", filecheck))
