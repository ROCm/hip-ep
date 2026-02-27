#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

# ruff: noqa: F821 -- `config` is injected by LLVM's lit runner at runtime.
import lit.formats
import os

config.name = "hip-mlir-opt"
config.test_format = lit.formats.ShTest(preamble_commands=[])
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)

hip_mlir_opt = os.path.join(
    os.path.dirname(__file__), "..", "build", "bin", "hip-mlir-opt"
)
config.substitutions.append(("%hip-mlir-opt", hip_mlir_opt))

llvm_build_bin = os.environ.get("LLVM_BUILD_BIN")
if not llvm_build_bin:
    llvm_dir = os.environ.get("LLVM_DIR", "")
    if llvm_dir:
        llvm_build_bin = os.path.normpath(
            os.path.join(llvm_dir, "..", "..", "..", "bin")
        )
    else:
        llvm_build_bin = ""
filecheck = os.path.join(llvm_build_bin, "FileCheck")
config.substitutions.append(("%FileCheck", filecheck))
