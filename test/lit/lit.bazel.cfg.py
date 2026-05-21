# -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Bazel-specific lit configuration.
#
# This file is used instead of lit.site.cfg.py (which is CMake-generated) when
# running lit tests via Bazel. It is referenced directly by the lit_test target
# in BUILD.bazel.
#
# In Bazel's runfiles layout, all data dependencies (hip-mlir-opt, FileCheck,
# not) land as sibling executables alongside this config in the test's runfiles
# tree. lit resolves tool substitutions via PATH, which we extend here.

import os
import sys
import lit.formats
import lit.llvm

# ruff: noqa: F821  (config/lit_config injected by lit framework)

config.name = "MorphizenMLIR"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]

# The test source root is this file's directory — the test/lit/ source tree.
# All .mlir srcs are passed to lit as absolute paths via $(execpath) in
# lit_test, so lit resolves them from the runfiles tree.
config.test_source_root = os.path.dirname(os.path.abspath(__file__))
config.test_exec_root = config.test_source_root

# In Bazel's runfiles, tools land in the same directory as the test runner or
# in the runfiles tree. We find them via the PATH that Bazel sets up, which
# includes the directories of all data deps that are executables.
#
# Additionally, add the directory of this script itself (where Bazel places
# data-dep executables that are in the same package) and common Bazel tool dirs.
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Collect candidate tool directories from the PATH and from Bazel runfiles.
_tool_dirs = []

# Bazel sets RUNFILES_DIR or TEST_SRCDIR; tools land under these.
for _env_var in ("RUNFILES_DIR", "TEST_SRCDIR"):
    _rd = os.environ.get(_env_var, "")
    if _rd and os.path.isdir(_rd):
        # Walk up to 3 levels deep to find directories with our tools.
        for _root, _dirs, _files in os.walk(_rd):
            if any(f in ("hip-mlir-opt", "hip-mlir-opt.exe",
                         "FileCheck", "FileCheck.exe") for f in _files):
                _tool_dirs.append(_root)
            _dirs[:] = _dirs[:3]  # limit depth

if not _tool_dirs:
    # Fallback: use PATH as-is (works when tools are installed system-wide)
    _tool_dirs = [p for p in os.environ.get("PATH", "").split(os.pathsep)
                  if os.path.isdir(p)]

config.environment["PATH"] = os.pathsep.join(
    _tool_dirs + [os.environ.get("PATH", "")]
)

tools = [
    "hip-mlir-opt",
    "FileCheck",
    "not",
]

# Register tool substitutions so %{tool} works in RUN lines.
llvm_config = lit.llvm.config.LLVMConfig(lit_config, config)
llvm_config.add_tool_substitutions(tools, _tool_dirs)
