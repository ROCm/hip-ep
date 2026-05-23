# -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Bazel-specific lit configuration.
#
# Used instead of lit.site.cfg.py (CMake-generated) when running via Bazel.
# RUN lines use bare tool names (hip-mlir-opt, FileCheck, not); we resolve
# their directories from the Bazel runfiles manifest and prepend to PATH.

import os
import lit.formats
import lit.llvm

# ruff: noqa: F821  (config/lit_config injected by lit framework)

config.name = "MorphizenMLIR"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(os.path.abspath(__file__))
config.test_exec_root = config.test_source_root

# ---------------------------------------------------------------------------
# Locate tool directories via RUNFILES_MANIFEST_FILE.
#
# Bazel writes a manifest mapping rlocation keys to on-disk paths. This is
# the most reliable resolution method on Windows remote executors, where
# --remote_download_outputs=minimal means not all outputs exist as files in
# the runfiles symlink forest.
#
# Each manifest line: <rlocation_key> <absolute_path>
# We look for lines whose basename is one of our tool executables.
# ---------------------------------------------------------------------------
_TOOL_BASENAMES = frozenset([
    "hip-mlir-opt.exe", "hip-mlir-opt",
    "FileCheck.exe",    "FileCheck",
    "not.exe",          "not",
])

_tool_dirs = []
_seen_dirs = set()

def _add_dir(d):
    if d and d not in _seen_dirs and os.path.isdir(d):
        _tool_dirs.append(d)
        _seen_dirs.add(d)

_manifest = os.environ.get("RUNFILES_MANIFEST_FILE", "")
if _manifest and os.path.isfile(_manifest):
    with open(_manifest, encoding="utf-8") as _f:
        for _line in _f:
            _parts = _line.rstrip("\n").split(" ", 1)
            if len(_parts) == 2:
                _key, _path = _parts
                if os.path.basename(_key) in _TOOL_BASENAMES:
                    _add_dir(os.path.dirname(_path))

# Fallback: walk runfiles tree (works when manifest is absent, e.g. Linux).
if not _tool_dirs:
    for _env_var in ("RUNFILES_DIR", "TEST_SRCDIR"):
        _rd = os.environ.get(_env_var, "")
        if not (_rd and os.path.isdir(_rd)):
            continue
        for _root, _dirs, _files in os.walk(_rd):
            if _TOOL_BASENAMES & set(_files):
                _add_dir(_root)
            if _root[len(_rd):].count(os.sep) >= 8:
                _dirs[:] = []

# config.llvm_tools_dir must be set: LLVMConfig.add_tool_substitutions falls
# back to it on unresolved tools (unresolved="warn").
config.llvm_tools_dir = _tool_dirs[0] if _tool_dirs else ""

config.environment["PATH"] = os.pathsep.join(
    _tool_dirs + [os.environ.get("PATH", "")]
)
# Ensure .exe is in PATHEXT so lit.util.which() finds hip-mlir-opt.exe etc.
# The remote executor may not have PATHEXT set in its environment.
#
# IMPORTANT: set both config.environment (for child bash processes) AND
# os.environ (for lit.util.which(), which reads os.environ directly).
_pathext = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD;.COM")
config.environment["PATHEXT"] = _pathext
os.environ.setdefault("PATHEXT", _pathext)

# ---------------------------------------------------------------------------
# Register tool substitutions (bare names — RUN lines use them directly via
# PATH, but substitutions are also registered for %{name} style if needed).
# ---------------------------------------------------------------------------
llvm_config = lit.llvm.config.LLVMConfig(lit_config, config)
llvm_config.add_tool_substitutions(
    ["hip-mlir-opt", "FileCheck", "not"],
    _tool_dirs,
)
