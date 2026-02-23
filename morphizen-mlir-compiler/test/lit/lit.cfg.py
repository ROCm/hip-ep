# -*- Python -*-
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

import os
import lit.formats
import lit.llvm

# Configuration file for the 'lit' test runner for morphizen-mlir-compiler
# Note: 'config' is provided by the lit framework at runtime
# ruff: noqa: F821

# name: The name of this test suite.
config.name = "MorphizenMLIR"

# testFormat: The test format to use to interpret tests.
# Use execute_external=False to run commands using Python's subprocess instead of bash
# This avoids issues with Windows paths in bash on Windows
config.test_format = lit.formats.ShTest(execute_external=False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.test_source_root)

# Find the build directory (tools are in ../../build/<worktree>/bin/Debug or Release)
# source_root is morphizen-mlir-compiler/, go up one more to get the worktree root
morphizen_root = os.path.abspath(os.path.join(config.test_source_root, "../.."))
worktree_root = os.path.abspath(os.path.join(morphizen_root, ".."))
worktree_name = os.path.basename(worktree_root)
build_root = os.path.join(worktree_root, "..", "build", worktree_name)

# Locate tool directories
# Priority order: build directory first (for newly built tools), then local (for LLVM tools)
bin_dir = os.path.join(build_root, "bin", "Debug")
if not os.path.exists(bin_dir):
    bin_dir = os.path.join(build_root, "bin", "Release")

local_dir = os.path.join(worktree_root, "..", "..", "local")
llvm_bin = os.path.join(local_dir, "bin") if os.path.exists(local_dir) else None

# Set up tool directories for PATH
# IMPORTANT: Order matters! Build directory must come BEFORE local directory
# to ensure newly built tools (morphizen-opt, morphizen-compile) are found before
# any old versions that might exist in ../../local/bin
tool_dirs = []
if os.path.exists(bin_dir):
    tool_dirs.append(bin_dir)
if llvm_bin and os.path.exists(llvm_bin):
    tool_dirs.append(llvm_bin)

# Use LLVM's LLVMConfig helper for portable PATH and tool management
# This handles:
# - Path separators (/ vs \) automatically via os.path.normpath()
# - Environment separators (; on Windows, : on Unix) via os.path.pathsep
# - Executable extensions (.exe on Windows, none on Unix) via ToolSubst.resolve()
llvm_config = lit.llvm.config.LLVMConfig(lit_config, config)

# Set llvm_tools_dir for tool resolution (required by add_tool_substitutions)
# Use the LLVM bin directory from local installation
if llvm_bin and os.path.exists(llvm_bin):
    config.llvm_tools_dir = llvm_bin

# Add tool directories to PATH (cross-platform)
# with_environment() handles path normalization and proper separators automatically
llvm_config.with_environment("PATH", tool_dirs, append_path=True)

# Define tools to substitute
# List our custom tools first, then LLVM tools
tools = [
    "morphizen-opt",  # Our MLIR optimization tool (renamed from hip-opt)
    "morphizen-compile",  # Our MLIR compiler (renamed from mlir-hip-compiler)
    "FileCheck",  # LLVM tool for test validation
    "not",  # LLVM tool for inverting exit codes
]

# Add tool substitutions (cross-platform)
# add_tool_substitutions() automatically:
# - Searches for executables in tool_dirs
# - Adds .exe extension on Windows
# - Creates substitutions mapping tool names to absolute paths
# - Ensures tests use the correct tools regardless of system PATH
llvm_config.add_tool_substitutions(tools, tool_dirs)
