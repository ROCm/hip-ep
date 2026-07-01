/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Tiny "vendor runtime" stand-in for the sample plugin.
//
// This source file is *not* part of the plugin's normal C++
// build. CMake compiles it separately to LLVM bitcode (via clang
// -emit-llvm) and then embeds the resulting `.bc` bytes into the
// plugin as a static byte array (see sample_plugin/CMakeLists.txt).
// The plugin's registration hands that buffer to hip-compiler
// through `HipEpPluginRegistry::addRuntimeBitcode`, which links it
// into the model module after the in-tree runtime with
// `Linker::Flags::OverrideFromSrc`.
//
// We deliberately keep the symbol name namespaced ("hip_ep_sample_*")
// so it cannot collide with any in-tree runtime entry point. Real
// vendor plugins will use this same mechanism to ship `wrap_*`
// overrides; the override semantics are governed by the link flag,
// not by anything visible from this source.

extern "C" int hip_ep_sample_plugin_runtime_marker(int x) {
  // The constant return value isn't observable in any current test;
  // the unit test only asserts the buffer round-trips through the
  // plugin -> registry -> linker path. Keeping the body trivial means
  // the bitcode is small (~1KB) and parses cleanly across LLVM
  // versions without dragging in standard-library declarations.
  return x + 1;
}
