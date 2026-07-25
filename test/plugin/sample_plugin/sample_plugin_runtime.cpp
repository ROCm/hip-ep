/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Tiny "vendor runtime" stand-in. NOT part of the plugin's C++ build: CMake
// compiles it to LLVM bitcode and embeds the bytes into the plugin (see
// CMakeLists.txt); registration hands the buffer to addRuntimeBitcode, which
// links it into the model module with OverrideFromSrc. The symbol is
// namespaced so it cannot collide with an in-tree runtime entry point.

extern "C" int hip_ep_sample_plugin_runtime_marker(int x) {
  // Trivial body keeps the bitcode tiny and standard-library-free.
  return x + 1;
}
