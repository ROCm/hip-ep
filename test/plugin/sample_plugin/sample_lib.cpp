/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Tiny static library exercising addLibraryPath / addLibrary. CMake builds it
// as `hip_ep_sample_lib`; the plugin contributes its dir + bare name, which
// CompilerDriver::discoverLibraries appends to the lld-link line. The single
// uniquely-named symbol only needs to exist -- the round-trip is checked via
// pluginLibraryPaths() / pluginLibraries() in the unit test.

extern "C" int hip_ep_sample_lib_marker(int x) { return x + 2; }
