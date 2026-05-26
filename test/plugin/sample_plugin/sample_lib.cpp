/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Tiny static library used by the sample plugin to exercise
// `addLibraryPath` / `addLibrary`.
//
// At configure time CMake builds this source as `hip_ep_sample_lib`
// (a static library, .lib on Windows / .a on Linux). The sample
// plugin's RegisterCallbacks then hands the library's parent
// directory to `addLibraryPath` and its bare name (`hip_ep_sample_lib`)
// to `addLibrary`. CompilerDriver::discoverLibraries appends both to
// the lld-link argument vector.
//
// The library exposes a single uniquely-named symbol so it cannot
// collide with anything in-tree. It is *not* required to be called
// from the model module to validate the API surface -- the
// round-trip is checked in the unit test by reading
// `pluginLibraryPaths()` and `pluginLibraries()` directly. The
// symbol exists primarily so a real model that wishes to test
// override behaviour has something concrete to call.

extern "C" int hip_ep_sample_lib_marker(int x) {
  // Same constant-return shape as sample_plugin_runtime.cpp so the
  // bitcode + library mechanisms stay symmetric in the sample.
  return x + 2;
}
