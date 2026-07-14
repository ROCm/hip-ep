<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# glog Integration in MorphiZen

This document explains how glog is integrated into the MorphiZen project and how to work with it on different platforms.

## Overview

glog (Google Logging Library) is now enabled in the MorphiZen project to provide robust logging capabilities. The integration is cross-platform and works on Windows, Linux, and macOS.

## Configuration

### Bazel Configuration

glog is configured as a dependency in the following files:

1. **Root MODULE.bazel**:
   ```starlark
   bazel_dep(name = "glog", version = "0.7.1")
   ```

2. **morphizen-utils BUILD.bazel**:
   ```starlark
   deps = [
       "@glog//:glog",
   ],
   ```

### Platform-Specific Setup

#### Windows

On Windows, glog requires bash for building its internal components (gflags). The `.bazelrc` file is configured to use Git bash:

```bazelrc
# Configure bash for genrules (glog/gflags require bash)
build:windows --action_env=BAZEL_SH="C:/Program Files/Git/bin/bash.exe"
```

**Prerequisites for Windows:**
- Git for Windows (includes bash) must be installed
- Git bash should be available at `C:/Program Files/Git/bin/bash.exe`

If Git is installed in a different location, update the path in `.bazelrc`.

#### Linux/macOS

No special configuration is required for Linux and macOS as bash is available by default.

## Usage in Code

glog is currently used in the following source files:

- `morphizen-utils/src/morphizen_plugin.cpp`
- `morphizen-utils/src/cleanup.cpp`

### Include Order

When using glog, it must be included before other MorphiZen headers:

```cpp
// we must include glog before morphizen headers
#include <glog/logging.h>
#include "morphizen-utils/some_header.hpp"
```

### Basic Logging

```cpp
#include <glog/logging.h>

// Initialize glog (typically done once at startup)
google::InitGoogleLogging(argv[0]);

// Log messages
LOG(INFO) << "This is an info message";
LOG(WARNING) << "This is a warning message";
LOG(ERROR) << "This is an error message";

// Conditional logging
LOG_IF(INFO, condition) << "Message when condition is true";

// Debug logging (only in debug builds)
DLOG(INFO) << "Debug message";

// Check macros
CHECK(condition) << "Error message if condition is false";
CHECK_EQ(a, b) << "Error if a != b";
```

## Building

### Standard Build

```bash
# Windows (PowerShell)
bazelisk build //morphizen-utils:morphizen_utils --config=windows

# Linux/macOS
bazelisk build //morphizen-utils:morphizen_utils --config=linux   # or --config=macos
```

### Build Without glog (if needed)

If you need to temporarily disable glog:

1. Comment out the dependency in `MODULE.bazel`:
   ```starlark
   # bazel_dep(name = "glog", version = "0.7.1")
   ```

2. Comment out the dependency in `BUILD.bazel`:
   ```starlark
   deps = [
       # "@glog//:glog",
   ],
   ```

3. Comment out or remove glog includes and calls in source files.

## Troubleshooting

### Windows Issues

1. **"bash.exe failed" error**:
   - Ensure Git for Windows is installed
   - Verify bash is available at `C:/Program Files/Git/bin/bash.exe`
   - Update the path in `.bazelrc` if Git is installed elsewhere

2. **MSYS2/WSL bash conflicts**:
   - The `.bazelrc` is configured to use Git bash specifically
   - Avoid having multiple bash implementations in PATH

3. **Build warnings about C++ standard**:
   - These are expected when glog/gflags try to use `-std=c++14` with MSVC
   - The warnings don't affect the build and can be ignored

### Linux/macOS Issues

1. **Missing bash**:
   - Ensure bash is installed and available in PATH
   - This is rare as bash is standard on these platforms

## Performance Considerations

- glog is highly optimized for performance
- Log messages are only formatted if the log level is enabled
- Use `DLOG` for debug-only logging to avoid runtime overhead in release builds
- Consider using `LOG_IF` and `VLOG` for conditional logging

## Integration Notes

- glog is thread-safe and can be used in multi-threaded code
- It integrates well with the existing MorphiZen error handling
- Log files are written to `/tmp` on Linux/macOS and `%TEMP%` on Windows by default
- Log rotation and cleanup is handled automatically by glog

## Version Information

- **glog version**: 0.7.1 (as specified in MODULE.bazel)
- **gflags version**: Automatically resolved by Bazel Central Registry
- **Compatibility**: C++17 and later (matches MorphiZen requirements)
