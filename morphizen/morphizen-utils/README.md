<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen Utils

A utility library for the MorphiZen project providing general-purpose C++ utilities and patterns.

## Features

### Environment Configuration
- Type-safe environment variable access
- Compile-time default values
- Support for common types (int, string, bool, vectors)
- Hex number parsing support
- Macro-based configuration definition

### Weak Reference Patterns
- **WeakSingleton**: Singleton pattern that doesn't prevent destruction
- **WeakStore**: Key-value store with automatic cleanup of unused objects
- Memory-safe patterns to avoid static destruction order issues

### String Parsing
- Robust string-to-type conversion
- Hex number support for integers
- Boolean parsing with multiple text formats
- Error checking with assertions

## Usage

### Environment Configuration

```cpp
#include <morphizen-utils/morphizen-utils.hpp>

// Define environment parameters
DEF_ENV_PARAM(DEBUG_LEVEL, "0");                    // int parameter
DEF_ENV_PARAM_2(LOG_FILE, "app.log", std::string);  // string parameter
DEF_ENV_PARAM_2(ENABLE_CACHE, "true", bool);        // bool parameter

void example() {
    int debug_level = ENV_PARAM(DEBUG_LEVEL);
    std::string log_file = ENV_PARAM(LOG_FILE);
    bool cache_enabled = ENV_PARAM(ENABLE_CACHE);

    // Values are cached after first access
}
```

### Weak Singleton

```cpp
#include <morphizen-utils/morphizen-utils.hpp>

class MyService {
public:
    void do_work() { /* ... */ }
};

void example() {
    // Create or get singleton instance
    auto service = morphizen::utils::WeakSingleton<MyService>::create();
    service->do_work();

    // Instance is automatically destroyed when no references remain
}
```

### Weak Store

```cpp
#include <morphizen-utils/morphizen-utils.hpp>

class DatabaseConnection {
public:
    DatabaseConnection(const std::string& url) : url_(url) {}
    void initialize() { /* Connect to database */ }
private:
    std::string url_;
};

void example() {
    // Create or get cached connection
    auto conn = morphizen::utils::WeakStore<std::string, DatabaseConnection>
                    ::create("postgresql://localhost", "postgresql://localhost");

    // Connection is reused if still alive, otherwise created fresh
}
```

## Building

```bash
mkdir build && cd build
cmake .. -DMORPHIZEN_UTILS_BUILD_TESTS=ON
cmake --build .
ctest  # Run tests
```

## Integration

### As Subdirectory
```cmake
add_subdirectory(morphizen-utils)
target_link_libraries(your-target morphizen-utils)
```

### Using find_package (after installation)
```cmake
find_package(morphizen-utils REQUIRED)
target_link_libraries(your-target morphizen::morphizen-utils)
```

## Requirements

- C++17 or later
- CMake 3.16 or later

## License

Licensed under the MIT License. See LICENSE file for details.

## Design Principles

1. **Header-mostly**: Most functionality is in headers for easy integration
2. **Type Safety**: Compile-time type checking where possible
3. **Memory Safety**: RAII patterns and weak references to prevent leaks
4. **Performance**: Cached values and efficient lookup patterns
5. **Simplicity**: Easy-to-use macros and templates

## Thread Safety

- Environment configuration: Thread-safe (values are cached at startup)
- WeakSingleton: Thread-safe construction and access
- WeakStore: Requires external synchronization for multi-threaded access
