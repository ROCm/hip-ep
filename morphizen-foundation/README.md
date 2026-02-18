<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen Foundation

**Generic, reusable C++ utilities with ZERO MorphiZen dependencies.**

## ⚠️ CRITICAL DEPENDENCY RULE ⚠️

**This library MUST have ZERO dependencies on other MorphiZen components.**

### Forbidden Dependencies
- ❌ morphizen-utils
- ❌ morphizen-core
- ❌ morphizen-graph
- ❌ morphizen-pattern
- ❌ Any other morphizen-* component

### Allowed Dependencies
- ✅ External libraries only (glog, GSL, zlib, OpenSSL)
- ✅ C++ standard library

### Rationale
morphizen-foundation is designed to be reusable in ANY C++ project without
requiring the MorphiZen framework. All components here are generic infrastructure
utilities with no domain-specific logic.

### Code Review Checklist
When reviewing changes to morphizen-foundation:
- [ ] No `#include` statements reference morphizen-utils, morphizen-core, etc.
- [ ] No `target_link_libraries` includes MorphiZen components
- [ ] All dependencies are external libraries or standard library
- [ ] No domain-specific or framework-specific logic

---

## Purpose

morphizen-foundation provides the lowest architectural layer of the MorphiZen
project, containing generic, reusable utilities that can be used in any C++
project.

**Architectural layers:**
```
morphizen-foundation (generic utilities, zero MorphiZen coupling)
    ↓
morphizen-utils (framework-specific utilities)
    ↓
morphizen-core (compilation engine)
    ↓
applications (execution providers, demos)
```

---

## Components

### 1. env_config - Type-Safe Environment Variable Access

Provides compile-time type-safe access to environment variables with caching
and default values.

**Features:**
- Type-safe conversion (string, int, bool, etc.)
- Cached after first access for performance
- Supports default values
- Cross-platform (uses `vitis_ai_getenv_s` when available)

**Example usage:**
```cpp
#include <morphizen-foundation/env_config.hpp>

// Define environment parameter
DEF_ENV_PARAM(DEBUG_LEVEL, "0")

// Use it
int level = ENV_PARAM(DEBUG_LEVEL);
if (level > 0) {
    LOG(INFO) << "Debug level: " << level;
}
```

**Namespace:** `morphizen::foundation`

### 2. parse_value - Generic String Parsing

Robust string-to-type conversion utilities with error checking.

**Features:**
- Parse strings to common types (int, double, bool, etc.)
- Error handling for invalid conversions
- Used internally by env_config

**Example usage:**
```cpp
#include <morphizen-foundation/parse_value.hpp>

std::string str = "42";
int value = morphizen::foundation::parse_value<int>(str);
```

**Namespace:** `morphizen::foundation`

### 3. file_io - Stream-Based File I/O Interfaces

Generic abstractions for streaming file I/O, enabling memory-efficient processing
and inter-DLL data transfer.

**Features:**
- FileReader: Abstract interface for sequential file reading
- FileWriter: Abstract interface for sequential file writing
- Optional memory-mapped access (zero-copy)
- Minimal interface ideal for DLL boundaries
- Pipeline-style processing support

**Use cases:**
- Processing large files without loading into memory
- Inter-DLL data transfer without serialization overhead
- Pipeline patterns: `compile(FileReader* input, FileWriter* output)`
- Abstracting file sources (disk, memory, network)

**Example usage:**
```cpp
#include <morphizen-foundation/file_io.hpp>

// Stream-based processing
void process(const morphizen::FileReader& reader,
             morphizen::FileWriter& writer) {
    char buffer[4096];
    reader.rewind();

    while (true) {
        size_t bytes = reader.fread(buffer, sizeof(buffer));
        if (bytes == 0) break;

        // Transform data...

        writer.fwrite(buffer, bytes);
    }
}

// Zero-copy access when available
void process_fast(const morphizen::FileReader& reader) {
    void* data = reader.mmap();
    if (data != nullptr) {
        // Direct memory access, no copy
        process_data(data, reader.size());
    } else {
        // Fallback to streaming
        process_streaming(reader);
    }
}
```

**Namespace:** `morphizen`

**Design rationale:**
Unlike std::istream/ostream, these interfaces are minimal and focused on binary
data transfer, making them ideal for DLL boundaries and simple streaming scenarios.
The optional mmap() support enables zero-copy access when available.

### 4. mem_binary - Build-Time Binary Resource Embedding

Embeds binary files directly into the library at build time with optional
compression support.

**Features:**
- Embed any binary file at build time
- Optional zlib compression (if available)
- Simple API: `get_mem_binary()`, `has_mem_binary()`, `get_mem_binary_span()`
- Python-based code generation

**Build-time configuration:**
- Set `MORPHIZEN_EMBEDDED_RESOURCE_PATH` to path of `embedded_resource.txt`
- Resource file format: Python literal eval (JSON-like with comments)

**Example usage:**
```cpp
#include <morphizen-foundation/mem_binary.hpp>

// Check if resource exists
if (morphizen::has_mem_binary("config.json")) {
    // Get resource as vector
    auto data = morphizen::get_mem_binary("config.json");

    // Or get as span (zero-copy)
    auto span = morphizen::get_mem_binary_span("config.json");
    std::string content(span->data(), span->size());
}
```

**Namespace:** `morphizen`

### 5. encryption - AES-256 Encryption/Decryption

Provides AES-256 ECB encryption and decryption for streams.

**Features:**
- AES-256 ECB mode encryption/decryption
- Stream-based API (works with any std::istream/ostream)
- Optional dependency (requires OpenSSL)
- Exception-based error handling

**Example usage:**
```cpp
#include <morphizen-foundation/encryption.hpp>

// Check if encryption support is available
if (morphizen_encryption::has_encryption_support()) {
    std::ifstream src("plaintext.txt", std::ios::binary);
    std::ofstream dst("encrypted.bin", std::ios::binary);

    morphizen_encryption::aes_encryption(src, dst, "my-secret-key");
}
```

**Namespace:** `morphizen_encryption`

---

## Building

morphizen-foundation is built as part of the MorphiZen project.

**Dependencies:**
- **Required:** glog, Microsoft.GSL
- **Optional:** zlib (for mem_binary compression), OpenSSL (for encryption)

**Build:**
```bash
cmake -S . -B ../../build/$(basename $PWD) -DCMAKE_BUILD_TYPE=Debug
cmake --build ../../build/$(basename $PWD) --config Debug
```

---

## Architecture Context

morphizen-foundation sits at the bottom of the MorphiZen architectural stack:

**Layer 0: Foundation** (this library)
- Generic utilities with zero MorphiZen coupling
- Reusable in any C++ project
- Components: env_config, parse_value, file_io, mem_binary, encryption

**Layer 1: morphizen-utils**
- Framework-specific utilities
- Depends on foundation
- Components: morphizen_plugin, cleanup, weak_refs

**Layer 2+: morphizen-core, morphizen-graph, etc.**
- Core framework components
- Depend on utils and foundation

See `docs/architecture.md` for complete architectural overview.

---

## Adding New Components

Before adding a component to morphizen-foundation, verify it meets ALL criteria:

**✅ Should go in morphizen-foundation if:**
- Generic, domain-agnostic utility
- No MorphiZen-specific logic or dependencies
- Reusable in other C++ projects
- Only depends on external libraries (glog, GSL, etc.)
- Size: typically <300 LOC per component

**❌ Should NOT go in morphizen-foundation if:**
- MorphiZen-specific functionality
- Depends on morphizen-utils, morphizen-core, etc.
- Domain-specific logic (graph manipulation, ONNX, etc.)
- Framework coupling required

See `docs/technical/component-organization-guidelines.md` for detailed decision criteria.

---

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
