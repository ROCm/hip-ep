<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Array Lifetime Management in MorphiZen ORT-Bridge

## Problem Statement

The original `convert_to_span` design had a critical resource leak because it didn't call `ReleaseArrayOfConstObjects`, leading to memory leaks when working with ORT arrays.

## New Design: OrtArraySpan<T>

### RAII Wrapper Class

The new `OrtArraySpan<T>` class provides automatic lifetime management for `OrtArrayOfConstObjects`:

```cpp
template <typename T>
class OrtArraySpan {
  // Automatically calls ReleaseArrayOfConstObjects in destructor
  // Provides span-like interface with proper resource management
};
```

### Key Features

1. **Automatic Resource Management**: Destructor automatically calls `ReleaseArrayOfConstObjects`
2. **Move-Only Semantics**: Prevents accidental copying and double-free issues
3. **Span Interface**: Provides familiar span-like access (`begin()`, `end()`, `[]`, etc.)
4. **Exception Safety**: Resources are released even if exceptions occur

## Usage Patterns

### For Long-Lived Access (Preferred)

Use `*_managed()` methods when you need to keep the array data around:

```cpp
// Good: RAII wrapper manages lifetime
auto nodes_managed = graph.nodes_managed();
for (const OrtNode* node : nodes_managed) {
    // Process node...
    // Array is automatically released when nodes_managed goes out of scope
}
```

### For Short-Lived Access (Convenience)

Use regular methods that return vectors for simple iteration:

```cpp
// Good: Immediate copy to vector, array released immediately
auto nodes = graph.nodes();  // Returns std::vector<const OrtNode*>
for (const OrtNode* node : nodes) {
    // Process node...
}
```

### Anti-Patterns (Don't Do This)

```cpp
// BAD: Storing span beyond the wrapper's lifetime
gsl::span<const OrtNode* const> bad_span;
{
    auto nodes_managed = graph.nodes_managed();
    bad_span = nodes_managed.span();  // DON'T DO THIS!
} // nodes_managed destroyed here, bad_span now points to freed memory

// BAD: Trying to copy the wrapper
auto nodes1 = graph.nodes_managed();
auto nodes2 = nodes1;  // COMPILATION ERROR - copy constructor deleted
```

## Migration Guide

### Before (Resource Leak)
```cpp
gsl::span<const OrtNode* const> Graph::nodes() const {
  OrtArrayOfConstObjects* nodes_array = nullptr;
  throw_if_error(ort_api.Graph_GetNodes(p_, &nodes_array));
  return convert_to_span<const OrtNode* const>(nodes_array);
  // LEAK: nodes_array never released!
}
```

### After (Proper Resource Management)
```cpp
// Option 1: RAII wrapper for long-lived access
OrtArraySpan<const OrtNode* const> Graph::nodes_managed() const {
  OrtArrayOfConstObjects* nodes_array = nullptr;
  throw_if_error(ort_api.Graph_GetNodes(p_, &nodes_array));
  return make_array_span<const OrtNode* const>(nodes_array);
  // Array will be released when returned wrapper is destroyed
}

// Option 2: Vector copy for simple iteration
std::vector<const OrtNode*> Graph::nodes() const {
  auto managed = nodes_managed();
  return std::vector<const OrtNode*>(managed.begin(), managed.end());
  // Array released immediately after copy
}
```

## Performance Considerations

- **Managed Access**: Zero-copy until the wrapper is destroyed
- **Vector Copy**: Small overhead for copying pointers, but immediate resource release
- **Choose Based on Usage**: Use managed for complex processing, vector for simple iteration

## Thread Safety

- `OrtArraySpan` is not thread-safe for modification
- Multiple readers can safely access the same span
- Don't share `OrtArraySpan` instances across threads

## Benefits

1. **Memory Safety**: Eliminates resource leaks
2. **Exception Safety**: Resources released even during exceptions
3. **Clear Ownership**: RAII makes lifetime management explicit
4. **Performance**: Zero-copy access when using managed interface
5. **Familiar Interface**: Span-like access patterns
