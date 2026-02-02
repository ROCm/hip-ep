<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #022: Remove fix_info Dead Code

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Refactoring
- **Owner:** TBD
- **Created:** 2026-02-02
- **Dependencies:** None
- **Related:** #010 (cache_files dead code), #020 (suffix_counter dead code)

## Description

Remove the entire `fix_info` system from ContextProto and IPass interface. The proto field `map<string, int32> fix_info` and all associated API methods are completely unused.

## Problem

**Current design:**
```protobuf
// pass_context.proto:50
message ContextProto {
  // ... other fields ...
  map<string, int32> fix_info = 6;
}
```

```cpp
// pass.hpp - IPass interface
virtual void set_fix_info(const char* name, int fix_pos) = 0;
virtual int get_fix_info(const char* name) const = 0;
virtual bool has_fix_info(const char* name) const = 0;
virtual void dump_fix_info(const char* name) const = 0;
void copy_fix_info(const Node& from_node, const Node& to_node);
void copy_fix_info(const std::string& from, const std::string& to);
void copy_fix_info(const char* from, const char* to);
```

**Why this is wrong:**
1. Dead code bloat - entire API (~50 LOC) is implemented but never called
2. Proto field wastes space in serialization (even if empty, still part of schema)
3. Misleads developers - suggests fix_info is a supported feature
4. Maintenance burden - code that needs to be tested, understood, and maintained

**Current behavior:**
- `copy_fix_info()` - never called from any pass
- `set_fix_info()` - only called internally by copy_fix_info
- `get_fix_info()` - only called internally by copy_fix_info
- `has_fix_info()` - only called internally by copy_fix_info
- `dump_fix_info()` - never called at all
- Proto field `fix_info` - never written or read in practice

## Solution

**Approach:**
1. Remove `map<string, int32> fix_info = 6;` from `pass_context.proto`
2. Remove virtual methods from `IPass` interface in `pass.hpp`:
   - `set_fix_info()`
   - `get_fix_info()`
   - `has_fix_info()`
   - `dump_fix_info()`
3. Remove `copy_fix_info()` helper methods from `IPass`
4. Remove implementations from `pass_imp.hpp` and `pass_imp.cpp`
5. Remove implementation from `pass.cpp`

**Benefits:**
- ✅ Reduces codebase by ~50 LOC
- ✅ Simplifies IPass interface (7 fewer methods)
- ✅ Removes proto field from ContextProto
- ✅ Eliminates maintenance burden for unused code
- ✅ Clarifies what features are actually supported

**Migration path:**
None required - code is completely unused.

## Evidence

**Proto definition:**
- `pass_context.proto:50` - Proto field definition

**Interface declarations:**
- `pass.hpp:139` - `set_fix_info()` declaration
- `pass.hpp:142` - `get_fix_info()` declaration
- `pass.hpp:145` - `has_fix_info()` declaration
- `pass.hpp:152` - `dump_fix_info()` declaration
- `pass.hpp:233-237` - `copy_fix_info()` overloads

**Implementations:**
- `pass_imp.hpp:53-56` - Override declarations
- `pass_imp.cpp:301-324` - All method implementations
- `pass.cpp:14-27` - `copy_fix_info()` helper implementations

**Usage verification:**
```bash
git grep -n "copy_fix_info" -- "*.cpp" "*.hpp" | grep -v "^morphizen-core/src/pass" | grep -v "^morphizen-core/include/morphizen/pass.hpp"
# Returns: (empty) - no external callers
```

## Notes

Similar to #010 (cache_files) and #020 (suffix_counter) - part of ongoing dead code cleanup to reduce ContextProto bloat and simplify the codebase.
