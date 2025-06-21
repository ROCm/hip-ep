<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Migration Guide: Moving to morphizen-utils

This guide helps you migrate from the old scattered utility headers to the new `morphizen-utils` sub-project.

## Changes Summary

### File Relocations
| Old File | New File |
|----------|----------|
| `vaip-core/include/morphizen/env_config.hpp` | `morphizen-utils/include/morphizen-utils/env_config.hpp` |
| `vaip-core/include/morphizen/weak.hpp` | `morphizen-utils/include/morphizen-utils/weak_refs.hpp` |
| `vaip-core/include/morphizen/parse_value.hpp` | `morphizen-utils/include/morphizen-utils/parse_value.hpp` |

### Namespace Changes
- **Old**: `morphizen::`
- **New**: `morphizen::utils::`

### Header Changes
- **Old**: Individual headers
- **New**: Single convenience header `#include <morphizen-utils/morphizen-utils.hpp>`

## Step-by-Step Migration

### 1. Update CMakeLists.txt

Add the new sub-project dependency:

```cmake
# Add morphizen-utils as subdirectory
add_subdirectory(morphizen-utils)

# Link to your targets
target_link_libraries(your-target morphizen-utils)
```

### 2. Update Include Statements

**Option A: Use individual headers (minimal changes)**
```cpp
// Old
#include <morphizen/env_config.hpp>
#include <morphizen/weak.hpp>

// New
#include <morphizen-utils/env_config.hpp>
#include <morphizen-utils/weak_refs.hpp>
```

**Option B: Use convenience header (recommended)**
```cpp
// Old
#include <morphizen/env_config.hpp>
#include <morphizen/weak.hpp>
#include <morphizen/parse_value.hpp>

// New
#include <morphizen-utils/morphizen-utils.hpp>
```

### 3. Update Namespace Usage

**Environment Configuration (no change needed)**
```cpp
// Macros work the same
DEF_ENV_PARAM(DEBUG_LEVEL, "0");
int level = ENV_PARAM(DEBUG_LEVEL);
```

**Weak References**
```cpp
// Old
morphizen::WeakSingleton<MyClass>::create();
morphizen::WeakStore<Key, Value>::create(key);

// New
morphizen::utils::WeakSingleton<MyClass>::create();
morphizen::utils::WeakStore<Key, Value>::create(key);
```

**Parse Value**
```cpp
// Old
morphizen::parse_value(text, value);

// New
morphizen::utils::parse_value(text, value);
```

### 4. Update Source Files

Find all files that use the old headers:

```bash
# Find files using old headers
grep -r "#include.*morphizen/env_config.hpp" .
grep -r "#include.*morphizen/weak.hpp" .
grep -r "#include.*morphizen/parse_value.hpp" .

# Find files using old namespace
grep -r "morphizen::WeakSingleton" .
grep -r "morphizen::WeakStore" .
grep -r "morphizen::parse_value" .
```

### 5. Automated Migration Script

You can use this sed script to automate most changes:

```bash
#!/bin/bash
# migrate-to-utils.sh

for file in $(find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h"); do
    # Update include statements
    sed -i 's|#include.*<morphizen/env_config.hpp>|#include <morphizen-utils/env_config.hpp>|g' "$file"
    sed -i 's|#include.*<morphizen/weak.hpp>|#include <morphizen-utils/weak_refs.hpp>|g' "$file"
    sed -i 's|#include.*<morphizen/parse_value.hpp>|#include <morphizen-utils/parse_value.hpp>|g' "$file"

    # Update namespace usage (be careful with these)
    sed -i 's|morphizen::WeakSingleton|morphizen::utils::WeakSingleton|g' "$file"
    sed -i 's|morphizen::WeakStore|morphizen::utils::WeakStore|g' "$file"
    sed -i 's|morphizen::parse_value|morphizen::utils::parse_value|g' "$file"
done
```

## Testing Migration

After migration, verify everything works:

1. **Build Test**
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```

2. **Run Tests**
   ```bash
   ctest
   ```

3. **Check for Missing Symbols**
   Look for linker errors about undefined references to the old symbols.

## Common Issues

### Issue 1: Namespace Conflicts
**Problem**: Using both old and new headers simultaneously
**Solution**: Complete the migration in one step per module

### Issue 2: Template Instantiation Errors
**Problem**: Template specializations not found
**Solution**: Make sure to include the new headers before any template usage

### Issue 3: Macro Redefinition
**Problem**: Old and new macro definitions conflict
**Solution**: Remove old headers completely before adding new ones

## Benefits After Migration

1. **Cleaner Dependencies**: Clear separation of utility code
2. **Better Organization**: Related utilities grouped together
3. **Improved Documentation**: Better API documentation and examples
4. **Enhanced Type Safety**: Improved C++17 template patterns
5. **Easier Testing**: Dedicated test suite for utilities

## Rollback Plan

If you need to rollback:

1. Remove `morphizen-utils` from CMakeLists.txt
2. Restore old header files to their original locations
3. Revert namespace changes using reverse sed script
4. Remove the `morphizen-utils` directory

Keep the old headers available during migration period for easier rollback.
