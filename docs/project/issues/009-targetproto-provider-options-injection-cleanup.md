<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #009: TargetProto Provider Options Injection Cleanup

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Cleanup
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-30 (Decision made - remove NPU-specific features entirely)
- **Dependencies:** Related to Issue #006 (cache_dir removal - xclbin functions use get_log_dir())

## Description

Remove NPU-specific TargetProto features entirely.

**Decision after investigation:** All three features are NPU/FPGA-specific and not needed for GPU. Remove completely.

**What these were:**
- `xlnx_enable_py3_round` - Xilinx-specific rounding mode for quantization
- `xlnx_enable_old_qdq` - Xilinx-specific legacy quantize/dequantize handling
- `xclbin` - FPGA firmware files for Xilinx devices

**Why remove:**
- **NPU/FPGA-specific** - Not applicable to GPU project
- **Already partially removed** - Injection code and two fields gone, only xclbin API remains
- **Dead code** - xclbin API functions never called anywhere
- **Obsolete dependency** - xclbin functions use cache_dir system (Issue #006)

**Current status:**
- ✅ xlnx_enable_py3_round - Completely removed (no references)
- ✅ xlnx_enable_old_qdq - Completely removed (no references)
- ❌ xclbin API functions - Still exist but never called (dead code)
  - `xclbin_path_to_cache_files()` - pass_context_imp.cpp:659
  - `read_xclbin()` - pass_context_imp.cpp:686

## Context

**Part of provider_options cleanup effort** (see docs/technical/cleanup-provider-options.md).

**provider_options pollution problem:** provider_options was modified at multiple stages:
1. User explicit values
2. Config file provider_options section
3. MEP table hit (Issue #008 - removed)
4. TargetProto hit (this issue - removing)

**From Issue #007 discussion:**
> User: "xlnx_enable_py3_round, xlnx_enable_old_qdq, xclbin are different issues. we need to clean them up also. let's put them aside and remind me to create issues later on."

### NPU vs GPU Context

**NPU (original project):**
- Xilinx FPGA/NPU devices
- xlnx_enable_py3_round - Xilinx-specific quantization rounding
- xlnx_enable_old_qdq - Legacy quantize/dequantize operator handling
- xclbin - FPGA firmware binary files (required for NPU execution)

**GPU (this new project):**
- Standard GPU devices
- No Xilinx-specific features needed
- No FPGA firmware (xclbin) needed
- These features are not applicable

**User confirmation:**
> "no xclbin needed for GPU"

### What Was Found

**1. xlnx_enable_py3_round and xlnx_enable_old_qdq:**
- ✅ Completely removed already (no code references found)
- Injection code removed in previous cleanup

**2. xclbin:**
- ✅ Injection code removed (config.cpp:123 comment: "Removed: update_xclbin - NPU-specific xclbin firmware handling")
- ❌ API functions still exist in public interface (pass_context.hpp:322-336)
- ❌ Implementation still exists (pass_context_imp.cpp:659-697)
- ❌ **Dead code** - Never called anywhere in the codebase

**3. xclbin functions depend on obsolete cache_dir:**
```cpp
// pass_context_imp.cpp:659
std::filesystem::path PassContextImp::xclbin_path_to_cache_files(
    const std::filesystem::path& path) const {
  auto filename = path.filename().u8string();
  auto ret = get_log_dir() / filename;  // Uses obsolete cache_dir system (Issue #006)
  ...
}
```

**Relationship to Issue #006:**
- xclbin functions call `get_log_dir()` which returns fake paths from obsolete cache_dir system
- Issue #006 removes cache_dir entirely
- These xclbin functions won't work after #006 anyway

## Solution

### Complete Removal Strategy

**Decision:** Remove remaining xclbin API functions entirely.

**What to remove:**

1. **Remove xclbin from public API** (pass_context.hpp)
   ```cpp
   // DELETE these two virtual functions:
   virtual std::filesystem::path
   xclbin_path_to_cache_files(const std::filesystem::path& path) const = 0;

   virtual std::optional<std::vector<char>>
   read_xclbin(const std::filesystem::path& path) const = 0;
   ```

2. **Remove xclbin implementations** (pass_context_imp.cpp:659-697)
   ```cpp
   // DELETE:
   std::filesystem::path PassContextImp::xclbin_path_to_cache_files(
       const std::filesystem::path& path) const { ... }

   std::optional<std::vector<char>>
   PassContextImp::read_xclbin(const std::filesystem::path& path) const { ... }
   ```

3. **Remove xclbin from PassContextImp header** (pass_context_imp.hpp)
   ```cpp
   // DELETE declarations:
   virtual std::filesystem::path xclbin_path_to_cache_files(...) override final;
   virtual std::optional<std::vector<char>> read_xclbin(...) const override final;
   ```

4. **Update documentation** (cleanup-provider-options.md:55-63)
   ```markdown
   # ~~set when TargetProto hit~~ REMOVED

   All three fields removed entirely (Issue #009):
   - xlnx_enable_py3_round - Xilinx quantization rounding (NPU-specific)
   - xlnx_enable_old_qdq - Legacy QDQ handling (NPU-specific)
   - xclbin - FPGA firmware files (NPU-specific)

   Not needed for GPU project.
   ```

### What xlnx_enable fields were (already removed)

**xlnx_enable_py3_round:**
- Enabled Python 3 compatible rounding for quantization
- Xilinx-specific quantization behavior
- Affected how quantize operators rounded values

**xlnx_enable_old_qdq:**
- Enabled legacy quantize/dequantize operator handling
- Compatibility mode for older Xilinx models
- Different QDQ fusion strategy

**Both already completely removed** - no code cleanup needed.

### What xclbin API was (to be removed)

**xclbin files:**
- Binary firmware files for Xilinx FPGA/NPU devices
- Loaded at runtime for hardware configuration
- Not applicable to GPU execution

**xclbin_path_to_cache_files():**
- Purpose: Copy xclbin file into cache for reuse
- Implementation: Uses obsolete get_log_dir() (Issue #006)
- Status: Never called (dead code)

**read_xclbin():**
- Purpose: Read xclbin file from cache
- Implementation: Reads from cache file
- Status: Never called (dead code)

### Implementation Steps

**Step 1: Verify no callers**
```bash
# Search for any calls to these functions
grep -r "xclbin_path_to_cache_files\|read_xclbin" morphizen-core/src/ --include="*.cpp"
grep -r "xclbin_path_to_cache_files\|read_xclbin" morphizen-core/include/ --include="*.hpp"

# Should find only definitions, no calls
```

**Step 2: Remove from public API**
- Edit pass_context.hpp
- Delete both virtual function declarations (lines ~322-336)
- Renumber or adjust surrounding code if needed

**Step 3: Remove implementations**
- Edit pass_context_imp.cpp
- Delete xclbin_path_to_cache_files() implementation (lines 659-683)
- Delete read_xclbin() implementation (lines 685-697)

**Step 4: Remove from PassContextImp header**
- Edit pass_context_imp.hpp
- Delete both function declarations (lines ~331-335)

**Step 5: Update documentation**
- Edit cleanup-provider-options.md (lines 55-63)
- Mark section as REMOVED
- Explain why removed (NPU-specific, not needed for GPU)

**Step 6: Verify compilation**
- Build the project
- Confirm no linker errors (no external callers)
- All tests pass

### Benefits

- ✅ Remove ~40 lines of dead code
- ✅ Remove NPU-specific dependencies
- ✅ Simplify PassContext API
- ✅ Remove dependency on obsolete cache_dir (Issue #006)
- ✅ Cleaner codebase focused on GPU

### No Breaking Changes Risk

**Why safe to remove:**
- Functions never called internally (verified by grep)
- NPU-specific (only relevant for Xilinx devices)
- GPU project has no external users yet
- Dead code since NPU → GPU transition

**If external code exists that calls these:**
- Would get compilation error (function not found)
- Clear indication to remove the calls
- No silent breakage

## Plans

_No plans needed - straightforward removal (delete functions, update docs)._

## Sessions

### 2026-01-30: Issue Created as Placeholder

**Context:** Identified during Issue #007 discussion about target cleanup.

**User guidance:**
> "xlnx_enable_py3_round, xlnx_enable_old_qdq, xclbin are different issues. we need to clean them up also. let's put them aside and remind me to create issues later on."

**Purpose:** Document that TargetProto injection needs cleanup. Defer detailed investigation and solution design.

### 2026-01-30: Investigation and Decision

**User:** "lets discuss about 009"

**Initial investigation:**
- Checked cleanup-provider-options.md reference to config.cpp:307-308
- Found code doesn't exist (file only 172 lines)
- Searched for three fields: xlnx_enable_py3_round, xlnx_enable_old_qdq, xclbin

**Findings:**

**xlnx_enable_py3_round and xlnx_enable_old_qdq:**
- Completely removed (no code references)
- No cleanup needed

**xclbin:**
- Injection code removed (config.cpp:123 comment confirms)
- But API functions still exist:
  - `xclbin_path_to_cache_files()` - pass_context_imp.cpp:659
  - `read_xclbin()` - pass_context_imp.cpp:686
  - Part of public PassContext API
- **Dead code** - Never called anywhere

**Discovery:** xclbin functions use obsolete cache_dir:
```cpp
auto ret = get_log_dir() / filename;  // Uses cache_dir from Issue #006
```

**Question asked:**
> "What is xclbin and is it still needed for GPU?"

**User clarification:**
> "it is @docs/technical/cleanup-provider-options.md for config.cpp:307-308. no xclbin needed for GPU."

**Follow-up question:**
> "Should we remove the xclbin API functions entirely?"

Elaborated the question:
- Functions are in public API (PassContext interface)
- Never called (dead code)
- NPU/FPGA-specific (not needed for GPU)
- Use obsolete cache_dir system

**Options presented:**
- A) Remove entirely (clean break, no dead code)
- B) Keep for compatibility (maintain dead code forever)
- C) Deprecate gradually (warning period)

**Asked for clarification:**
- Is PassContext public API used externally?
- Do you care about API stability?
- Any plugins calling these?

**User decision:**
> "we can remove `xclbin_path_to_cache_files` and `read_xclbin`"

**Final decision:** Remove both xclbin API functions entirely.

**Rationale:**
- NPU/FPGA-specific (Xilinx devices)
- Not needed for GPU project
- Dead code (never called)
- Depends on obsolete cache_dir (Issue #006)
- Clean removal is better than maintaining dead code

## Related PRs

- PR #159, #155 - Mentioned as deprecation effort (to be investigated)

## Related Branches

_None yet._

## Notes

### Current State

**From docs/technical/cleanup-provider-options.md (lines 55-63):**
```markdown
# set when TargetProto hit

[see here]()../morphizen-core/src/config.cpp#L307-L308)  # Code doesn't exist anymore

It is to be deprecated, see PR #159 #155

* `xlnx_enable_py3_round`  # Already removed
* `xlnx_enable_old_qdq`    # Already removed
* `xclbin`                  # API still exists, never called
```

**Injection code removed:**
```cpp
// config.cpp:123 (comment showing removal)
// Removed: update_xclbin - NPU-specific xclbin firmware handling
```

### What These Features Were

**xlnx_enable_py3_round:**
- Xilinx-specific quantization rounding mode
- Python 3 compatible rounding behavior
- Affected how quantize operators rounded floating-point values
- NPU/FPGA-specific optimization

**xlnx_enable_old_qdq:**
- Xilinx-specific legacy quantize/dequantize handling
- Compatibility mode for older Xilinx ONNX models
- Different QDQ operator fusion strategy
- NPU/FPGA-specific compatibility

**xclbin:**
- FPGA firmware binary files (Xilinx Device Binary)
- Required for Xilinx NPU/FPGA execution
- Contains hardware configuration and kernel code
- Loaded at runtime to configure FPGA device

### Code Locations

**Public API (to be removed):**
- `morphizen-core/include/morphizen/pass_context.hpp:322-336`
  - `virtual std::filesystem::path xclbin_path_to_cache_files(...) const = 0;`
  - `virtual std::optional<std::vector<char>> read_xclbin(...) const = 0;`

**Implementation (to be removed):**
- `morphizen-core/src/pass_context_imp.hpp:331-335`
  - Function declarations
- `morphizen-core/src/pass_context_imp.cpp:659-697`
  - `xclbin_path_to_cache_files()` implementation (25 lines)
  - `read_xclbin()` implementation (13 lines)

**References (comments only):**
- `morphizen-core/src/config.cpp:123` - Comment: "Removed: update_xclbin"
- `morphizen-core/src/binary/config_reader.cpp:38` - Comment referencing xclbin path

### Why These Functions Use Obsolete cache_dir

**xclbin_path_to_cache_files() implementation:**
```cpp
std::filesystem::path PassContextImp::xclbin_path_to_cache_files(
    const std::filesystem::path& path) const {
  auto filename = path.filename().u8string();
  auto ret = get_log_dir() / filename;  // ← Uses obsolete cache_dir!

  if (has_cache_file(filename)) {
    return ret;
  }

  // Copy xclbin to cache...
  const_cast<PassContextImp*>(this)->write_file(filename, buffer);
  return ret;
}
```

**Problem:**
- `get_log_dir()` is from obsolete cache_dir system (Issue #006)
- Returns fake path that doesn't exist on disk
- After Issue #006, get_log_dir() will be removed
- These xclbin functions would break anyway

### Verification of No Callers

**Search results:**
```bash
grep -r "xclbin_path_to_cache_files\|read_xclbin" morphizen-core/src/ --include="*.cpp"
```

**Found:**
- pass_context_imp.cpp:659 - Definition of xclbin_path_to_cache_files
- pass_context_imp.cpp:686 - Definition of read_xclbin
- pass_context_imp.hpp:331 - Declaration of xclbin_path_to_cache_files
- pass_context_imp.hpp:334 - Declaration of read_xclbin

**NOT found:**
- No calls to these functions anywhere
- No usage outside of definition/declaration
- Dead code confirmed

### Why Safe to Remove

1. **Never called** - Grep confirms zero callers in entire codebase
2. **NPU-specific** - Only relevant for Xilinx FPGA/NPU devices
3. **GPU irrelevant** - User confirmed "no xclbin needed for GPU"
4. **Broken dependency** - Relies on obsolete cache_dir being removed
5. **No external users** - GPU project is new, no compatibility burden

### Related to Other Issues

**Issue #006 (cache_dir removal):**
- xclbin functions call `get_log_dir()` which will be removed
- After #006, these functions won't work anyway
- Removing together is cleaner

**Issue #008 (MEP table removal):**
- Similar pattern: NPU features not needed for GPU
- Complete removal better than partial cleanup
- Keep codebase focused on GPU
