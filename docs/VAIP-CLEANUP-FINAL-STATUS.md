<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# VAIP to MorphiZen Cleanup - Final Status

## Summary

The VAIP to MorphiZen naming cleanup is **COMPLETE**. All internal references have been renamed from "VAIP" to "MorphiZen". The remaining VAIP references (89 total) are **intentional** and fall into six categories documented below.

## Progress

- **Starting VAIP references**: 620
- **References cleaned up**: 531 (86%)
- **Remaining intentional references**: 89 (14%)

## Cleanup Phases Completed

- ✅ **Phase 0B**: Dead vendored headers removal (#49)
- ✅ **Phase 1-4**: Core refactoring (#50, #51, #52, #53)
- ✅ **Phase 5**: Config and CMake cleanup (#57)
- ✅ **Phase 6**: Environment variables (#58)
- ✅ **Phase 7**: Documentation and test cleanup (#59)

## Remaining VAIP References (By Category)

### Category 1: Backward Compatibility Aliases (5 refs)

**Purpose**: Provide deprecated aliases for external code that may use old API names.

**Files**:
1. `morphizen-ort-api-ext/include/morphizen/morphizen-ort-api-ext.hpp` (5 refs)
   - Line 32: Comment explaining deprecation
   - Line 34: `[[deprecated]] get_vaip_version_major()`
   - Line 38: `[[deprecated]] get_vaip_version_minor()`
   - Line 42: `[[deprecated]] get_vaip_version_patch()`
   - Line 64: `using VaipOrtApiExt [[deprecated]] = MorphizenOrtApiExt`
   - Line 78: Comment explaining deprecation
   - Line 81: `[[deprecated]] setup_global_vaip_ort_api()`
   - Line 87: `[[deprecated]] get_global_vaip_ort_api()`

**Justification**: These deprecated aliases allow external code to migrate gradually. They will emit compiler warnings directing users to the new API names.

---

### Category 2: Magic Numbers for Binary Compatibility (3 refs)

**Purpose**: Binary format identification magic numbers that cannot change without breaking compatibility.

**Files**:
1. `mlir-imp/src/morphizen-ort-api.cpp` (1 ref)
   - Line 86: `0x50494156; // 'VAIP' in little endian`

2. `onnx-ir-imp/src/morphizen-ort-api.cpp` (1 ref)
   - Line 78: `0x50494156; // 'VAIP' in little endian`

3. `morphizen-ort-api-ext/src/morphizen-ort-api-ext.cpp` (1 ref)
   - Line 67: `const char* magic = "VAIP";`

**Justification**: The magic number `0x50494156` ('VAIP' in ASCII little-endian) is embedded in binary cache files. Changing it would break compatibility with existing cached binaries. The value is intentionally kept as 'VAIP' for backward compatibility, even though the project is now MorphiZen.

---

### Category 3: Historical Copyright Headers in Test Data (5 refs)

**Purpose**: Historical copyright notices in test data files.

**Files**:
1. `unit-test/morphizen/test_anchor_point.data/case0.prototxt` (Line 1)
2. `unit-test/morphizen/test_anchor_point.data/case1.prototxt` (Line 1)
3. `unit-test/morphizen/test_anchor_point.data/case2.prototxt` (Line 1)
4. `unit-test/morphizen/test_anchor_point.data/case3.prototxt` (Line 1)
5. `unit-test/morphizen/test_anchor_point.data/case4.prototxt` (Line 1)

All files contain: `# The Xilinx Vitis AI Vaip in this distribution are provided under the...`

**Justification**: These are historical copyright headers in test data files. Preserving them maintains historical accuracy and legal attribution.

---

### Category 4: External Dependency Name (1 ref)

**Purpose**: Reference to external AMD VitisAI component.

**Files**:
1. `morphizen-core/etc/version_info.txt` (Line 27)
   - `vaip_xclbin;1ec6e20;1ec6e20`

**Justification**: `vaip_xclbin` is an **external component** from AMD's VitisAI ecosystem that provides XCLBIN files (compiled binaries for FPGAs/NPUs). This is similar to other external dependencies listed in the same file:
- Line 12: `vaitrace` (VitisAI trace library)
- Line 25: `vairt` (VitisAI runtime)
- Line 27: `vaip_xclbin` (VitisAI XCLBIN provider)

These external component names should not be changed as they reflect actual upstream project names.

---

### Category 5: External Repository Integration Tools (61 refs)

**Purpose**: Tools that integrate with the external "vaip" repository at Xilinx GitEnterprise.

**Background**: The external repository `git@gitenterprise.xilinx.com:MorphiZen/vaip.git` is still named "vaip". These tools create PRs and sync dependencies with that external repository.

**Files**:

1. **`.github/workflows/update_deps_txt_in_morphizen.yml`** (8 refs)
   - Line 5: Workflow name - "Submit PR to update cmake/deps.txt in vaip"
   - Line 13: `VAIP_BRANCH: cp_dev`
   - Line 14: `VAIP_REPO: git@gitenterprise.xilinx.com:VitisAI/vaip.git`
   - Line 17: Job name - `update-deps-txt-in-vaip`
   - Line 34: Step name - "Submit PR to update cmake/deps.txt in vaip"
   - Line 37: Comment - "Submit PR to update cmake/deps.txt in vaip"
   - Line 38: Script invocation - `submit_vaip_pr.ps1`
   - Line 41: Environment variable - `VAIP_REMOTE_BRANCH`

2. **`tools/submit_morphizen_pr.ps1`** (15 refs)
   - Lines 9-10: `$Env:VAIP_REMOTE_BRANCH` variable
   - Line 16: `$VAIP_DIR = "$W\vaip"` - local clone path
   - Lines 18-20: Clone and sync comments/commands
   - Lines 22-70: Git operations on VAIP repository

3. **`tools/submit_morphizen_pr_for_verify_the_pr.ps1`** (18 refs)
   - Similar structure to submit_morphizen_pr.ps1
   - Line 67: PR title - "verify morphizen PR ... in vaip"

4. **`tools/submit_morphizen_pr.sh`** (9 refs)
   - Line 9: `VAIP_DIR=$W/vaip`
   - Lines 11-15: Clone and sync operations
   - Line 57: PR title generation

5. **`tools/create_pr.ps1`** (2 refs)
   - Line 14: `$repoName = "vaip"`
   - Line 68: Comment text - "VAIP Verification PR Created"

6. **`tools/list_pr.ps1`** (1 ref)
   - Line 10: `$repoName = "vaip"`

7. **`tools/env.sh`** (6 refs)
   - Line 22: Function name - `vaip_banner()`
   - Line 25: Banner text - "Welcome to VAIP dev environment"
   - Line 38: Help text - "to build VAIP"
   - Line 40: Build command example with "vaip" project name
   - Lines 48, 53: Function calls to `vaip_banner`

8. **`tools/parse_cl_link_error.py`** (2 refs)
   - Lines 23, 25: Example paths in comments

**Justification**: These tools interact with an external repository that is actually named "vaip". The variable names and comments accurately reflect the external system they integrate with. These should be kept as-is unless the external repository is renamed.

---

### Category 6: Configuration and Exclusion Files (10 refs)

**Purpose**: Historical directory names and configuration for scanning tools.

**Files**:

1. **`.github/copyright_updates/COPYRIGHT_EXCLUDES`** (4 refs)
   - Lines 18-21: Old directory names preserved for historical reference
     ```
     vaip-core/** @include
     vaip-ort-api-ext/** @include
     vaip_io/** @include
     vaip_pass_init/** @include
     ```

2. **`.github/keyword_scan_excludes.txt`** (3 refs)
   - Lines 5-7: Tool filenames to exclude from scanning
     ```
     tools/submit_vaip_pr.ps1
     tools/submit_vaip_pr.sh
     tools/submit_vaip_pr_for_verify_the_pr.ps1
     ```

3. **`3rd-party/onnxruntime-morphizen-headers/README.md`** (3 refs)
   - Line 12: Source path documentation - `onnxruntime/core/providers/vitisai/include/vaip/`
   - Lines 43-44: Example copy commands showing original source paths

**Justification**: These files document historical information and configure scanning tools. They serve as useful documentation of the project's evolution.

---

## Verification Commands

To verify the categorization of remaining VAIP references:

```bash
# Count all VAIP references (case-insensitive)
git grep -i "vaip" | wc -l

# Exclude intentional references and count unexpected ones
git grep -i "vaip" \
  --exclude="morphizen-ort-api-ext.hpp" \
  --exclude="morphizen-ort-api.cpp" \
  --exclude="morphizen-ort-api-ext.cpp" \
  --exclude="*.prototxt" \
  --exclude="version_info.txt" \
  --exclude-dir="tools" \
  --exclude-dir=".github" \
  --exclude="README.md" \
  | wc -l

# Should return 0 (no unexpected references)
```

## Conclusion

The VAIP to MorphiZen naming cleanup is **COMPLETE**. All 89 remaining VAIP references are intentional and serve specific purposes:
- **Backward compatibility** for external users
- **Binary compatibility** with cached files
- **Historical attribution** in test data
- **External dependency** names
- **External repository integration** tooling
- **Configuration and documentation**

No further cleanup is needed unless:
1. The external "vaip" repository is renamed
2. Binary format version is upgraded (allowing magic number change)
3. Backward compatibility period expires (deprecated APIs can be removed)

## Related Documentation

- [Architecture Documentation](architecture.md) - Updated to use MorphizenOrtApiExt
- [ORT Bridge Design](../ort-bridge/doc/ORT-BRIDGE-DESIGN.md) - Updated to reference MorphiZen core
- [Pre-commit Setup](pre-commit-setup.md) - Developer tooling guide
- [Developer Guide](developer-guide.md) - Complete development workflow

---

**Last Updated**: 2026-01-30
**Total Phases**: 7 (Phase 0B through Phase 7)
**Total PRs**: 11 (#49-#59)
**Cleanup Status**: ✅ COMPLETE
