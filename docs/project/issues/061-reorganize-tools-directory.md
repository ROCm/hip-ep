<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #061: Reorganize tools/ Directory Structure

## Metadata
- **Type:** Refactoring / Cleanup
- **Priority:** MEDIUM
- **Created:** 2026-02-06
- **Estimated Effort:** 2-3 hours
- **Component:** Build System / Project Organization

## Description

The tools/ directory currently contains a mix of build scripts, debugging utilities, and Bazel files (13 files total) rather than actual MorphiZen end-user tool executables. Meanwhile, 4 tool component directories (graph-opt/, onnx-grep/, pattern-gen/, tar/) are scattered at the top level. This creates naming confusion and organizational mess.

## Problem

**Current state:**
```
tools/                              # Mix of scripts (NOT tools!)
├── morphizen_check_version.py      # Debugging utility
├── xxd.py                          # CMake code generation
├── parse_cl_link_error.py          # MSVC linker helper
├── patch_wrapper.py                # Bazel utility
├── convert_onnx_to_external_data_mode.py  # ONNX utility
├── run-external-command.ps1        # CI helper
├── setup_msvc_env.ps1              # CI helper
├── build_and_test.ps1              # CI script
├── build_llvm.ps1                  # CI script
├── build_ort_and_deps.ps1          # CI script
├── BUILD.bazel                     # Bazel build file
└── collect_pb.bzl                  # Bazel macro

Top level (scattered):
├── graph-opt/                      # Actual tool (executable)
├── onnx-grep/                      # Actual tool (executable)
├── pattern-gen/                    # Actual tool (executable)
└── tar/                            # Actual tool (executable)
```

**Issues:**
1. **Naming confusion**: "tools/" implies end-user tools, but contains build/debug scripts
2. **Top-level clutter**: 4 tool directories scattered among libraries (flat organization)
3. **Poor organization**: Scripts mixed together without logical grouping
4. **Misleading structure**: Developers expect tools in tools/, find scripts instead

## Solution

Reorganize into logical structure where:
- **tools/** contains only end-user tool executables
- **scripts/** organized by purpose (debug/, build/, bazel/)
- **cmake/scripts/** contains CMake-specific utilities

**Target state:**
```
tools/                              # Only executables (tools!)
├── graph-opt/
├── onnx-grep/
├── pattern-gen/
└── tar/

scripts/
├── setup/                          # Dev environment (existing)
│   ├── setup-dev-env.ps1
│   └── setup-dev-env.sh
├── build/                          # CI build scripts (keep for now)
│   ├── run-external-command.ps1
│   ├── setup_msvc_env.ps1
│   ├── build_and_test.ps1
│   ├── build_llvm.ps1
│   └── build_ort_and_deps.ps1
├── debug/                          # NEW - Debugging utilities
│   ├── README.md                   # NEW - Documentation
│   ├── morphizen_check_version.py
│   └── convert_onnx_to_external_data_mode.py
└── bazel/                          # NEW - Bazel utilities
    ├── BUILD.bazel                 # NEW - Bazel targets
    ├── patch_wrapper.py
    └── collect_pb.bzl

cmake/
└── scripts/                        # NEW - CMake utilities
    └── xxd.py
```

**Files to delete:**
- `tools/parse_cl_link_error.py` (AI can help manually when needed)
- `tools/BUILD.bazel` (replaced by scripts/bazel/BUILD.bazel)

## Why It Matters

- **Improves developer experience**: Easier to find scripts/tools in logical locations
- **Fixes naming confusion**: tools/ directory actually contains tools (executables)
- **Reduces top-level clutter**: 4 scattered tool dirs → 1 consolidated directory
- **Better organization**: Scripts grouped by purpose (debug, build, bazel, cmake)
- **Clearer intent**: Directory names match their contents

## Evidence

- tools/ contains 13 files, none are executables
- 4 tool component directories at top level among 27 total directories
- Naming inconsistency confirmed during codebase exploration session

## Implementation

See [detailed plan](../plans/061-reorganize-tools-directory-plan.md) for step-by-step implementation guide.

**Key steps:**
1. Create new directory structure
2. Move files to new locations
3. Delete obsolete files
4. Update all references (BUILD.bazel, CI workflows)
5. Create documentation
6. Update CLAUDE.md

**Files affected:**
- Move: 10 files
- Delete: 2 files
- Move directories: 4 (graph-opt/, onnx-grep/, pattern-gen/, tar/)
- Create: 2 files (README.md, BUILD.bazel)
- Update: morphizen-core/BUILD.bazel, .github workflows, CLAUDE.md

## Acceptance Criteria

- [ ] tools/ contains only executable tool directories (graph-opt/, onnx-grep/, pattern-gen/, tar/)
- [ ] scripts/debug/ exists with README.md documenting utilities
- [ ] scripts/bazel/ exists with BUILD.bazel defining targets
- [ ] cmake/scripts/ exists with xxd.py
- [ ] All Bazel builds still work (morphizen-core references updated)
- [ ] All CI workflows still work (GitHub Actions references updated)
- [ ] CLAUDE.md updated to reflect new structure
- [ ] No broken references to moved files
- [ ] Pre-commit passes

## Notes

- CI build scripts (build_*.ps1, run-external-command.ps1, setup_msvc_env.ps1) are kept for now but marked for future refactoring (separate issue to be created)
- This is part of larger top-level directory organization cleanup
- Related to flattening top-level structure (reducing from 35+ items)
