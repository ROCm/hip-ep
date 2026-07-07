<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Component Organization Guidelines

**Purpose:** Provide objective criteria for deciding when to create standalone components versus consolidating into existing libraries.

**Audience:** Contributors, code reviewers, maintainers

**Last Updated:** 2026-02-18

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Layers](#architectural-layers)
3. [Decision Framework](#decision-framework)
4. [Component Types](#component-types)
5. [Size Thresholds](#size-thresholds)
6. [Examples](#examples)
7. [Review Checklist](#review-checklist)

---

## Overview

MorphiZen follows a **layered architecture** with clear separation of concerns. When adding new functionality, contributors should:

1. **Identify the appropriate layer** (Foundation, Utilities, Core, Applications)
2. **Evaluate size and scope** against defined thresholds
3. **Apply decision criteria** from this document
4. **Document rationale** in PR description

**Goals:**
- Minimize top-level directory proliferation
- Maintain clear architectural boundaries
- Enable code reuse across layers
- Reduce subjective organizational debates

---

## Architectural Layers

MorphiZen follows a **strict layered architecture**:

```
Layer 3: Applications
├── Execution providers (ONNX Runtime integration)
├── Demos and examples
└── Command-line tools
    ↓ (depends on)
Layer 2: Core
├── morphizen-core (compilation engine)
├── morphizen-graph (graph manipulation)
└── morphizen-pattern (pattern matching)
    ↓ (depends on)
Layer 1: Utilities (Framework-Specific)
├── morphizen-utils (plugin system, cleanup, weak refs)
└── Framework-specific helper utilities
    ↓ (depends on)
Layer 0: Foundation (Generic, Reusable)
├── morphizen-foundation (env_config, file_io, mem_binary, etc.)
└── ZERO MorphiZen dependencies, reusable in ANY C++ project
    ↓ (depends on)
External Dependencies
└── glog, GSL, protobuf, ONNX Runtime, LLVM/MLIR, etc.
```

**Dependency Rules:**
- ✅ Higher layers MAY depend on lower layers
- ❌ Lower layers MUST NOT depend on higher layers
- ❌ Layers MUST NOT depend on peers (same level)

---

## Decision Framework

### Two-Tier Utilities Pattern

MorphiZen uses a **two-tier utilities pattern**:

**Tier 0: morphizen-foundation**
- Generic, domain-agnostic utilities
- ZERO dependencies on MorphiZen components
- Reusable in ANY C++ project
- Examples: env_config, file_io, mem_binary, encryption

**Tier 1: morphizen-utils**
- Framework-specific utilities
- May depend on morphizen-foundation
- MorphiZen-specific logic allowed
- Examples: plugin system, cleanup helpers, weak singleton patterns

### When to Add to morphizen-foundation

Use this checklist to determine if a utility belongs in foundation:

✅ **YES - Add to foundation IF ALL are true:**
- [ ] Generic, domain-agnostic functionality
- [ ] No MorphiZen-specific logic or dependencies
- [ ] Reusable in other C++ projects outside MorphiZen
- [ ] Only depends on external libraries (glog, GSL, etc.)
- [ ] Size: typically <300 LOC per component
- [ ] Well-defined, focused purpose

❌ **NO - Use morphizen-utils IF ANY are true:**
- [ ] MorphiZen-specific functionality
- [ ] Depends on morphizen-core, morphizen-graph, etc.
- [ ] Domain-specific logic (ONNX, graph manipulation, compilation)
- [ ] Framework coupling required
- [ ] Tightly coupled to MorphiZen architecture

### When to Add to morphizen-utils

✅ **YES - Add to morphizen-utils IF:**
- [ ] Framework-specific utility (but not part of core logic)
- [ ] Used across multiple MorphiZen components
- [ ] Size: <500 LOC for the addition
- [ ] Not specific to a single layer (if specific, put in that layer)
- [ ] Provides infrastructure/helper functionality

❌ **NO - Put elsewhere IF:**
- [ ] Generic utility → morphizen-foundation
- [ ] Core compilation logic → morphizen-core
- [ ] Graph-specific logic → morphizen-graph
- [ ] Application-specific → respective application directory

### When to Create a Standalone Component

Create a new top-level component (e.g., `new-component/`) ONLY IF:

✅ **Required conditions (ALL must be true):**
- [ ] **Substantial size**: >1000 LOC or expected to grow beyond 1000 LOC
- [ ] **Clear architectural layer**: Fits cleanly into Layer 0, 1, 2, or 3
- [ ] **Cohesive purpose**: Single, well-defined responsibility
- [ ] **Reuse potential**: Used by multiple other components
- [ ] **Distinct lifecycle**: May evolve independently of other components

**Additional considerations:**
- [ ] **External exposure**: Will this be exposed as a public API?
- [ ] **Testing needs**: Requires substantial independent test suite?
- [ ] **Documentation scope**: Needs dedicated documentation?

❌ **DO NOT create standalone IF:**
- [ ] Size <500 LOC (consolidate into existing library)
- [ ] Single-use utility (put in the component that uses it)
- [ ] Fits naturally into existing component

---

## Component Types

### Type A: Generic Infrastructure (Layer 0)

**Location:** `morphizen-foundation/`

**Characteristics:**
- Generic C++ utilities
- Zero MorphiZen coupling
- Reusable outside MorphiZen
- Minimal dependencies (only external libs)

**Examples:**
- Environment variable access (env_config)
- File I/O abstractions (file_io)
- Binary resource embedding (mem_binary)
- Encryption/decryption (encryption)

**Size guideline:** <300 LOC per component, aggregate library can be larger

**Decision criteria:**
```
IF (generic AND no_morphizen_deps AND reusable_elsewhere)
  THEN add_to_foundation
```

### Type B: Framework Utilities (Layer 1)

**Location:** `morphizen-utils/`

**Characteristics:**
- MorphiZen-specific helpers
- May depend on foundation
- Used across multiple framework components
- Infrastructure/support functionality

**Examples:**
- Plugin loading system
- Framework cleanup utilities
- Weak singleton patterns
- Framework-wide configuration helpers

**Size guideline:** Individual additions <500 LOC, library can grow as needed

**Decision criteria:**
```
IF (morphizen_specific AND used_across_components AND not_core_logic)
  THEN add_to_utils
```

### Type C: Core Framework (Layer 2)

**Location:** `morphizen-core/`, `morphizen-graph/`, `morphizen-pattern/`

**Characteristics:**
- Core compilation/graph logic
- Domain-specific (ONNX, graph manipulation)
- Substantial size and complexity
- Standalone architectural significance

**Examples:**
- Compilation engine (morphizen-core)
- Graph data structures (morphizen-graph)
- Pattern matching engine (morphizen-pattern)

**Size guideline:** >1000 LOC, cohesive architectural component

**Decision criteria:**
```
IF (core_functionality AND substantial_size AND cohesive_purpose)
  THEN standalone_component
```

### Type D: Applications (Layer 3)

**Location:** `tools/`, `morphizen-demo/`, execution provider directories

**Characteristics:**
- End-user facing applications
- Depends on lower layers
- May be executables or libraries
- Specific use cases

**Examples:**
- Execution providers (ONNX Runtime integration)
- Command-line tools (graph-opt, tar, onnx-grep)
- Demos and examples

**Decision criteria:**
```
IF (application_layer AND end_user_facing)
  THEN app_directory
```

---

## Size Thresholds

Use these thresholds as **guidelines, not strict rules**:

| Size | Recommendation |
|------|---------------|
| <200 LOC | Add to existing library (foundation or utils) |
| 200-500 LOC | Evaluate: foundation, utils, or existing component |
| 500-1000 LOC | Strong case for consolidation unless strong separation rationale |
| >1000 LOC | Consider standalone component if cohesive and reusable |

**Important Notes:**
- These are **guidelines** - use architectural judgment
- Consider **expected growth** - will this 300 LOC utility grow to 2000 LOC?
- **Cohesion matters** - 10 unrelated 100 LOC utilities shouldn't be lumped together
- **Dependency graph** - minimize dependency complexity

---

## Examples

### Example 1: Where to Put a New Utility?

**Scenario:** You've written a 150 LOC helper for parsing JSON configuration files.

**Analysis:**
1. **Is it generic?** Yes, JSON parsing is domain-agnostic
2. **MorphiZen dependencies?** No, uses only standard library + external JSON lib
3. **Reusable elsewhere?** Yes, any C++ project could use it
4. **Size?** 150 LOC (small)

**Decision:** ✅ Add to `morphizen-foundation/`

**Rationale:** Generic, small, reusable utility → foundation

---

### Example 2: MorphiZen-Specific Helper

**Scenario:** You've written a 200 LOC helper for managing ONNX graph metadata specific to MorphiZen's optimization passes.

**Analysis:**
1. **Is it generic?** No, specific to MorphiZen's optimization logic
2. **MorphiZen dependencies?** Yes, depends on morphizen-graph
3. **Reusable elsewhere?** No, tightly coupled to MorphiZen architecture
4. **Where is it used?** Multiple optimization passes

**Decision:** ❌ NOT foundation. ✅ Add to `morphizen-core/` (where optimization passes live)

**Rationale:** Specific to core optimization logic, belongs in core layer

---

### Example 3: Should This Be Standalone?

**Scenario:** You've built a 1500 LOC tensor shape inference library with its own data structures and algorithms.

**Analysis:**
1. **Size?** 1500 LOC (substantial)
2. **Cohesive?** Yes, single well-defined purpose
3. **Reuse potential?** Used by multiple components (graph, core, pattern)
4. **Independent evolution?** Yes, shape inference rules may evolve independently
5. **Testing needs?** Requires comprehensive test suite

**Decision:** ✅ Create standalone `morphizen-shape-inference/`

**Rationale:** Substantial, cohesive, reusable component with independent lifecycle

---

### Example 4: File I/O Abstractions

**Scenario:** You need generic FileReader/FileWriter interfaces for stream-based processing.

**Analysis:**
1. **Is it generic?** Yes, file I/O is domain-agnostic
2. **MorphiZen dependencies?** No, pure virtual interfaces
3. **Reusable elsewhere?** Yes, DLL boundaries, any C++ project
4. **Size?** ~200 LOC (interface definitions + documentation)
5. **Purpose?** Enable `compile(FileReader*, FileWriter*)` pattern

**Decision:** ✅ Add to `morphizen-foundation/` as `file_io.hpp`

**Rationale:** Generic, minimal, enables memory-efficient patterns across codebase

---

## Review Checklist

When reviewing PRs that add new components or utilities, check:

### For Additions to morphizen-foundation

- [ ] **No MorphiZen dependencies**: Verified zero dependencies on morphizen-* components
- [ ] **Generic purpose**: Can be used in other C++ projects
- [ ] **External deps only**: Only depends on glog, GSL, or other external libraries
- [ ] **Documented**: Clear documentation of purpose and usage
- [ ] **Size appropriate**: <300 LOC per component (or justified if larger)

### For Additions to morphizen-utils

- [ ] **Framework-specific**: Clearly tied to MorphiZen architecture
- [ ] **Multi-component use**: Used across multiple MorphiZen components (or planned to be)
- [ ] **Not core logic**: Infrastructure/support, not compilation/graph core logic
- [ ] **Size appropriate**: Addition is <500 LOC (or justified if larger)

### For New Standalone Components

- [ ] **Substantial size**: >1000 LOC or clear growth trajectory
- [ ] **Cohesive purpose**: Single, well-defined responsibility
- [ ] **Reuse potential**: Used by multiple other components
- [ ] **Clear layer**: Fits into Layer 0, 1, 2, or 3
- [ ] **Distinct lifecycle**: Can evolve independently
- [ ] **Justified**: PR description explains why standalone vs consolidated

### General

- [ ] **Architecture document updated**: If creating new patterns
- [ ] **Dependency graph reviewed**: No circular dependencies introduced
- [ ] **Build system updated**: CMakeLists.txt properly configured
- [ ] **Tests added**: Appropriate test coverage

---

## References

- **Issue #059**: Component Organization Guidelines (this document)
- **Issue #060**: Create morphizen-foundation library (implements two-tier pattern)
- **Architecture docs**: `docs/architecture.md`
- **Git workflow**: `docs/workflows/git-workflow.md`

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-18 | Initial | Created component organization guidelines (Issue #059) |

---

## Questions?

If you're unsure where to place new functionality:

1. **Review this document** - Apply the decision framework
2. **Check examples** - Find similar components for reference
3. **Ask in PR description** - Explain your reasoning for reviewers
4. **Propose alternatives** - It's okay to discuss trade-offs

When in doubt, err on the side of **consolidation** rather than creating new standalone components.
