<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #027: Eliminate C-Style APIs from morphizen-graph

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Tech Debt / Refactoring
- **Owner:** TBD
- **Created:** 2026-02-02
- **Dependencies:** None
- **Related:** Issue #044 (PR #44 - established morphizen-graph as single abstraction point)
- **Strategic Goal:** Complete C++ wrapper migration, eliminate all C-style free functions in favor of GraphRef/GraphConstRef/ModelRef APIs

## Description

Eliminate all 28 remaining C-style free functions from `morphizen-graph/include/morphizen/graph.hpp` in favor of the C++ wrapper APIs (GraphRef, GraphConstRef, ModelRef). This completes the architectural vision from PR #44 where morphizen-graph was established as the single abstraction point for MORPHIZEN_ORT_API.

Currently, 1 function is already deprecated (`graph_add_node`), 25 have C++ equivalents, and the codebase has a mix of both styles. This issue proposes a 4-phase migration to fully adopt C++ style APIs.

## Problem

**Current design:**
- 28 C-style free functions exist in `morphizen` namespace (graph.hpp lines 36-316)
- Major call sites: morphizen-core (pass.cpp, pass_imp.cpp, node_builder.cpp), ort-bridge (ir-converter-imp.cpp)
- These functions bypass the C++ wrapper architecture established in PR #44

**Why this is wrong:**
1. **Architectural inconsistency** - PR #44 established morphizen-graph as the single abstraction layer over MORPHIZEN_ORT_API, but C-style functions bypass this design
2. **API duplication** - Two ways to do the same thing: `graph_get_inputs(graph)` vs `GraphConstRef(graph).inputs()`
3. **Type safety** - C-style functions return raw pointers (`const NodeArg*`, `const Node*`) instead of type-safe wrappers (`NodeArgConstRef`, `NodeConstRef`)
4. **Maintenance burden** - Need to maintain parallel implementations

**Current behavior:**
- Codebase uses both styles inconsistently
- Some files use GraphConstRef, others use C-style functions
- Graph.cpp has internal call chains mixing both approaches

## Solution

**Proposed design:**

Complete 4-phase multi-PR migration:

**Phase 1: Graph Query Functions (11 functions)**
- Deprecate and migrate read-only queries: `graph_get_inputs`, `graph_get_outputs`, `graph_nodes`, `graph_get_name`, `graph_as_string`, etc.
- Lowest risk, highest usage

**Phase 2: Graph Mutations + Model Operations (10 functions)**
- Migrate state-changing operations: `graph_set_name`, `graph_resolve`, `graph_gc`
- Create new `ModelRef` wrapper class for model operations

**Phase 3: Advanced Algorithms (5 functions)**
- Migrate DFS traversal and fusion: `graph_reverse_dfs_from`, `graph_reverse_dfs_from_multi`, `graph_fuse`
- Add missing `reverse_dfs_from_multi()` to GraphConstRef

**Phase 4: Final Cleanup**
- Remove all deprecated functions
- Update documentation

**Benefits:**
- ✅ Consistent C++ API throughout codebase
- ✅ Improved type safety with wrapper classes
- ✅ Single abstraction layer over MORPHIZEN_ORT_API
- ✅ Easier maintenance with one API style
- ✅ Better encapsulation and RAII-friendly design

**Migration path:**
- Incremental 4-PR approach minimizes risk
- Each phase independently testable
- Clear rollback points if issues arise

## Evidence

**Primary API Definitions:**
- `morphizen-graph/include/morphizen/graph.hpp:36-316` - All 28 C-style functions

**Major Call Sites:**
- `morphizen-core/src/pass.cpp` - 6+ C-style API calls
- `morphizen-core/src/pass_imp.cpp` - graph_resolve, graph_gc, fusion
- `morphizen-core/src/node_builder.cpp` - graph_resolve, DFS, fusion
- `morphizen-core/src/morphizen_compile_model.cpp` - graph I/O operations
- `ort-bridge/src/ir-converter-imp.cpp` - model operations, graph_save

## Context

PR #44 established morphizen-graph as the single abstraction point for MORPHIZEN_ORT_API with a clean C++ wrapper layer (GraphRef/GraphConstRef). However, the legacy C-style free functions remained for backward compatibility. This issue proposes completing that migration by eliminating all C-style functions.

The 4-phase approach was chosen to:
1. Reduce risk through incremental changes
2. Enable focused testing per category
3. Provide clean git history showing progression
4. Allow rollback of individual phases if needed

## Plans

- [027-eliminate-c-style-apis-migration-plan.md](../plans/027-eliminate-c-style-apis-migration-plan.md) - Created 2026-02-02, READY

## Notes

This is a continuation of the architectural work started in PR #44. The migration maintains backward compatibility during the transition by using deprecation warnings before final removal.

Each PR in the plan includes specific file modifications, testing strategies, and verification checklists to ensure a smooth migration.
