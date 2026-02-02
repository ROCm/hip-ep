<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Migration Plan: Eliminate C-Style APIs from morphizen-graph

**Issue:** #027
**Created:** 2026-02-02
**Status:** READY

## Objective

Complete the C-style API mitigation work started in PR #44 by eliminating all 28 C-style free functions from `morphizen-graph/include/morphizen/graph.hpp` in favor of C++ style APIs (GraphRef/GraphConstRef/ModelRef).

## Context

**Current State:**
- 28 C-style free functions in `morphizen` namespace (graph.hpp lines 36-316)
- 1 already deprecated: `graph_add_node` (line 42)
- 25 functions have C++ equivalents in GraphRef/GraphConstRef
- Major call sites: morphizen-core (pass.cpp, pass_imp.cpp, node_builder.cpp, morphizen_compile_model.cpp), ort-bridge (ir-converter-imp.cpp), graph.cpp internals

**Motivation:**
- PR #44 established morphizen-graph as single abstraction point for MORPHIZEN_ORT_API
- Current C-style functions bypass the C++ wrapper architecture
- Goal: Make morphizen-graph truly C++ style, consistent with PR #44 principles

## Strategy: 4-Phase Multi-PR Approach

### Why Multi-PR?
- Reduces risk through incremental changes
- Enables focused testing per category
- Clean git history showing progression
- Allows rollback of individual phases

---

## PR 1: Graph Query Functions (11 functions)

**Scope:** Read-only graph queries - lowest risk, highest usage

### Functions to Migrate

```cpp
// Graph information
const std::string& graph_get_name(const Graph&);
const Model& graph_get_model(const Graph&);

// Structure queries
std::vector<const NodeArg*> graph_get_inputs(const Graph&);
std::vector<const NodeArg*> graph_get_outputs(const Graph&);
std::vector<const Node*> graph_nodes(const Graph&);
std::vector<size_t> graph_get_node_in_topoligical_order(const Graph&);
std::vector<const Node*> graph_get_output_nodes(const Graph&);

// Node argument queries
const NodeArg* graph_get_node_arg(const Graph&, const std::string& name);
const Node* graph_producer_node(const Graph&, const std::string& node_arg_name);
std::vector<const Node*> graph_get_consumer_nodes(const Graph&, const std::string& node_arg_name);

// Debugging
std::string graph_as_string(const Graph&);
```

### Migration Pattern

```cpp
// Before:
auto name = graph_get_name(graph);
auto inputs = graph_get_inputs(graph);  // std::vector<const NodeArg*>

// After:
auto name = GraphConstRef(graph).name();
auto inputs_cxx = GraphConstRef(graph).inputs();  // std::vector<NodeArgConstRef>

// If raw pointers needed:
std::vector<const NodeArg*> inputs;
for (auto& inp : inputs_cxx) inputs.push_back(inp.ptr());
```

### Critical Call Sites

- `morphizen-core/src/pass.cpp` (lines 35, 135, 169, 349-350, 405)
- `morphizen-core/src/morphizen_compile_model.cpp` (lines 260, 270, 1065-1066)
- `morphizen-graph/src/graph.cpp` (internal usage in multiple functions)

### Implementation Steps

1. Create helper conversion functions if needed (NodeArgConstRef vector → raw pointer vector)
2. Update morphizen-core call sites (pass.cpp, morphizen_compile_model.cpp)
3. Refactor graph.cpp internals to use GraphConstRef methods
4. Mark all 11 functions as `[[deprecated]]` with C++ API guidance
5. Build and test

### Files Modified

- `morphizen-graph/include/morphizen/graph.hpp` - Add deprecation warnings
- `morphizen-core/src/pass.cpp` - Migrate to GraphConstRef
- `morphizen-core/src/morphizen_compile_model.cpp` - Migrate to GraphConstRef
- `morphizen-graph/src/graph.cpp` - Refactor internal usage

---

## PR 2: Graph Mutations + Model Operations (10 functions)

**Scope:** State-changing operations on graphs and models

### Functions to Migrate

```cpp
// Graph mutations
void graph_set_name(Graph&, const std::string& name);
void graph_set_inputs(Graph&, const std::vector<NodeArg*>& inputs);
void graph_set_outputs(Graph&, const std::vector<NodeArg*>& outputs);
void graph_add_initialized_tensor(Graph&, const TensorProto&);
void graph_save(Graph&, const std::string&, const std::string&, size_t);
void graph_resolve(Graph&, bool force = false);
void graph_gc(Graph&);

// Model operations (need ModelRef wrapper)
Graph& model_main_graph(Model&);
const std::string& model_get_meta_data(const Model&, const std::string&);
bool model_has_meta_data(const Model&, const std::string&);
Model* model_clone(const Model&);
```

### NEW: ModelRef Wrapper Class

Create `morphizen_cxx::ModelRef` class (similar to GraphRef):

```cpp
namespace morphizen_cxx {
class MORPHIZEN_DLL_SPEC ModelRef {
public:
    ModelRef(morphizen::Model& model) : model_(model) {}

    GraphRef main_graph();
    const std::string& get_meta_data(const std::string& key);
    bool has_meta_data(const std::string& key);
    ModelRef clone() const;

private:
    morphizen::Model& model_;
};
}
```

### Migration Pattern

```cpp
// Before:
graph_set_name(graph, "my_graph");
graph_resolve(graph, false);
auto& main_graph = model_main_graph(model);

// After:
GraphRef(graph).set_name("my_graph");
GraphRef(graph).resolve(false);
auto& main_graph = ModelRef(model).main_graph();
```

### Critical Call Sites

- `morphizen-core/src/pass_imp.cpp` (lines 162, 165, 273, 338 - graph_resolve, graph_gc)
- `morphizen-core/src/node_builder.cpp` (line 477 - graph_resolve)
- `ort-bridge/src/ir-converter-imp.cpp` (lines 52, 86, 88 - model operations, graph_save)

### Implementation Steps

1. Create ModelRef wrapper class in graph.hpp
2. Update morphizen-core call sites (pass_imp.cpp, node_builder.cpp)
3. Update ort-bridge (ir-converter-imp.cpp)
4. Refactor graph.cpp to make C-style functions call GraphRef/ModelRef methods
5. Mark all 10 functions as `[[deprecated]]`
6. Build and test

### Files Modified

- `morphizen-graph/include/morphizen/graph.hpp` - Add ModelRef class, deprecation warnings
- `morphizen-graph/src/graph.cpp` - Implement ModelRef, refactor internals
- `morphizen-core/src/pass_imp.cpp` - Migrate to GraphRef
- `morphizen-core/src/node_builder.cpp` - Migrate to GraphRef
- `ort-bridge/src/ir-converter-imp.cpp` - Migrate to ModelRef, GraphRef

---

## PR 3: Advanced Algorithms (5 functions)

**Scope:** DFS traversal and graph fusion operations

### Functions to Migrate

```cpp
void graph_reverse_dfs_from(const Graph&, size_t node_index,
    const std::function<bool(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& comp = nullptr,
    bool subgraph_sensitive = false);

void graph_reverse_dfs_from_multi(const Graph&, gsl::span<const Node* const> from,
    const std::function<void(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& stop);

// Two graph_fuse overloads
void graph_fuse(Graph&, const std::string& name, const std::string& op_type,
    const std::vector<const Node*>& nodes, ...);
Node& graph_fuse(Graph&, const std::string& name, const std::string& op_type,
    const std::vector<size_t>& nodes, ...);
```

### NEW: Add Missing Method to GraphConstRef

Add `reverse_dfs_from_multi()` to GraphConstRef class (currently missing):

```cpp
void reverse_dfs_from_multi(
    gsl::span<const Node* const> from,
    const std::function<void(const Node*)>& enter,
    const std::function<void(const Node*)>& leave,
    const std::function<bool(const Node*, const Node*)>& stop) const;
```

### Migration Pattern

```cpp
// Before:
graph_reverse_dfs_from(graph, node_idx, enter_fn, leave_fn);
graph_reverse_dfs_from_multi(graph, nodes_span, enter_fn, leave_fn, stop_fn);

// After:
GraphConstRef(graph).reverse_dfs_from(node_idx, enter_fn, leave_fn);
GraphConstRef(graph).reverse_dfs_from_multi(nodes_span, enter_fn, leave_fn, stop_fn);
```

### Critical Call Sites

- `morphizen-core/src/node_builder.cpp` (lines 539, 582 - DFS and fusion)
- `morphizen-core/src/pass.cpp` (lines 252, 366, 417 - DFS traversal)
- `morphizen-core/src/pass_imp.cpp` (lines 336, 373 - fusion operations)

### Implementation Steps

1. Add `reverse_dfs_from_multi()` to GraphConstRef in graph.hpp
2. Implement the new method in graph.cpp
3. Update all morphizen-core call sites to use GraphConstRef methods
4. Mark all 5 functions as `[[deprecated]]`
5. Build and test (especially pass system tests)

### Files Modified

- `morphizen-graph/include/morphizen/graph.hpp` - Add reverse_dfs_from_multi, deprecations
- `morphizen-graph/src/graph.cpp` - Implement new method
- `morphizen-core/src/node_builder.cpp` - Migrate to GraphConstRef
- `morphizen-core/src/pass.cpp` - Migrate to GraphConstRef
- `morphizen-core/src/pass_imp.cpp` - Migrate to GraphConstRef

---

## PR 4: Final Cleanup and Documentation

**Scope:** Remove deprecated functions, update documentation

### Tasks

1. **Remove all deprecated C-style functions** from graph.hpp (28 total)
2. **Remove implementations** from graph.cpp
3. **Verify no remaining call sites** - grep for any stragglers
4. **Update documentation:**
   - CLAUDE.md: Document C++ API migration completion
   - docs/architecture.md: Emphasize C++ wrapper layer
   - graph.hpp: Update file header comments
5. **Clean up utilities:**
   - Remove `node_inputs_2_node_args()` if no longer needed
   - Evaluate `graph_get_output_nodes()` - keep as utility or remove?

### Verification Checklist

- [ ] No `graph_*` free functions in `morphizen` namespace (lines 36-316)
- [ ] No `model_*` free functions in `morphizen` namespace
- [ ] All morphizen-core code uses GraphRef/GraphConstRef/ModelRef
- [ ] All ort-bridge code uses C++ wrappers
- [ ] All unit tests pass
- [ ] No compiler warnings about deprecated APIs
- [ ] Zero calls to removed functions (`grep -r "graph_get_\|graph_set_" morphizen-core morphizen-graph ort-bridge`)

### Files Modified

- `morphizen-graph/include/morphizen/graph.hpp` - Remove deprecated functions
- `morphizen-graph/src/graph.cpp` - Remove implementations
- `CLAUDE.md` - Document migration completion
- `docs/architecture.md` - Update architecture description

---

## Testing Strategy

### Per-PR Testing

After each PR:

```bash
# Build with warnings
cmake --build ../../build/$(basename $PWD) --config Debug --parallel

# Run full test suite
../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe

# Check for deprecated API usage
grep -r "graph_get_\|graph_set_\|model_\|graph_resolve\|graph_gc" \
  morphizen-core/src morphizen-graph/src ort-bridge/src
```

### Integration Testing

After all PRs merged:
- Full build + test on clean workspace
- Verify no performance regressions
- Run example applications (if any)

---

## Rollback Plan

Each PR can be independently rolled back:

```bash
git revert <commit-sha>
git push fork <branch>
```

---

## Timeline Estimate

- **PR 1 (Queries):** 4-6 hours
- **PR 2 (Mutations/Model):** 6-8 hours (includes ModelRef creation)
- **PR 3 (Algorithms):** 4-6 hours
- **PR 4 (Cleanup):** 2-4 hours

**Total: 16-24 hours** over 4 PRs

---

## Success Criteria

1. ✅ Zero C-style `graph_*`/`model_*` functions in morphizen namespace
2. ✅ All morphizen-core uses GraphRef/GraphConstRef/ModelRef exclusively
3. ✅ All ort-bridge uses C++ API
4. ✅ All unit tests pass
5. ✅ No compiler warnings
6. ✅ Documentation reflects C++ API architecture
7. ✅ Architecture maintains PR #44 principle: morphizen-graph as single abstraction layer

---

## Critical Files Reference

**Primary API Definitions:**
- `morphizen-graph/include/morphizen/graph.hpp` - All 28 C-style functions, GraphRef/GraphConstRef classes

**Implementations:**
- `morphizen-graph/src/graph.cpp` - Function implementations, internal call chains

**Major Call Sites:**
- `morphizen-core/src/pass.cpp` - 6+ C-style API calls in pass system
- `morphizen-core/src/pass_imp.cpp` - graph_resolve, graph_gc, fusion
- `morphizen-core/src/node_builder.cpp` - graph_resolve, DFS, fusion
- `morphizen-core/src/morphizen_compile_model.cpp` - graph I/O
- `ort-bridge/src/ir-converter-imp.cpp` - model operations, graph_save

**Documentation:**
- `CLAUDE.md` - Architecture guidance
- `docs/architecture.md` - Layer descriptions
