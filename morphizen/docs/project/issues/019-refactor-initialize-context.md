<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #019: Refactor initialize_context() - God Function Cleanup

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Refactoring
- **Dependencies:** Issue #005, #006, #017 should complete first
- **Strategic Goal:** Immutable ConfigProto (optional quality improvement)

## Description

Refactor `initialize_context()` function by extracting complex cache_key computation logic into a separate, testable function. This is the main source of complexity in the function (~23 lines of 4-way conditional logic).

## Problem

**Current implementation (morphizen_compile_model.cpp:362-436):**
```cpp
std::shared_ptr<PassContext> initialize_context(
    const ConfigProto& config_proto,
    const Model& model,
    const std::string& file_path,
    const std::unordered_map<std::string, std::string>& provider_options,
    const int model_load_type) {

  // 1. Create context from proto (lines 362-385)
  auto context = PassContext::create(config_proto, provider_options);

  // 2. Complex 4-way cache_key logic (lines 387-409)
  // - Compute from model metadata
  // - OR compute from file hash
  // - OR compute from memory signature
  // - OR compute from EP context node

  // 3. Call obsolete function (line 417 - Issue #017)
  update_config_by_target(...);

  // 4. Dangerous const-cast (line 427)
  Model& mutable_model = const_cast<Model&>(model);

  // 5. Setup cache files and other side effects (lines 428-436)

  return context;
}
```

**Why this is a problem:**

1. **Complex cache_key Computation (Lines 387-409)**
   - ~23 lines of 4-way conditional logic
   - Repeated pattern: try metadata, try file hash, try memory, try EP node
   - Difficult to reason about priority and fallback logic
   - Cannot test cache_key logic in isolation
   - Hard to verify correctness of each branch

2. **After Issues #005, #006, #017 Complete**
   - Line 417: update_config_by_target() removed (Issue #017)
   - Lines 425, 427-429: cache_dir + const-cast removed (Issue #006 - dead code)
   - Lines 387-409: cache_key computation remains (the main complexity)
   - **Result:** cache_key logic is the primary remaining complexity

3. **Function Still ~52 Lines After Cleanups**
   - Lines 362-385: Context creation (~24 lines)
   - Lines 387-409: cache_key computation (~23 lines) ← COMPLEX
   - Lines 415-436: Remaining logic (~5 lines after #006, #017)
   - Main complexity is cache_key computation

**Current behavior:**
- Cannot test cache_key computation in isolation
- Testing requires full PassContext setup
- Hard to verify each cache_key source works correctly
- Difficult to add new cache_key sources

## Solution

Extract cache_key computation into a separate, testable function.

**Step 1: Extract cache_key Computation**

```cpp
// NEW function: Pure computation, no side effects, testable
static std::string compute_cache_key(
    const Model& model,
    const std::string& file_path,
    const std::unordered_map<std::string, std::string>& provider_options,
    const int model_load_type) {

  // Try 1: Model metadata (highest priority)
  if (auto cache_key = extract_cache_key_from_metadata(model); !cache_key.empty()) {
    return cache_key;
  }

  // Try 2: File hash (for file-based loading)
  if (!file_path.empty()) {
    return compute_file_hash(file_path);
  }

  // Try 3: Memory signature (for memory-based loading)
  if (model_load_type == MODEL_LOAD_TYPE_MEMORY) {
    return compute_memory_signature(model);
  }

  // Try 4: EP context node (fallback)
  if (auto ep_cache_key = extract_cache_key_from_ep_context(model); !ep_cache_key.empty()) {
    return ep_cache_key;
  }

  return "";  // No cache_key available
}
```

**Step 2: Use in initialize_context()**

```cpp
// AFTER refactoring: ~30 lines (was ~52 after #005, #006, #017)
std::shared_ptr<PassContext> initialize_context(
    const ConfigProto& config_proto,
    const Model& model,
    const std::string& file_path,
    const std::unordered_map<std::string, std::string>& provider_options,
    const int model_load_type) {

  // 1. Create context (~24 lines - unchanged)
  auto context = PassContext::create(config_proto, provider_options);

  // 2. Compute cache_key (extracted - was ~23 lines, now 1 line)
  std::string cache_key = compute_cache_key(model, file_path, provider_options, model_load_type);

  // 3. Store cache_key in ContextProto (Issue #005)
  *context->context_proto.mutable_cache_key() = cache_key;

  // 4. Remaining logic (~5 lines - target_auto_discovery, suffix_counter, etc.)
  // ... (unchanged)

  return context;
}
```

**Benefits:**
- ✅ **Better testability** - Can test compute_cache_key() in isolation (unit tests)
- ✅ **Clear priority order** - Comments document fallback logic (metadata > file > memory > EP)
- ✅ **Easier to extend** - Adding new cache_key source is straightforward
- ✅ **Reduced complexity** - Main function shorter (~30 lines vs ~52)
- ✅ **Pure function** - No side effects, easy to reason about
- ✅ **Better documentation** - Function name and structure clarify intent

**Migration path:**
- Incremental refactoring - extract functions one by one
- No API changes (initialize_context signature stays the same)
- Tests pass at each step

## Evidence

**initialize_context() function:**
- morphizen_compile_model.cpp:362-436 - Full implementation (~75 lines currently)

**Complex cache_key logic (main complexity):**
- morphizen_compile_model.cpp:387-409 - 4-way cache_key computation (~23 lines)
- Lines 395, 402, 407, 412 - Four different cache_key sources (metadata > file > memory > EP)

**Dead code (removed by other issues):**
- Line 417 - update_config_by_target() call (Issue #017 removes ~15 lines)
- Lines 425, 427-429 - cache_dir + const-cast + morphizen_log_dir (Issue #006 removes ~5 lines - dead code, never read)

**After #005, #006, #017 complete:**
- Lines 362-385: Context creation (~24 lines)
- Lines 387-409: cache_key computation (~23 lines) ← EXTRACT THIS
- Lines 415-436: Remaining logic (~5 lines)
- **Total:** ~52 lines → extract cache_key → ~30 lines

## Context

### Why Refactor After #005, #006, #017?

**After Issue #005 (cache_key to ContextProto):**
- Lines 395, 402, 407 change from ConfigProto mutations to ContextProto mutations
- cache_key logic becomes clearer (not mixed with ConfigProto immutability)
- Better time to extract compute_cache_key() function

**After Issue #006 (remove cache_dir):**
- Lines 425, 427-429 deleted entirely (update_cache_dir, const-cast, morphizen_log_dir - all dead code)
- Removes ~5 lines of dead code that was never read

**After Issue #017 (remove update_config_by_target):**
- Line 417 deleted entirely
- Removes ~15 lines

**Result: Function shrinks from ~75 to ~52 lines**
- Context creation: ~24 lines (unchanged)
- cache_key computation: ~23 lines (the remaining complexity)
- Other logic: ~5 lines (simple, clear)

**After extracting cache_key: ~30 lines total**
- Much easier to understand and maintain

### Why Only Extract cache_key?

**Dead code already removed by #006:**
- const-cast (line 427) - DELETED by #006 (dead code)
- morphizen_log_dir (line 428-429) - DELETED by #006 (never read)

**Remaining code after #005, #006, #017:**
- suffix_counter reading (4 lines) - simple, clear
- target_auto_discovery (1 line) - simple
- print_version_info (1 line) - simple
- maybe_create_tar_file_for_write (3 lines) - simple

**Only cache_key computation is complex (~23 lines, 4-way logic)**
- Extracting it is sufficient to simplify the function

### Testing Strategy

**Before refactoring (current state):**
- Cannot test cache_key computation without full PassContext setup
- Must create Model, ConfigProto, provider_options, etc. just to test cache_key logic
- Integration test only - hard to test individual branches

**After refactoring:**
```cpp
// Unit test for cache_key computation (ISOLATED, no PassContext needed)
TEST(CacheKeyTest, ComputeFromMetadata) {
  Model model = create_model_with_metadata("cache_key_123");
  auto cache_key = compute_cache_key(model, "", {}, MODEL_LOAD_TYPE_MEMORY);
  EXPECT_EQ(cache_key, "cache_key_123");
}

TEST(CacheKeyTest, FallbackToFileHash) {
  Model model = create_model_without_metadata();
  auto cache_key = compute_cache_key(model, "/path/to/model.onnx", {}, MODEL_LOAD_TYPE_FILE);
  EXPECT_EQ(cache_key, compute_file_hash("/path/to/model.onnx"));
}

TEST(CacheKeyTest, FallbackToMemorySignature) {
  Model model = create_model_without_metadata_or_file();
  auto cache_key = compute_cache_key(model, "", {}, MODEL_LOAD_TYPE_MEMORY);
  EXPECT_EQ(cache_key, compute_memory_signature(model));
}

TEST(CacheKeyTest, FallbackToEPContext) {
  Model model = create_model_with_ep_context("ep_cache_key_456");
  auto cache_key = compute_cache_key(model, "", {}, MODEL_LOAD_TYPE_MEMORY);
  EXPECT_EQ(cache_key, "ep_cache_key_456");
}

TEST(CacheKeyTest, PriorityOrder) {
  // Verify metadata takes priority over other sources
  Model model = create_model_with_all_cache_sources("metadata", "file", "memory", "ep");
  auto cache_key = compute_cache_key(model, "/path", {}, MODEL_LOAD_TYPE_MEMORY);
  EXPECT_EQ(cache_key, "metadata");  // Metadata wins
}
```

**Benefits:**
- Easy to test each cache_key source in isolation
- Easy to verify priority order
- No need for PassContext, ConfigProto, etc.
- Fast unit tests (no heavy initialization)

## Acceptance Criteria

**Prerequisites (should complete first):**
- [ ] Issue #005 completed (cache_key moved to ContextProto)
- [ ] Issue #006 completed (cache_dir removed - cleans up dead code)
- [ ] Issue #017 completed (update_config_by_target removed)

**Implementation:**
- [ ] Extract compute_cache_key() function as static function (pure, no side effects)
- [ ] Update initialize_context() to call compute_cache_key()
- [ ] Function reduced from ~52 lines to ~30 lines
- [ ] All existing tests pass

**Testing:**
- [ ] Add unit tests for compute_cache_key() covering all 4 sources
- [ ] Add test for priority order (metadata > file > memory > EP)
- [ ] Add test for empty cache_key case (no sources available)
- [ ] Existing integration tests still pass
- [ ] Code coverage maintained or improved

**Documentation:**
- [ ] Add function comment for compute_cache_key() describing purpose and priority order
- [ ] Document cache_key priority: metadata > file hash > memory signature > EP context node
- [ ] Add comments in function explaining each fallback step

## Notes

### Not a Blocker for Immutable ConfigProto

**This issue is OPTIONAL for achieving immutable ConfigProto:**
- Issues #003-#018 are sufficient to make ConfigProto immutable
- This issue improves code quality but doesn't change ConfigProto behavior
- Can be done anytime after #005 and #017 complete

**Why create this issue:**
- initialize_context() will still have complex cache_key logic after #005, #006, #017
- Refactoring now prevents future maintenance debt
- compute_cache_key() can be unit tested in isolation
- Better testability helps prevent cache_key bugs
- May be useful elsewhere (other code that needs cache_key computation)

### Execution Order

**Recommended order:**
1. Complete Issue #005 (cache_key to ContextProto)
2. Complete Issue #006 (remove cache_dir - deletes dead code)
3. Complete Issue #017 (remove update_config_by_target)
4. **Then this issue** (extract compute_cache_key)

**Why this order:**
- #005: cache_key logic becomes clearer (ContextProto vs ConfigProto)
- #006: Removes const-cast dead code (lines 427-429)
- #017: Removes update_config_by_target (line 417)
- After all three: Function is ~52 lines, cache_key is the main remaining complexity
- Extract cache_key: Function becomes ~30 lines, much clearer

### Low Risk Refactoring

**Risk level:** LOW (pure refactoring, no behavior changes)

**Safety measures:**
- Extract functions one at a time
- Run tests after each extraction
- No API changes (initialize_context signature unchanged)
- Incremental commits (easy to rollback if needed)

### Estimated Effort

**Effort:** 1-2 hours (one session)

**Breakdown:**
- Extract compute_cache_key() function: 30 minutes
- Update initialize_context() to use it: 15 minutes
- Write unit tests (5 test cases): 30 minutes
- Documentation (function comment + inline comments): 15 minutes

**Complexity:** LOW (simple extraction refactoring, single function)

**Why low complexity:**
- Only one function to extract (compute_cache_key)
- Pure function (no side effects, no state changes)
- No API changes (initialize_context signature unchanged)
- Existing tests verify correctness automatically
