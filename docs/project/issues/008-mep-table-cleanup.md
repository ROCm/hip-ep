<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #008: MEP Table Provider Options Injection Cleanup

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Cleanup
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-30 (Decision made after evaluation - remove entirely)
- **Dependencies:** None

## Description

Remove MEP (Model Entry Point) table feature entirely.

**Decision after evaluation:** MEP table does not scale and has high maintenance burden. Remove completely.

**What was MEP table:**
- Lookup table: MD5 checksum of ONNX model → MEPProto (model-specific configuration)
- Automatically injected provider_options when model fingerprint matched
- Intended for performance optimization, auto-configuration, and compatibility

**Why remove:**
- **Doesn't scale** - Can't maintain table for thousands of models released daily
- **Fragile** - MD5 breaks with any model modification
- **Too early for GPU** - This is a new GPU project (original was NPU), requirements unclear
- **High maintenance burden** - Each entry requires testing and validation
- **Better alternatives exist** - Smart defaults + user override + documentation

**Current status:**
- Partially removed (mepTable deleted from built-in config, MEPProto removed, injection code removed)
- Need to complete cleanup (update docs, remove references)

## Context

**Part of provider_options cleanup effort** (see docs/technical/cleanup-provider-options.md).

**provider_options pollution problem:** provider_options is modified at multiple stages:
1. User explicit values
2. Config file provider_options section
3. MEP table hit (this issue - being removed)
4. TargetProto hit (Issue #009)

**From Issue #007 discussion:**
> User: "MEP Table again is another issue, we need another discussion thread."

### What MEP Table Was

**MEP (Model Entry Point) Table:**
- Table of key-value pairs in config file
- **Key:** MD5 checksum of ONNX model
- **Value:** MEPProto object containing model-specific configuration

**Example (conceptual):**
```json
{
  "mepTable": {
    "abc123def456...": {  // MD5 of ResNet50
      "model_name": "resnet50",
      "model_category": "classification",
      "model_variant": "v1.5",
      "is_preemptible": "true",
      "qos_priority": "high",
      "dd_use_lazy_scratch_bo": "false"
    },
    "789ghi012jkl...": {  // MD5 of BERT
      ...
    }
  }
}
```

**How it worked:**
1. Load model → compute MD5 signature
2. Lookup MD5 in MEP table
3. If found → inject MEPProto fields into provider_options
4. Model runs with auto-configured settings

**Original purpose (NPU project):**
- **Performance optimization:** Different models need different runtime settings
- **Auto-configuration:** Users don't need technical knowledge, system auto-configures
- **Compatibility:** Specific models require specific settings to work correctly

**Maintained by:** AMD/Vendor (pre-populated known models, shipped in built-in config)

### NPU vs GPU Context

**NPU (original project):**
- MEP table was valuable
- AMD knew NPU hardware well, could pre-tune for specific models
- Model-specific settings (qos_priority, is_preemptible) were important

**GPU (this new project):**
- Different hardware characteristics
- Uncertain if model-specific tuning needed
- Hard to predict user requirements
- Too early to commit to MEP table approach

### Why MEP Table Doesn't Scale

1. **Maintenance explosion** - New models released daily (HuggingFace: 500k+ models)
2. **Fragile identification** - MD5 breaks if user modifies model (quantization, shape changes)
3. **Hidden magic** - Users don't understand why settings applied, hard to debug
4. **Testing burden** - Each entry needs validation across hardware/driver versions
5. **Configuration drift** - Built-in table becomes stale, can't update without new binary

### Better Alternatives

**Smart defaults + user override:**
```cpp
// Good defaults work for 80% of cases
// Users override when needed
provider_options["qos_priority"] = "high";  // Explicit, transparent
```

**Runtime heuristics:**
```cpp
if (model_size > 1GB) { /* adjust settings */ }
if (contains_attention_ops) { /* transformer optimizations */ }
```

**Documentation + examples:**
- Best practices for different model types
- Configuration examples
- Users learn and configure themselves

## Solution

### Complete Removal Strategy

**Decision:** Remove MEP table feature entirely.

**What to remove:**

1. **Delete mepTable from config** (✅ Already done)
   - config_json_binary.hpp.py:28-29 deletes mepTable from built-in config
   - Prevents shipping MEP table in binary

2. **Update documentation** (To do)
   - docs/technical/cleanup-provider-options.md lines 42-53
   - Remove MEP table section or mark as REMOVED
   - Explain why removed and alternatives

3. **Update target-auto-discovery.md** (To do)
   - Priority 2 mentions MEP table
   - Remove or mark as deprecated
   - Clarify that only 3 priorities remain (user, plugin, config default)

4. **Verify removal complete** (To do)
   - Search for any remaining MEPProto references
   - Search for MEP table lookup code
   - Confirm injection code removed

**What to keep:**

1. **MD5 signature infrastructure** (Keep)
   - morphizen_compile_model.cpp:251-360
   - `get_model_signature()`, `get_model_signature_with_graph_inputs_and_outputs()`
   - Still useful for cache keys and debugging
   - Low maintenance cost

2. **Config file flexibility** (Keep)
   - Users can still provide custom config files
   - Just without MEP table support

### Implementation Steps

**Step 1: Verify current state**
```bash
# Search for any remaining MEP/mep references
grep -r "MEP\|mep" morphizen-core/src/
grep -r "MEPProto" morphizen-core/src/

# Confirm MEPProto message doesn't exist
grep "message MEPProto" morphizen-core/src/*.proto
```

**Step 2: Update documentation**
```markdown
# docs/technical/cleanup-provider-options.md

## ~~set when MEP table hit~~ REMOVED

MEP table feature has been removed entirely. See Issue #008.

**Why removed:**
- Doesn't scale (maintenance burden)
- Fragile (MD5-based identification)
- GPU requirements unclear (too early to commit)

**Alternatives:**
- Use smart defaults
- Users set provider_options explicitly when needed
- Consult documentation for model-specific best practices
```

**Step 3: Update target-auto-discovery.md**
```markdown
# Target auto-discovery priorities

1. provider_options["target"] - User explicit
~~2. MEP table target - REMOVED~~
3. Plugin auto-discovery
4. ConfigProto.target default
```

**Step 4: Final verification**
- No MEPProto references
- No MEP table lookup code
- Documentation updated
- Issue marked complete

### Benefits of Removal

- ✅ No maintenance burden
- ✅ Simpler system (one less layer)
- ✅ More transparent (users know what settings they have)
- ✅ Flexible (can re-add if GPU requirements clarify later)
- ✅ MD5 infrastructure preserved (useful for other purposes)

### Reversibility

**If we discover we need MEP table later:**
- MD5 signature code still exists
- Can define MEPProto message again
- Re-implement lookup and injection
- Low cost to restore if truly needed

**This is a reversible decision** - removing now doesn't prevent adding back later with more information.

## Sessions

### 2026-01-30: Issue Created as Placeholder

**Context:** Identified during Issue #007 discussion about target cleanup.

**User guidance:**
> "MEP Table again is another issue, we need another discussion thread."

**Purpose:** Document that MEP table needs cleanup. Defer detailed investigation and solution design.

### 2026-01-30: Investigation and Evaluation

**User:** "let's discuss about 008"

**Initial investigation:**
- Searched for MEP table code - found it's already partially removed
- mepTable deleted from built-in config (config_json_binary.hpp.py:28-29)
- No MEPProto message found
- No injection code found in morphizen_compile_model.cpp

**Discovery:** MEP table appears already removed, but needed to understand original design to evaluate completion.

**User explained original design:**
> "let me explain the original design or MEP table. it is a table of key value pairs. key is the md5 check sum of a onnx model, and value is a MEPPRoto object."

**Understanding built:**
- MD5 signature infrastructure still exists (lines 251-360)
- Table structure: MD5 → MEPProto (model-specific config)
- Auto-injection of provider_options when model fingerprint matched

**Evaluation questions explored:**

**Q1: What problem did MEP table solve?**
- User: "A, B, and C. all are correct."
  - A) Performance optimization
  - B) Automatic configuration
  - C) Model compatibility

**Q2: Who populated the MEP table?**
- User: "A" (AMD/Vendor)
- Pre-configured known models, shipped in built-in config
- Not user-extensible

**Q3: What changed that led to partial removal?**
- User: "this is a new project. original one is for NPU. now it is for GPU. I am not sure for GPU, it is also an issue, it is hard to predict user requirement."

**Key insight:** NPU → GPU migration, requirements uncertain for new hardware.

**Q4: Should we keep it for GPU?**
- User: "too early to tell. what do you think about this feature in general. it is a burden for developers to maintain this table."

**Analysis presented:**
- MEP table doesn't scale (can't maintain for all models)
- Fragile (MD5 breaks easily)
- Hidden magic (users confused)
- High testing/maintenance burden
- Better alternatives exist (defaults + user override + docs)

**Recommendation:** Remove entirely because:
- Maintenance burden >> benefit
- Too early to commit for GPU
- Better alternatives available
- Reversible decision (can re-add if needed)

**User decision:**
> "yes, remove it entirely."

**Final decision:** Complete removal of MEP table feature.

**Rationale agreed upon:**
- Doesn't scale to modern model ecosystem
- GPU requirements unclear (too early)
- Smart defaults + documentation is better approach
- Can restore later if proven necessary (MD5 infrastructure preserved)

## Related PRs

- PR #159, #155 - Mentioned as deprecation effort (to be investigated)

## Related Branches

_None yet._

## Notes

### Current State (Partially Removed)

**Already removed:**
- ✅ mepTable deleted from built-in config (config_json_binary.hpp.py:28-29)
- ✅ MEPProto message removed (no proto definition found)
- ✅ Injection code removed (no references in morphizen_compile_model.cpp)
- ✅ Related fields removed (qos_priority, etc. - see config.cpp:121-128 comments)

**Still exists:**
- MD5 signature infrastructure (morphizen_compile_model.cpp:251-360)
  - `get_model_signature()` - Based on topologically ordered nodes
  - `get_model_signature_with_graph_inputs_and_outputs()` - Based on I/O signatures
  - Used for debugging, potentially useful for cache keys

**Still to do:**
- Update docs/technical/cleanup-provider-options.md (remove MEP table section)
- Update docs/technical/target-auto-discovery.md (remove Priority 2 reference)
- Final verification (no remaining references)

### MD5 Signature Infrastructure (Keep)

**Location:** morphizen_compile_model.cpp:251-360

**Two signature algorithms:**

1. **Algorithm A - Topological (line 310-339):**
   ```cpp
   static std::string get_model_signature(const Graph& onnx_graph);
   ```
   - Iterate nodes in topological order
   - Hash: op_type, node_arg names, shapes, types
   - Skip ops in XLNX_MD5_SIG_SKIP_OPS (default: QuantizeLinear, DequantizeLinear)

2. **Algorithm B - I/O Signature (line 284-308):**
   ```cpp
   static std::string get_model_signature_with_graph_inputs_and_outputs(const Graph& onnx_graph);
   ```
   - Hash only graph inputs and outputs (names + shapes)
   - Simpler, less sensitive to internal changes

**Usage (line 341-360):**
```cpp
static std::string get_signature(const std::string& model_path,
                                 const Graph& onnx_graph,
                                 ConfigProto& /*proto*/) {
  auto md5_file_base = get_md5_of_file(model_path);       // File-based
  auto md5_in_memory_a = get_model_signature(onnx_graph); // Algorithm A
  auto md5_in_memory_b = get_model_signature_with_graph_inputs_and_outputs(onnx_graph); // Algorithm B

  LOG << "File base signature : " << md5_file_base;
  LOG << "Algorithm-A: based on topologically ordered signature : " << md5_in_memory_a;
  LOG << "Algorithm-B: based on graph inputs/outputs signature : " << md5_in_memory_b;

  return md5_in_memory_a;  // Uses Algorithm A
}
```

**Why keep:**
- Useful for cache keys (identify models uniquely)
- Debugging (model fingerprinting)
- Low maintenance cost (self-contained)
- Future use cases (cache hit detection, model versioning)

### Removed Fields (config.cpp:121-128)

**Comments showing removal:**
```cpp
// Removed: update_target_attr - depended on removed target_opts field
// Removed: update_xclbin - NPU-specific xclbin firmware handling
// Removed: update_hw_context_share - depended on removed share_hw_context field
// Removed: update_graph_engine_qos_priority - depended on removed graph_engine_qos_priority field
```

**These were likely MEP table-injected fields** - already cleaned up in previous work.

### Documentation to Update

**1. docs/technical/cleanup-provider-options.md (lines 42-53):**

Current:
```markdown
## set when MEP table hit
[see here](../morphizen-core/src/morphizen_compile_model.cpp#L502)
it is to be deprecated, see PR #159 #155
```

Update to:
```markdown
## ~~set when MEP table hit~~ REMOVED

MEP table feature removed entirely (Issue #008).
See issue for rationale and alternatives.
```

**2. docs/technical/target-auto-discovery.md:**

Current mentions Priority 2 as MEP table. Update to:
```markdown
1. provider_options["target"] - User explicit
~~2. MEP table target - REMOVED (Issue #008)~~
3. Plugin auto-discovery
4. ConfigProto.target default
```

### Why Removal is Correct

**Fundamental scaling problem:**
- Modern ML ecosystem: 500k+ models on HuggingFace alone
- New models daily (fine-tuning, quantization, custom architectures)
- Cannot maintain hand-curated table
- MD5 fragile (any modification breaks fingerprint)

**NPU to GPU transition:**
- MEP table designed for NPU (AMD knew hardware well)
- GPU is new territory (requirements unclear)
- Too early to lock in pre-configuration approach
- Better to learn from users first

**Better alternatives proven:**
- Smart defaults work for most cases
- Users override when needed (explicit, debuggable)
- Documentation scales better than pre-configuration
- Runtime heuristics can adapt automatically

**Reversible:**
- MD5 infrastructure preserved
- Can restore if proven necessary
- Low risk decision
