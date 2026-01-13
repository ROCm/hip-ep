# ResNet50 Test Execution Work Log

**Date:** January 13, 2026  
**Project:** morphizen-hipdnn  
**Operator:** mingyue  
**Objective:** Run ResNet50 model test using test_onnx_runner (new ORT API 2.0)

---

## Test Environment

### Hardware
- **GPU:** AMD Radeon(TM) 8050S Graphics (gfx1151, RDNA 3)
- **VRAM:** 24.26 GB

### Software
- **OS:** Windows 10 (Build 26100)
- **IDE:** Visual Studio 2022
- **ONNX Runtime:** API 2.0 (new ABI)
- **Test Program:** `D:\Users\mingyue\hipdnn\workspace\build\test_onnx_runner\Debug\test_onnx_runner.exe`
- **Test Model:** `pt_resnet50.onnx`

### Key Environment Variables

```bash
# ORT API 2.0 (Critical)
USE_ORT_API_2_0=1
MORPHIZEN_VITISAI_EP=D:\Users\mingyue\hipdnn\workspace\local\bin\onnxruntime_vitisai_ep.dll

# Debug Flags
DEBUG_VAIP_PASS=1
MORPHIZEN_DEBUG_HIPDNN=1
XLNX_ONNX_EP_VERBOSE=2
DEBUG_DPU_CUSTOM_OP=1

# Cache (Disabled for debugging)
XLNX_ENABLE_CACHE=0
XLNX_ENABLE_CACHE_CONTEXT=0

# Paths
VITISAI_EP_JSON_CONFIG=D:\Users\mingyue\hipdnn\workspace\local\bin\vaip_config.json
PATH=D:\therock\bin;D:\Users\mingyue\hipdnn\workspace\local\bin;...
```

---

## Code Modification

### Changed to Reverse Node Traversal

**File:** `morphizen-hipdnn/level-1-pass-hipdnn/src/pass_main.cpp`

**Changes:**
```diff
- for (auto node_idx : node_indices) {
+ for (auto it = node_indices.rbegin(); it != node_indices.rend(); ++it) {
+   auto node_idx = *it;
    auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
    auto node_ref = NodeConstRef::from_node(ort_graph, *node);
-   MY_LOG(1) << "node_idx: " << node_idx;
+   MY_LOG(1) << "node_idx: " << node_idx << " node_name:" << node_ref.name();
```

**Compilation:**
```powershell
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

---

## Test Execution

### Run 1: ❌ Node Index Error

**Error:**
```
Invalid NodeIndex: NodeIndex(index: 342, ...)
Check failed: is_valid() NodeIndex is invalid
```

**Cause:** Node fusion deletes nodes while traversal uses pre-fusion index list.

**Solution:** Keep reverse traversal, framework handles node validity.

---

### Run 2: ✅ Session Creation Success

**Result:** Session created successfully

**Achievements:**
- ✅ VitisAI EP loaded
- ✅ hipDNN Pass executed
- ✅ Conv nodes fused to HIPDNN operators
- ✅ hipDNN graph files generated
- ✅ ONNX Runtime Session ready

---

### Run 3: ❌ Infinite Loop at Session.Run()

**Status:** ⏳ **Under Investigation**

**Symptoms:**
- Session creation successful
- Hangs when executing `session.Run()`
- Program unresponsive
- Abnormal CPU usage

**Possible Causes:**
1. Custom operator internal loop
2. HIP device synchronization stuck
3. hipDNN graph execution issue
4. GPU kernel waiting timeout

**Debugging Steps:**
1. Pause in VS2022 (Ctrl+Alt+Break)
2. Check Call Stack location
3. Analyze last log messages
4. Check for repeated patterns
5. Verify GPU status with `hipInfo.exe`

---

## Issues Summary

| Issue | Status | Solution |
|-------|--------|----------|
| Node Index Invalid | ✅ Resolved | Reverse traversal + enhanced logging |
| Session Creation | ✅ Success | Code modification worked |
| Session Run Loop | ❌ Open | Under investigation |

---

## Current Status

### Completed ✅
- Modified code for reverse traversal
- Configured ORT API 2.0 environment
- Fixed node index issue
- Session creation working

### In Progress ⏳
- **Debugging infinite loop at session.Run()**
- Need to locate stuck position
- May involve custom operator or GPU sync

### Next Steps
1. Use VS2022 debugger to pause and inspect
2. Analyze call stack and threads
3. Check HIP API calls and GPU operations
4. Add timeout or simplify test case

---

## Quick Reference

### Debug Commands

```powershell
# Check GPU
D:\therock\bin\hipInfo.exe

# Monitor process
Get-Process test_onnx_runner | Select-Object CPU, WorkingSet

# VS2022: Pause → Call Stack → Threads
```

### Modified Files
- `morphizen-hipdnn/level-1-pass-hipdnn/src/pass_main.cpp` (lines 312, 315-316, 319)

---

**Document Generated:** January 13, 2026  
**Status:** 🔄 Session creation ✅ | Inference execution ❌ (infinite loop)
