# morphizen-hipdnn Complete Work Log

**Date:** January 12-13, 2026  
**Project:** morphizen-hipdnn and hipDNNEP  
**Operator:** mingyue  
**Objective:** Build hipDNNEP and test ResNet50 on Windows with AMD GPU

---

## Environment Information

### Hardware
- **GPU:** AMD Radeon(TM) 8050S Graphics
- **Architecture:** gfx1151 (RDNA 3)
- **VRAM:** 24.26 GB (integrated graphics)
- **Driver:** 32.0.22029.9039

### Software
- **OS:** Windows 10 (Build 26100)
- **IDE:** Visual Studio 2022
- **CMake:** 3.29
- **Compiler:** MSVC
- **TheRock SDK:** D:\therock (version 7.11.0)
- **HIP:** 7.2.53150-e5316dcbd9
- **ONNX Runtime:** API 2.0 (new ABI)

### Directory Structure
```
D:\Users\mingyue\hipdnn\workspace\
├── hipDNNEP/              # hipDNN Execution Provider
├── morphizen-hipdnn/      # Main project
├── MorphiZen/             # Core framework
├── onnxruntime/           # ONNX Runtime source
├── build/                 # Build outputs
└── local/                 # Installation prefix
```

---

## Phase 1: Build hipDNNEP (Jan 12)

### 1.1 Clone and Configure

```powershell
cd D:\Users\mingyue\hipdnn\workspace
git clone https://github.com/zpye/hipDNNEP.git
cd hipDNNEP

# Configure environment
set THEROCK_DIST=D:\therock
set ONNXRUNTIME_ROOT=D:\Users\mingyue\hipdnn\workspace\local
set PATH=D:\therock\bin;%PATH%

# Configure CMake
cmake --preset Debug -DTHEROCK_DIST=D:\therock -DONNXRUNTIME_ROOT=D:\Users\mingyue\hipdnn\workspace\local
```

### 1.2 Issues Encountered

#### Issue 1: UTF-8 Support ❌ → ✅

**Error:**
```
error C2338: 'Unicode support requires compiling with /utf-8'
```

**Fix:** Modified `CMakeLists.txt`:
```cmake
if(MSVC)
  target_compile_options(hipdnn_ep PRIVATE /utf-8)
endif()
```

#### Issue 2: hip_bfloat16 Conversion ❌ → ✅

**Error:**
```
error C2440: cannot convert from 'float' to 'hip_bfloat16'
```

**Fix:** Pulled latest upstream code (already fixed).

#### Issue 3: Test DLL Dependencies ⚠️

**Error:** Exit code 0xc0000135 (DLL not found)

**Fix:** Build only main target:
```powershell
cmake --build --preset Debug --target hipdnn_ep
```

### 1.3 Build Result

**Status:** ✅ Success

**Output:**
```
D:\Users\mingyue\hipdnn\workspace\hipDNNEP\build\hipDNNEP\Debug\Debug\
├── hipdnn_ep.dll
├── hipdnn_ep.lib
└── hipdnn_ep.pdb
```

---

## Phase 2: Test ResNet50 (Jan 13)

### 2.1 Code Modification

**File:** `morphizen-hipdnn/level-1-pass-hipdnn/src/pass_main.cpp`

**Change:** Reverse node traversal

```diff
- for (auto node_idx : node_indices) {
+ for (auto it = node_indices.rbegin(); it != node_indices.rend(); ++it) {
+   auto node_idx = *it;
    auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
    auto node_ref = NodeConstRef::from_node(ort_graph, *node);
-   MY_LOG(1) << "node_idx: " << node_idx;
+   MY_LOG(1) << "node_idx: " << node_idx << " node_name:" << node_ref.name();
```

**Compile:**
```powershell
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

### 2.2 Test Environment

**Test Program:**
```
D:\Users\mingyue\hipdnn\workspace\build\test_onnx_runner\Debug\test_onnx_runner.exe
```

**Test Model:** `pt_resnet50.onnx`

**Key Environment Variables:**
```bash
USE_ORT_API_2_0=1                     # New ORT ABI
MORPHIZEN_DEBUG_HIPDNN=1              # Debug output
XLNX_ONNX_EP_VERBOSE=2                # Verbose logging
XLNX_ENABLE_CACHE=0                   # Disable cache
MORPHIZEN_VITISAI_EP=D:\...\onnxruntime_vitisai_ep.dll
```

### 2.3 Test Execution

#### Run 1: ❌ Node Index Error

**Error:**
```
Invalid NodeIndex: NodeIndex(index: 342, ...)
Check failed: is_valid() NodeIndex is invalid
```

**Cause:** Node fusion deletes nodes during traversal.

**Fix:** Reverse traversal handles this correctly.

#### Run 2: ✅ Session Creation Success

**Result:** Session created successfully

**Achievements:**
- ✅ VitisAI EP loaded
- ✅ hipDNN Pass executed
- ✅ Conv nodes fused
- ✅ Graph files generated
- ✅ Session ready

#### Run 3: ❌ Infinite Loop

**Status:** ⏳ Under investigation

**Symptoms:**
- Session creation OK
- Hangs at `session.Run()`
- Program unresponsive
- High CPU usage

**Possible Causes:**
1. Custom operator internal loop
2. HIP device synchronization
3. GPU kernel timeout
4. Thread deadlock

**Debugging Steps:**
1. Pause in VS2022 (Ctrl+Alt+Break)
2. Check Call Stack
3. Analyze log patterns
4. Verify GPU status

---

## Complete Status Summary

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| **Phase 1** | hipDNNEP Build | ✅ Complete | `hipdnn_ep.dll` built |
| | UTF-8 Fix | ✅ Fixed | Added `/utf-8` flag |
| | bf16 Conversion | ✅ Fixed | Upstream fix |
| **Phase 2** | Code Modification | ✅ Complete | Reverse traversal |
| | Session Creation | ✅ Success | All passes executed |
| | Inference Run | ❌ **Blocked** | Infinite loop |

---

## Issues Summary

### Resolved ✅

1. **MSVC UTF-8 compilation** - Added compiler flag
2. **hip_bfloat16 type conversion** - Upstream fixed
3. **Node index invalid** - Reverse traversal solved
4. **Session creation** - Successfully working

### Open ❌

1. **Session.Run() infinite loop**
   - Location: Unknown (need debugging)
   - Impact: Cannot run inference
   - Priority: High
   - Next: Use VS2022 debugger to locate

---

## Key Files Modified

```
hipDNNEP/CMakeLists.txt
└── Added MSVC /utf-8 flag

morphizen-hipdnn/level-1-pass-hipdnn/src/pass_main.cpp
└── Lines 312, 315-316, 319: Reverse traversal + logging
```

---

## Next Steps

### Immediate (Debugging)
1. Use VS2022 to pause execution
2. Examine call stack and threads
3. Check HIP API calls
4. Look for repeated log patterns

### Future (After Fix)
1. Complete ResNet50 inference
2. Verify results correctness
3. Performance testing
4. Document final solution

---

## Quick Reference

### Build Commands

```powershell
# hipDNNEP
cd hipDNNEP
cmake --preset Debug -DTHEROCK_DIST=D:\therock -DONNXRUNTIME_ROOT=D:\...\local
cmake --build --preset Debug --target hipdnn_ep

# morphizen-hipdnn
cd morphizen-hipdnn
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

### Debug Commands

```powershell
# Check GPU
D:\therock\bin\hipInfo.exe

# Monitor process
Get-Process test_onnx_runner | Select-Object CPU, WorkingSet

# VS2022: Debug > Break All > Call Stack > Threads
```

### Environment Setup

```powershell
$env:THEROCK_DIST = "D:\therock"
$env:ONNXRUNTIME_ROOT = "D:\Users\mingyue\hipdnn\workspace\local"
$env:USE_ORT_API_2_0 = "1"
$env:MORPHIZEN_DEBUG_HIPDNN = "1"
$env:PATH = "D:\therock\bin;..." + $env:PATH
```

---

## Achievements

✅ Successfully built hipDNNEP on Windows with MSVC  
✅ Resolved multiple compilation issues  
✅ Modified morphizen-hipdnn for correct node traversal  
✅ Session creation working with VitisAI EP  
✅ hipDNN Pass successfully fusing Conv nodes  
⏳ Debugging inference execution hang  

---

**Document Generated:** January 13, 2026  
**Status:** Phase 1 ✅ Complete | Phase 2 🔄 In Progress (debugging infinite loop)
