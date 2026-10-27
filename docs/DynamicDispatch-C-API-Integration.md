<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# DynamicDispatch C API Integration for hip-ep

## Problem Statement

When calling C++ virtual functions across DLL boundaries (hip-ep → DynamicDispatch), we encountered access violations due to:
- **Vtable issues**: Virtual function tables not properly resolved across DLL boundaries
- **ABI incompatibilities**: Template instantiations and STL types causing memory layout mismatches
- **Symbol resolution**: C++ name mangling and template specialization issues

Error observed: `exception 0xc0000005` when calling `gemm_state->op->execute(inputs, outputs)`

## Solution: Use DynamicDispatch C API

DynamicDispatch provides a stable C API (`dd_c_api.h`) that eliminates these issues by:
- Using **opaque handles** instead of C++ objects
- Providing **C function calls** instead of virtual methods
- Using **simple C types** for parameters

## Implementation Changes

### 1. Updated Includes in `dynamic_dispatch.cpp`

**Before:**
```cpp
#include <ops/dmacompiler/combined_gemm/combined_gemm.hpp>
#include <ops/dmacompiler/iconv/iconv.hpp>
#include <ops/op_interface.hpp>
#include <xrt_context/xrt_context.hpp>
```

**After:**
```cpp
#include <ops/dd_c_api.h>
```

### 2. Changed Operator State to Use Handles

**Before:**
```cpp
struct DDGemmState : OpStateT<DDGemmState> {
  using GemmOp = ryzenai::combined_gemm<uint16_t, uint8_t, uint16_t>;
  std::unique_ptr<GemmOp> op;
  DDGemmState(std::unique_ptr<GemmOp> &&gemm_op) : op(std::move(gemm_op)) {}
};
```

**After:**
```cpp
struct DDGemmState : OpStateT<DDGemmState> {
  dd_gemm_handle_t handle;
  DDGemmState(dd_gemm_handle_t h) : handle(h) {}
  ~DDGemmState() {
    if (handle) {
      dd_combined_gemm_destroy(handle);
    }
  }
};
```

### 3. Rewrote `wrap_dd_matmul` to Use C API

**Key changes:**
- Use `dd_combined_gemm_create()` instead of C++ constructor
- Use `dd_combined_gemm_initialize_weights()` for weights
- Use `dd_combined_gemm_execute()` instead of virtual function call
- Use `dd_tensor_t` struct instead of C++ `Tensor` class

**Current API** (DD commit `57b0f8604`, "Pass full op configuration through
the C API wrappers"): `create` takes a config struct, and
`initialize_weights` takes an array of const-param tensors.

```cpp
dd_combined_gemm_config_t cfg{};
cfg.a_dtype = "uint16"; cfg.b_dtype = "uint8"; cfg.c_dtype = "uint16";
cfg.design_param = "8x4";
cfg.model_name = "8x4_PSMU_ST1_v3.2";      // selects the xclbin; required
int shape[5] = {1, 6, 1, 1024, 64};        // Bin, B, M, K, N
std::copy(shape, shape + 5, cfg.input_shape);
dd_gemm_handle_t h = dd_combined_gemm_create(&cfg);

dd_tensor_t consts[3] = {weights /*[B,K,N] uint8*/, qdq /*[B,N] int64*/,
                         qdq_params /*[B,16] int32*/};
dd_combined_gemm_initialize_weights(h, consts, 3);
dd_combined_gemm_execute(h, &input /*[Bin,M,K]*/, &output /*[B,M,N]*/);
```

### 4. Current hardcoded mapping in `wrap_dd_matmul` / `wrap_dd_conv2d`

The compiler does not yet pass model selection or QDQ data to the runtime, so
`lib/Runtime/real/dynamic_dispatch.cpp` hardcodes them. The result exercises
the DD call sequence but is **not numerically correct**:

- Only float16 activations are accepted (`"uint16"`); weights are
  reinterpreted as `"uint8"`; qdq / qdq_params are zero-filled.
- combined_gemm only runs exact `{Bin,B,M,K,N}` shapes. A 2-D GEMM
  `[1,K] x [K,B*N]` maps to the `Bin=1, M=1` kernels `{1,3,1,1024,64}` and
  `{1,6,1,1024,64}` (output `[B,1,N]` is byte-identical to `[1,B*N]`). Other
  shapes return an error.
- iconv uses model `"8x4PSP1"`, NCHW, the op's strides, and requires `n == 1`,
  `group == 1`, no dilation, and channel counts that are multiples of 8 (DD
  pads activations to 8 channels). Padding and bias are ignored.
- The handle and its weights are created once and cached in the op-state slot.
  The slot may already hold another op's state, because `hip.matmul`'s
  `OpStateOpInterface` constructor installs a hipBLASLt `MatmulState` at
  session init even when the op is lowered to `wrap_dd_matmul`. The DD
  wrappers recognize their own state by its `deletor` and replace any other
  occupant on first use.
- DD reads and writes tensors from the host, but the EP's buffers are HIP
  device allocations. Weights (once), inputs and outputs are therefore staged
  through host buffers with synchronized `hipMemcpyAsync` copies on the
  runtime stream. Passing device pointers directly faults inside
  `combined_gemm::execute`.
- Generated code currently ignores the `wrap_dd_*` return code, so a DD failure
  does not fail inference.

`test/python/test_dd_matmul.py` drives a MatMul through `wrap_dd_matmul`;
DD's `tests/test_c_api.cpp` (`test_wrap_dd_matmul`) issues the identical dd_*
calls directly, isolating DD failures from DLL-boundary issues.

## Build Configuration

### CMakeLists.txt Changes

Added DD source include directory:
```cmake
if(DEFINED ENV{DD_ROOT})
    list(APPEND DD_SEARCH_PATHS $ENV{DD_ROOT}/include)
endif()
```

### Required Environment Variables

- `DD_ROOT`: DynamicDispatch source directory (e.g., `c:\vai-rt-0\vai-rt-build\workspace\dod`)
- `DYNAMICDISPATCH_ROOT`: DynamicDispatch build directory (e.g., `c:\vai-rt-0\vai-rt-build\workspace\dod\build\Release`)
- `XILINX_XRT`: XRT installation directory

### Build Command

```bash
python build.py \
  --hip_arch gfx1150 \
  --therock_dist "d:\DEVELOP\WORKFINAL\_therock" \
  --install_dir "d:\DEVELOP\WORKFINAL\install" \
  --cmake_prefix_path "d:\DEVELOP\WORKFINAL\local" \
  --skip_tests
```

## Testing

### Test Model Requirements

Models must use **float16** (not float32):
```python
A = helper.make_tensor_value_info('A', TensorProto.FLOAT16, [M, K])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT16, [K, N])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT16, [M, N])
```

### Environment Variables for Testing

```bash
export HIPEP_USE_DYNAMIC_DISPATCH=1
export MORPHIZEN_DEBUG_MLIR_BACKEND=1  # Optional: for debug output
```

## Status

- ✅ C API integration code complete
- ✅ Data type mapping fixed
- ✅ C API header (`dd_c_api.h`) installed to DYNAMICDISPATCH_ROOT
- ⏳ Build integration in progress
- ⏳ Runtime testing pending

## Next Steps

1. Complete build with C API integration
2. Test with float16 MatMul model
3. Verify no access violations when calling execute()
4. Add Conv2D support using `dd_iconv_*` functions
5. Handle additional parameters (bias, alpha, beta, transpose flags)

## Linking Against a Default DynamicDispatch Build

This test checks whether a third-party DLL can link against the stock
`dyn_dispatch_core.dll` from DynamicDispatch `main`, which does not include
the `dd_*` C API, without protobuf, C runtime (CRT) or other link conflicts.

### Setup

- hip-ep objects are compiled against a DD tree that has `ops/dd_c_api.h`.
- `hipgpu.dll` is linked against the default DD build's `lib/`, which is
  `DYNAMICDISPATCH_ROOT` set to `<dd>/build/Release`.
- CMake detects `bin/dyn_dispatch_core.dll` and uses DLL mode. The link line
  then has only `dyn_dispatch_core.lib` and `dyn_bins.lib` (import
  libraries), plus `xrt_coreutil.lib`. DD's static helper libraries
  (`transaction`, `pm_package`, `xclbin`, `hashlib`) are not linked.

### Result

The only errors are the 8 C API entry points that the default build does not
export:

```
error LNK2019: unresolved external symbol dd_combined_gemm_create
error LNK2019: unresolved external symbol dd_combined_gemm_initialize_weights
error LNK2019: unresolved external symbol dd_combined_gemm_execute
error LNK2019: unresolved external symbol dd_combined_gemm_destroy
error LNK2019: unresolved external symbol dd_iconv_create
error LNK2019: unresolved external symbol dd_iconv_initialize_weights
error LNK2019: unresolved external symbol dd_iconv_execute
error LNK2019: unresolved external symbol dd_iconv_destroy
fatal error LNK1120: 8 unresolved externals
```

There are no protobuf, CRT or duplicate-symbol errors caused by DD. The
warnings in this link are not caused by DD either:

| Warning | Source | DD-related? |
|---|---|---|
| 50 x LNK4217 (`google::protobuf::...`) | hip-ep's `mlir_compilation_proto` was compiled with `PROTOBUF_USE_DLLS=0`. Protobuf tests `defined(PROTOBUF_USE_DLLS)`, so that turned on `dllimport` against a static `libprotobuf.lib` | No; also present in non-DD builds. Fixed by removing the define (see below) |
| LNK4098 (`defaultlib 'MSVCRT' conflicts`) | vai-rt's static `libcrypto.lib` (OpenSSL, built `/MD`), pulled in by `morphizen-foundation`. hip-ep is built `/MT` | No; `-VERBOSE` shows no member of any DD library loaded |
| LNK4070 (`/OUT` directive in `.EXP`) | EP export file names the DLL `onnxruntime_morphizen_ep.dll` | No |

### Linking Against a DD Build With `DD_EXPORT_ALL_SYMBOLS=ON`

`DD_EXPORT_ALL_SYMBOLS=ON` (also turned on by DD's `--en-tests`) makes
`dyn_dispatch_core.dll` export about 27,000 symbols, roughly 1,100 of them
protobuf. The default build exports about 2,150, with 1 protobuf symbol.

With `PROTOBUF_USE_DLLS=0` still on `mlir_compilation_proto`, the `dd_*`
symbols resolve, but the link fails:

```
dyn_dispatch_core.lib(dyn_dispatch_core.dll) : error LNK2005:
  google::protobuf::Message::~Message(void) already defined in
  morphizen-core-static.lib(capability.pb.obj)
  (also ParseContext::Done, EpsCopyOutputStream::Encode64,
   EpsCopyOutputStream::WriteStringMaybeAliased,
   CodedOutputStream::WriteVarint32ToArray)
fatal error LNK1169: one or more multiply defined symbols found
```

The `dllimport` references from `metadata.pb.obj` are satisfied by DD's
exported protobuf symbols, which then clash with hip-ep's own static protobuf.

After removing `PROTOBUF_USE_DLLS=0`, `hipgpu.dll` links cleanly against this
DD build. The only remaining warnings are LNK4098 and LNK4070, and
`dumpbin /imports` shows that the only imports from `dyn_dispatch_core.dll`
are the 8 `dd_*` functions.

### Guidance for Third-Party DLLs Using `dyn_dispatch_core.dll`

- If the consumer links protobuf statically, never define `PROTOBUF_USE_DLLS`
  in any form. It is a presence flag, so `PROTOBUF_USE_DLLS=0` still turns on
  `dllimport` and changes class layout (`arena_impl.h`). A DD build with
  `DD_EXPORT_ALL_SYMBOLS=ON` will then satisfy those imports with DD's own
  protobuf, which causes LNK2005 errors or two protobuf runtimes in one
  process.

- Link only the import libraries (`dyn_dispatch_core.lib`, plus
  `dyn_bins.lib` if needed). DD's protobuf, spdlog and other internals stay
  inside `dyn_dispatch_core.dll`. With the C API, only `extern "C"` functions
  and POD structs (`dd_tensor_t`, `dd_*_config_t`) cross the boundary, so the
  consumer does not need DD's protobuf version or matching STL types.
- Do not link DD's static helper libraries (`transaction.lib`,
  `pm_package.lib`, `xclbin.lib`, `hashlib.lib`). They are built with the
  static CRT (`/DEFAULTLIB:LIBCMT`, `libcpmt`) and are already inside the DLL.
  Linking them into a `/MD` consumer causes LNK4098 and duplicate CRT state.
- The consumer only needs DD to export the `dd_*` C API. Until DD `main`
  carries it, use a DD build that includes `dd_c_api.h` and its exports.

To reproduce the diagnosis:

```bash
# Which libraries request which CRT
dumpbin /directives <lib> | grep -i DEFAULTLIB
# Which library members the linker actually loads (use -VERBOSE in Git Bash;
# /VERBOSE gets rewritten to a path by MSYS)
LINK=-VERBOSE cmake --build <build> --config Release --target <ep-target> \
  | grep -iE "Loaded |LNK4098"
```

## References

- DynamicDispatch C API Guide: `DD_ROOT/DD-C-API-Integration-Guide.md`
- C API Header: `DD_ROOT/include/ops/dd_c_api.h`
- hip-ep Integration: `lib/Runtime/real/dynamic_dispatch.cpp`

## Test Results

### Build Status
✅ **SUCCESS** - C API integration builds successfully with C++17

### Runtime Status
✅ **C API Functions Called** - Successfully calling DD C API functions:
- `dd_combined_gemm_create()` - Works
- `dd_combined_gemm_initialize_weights()` - Called but crashes due to data mismatch

### Current Issue: Data Type Mismatch

**Problem**: `combined_gemm` expects quantized uint8 weights, but test provides float16 weights.

**Stack trace**:
```
#4 0x00007ffdd7b80bc4 in dd_combined_gemm_initialize_weights
#5 0x00007ffe08478f10 in wrap_dd_matmul
```

**Root cause**: DynamicDispatch `combined_gemm` is designed for quantized inference:
- Activations: float16 (uint16 representation) ✓
- **Weights: uint8 (quantized)** ✗ Our test uses float16
- Output: float16 (uint16 representation) ✓

**Next steps**:
1. Either quantize weights before passing to DD
2. Or check if DD has a float16x float16 MatMul operator
3. Or only use DD for operations it's designed for (quantized inference)

The C API integration itself is **working correctly** - the crash is in DD's weight initialization due to incorrect data format, not in the C API layer.
