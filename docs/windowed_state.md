<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Windowed recurrent state

hip-ep supports windowed state for models that use
`CausalConvWithState` and `LinearAttention`. A windowed export stores the
per-token recurrent/conv state in a fixed state window, allowing a decoder to
crop to an accepted speculative prefix instead of replaying that prefix in a
second forward pass.

This support is independent of the decoder/controller integration. An OGA
consumer must call its crop API after a partial acceptance; a model without
the `state_window` attributes continues to use the normal rewind-and-replay
path.

## Build from source

Use the normal Linux build prerequisites and source-build flow from the
[Linux quick start](quick_start_linux.md). The windowed-state changes are
included in the compiler, runtime, and per-architecture custom-kernel
targets.

For a native gfx1151 build:

```bash
python3 build.py \
  --config Release \
  --cmake_generator Ninja \
  --hip_arch gfx1151
```

For an incremental build from an existing CMake tree:

```bash
cmake --build <build-dir> --target libhipgpu.so -j8
cmake --install <build-dir> --config Release
```

The install prefix must contain both the provider and the matching
architecture kernel library:

```text
<install>/lib/libhipgpu.so
<install>/lib/libcustom_kernels_gfx1151.so
```

When using OGA on Linux, expose the provider under the filename expected by
the AMDGPU loader:

```bash
export ROOT=/path/to/hip-ep/install
export THEROCK_DIST=/path/to/therock-dist
export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"
export PATH="$ROOT/bin:$PATH"
cp "$ROOT/lib/libhipgpu.so" "$ROOT/lib/libamdgpu-ep.so"
export AMDGPU_EP_PATH="$ROOT/lib/libamdgpu-ep.so"
```

The ROCm runtime path is required when running tests directly from the build
tree. Without it, ELF loading tests can fail with an error such as
`libamdhip64.so.7: cannot open shared object file`.

## Focused tests

The change has coverage at each compiler/runtime boundary:

- ONNX-to-HIP conversion: `test/lit/Conversion/onnx-to-hip/`
- HIP-to-LLVM lowering: `test/lit/Conversion/hip-to-llvm/`
- HIP dialect verification/canonicalization:
  `test/lit/Dialect/`
- GPU-free numeric checks:
  `test/numeric/tests/test_causal_conv_with_state.py` and
  `test/numeric/tests/test_linear_attention.py`
- GPU state-slot promotion:
  `test/runtime/test_state_slot_copy.cpp`
- Provider export contract:
  `test/runtime/test_hipgpu_elf_contract.cpp`

From the build tree, run the focused runtime tests:

```bash
export LD_LIBRARY_PATH="/opt/rocm/core-7.11/lib:<install>/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir <build-dir> --output-on-failure \
  -R 'StateSlotCopy_gfx1151|HipGpuElfContract'
```

The state-slot test checks mixed descriptor sizes, alignment, stream ordering,
guards, and repeated launches. The ELF test checks provider visibility,
`DF_1_NODELETE`, plugin exports, and the versioned state-slot API exports.

Run the compiler lit suite and the two numeric tests using the repository's
standard test configuration described in the [numeric test guide](../test/numeric/README.md).
The expected numeric coverage includes both `state_window=0` and
`state_window>0` cases.

## State-slot API

The provider exports a versioned C API from `include/hip/device_api.h`:

- `hipdnn_ep_device_api_version()`
- `hipdnn_ep_copy_state_slots(...)`

The second function validates host descriptors, obtains the current provider
stream, and launches one per-architecture HIP kernel over all state
descriptors. It does not synchronize the stream. If the API or matching
kernel is unavailable, consumers should retain their compatibility fallback.

The API is deliberately versioned. Consumers must check version `1` before
calling the copy entry point, and future incompatible changes should use a
new API version rather than changing the existing signature.

## Packaging guidance

Do not commit build directories, installed packages, generated model files,
ONNX external-data files, caches, or profiling JSON. A reviewable change
contains the source, tests, and this documentation only. The OGA/DFlash
controller bridge is a separate consumer-side change and should be reviewed
in the DFlash/OGA change rather than copied into the HIP EP PR.
