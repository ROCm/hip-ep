<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Windowed recurrent state

## Purpose and supported operators

hip-ep supports windowed state for models that use
`CausalConvWithState` and `LinearAttention`. A windowed export stores the
per-token recurrent/conv state in a fixed state window. A decoder can then
crop to an accepted speculative prefix instead of replaying that prefix in a
second forward pass.

This feature is model- and export-dependent. A model must carry the
`state_window` attributes and use the supported operator forms. A model
without those attributes continues to use the normal rewind-and-replay path.
KV-cache cropping and recurrent-state promotion are separate mechanisms.

**Models without** `state_window` **attributes continue to use the existing non-windowed execution path; this feature does not change their behavior.**

## Implementation summary

This feature combines additions with targeted changes to existing execution
paths:

- **New compiler and lowering support:** preserve the `state_window` dimension
and attributes through ONNX-to-HIP conversion, HIP dialect processing,
bufferization, and HIP-to-LLVM lowering.
- **New batched state-slot promotion kernel:** add the per-architecture `hip_state_slot_copy` GPU kernel to copy the selected slot across all
recurrent and convolution state tensors in one asynchronous launch.
- **New provider contract:** add the versioned
`hipdnn_ep_copy_state_slots(...)` API and export it from the provider,
with a compatibility path for consumers that cannot use the API.
- **Existing runtime changes:** pass window dimensions and selected slots
through the recurrent/conv wrappers and publish the active model stream to
the promotion launcher.
- **Existing** `LinearAttention` **path change:** retain the burst optimization
when `state_window == 0`, while routing `state_window > 0` through per-token materialization so every retained source slot is valid before promotion. This is needed for accuracy purposes.
- **Unchanged consumer behavior:** crop/promotion integration belongs to the decoder consumer; this HIP EP feature does not modify the inference controller.

The sections below explain the state semantics and each contract in detail.

### Accepted-prefix example in wide speculative verification

For a windowed recurrent model, the per-token path used during wide
verification materializes one recurrent state for each verified position:

```text
wide verification:
  S0 -> S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> S7
```

If the decoder accepts the prefix through `S3`, the later states belong to
rejected speculative positions and must not become the committed state:

```text
accepted state:  S3
rejected states: S4, S5, S6, S7
```

The decoder performs two related operations. It crops the KV/position state to
the accepted length, then asks the provider to copy `S3` into the committed
destination slot `S7`:

```text
copy S3 -> S7
```

The `S3 -> S7` operation is a device-to-device copy across the recurrent and convolution state tensors, not a slot rename. The added batched HIP EP slot-copy kernel performs these copies; the decoder chooses the source and destination slots based on the accepted prefix.

`S4` through `S6` remain physically present and correctly materialized, but
they are logically stale because they represent rejected speculative
positions. They are ignored until a later verification overwrites them. The
next decoding step uses the committed state now stored in `S7`.

Without windowed recurrent-state support, the decoder could not recover `S3`
from the wide verification. It would have to snapshot the state before
verification, run the wide verification, determine that `S1` through `S3`
were accepted, restore the snapshot, and replay those accepted positions
sequentially:

```text
Process without windowed support:
snapshot S0
→ wide verification
→ accept S1, S2, S3
→ restore S0
→ replay S1 → S2 → S3
→ continue from the replayed state
```

Windowed state avoids that additional target-model replay. It retains the accepted recurrent state during the wide pass, promotes it with one batched device-to-device operation, and continues directly from the accepted prefix. This reduces replay overhead and can improve throughput.

## Model export requirements

hip-ep does not export or rewrite a model's state layout. The model exporter
must emit `state_window > 0` on the supported `CausalConvWithState` and
`LinearAttention` nodes, together with state tensors whose first dimension is
the window dimension. The exported graph must also preserve the normal
past/present state input and output names expected by the consumer.

## Compiler and lowering changes

The ONNX-to-HIP conversions preserve the window dimension and operator
attributes for both supported operators. HIP dialect verification,
canonicalization, bufferization, and HIP-to-LLVM lowering carry the extra
state-window dimension through the generated model interface.

The compiler tests cover both conversion boundaries and windowed attributes:

- `test/lit/Conversion/onnx-to-hip/`
- `test/lit/Conversion/hip-to-llvm/`
- `test/lit/Dialect/`



## Runtime and kernel/API design

During a wide verification, each supported hybrid operator writes its recurrent/conv result into one slot of the state window. After partial acceptance, the decoder must promote the accepted slot to the committed slot. For example, in the Qwen3.6-27B export this means copying one slot in 96 state tensors (48 convolution states and 48 recurrent states).

Promotion means copying the accepted slot into the committed destination slot;
it does not rename or clear the other window slots.

### Windowed materialization versus the burst path

The recurrent state must be produced correctly before promotion can be
correct. The short-burst `LinearAttention` kernel is optimized for
unwindowed execution: it reads the prior state and writes the final recurrent
state for the whole sequence. That is sufficient when `state_window == 0`,
because no intermediate state slot is retained.

For `state_window > 0`, the runtime uses the per-token path instead. Each token's recurrent update is materialized into its corresponding retained window slot before crop selects an accepted prefix. Using the burst path for a windowed sequence would produce the final state but could leave intermediate slots semantically incomplete, so promoting one of those slots could resume from the wrong recurrent state.

The selection is intentionally scoped in `lib/Runtime/real/linear_attention.cpp`:

```cpp
if (state_window == 0 && seq_len <= hipdnn_ep_la_burst_max_tokens())
```

This preserves the existing burst optimization for non-windowed recurrent
models. Dense models do not use this recurrent `LinearAttention` path and are
unaffected. Batched state-slot promotion remains unchanged; this guard fixes
how the source slots are materialized before that promotion runs.

The runtime wrappers pass the selected slot and state layout to the
per-architecture custom HIP kernel. The kernel performs all of those copies
on the model's active session stream.

OGA calls the provider through the small versioned C API in
`include/hip/device_api.h`:

- `hipdnn_ep_device_api_version()`
- `hipdnn_ep_copy_state_slots(...)`

The first function reports which API contract HIP EP supports. OGA checks that
the result is version `1` before using the second function. The copy function
receives the device descriptors plus matching host descriptors, validates the
host descriptors, obtains the provider's active GPU stream, and launches one
kernel for all state tensors. It returns without synchronizing, so the copy
stays ordered with the model work on that stream.

If the API or architecture-specific kernel is unavailable, OGA uses its
compatibility path with individual tensor copies. The API is versioned so a
future incompatible signature can be added as version `2` without breaking
older OGA binaries.

## Consumer contract for OGA

This HIP EP feature is independent of the decoder/controller integration. An
OGA consumer must:

1. Run the wide verification forward.
2. Determine the accepted prefix.
3. Crop the KV/position state and promote the corresponding recurrent-state
  window slot.
4. Avoid the immediate replay append.

The consumer must resolve the API from the already-loaded decoder provider,
use the provider's active session stream, and retain a per-tensor fallback for
older packages. The OGA/DFlash bridge is a separate consumer-side change and
does not belong in this HIP EP runtime contract.

## Tests and validation

Coverage is provided at each compiler/runtime boundary:

- GPU-free numeric checks:
`test/numeric/tests/test_causal_conv_with_state.py` and
`test/numeric/tests/test_linear_attention.py`
- GPU state-slot promotion:
`test/runtime/test_state_slot_copy.cpp`
- Provider export contract:
`test/runtime/test_hipgpu_elf_contract.cpp`

The state-slot test checks mixed descriptor sizes, alignment, stream ordering,
guards, and repeated launches. The ELF test checks provider visibility,
`DF_1_NODELETE`, plugin exports, and the versioned state-slot API exports.

The kernel-branch validation for the windowed materialization change also
includes:

- `StateSlotCopy_gfx1151` and `HipGpuElfContract`: 2/2 passed.
- The numeric `CausalConvWithState` and `LinearAttention` suite: 67 passed.
The generated test models cap their ONNX IR version at 13 to match the
packaged ONNX Runtime used for validation; this affects only temporary test
models.
- The compiler LIT suite covering the modified lowering and dialect paths.

These are native HIP EP regression checks. OGA/DFlash integration, crop
behavior, wide-versus-sequential diagnostics, and end-to-end performance
belong to the downstream consumer integration validation.

From the build tree, run the focused runtime tests:

```bash
export ROCM_ROOT=/path/to/rocm
export ROOT=/path/to/hip-ep/install
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROOT/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir <build-dir> --output-on-failure \
  -R 'StateSlotCopy_gfx1151|HipGpuElfContract'
```

Run the compiler lit suite and the two numeric tests using the repository's
standard test configuration described in the [numeric test guide](../test/numeric/README.md).
The expected numeric coverage includes both `state_window=0` and
`state_window>0` cases.

## Build and package notes

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
export ROCM_ROOT=/path/to/rocm
export THEROCK_DIST=/path/to/therock-dist
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROOT/lib:$THEROCK_DIST/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"
export PATH="$ROOT/bin:$PATH"
cp "$ROOT/lib/libhipgpu.so" "$ROOT/lib/libamdgpu-ep.so"
export AMDGPU_EP_PATH="$ROOT/lib/libamdgpu-ep.so"
```

The ROCm runtime path is required when running tests directly from the build
tree. Without it, ELF loading tests can fail with an error such as
`libamdhip64.so.7: cannot open shared object file`.

## How to utilize in practice

The OGA/DFlash controller bridge is a separate consumer-side change that will be referenced. It uses the windowed state functionality and crop optimization in hip-ep to accelerate speculative decoding algorithm DFlash.
