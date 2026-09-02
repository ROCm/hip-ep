<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# CLAUDE.md

This file gives coding agents the repository-wide rules and entry points needed to work safely in hip-ep. Detailed design, troubleshooting, model, and benchmark information belongs in the linked documentation rather than in this file.

## Documentation policy

- Treat current code, tests, and `docs/design/` as the sources of truth.
- Update affected documentation in the same change whenever behavior, interfaces, pass ordering, or architecture changes.
- Add guidance here only when it is a repository-wide rule or recurring workflow that an agent must know before locating the relevant code.
- Do not add investigation transcripts, dated performance results, model-specific debugging histories, or superseded approaches to this file.
- Do not use `.claude/memory`; durable project knowledge belongs in version-controlled documentation.

## Project overview

hip-ep is an ONNX Runtime Execution Provider for AMD GPUs. The compiler lowers ONNX operations through a custom MLIR HIP dialect to LLVM IR. Generated code dispatches to hipDNN, hipBLASLt, and custom HIP kernels.

The default per-model artifact is OS-portable LLVM bitcode. The Execution Provider JIT-loads it in-process together with embedded runtime bitcode. Native `.dll`/`.so` model artifacts are an opt-in mode.

The generated model ABI uses `inference_compute(state, inputs)`. Graph outputs are allocated in-graph through `hip.alloc_output` and the Execution Provider's output-allocation callback.

## Repository map

| Area | Path |
|---|---|
| ONNX-to-HIP conversion | `lib/Conversion/OnnxToHip/` |
| HIP dialect | `include/hip/Dialect/`, `lib/Dialect/` |
| HIP-to-LLVM lowering | `lib/Conversion/HipToLLVM/` |
| Compiler pipeline and driver | `lib/Dialect/Transforms/Pipelines.cpp`, `lib/Compiler/` |
| Runtime | `lib/Runtime/` |
| Custom HIP kernels | `lib/Runtime/Kernels/` |
| ONNX Runtime / MorphiZen integration | `backend-mlir-compiler/`, `morphizen/` |
| Command-line tools | `tools/` |
| Compiler tests | `test/lit/`, `test/e2e/`, `test/runtime/` |
| Numeric and model tests | `test/numeric/`, `test/python/` |
| Design documentation | `docs/design/` |

## Build, test, and lint

Windows and native Linux builds use `build.py`. It selects the platform toolchain, resolves dependencies through CMake, chooses the GPU architecture, builds, installs, and runs the default tests unless disabled.

```bash
# Windows
python build.py

# Linux native
python3 build.py

# Linux Docker (recommended/authoritative on the project build host)
./docker/run.sh build
```

`./docker/run.sh build` creates the build image when needed, detects the host GPU architecture, bind-mounts the workspace, and runs `build.py` in the compile-only container.

Common `build.py` options:

| Option | Purpose |
|---|---|
| `--mock` | Build the compiler and mock runtime without ROCm hardware |
| `--clean` | Remove the selected build and install trees, then exit |
| `--skip_tests` | Skip the post-install test run |
| `--hip_arch <gfx-arch>` | Override GPU architecture detection |

The default layout is:

```text
<workspace>/<repo>/          source
<workspace>/build/<repo>/    build tree
<workspace>/install/         install prefix
```

`build.py` runs the LIT suite plus the GPU-free plugin, output-allocator, and
symbolic-metadata unit tests by default. To run tests manually from Linux or Git
Bash at the repository root:

```bash
BUILD_DIR=../build/$(basename "$PWD")

# All configured CTest tests; some E2E tests may require a GPU.
ctest --test-dir "$BUILD_DIR" -C Release --verbose

# MLIR LIT pass-verification suite only.
ctest --test-dir "$BUILD_DIR" -C Release -R MorphizenMLIRLitTests --verbose
```

Use the dedicated guides for test-specific setup:

- [test/README.md](test/README.md)
- [test/lit/README.md](test/lit/README.md)
- [test/numeric/README.md](test/numeric/README.md)
- [test/e2e/README.md](test/e2e/README.md)

Before committing:

```bash
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for PR, formatting, AI-disclosure, and commit-trailer requirements.

## Architecture references

- Pass order and plugin slots: [docs/pipeline_pass_menu.md](docs/pipeline_pass_menu.md)
- Compiler/runtime metadata and ABI: [docs/design/compiler-runtime-contract.md](docs/design/compiler-runtime-contract.md)
- Output allocator: [docs/design/output-allocator-design.md](docs/design/output-allocator-design.md)
- Shape inference: [docs/design/hip-shape-inference.md](docs/design/hip-shape-inference.md)
- Memory planning: [docs/design/pool-allocs-memory-planning.md](docs/design/pool-allocs-memory-planning.md)
- Constant handling: [docs/design/constant-handling-design.md](docs/design/constant-handling-design.md)
- Compilation options and artifact format: [docs/design/compilation-options.md](docs/design/compilation-options.md)
- Plugin design and authoring: [docs/design/plugin-interface.md](docs/design/plugin-interface.md), [docs/plugin_authoring.md](docs/plugin_authoring.md)
- Windows and Linux setup: [docs/quick_start.md](docs/quick_start.md), [docs/quick_start_linux.md](docs/quick_start_linux.md)
- Remote workflow: [docs/remote-dev-workflow.md](docs/remote-dev-workflow.md)

## Non-negotiable invariants

### GPU targeting

- A real HIP build must target the architecture of the GPU that will execute it. Use `build.py` auto-detection or pass `--hip_arch`.
- A mismatched architecture can build successfully and fail only when a kernel launches.

### Proving GPU execution

- Compilation failures may allow ONNX Runtime to fall back to CPU. Accuracy results are not proof that the graph ran on the GPU.
- Use `HIPDNN_EP_STRICT=1` when validating that a graph expected to be fully offloaded compiles successfully.
- Use focused debug/runtime evidence when needed, but never benchmark with debug or per-op performance tracing enabled.

### Output allocator ABI

- The supported generated ABI is `inference_compute(state, inputs)`.
- `hip-use-output-allocator` must run before PoolAllocs so graph outputs remain runtime-owned and are not pooled. The production pipeline intentionally omits ownership-based buffer deallocation because every transient is pooled and outputs are runtime-owned.
- Keep the allocator callback model-agnostic; dynamic output shapes are computed in the generated graph.
- See [docs/design/output-allocator-design.md](docs/design/output-allocator-design.md).

### Host reads of GPU values

- A host read of a GPU-computed scalar must use a synchronized HIP readback operation. Do not introduce a bare `tensor.extract` or `memref.load` of device data.
- Keep tiny host-written shape buffers out of the GPU pool; `hip-materialize-host-scalars` redirects them to host-mapped scratch.
- See [docs/design/hip-shape-inference.md](docs/design/hip-shape-inference.md).

### Allocation and memory planning

- Every transient allocation must be pooled or rewritten as an output allocation. Leftover `hip.alloc`/`memref.alloc` paths are not a supported per-inference allocator strategy.
- Preserve the ordering and dominance requirements documented in [docs/design/pool-allocs-memory-planning.md](docs/design/pool-allocs-memory-planning.md).
- `HIPDNN_EP_BUFFERIZE_COPY_BEFORE_WRITE=1` is an opt-in compile-time escape hatch for extremely large single-function graphs: it skips One-Shot Bufferize's expensive RaW analysis by copying before writes. Keep it off by default because the extra copies can reduce runtime performance.
- HIP runtime calls consume contiguous bare pointers unless their ABI explicitly carries layout metadata. Materialize non-contiguous inputs before such calls.

### Runtime ABI and bitcode

- Functions called from generated code require `extern "C"` declarations in `lib/Runtime/hipdnn_ep_runtime.h`.
- Keep declaration and definition export attributes consistent for runtime sources that are also compiled natively.
- Add every newly included runtime header to the bitcode build dependencies in `lib/Runtime/CMakeLists.txt`.
- Runtime and custom-kernel changes may leave cached compiled models stale; invalidate the model cache when validating those changes.

### Kernel launch status

- Clear the HIP last-error state before launching a kernel and return `hipGetLastError()` afterward.
- Never return `hipSuccess` unconditionally after `<<<>>>` or `hipLaunchKernelGGL`.

### Benchmarking

- Run GPU benchmarks serially; concurrent runs invalidate results.
- Do not measure throughput with `HIPDNN_EP_PERF=1` or `HIPDNN_EP_DEBUG=1`.
- Prime persistent kernel-autotune caches after rebuilding custom kernels, and account for thermal drift in compute-bound measurements.

## Common workflows

### Add or change an ONNX operation

Check every affected layer:

1. ONNX-to-HIP conversion and registration.
2. HIP dialect definition and verification.
3. Shape inference/reification and DPS init construction.
4. HIP-to-LLVM lowering and runtime symbol declaration.
5. Real and mock runtime behavior.
6. Custom kernel declaration/implementation, when needed.
7. LIT coverage for conversion/lowering.
8. Numeric or E2E coverage for runtime correctness.
9. Supported-operation and design documentation.

### Change a compiler transformation

- Document non-obvious rewrites with a minimal `Before:` / `After:` IR example in the implementation.
- Update the example whenever the transformation changes.
- Prefer standard MLIR interfaces, analyses, and passes before adding project-specific infrastructure.
- Keep generic compiler comments focused on IR patterns and invariants rather than the model that exposed the issue.

### Change runtime code or ABI

- Keep runtime declarations in the C ABI header.
- Verify bitcode dependency tracking.
- Rebuild and invalidate cached model artifacts before testing.
- Add a GPU-free runtime unit test when the contract can be exercised without HIP libraries.

### Validate correctness

- Use LIT for IR transformations and ABI lowering.
- Use `test/numeric/` for per-operation GPU-vs-CPU correctness.
- Use `test/e2e/` and targeted `test/python/` tests for full model/runtime paths.
- Ensure the test proves GPU execution rather than silently comparing CPU fallback against CPU.

## Code conventions

- C++17; format C++ with clang-format and Python with Ruff.
- MIT license headers are enforced by pre-commit.
- Use MorphiZen C++ wrappers for graph APIs; do not use raw ONNX protobuf APIs.
- Match ONNX operations through MLIR's generic `Operation` API; do not add an onnx-mlir dependency.
- Explain why non-obvious code exists; do not comment obvious mechanics.
- In compiler code, prefer `llvm::seq`/`llvm::seq_inclusive` for straightforward integer ranges.
- Do not put hardware-specific identifiers or model anecdotes in generic compiler/dialect/pass comments. Put reproduction and performance details in tests, design docs, issues, or commit messages.
- When an experiment fails, revert it completely before trying another approach.

## Scope of this file

Do not add:

- per-model setup instructions;
- model download URLs or gated-repository details;
- dated benchmark tables;
- kernel-tuning campaign notes;
- host-specific paths or hardware inventory;
- historical/superseded investigations;
- full design explanations already maintained in `docs/`.

Add or update the authoritative documentation instead, then link it here only if the guidance is repository-wide and needed by most coding agents.
