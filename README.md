<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ROCm hip-ep

**hip-ep** is an ONNX Runtime Execution Provider for AMD GPUs. It compiles ONNX graphs through an MLIR pipeline—from ONNX dialect operations to a custom HIP dialect and LLVM IR—and executes them with hipDNN, MIOpen, hipBLASLt, and custom HIP kernels.

The provider integrates with ONNX Runtime through the MorphiZen pass framework. By default, compiled models are emitted as OS-portable LLVM bitcode and JIT-loaded in-process when a session is created. Native per-model libraries remain available as an opt-in artifact format.

## Highlights

- **MLIR compiler pipeline** — lowers ONNX graphs to HIP and LLVM IR.
- **ROCm execution backends** — dispatches to hipDNN, MIOpen, hipBLASLt, and custom HIP kernels.
- **Dynamic shapes** — supports runtime batch/sequence dimensions, shape refinement, and runtime-sized outputs.
- **GPU memory planning** — packs transient allocations into one or more grow-on-demand pool domains and keeps host-written shape scalars in separate host-mapped scratch.
- **In-graph output allocation** — allocates graph outputs through the Execution Provider callback once their runtime shapes are known.
- **Externalized constants** — stores large model weights in a sidecar constants file instead of embedding them in the model artifact.
- **Two artifact formats** — LLVM bitcode with in-process JIT by default, or a native `.dll`/`.so` when explicitly requested.
- **Extensible compiler pipeline** — supports registered plugin slots and out-of-tree dialect/pass plugins.
- **GPU-free development** — provides a mock runtime for compiler and LIT testing without ROCm hardware.

## Get started

| Goal | Guide |
|---|---|
| Build and run on Windows | [Windows quick start](docs/quick_start.md) |
| Build on Linux, use Docker, or download a prebuilt package | [Linux quick start](docs/quick_start_linux.md) |
| Understand artifact formats | [LLVM IR vs native artifacts](docs/native-vs-ir-comparison.md) |
| Extend the pipeline with a plugin | [Plugin authoring guide](docs/plugin_authoring.md) |
| Contribute to the project | [Contributing guide](CONTRIBUTING.md) |

## How it works

```text
ONNX model
    │
    ▼
ONNX Runtime + MorphiZen EP integration
    │
    ▼
ONNX MLIR
    │  ONNX-to-HIP lowering
    ▼
HIP MLIR
    │  shape refinement, bufferization, and memory planning
    ▼
LLVM dialect
    │  HIP-to-LLVM lowering + generated model interface
    ▼
Per-model artifact + constants file
    ├── LLVM bitcode (default) ──► in-process ORC JIT
    └── native .dll/.so (opt-in) ─► OS dynamic loader
    │
    ▼
hipDNN / MIOpen / hipBLASLt / custom HIP kernels
    │
    ▼
AMD GPU
```

The generated model interface initializes per-session runtime state, executes the compiled graph, and releases resources. Graph outputs are allocated in-graph through the provider's output-allocation callback; transient buffers use lifetime-aware GPU pool planning.

For the exact registered pass names and ordering, see the [pipeline pass menu](docs/pipeline_pass_menu.md). For compiler/runtime ABI details, see the [compiler-runtime contract](docs/design/compiler-runtime-contract.md).

## Supported operations

hip-ep supports common operation families used by transformer, vision, convolutional, quantized, and multimodal models:

- convolution, transposed convolution, and pooling;
- MatMul, Gemm, quantized MatMul, and mixture-of-experts operations;
- attention, grouped-query attention, rotary embeddings, and KV-cache updates;
- normalization and activation functions;
- elementwise, reduction, gather/scatter, padding, resize, and indexing operations;
- tensor shape/view operations and selected ONNX control flow.

See [Supported operations](docs/supported-operations.md) for per-operation domains, lowering paths, runtime backends, data types, dynamic-shape behavior, and known limitations.

## Developer tools

| Tool | Purpose |
|---|---|
| `hip-mlir-opt` | Run and inspect registered HIP/MLIR passes and pipelines. |
| `hip-compiler` | Compile ONNX MLIR into LLVM-bitcode or native model artifacts. |
| `hip-onnx-runner` | Run ONNX models through the installed Execution Provider. |
| `hip-inspect` | Inspect metadata embedded in a compiled model artifact. |
| `hip-test` | Load and execute compiled artifacts through the same runtime path used by the EP. |

Tool availability depends on the build configuration; see the platform quick-start guide for the corresponding CMake options and install layout.

## Documentation

### Architecture and compiler

- [Pipeline pass menu](docs/pipeline_pass_menu.md)
- [Compiler-runtime contract](docs/design/compiler-runtime-contract.md)
- [Shape inference design](docs/design/hip-shape-inference.md)
- [GPU memory planning](docs/design/pool-allocs-memory-planning.md)
- [Output allocator design](docs/design/output-allocator-design.md)
- [Constant handling](docs/design/constant-handling-design.md)
- [Compilation options](docs/design/compilation-options.md)
- [LLVM IR vs native artifacts](docs/native-vs-ir-comparison.md)

### Extending and operating

- [Plugin authoring](docs/plugin_authoring.md)
- [Plugin interface design](docs/design/plugin-interface.md)
- [Custom kernel design](docs/design/custom_kernel_design.md)
- [Per-operation profiling](docs/design/per-op-profiling.md)
- [Remote development workflow](docs/remote-dev-workflow.md)

## Developer entry points

- ONNX frontend: `lib/Conversion/OnnxToHip/`
- HIP dialect: `include/hip/Dialect/` and `lib/Dialect/`
- HIP-to-LLVM lowering: `lib/Conversion/HipToLLVM/`
- Compiler driver: `lib/Compiler/`
- Runtime and custom kernels: `lib/Runtime/`
- MorphiZen/ONNX Runtime integration: `backend-mlir-compiler/` and `morphizen/`
- Command-line tools: `tools/`
- Tests: `test/lit/`, `test/numeric/`, `test/python/`, and `test/e2e/`

## Issues and feedback

Report bugs and request features through [GitHub Issues](https://github.com/ROCm/hip-ep/issues).

## Contributing

This project follows LLVM's [incremental development](https://llvm.org/docs/DeveloperPolicy.html#incremental-development) and [AI tool use](https://llvm.org/docs/AIToolPolicy.html) policies. See [CONTRIBUTING.md](CONTRIBUTING.md) for project-specific requirements.

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

Licensed under the MIT License. See [LICENSE](LICENSE).
