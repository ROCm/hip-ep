<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ROCm hip-ep

**hip-ep** is an ONNX Runtime Execution Provider for AMD GPUs. It compiles ONNX graphs through an MLIR pipelineâ€”from ONNX dialect operations to a custom HIP dialect and LLVM IRâ€”and executes them with hipBLASLt and custom HIP kernels.

The provider integrates with ONNX Runtime through the MorphiZen pass framework. By default, compiled models are emitted as OS-portable LLVM bitcode and JIT-loaded in-process when a session is created. Native per-model libraries remain available as an opt-in artifact format.

See the [HIP EP website](https://rocm.github.io/hip-ep/) for documentation and benchmarks.

## What's new

### Mid-November 2026
- **[v0.5.3](https://github.com/ROCm/hip-ep/releases/tag/v0.5.3) released** â€” ONNX Runtime 1.31 and OGA 0.19.0, faster gpt-oss and Gemma 4 decode, and faster MoE prefill. The table below shows the latest `main` ([`9ee5e5b`](https://github.com/ROCm/hip-ep/commit/9ee5e5b)).
- **Large-model speedups at 16K** since October: Laguna-S-2.1 1.91Ã— TTFT and 2.76Ã— decode, Nemotron-3-Super-120B 1.76Ã— TTFT, Qwen3.8-Flash-Next 1.31Ã— TTFT and 1.54Ã— decode.
- **New model support:** Step-3.5-Flash.

| Model (Strix Halo) | Input (tokens) | TTFT (s) | Decode TPS | Speedup vs [Oct](https://github.com/ROCm/hip-ep/commit/d3df783) (TTFT / TPS) |
|---|---|--:|--:|--:|
| Qwen3.8-Flash-Next | 16K text + 1.8K image | 28.62 | 30.74 | 1.31Ã— / 1.54Ã— |
| Qwen3.5-122B-A10B | 16K text + 1.8K image | 39.30 | 22.02 | 1.13Ã— / 1.05Ã— |
| Nemotron-3-Super-120B-A12B | 16K text | 46.92 | 21.09 | 1.76Ã— / 1.03Ã— |
| Nemotron-3.5-Lightning-30B-A3B | 16K text | 9.37 | 68.65 | 1.62Ã— / 1.09Ã— |
| Laguna-S-2.1 (118B) | 16K text | 21.10 | 26.21 | 1.91Ã— / 2.76Ã— |
| Laguna-XS-2.1 | 16K text | 7.86 | 48.47 | 2.71Ã— / 3.14Ã— |
| Gemma 4 26B-A4B | 16K text + 0.25K image | 13.61 | 41.56 | 1.23Ã— / 1.16Ã— |
| Mistral-Small-4-119B-2603 | 16K text + 2.4K image | 26.10 | 11.81 | â€” |
| Qwen3.8-27B | 16K text + 1.8K image | 40.80 | 13.90 | â€” |
| Step-3.5-Flash | 16K text | 240.64 | 13.80 | New |

### October 2026
- **Faster** â€” 1.85Ã— prefill and 1.13Ã— decode at 16K context vs v0.3, averaged across the 15 models we track.
- **Speculative decoding** â€” EAGLE4, DFLASH2, and MTP, delivering up to 2.3Ã— the decode TPS on Qwen3.8-27B.
- **Broad model support**
  - **New models:** Qwen3.8-Flash-Next, Qwen3.8-27B, Qwen3.5-122B-A10B, Nemotron-3-Super-120B, Mistral-Small-4-119B, Laguna-S-2.1, Gemma 4.
  - Agile PyTorch-to-OGA flow brings up a new model in under a day.
  - Released models are listed in [models.json](docs/models.json).

## Key features

- **MLIR compiler pipeline** â€” lowers ONNX graphs to HIP and LLVM IR.
- **ROCm execution backends** â€” dispatches to hipBLASLt and custom HIP kernels.
- **Dynamic shapes** â€” supports runtime batch/sequence dimensions, shape refinement, and runtime-sized outputs.
- **GPU memory planning** â€” packs transient allocations into one or more grow-on-demand pool domains and keeps host-written shape scalars in separate host-mapped scratch.
- **In-graph output allocation** â€” allocates graph outputs through the Execution Provider callback once their runtime shapes are known.
- **Externalized constants** â€” stores large model weights in a sidecar constants file instead of embedding them in the model artifact.
- **Two artifact formats** â€” LLVM bitcode with in-process JIT by default, or a native `.dll`/`.so` when explicitly requested.
- **Extensible compiler pipeline** â€” supports registered plugin slots and out-of-tree dialect/pass plugins.

## Get started

| Goal | Guide |
|---|---|
| Build and run on Windows | [Windows quick start](docs/quick_start.md) |
| Build on Linux, use Docker, or download a prebuilt package | [Linux quick start](docs/quick_start_linux.md) |
| Run on AMD Instinct MI350X (gfx950, CDNA4) | [MI350X quick start](docs/quick_start_mi350.md) |
| Understand artifact formats | [LLVM IR vs native artifacts](docs/native-vs-ir-comparison.md) |
| Extend the pipeline with a plugin | [Plugin authoring guide](docs/plugin_authoring.md) |
| Contribute to the project | [Contributing guide](CONTRIBUTING.md) |

## How it works

```text
ONNX model
    â”‚
    â–¼
ONNX Runtime + MorphiZen EP integration
    â”‚
    â–¼
ONNX MLIR
    â”‚  ONNX-to-HIP lowering
    â–¼
HIP MLIR
    â”‚  shape refinement, bufferization, and memory planning
    â–¼
LLVM dialect
    â”‚  HIP-to-LLVM lowering + generated model interface
    â–¼
Per-model artifact + constants file
    â”œâ”€â”€ LLVM bitcode (default) â”€â”€â–º in-process ORC JIT
    â””â”€â”€ native .dll/.so (opt-in) â”€â–º OS dynamic loader
    â”‚
    â–¼
hipBLASLt / custom HIP kernels
    â”‚
    â–¼
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

- [OLMo native QSA/GQA integration](docs/design/olmo-qsa-native-gqa.md)
- [OLMo native QSA/GQA results](docs/experiments/olmo-qsa-native-gqa-results.md)
- [OLMo native GSM8K and agent benchmark accuracy](docs/experiments/olmo-qsa-agent-accuracy.md)
- [Qwen trained-selector port and native default evidence](docs/experiments/qwen-qsa-native-gqa-results.md)
<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->
- [Qwen eight-selector QSA: latest 4K/16K/32K performance](docs/experiments/qwen-qsa-eight-selector-results.md)
- [Qwen eight-selector paper packet and raw measurements](docs/reproducibility/qwen-eight-hip-ep/README.md)
<!-- QWEN_EIGHT_PERFORMANCE_END -->

- [Qwen isolated default/QSA memory-only matrix](docs/experiments/qwen-qsa-memory.md)
- [Native QSA audit, PR notes and paper data packet](docs/reproducibility/qwen-hip-ep/README.md)
- [Qwen native GQA port contract](docs/design/qwen-qsa-native-gqa.md)
- [OLMo QSA boundary prototype audit](docs/design/olmo-qsa-port.md)
- [OLMo QSA boundary prototype results](docs/experiments/olmo-qsa-results.md)
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
