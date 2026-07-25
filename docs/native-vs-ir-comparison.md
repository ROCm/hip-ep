<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Native DLL vs LLVM IR Storage — Design Decision

**Date:** 2026-05-29 (dual-format update 2026-06-08)
**Document Type:** Design (Alternatives Considered)
**Status:** Both formats supported — **LLVM IR bitcode is the production default**; **Native DLL is opt-in** (benchmarking/dev), selected by a single compile option.
**Related:** [design/morphizen-ep-integration.md](design/morphizen-ep-integration.md), [design/compiler-runtime-contract.md](design/compiler-runtime-contract.md), [design/compilation-options.md](design/compilation-options.md)

**Adapted from:** [ROCm/hip-compiler `docs/design/alternatives/NATIVE-VS-IR-COMPARISON.md`](https://github.com/ROCm/hip-compiler/blob/main/docs/design/alternatives/NATIVE-VS-IR-COMPARISON.md).

---

## Decision

`hip-ep` supports **two** per-model artifact formats behind a single
compile option (`artifact_format` / `CompilationOptions.output_mode`):

**LLVM IR (default, production).**
- `hip-compiler.dll` emits OS-portable `.bc` into the [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) tar via `morphizen::FileSystem` (see [design/compilation-options.md](design/compilation-options.md)). The bitcode is emitted with empty triple and empty datalayout.
- `onnxruntime_morphizen_ep.dll` JITs that bitcode in-process at session creation via `LlvmIrJit::create` (`backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.cpp`). The JIT stamps the host triple/datalayout onto both the per-model module and an embedded per-OS `runtime.bc`, then `addIRModule`s them into one ORC LLJIT JITDylib over a shared `LLVMContext`.
- At inference time the EP holds function pointers obtained via `LlvmIrJit::lookup_raw` and calls them per `Compute()`. No LLVM machinery runs on that hot path.

**Native DLL (opt-in, benchmarking/dev).**
- `hip-compiler.dll` merges the embedded `runtime.bc` at producer time (`LLVMBackend::linkRuntimeModule`), emits a host object (`compileToObjectFile`, PIC), and links a per-OS `.dll`/`.so` via `DLLLinker` (in-process `lld-link` COFF on Windows; `clang++ -shared -fuse-ld=lld` subprocess on Linux), linking the per-arch `custom_kernels_<arch>` import lib + ROCm import libs.
- `onnxruntime_morphizen_ep.dll` writes the artifact bytes to a temp file and loads it via `morphizen::Plugin` (`LoadLibraryW` / `dlopen`), resolving the same five-symbol C ABI through `get_method` (`GetProcAddress` / `dlsym`). The temp file is deleted on session teardown.
- The EP picks the loader from the `artifact_format` field in the EPContext metadata (`mlir_metadata::Metadata`); the compiler always records it, and an empty/unknown value is fatal.

Why native is **not** the production default — the signed-DLL-only loading
policy below. It remains available for internal benchmarking against the JIT
path. The two paths share the identical five-symbol C ABI, so the per-model
contract is format-independent.

---

## Why not Native DLL — three reasons in priority order

### 1. Microsoft signed-DLL-only loading policy (the binding reason)

Per-model machine code generated on the user's machine cannot be signed by AMD. Any path that requires the OS loader to map it as a DLL/SO is blocked at deployment time:

- `LoadLibrary("model.dll")` from disk: fails the policy directly.
- `MemoryModule` (Windows) / `memfd_create` + `dlopen` (Linux) of a per-model image embedded in the EPContext: bypasses the OS loader's signature check, which *is* the policy violation, not a workaround for it.

The IR/JIT path side-steps the policy entirely. The only DLL the OS loader maps is `onnxruntime_morphizen_ep.dll` (signed once, by AMD). Per-model bitcode is data; LLVM ORC allocates an executable code page inside the EP DLL's address space via its own memory manager and emits codegen output into it. No `LoadLibrary`, no `dlopen`, no per-model file on disk.

This is the constraint that closes the design. The other two reasons below would not, by themselves, be enough.

### 2. OS portability of the EPContext tar

A single `.bc` works on both Linux and Windows. The compile-time mechanism (`LLVMBackend::emitLlvmIr`) strips the triple and datalayout; the runtime mechanism (`LlvmIrJit::create`'s `parseAndStamp` in `LlvmIrJit.cpp`) stamps the host's values back on before `addIRModule`. CRT and platform runtime symbols (`operator new`, `__cxa_atexit`, atexit shims, etc.) resolve against the EP DLL's embedded per-OS `runtime.bc`, which is built per-OS as part of the EP DLL.

Native DLL requires either two artifacts in every EPContext (`.dll` + `.so`, doubling tar size) or two distribution pipelines.

### 3. JIT cost is bounded and off the inference hot path

JIT runs once per session in `LlvmIrJit::create`. The result is a `unique_ptr<LlvmIrJit>` whose `impl_->jit` holds the JITted code for the lifetime of the session. Every subsequent `Compute()` is a function-pointer call into JITted code that calls into ROCm libraries — no LLVM activity. The session-creation cost is observable for large models but amortized across all inferences in the process.

The hip-original framing "recompiles every process start" is technically correct but misleading as shorthand: it is not per-inference, it is per-process.

---

## Comparison table

| Aspect | Native DLL (rejected) | LLVM IR (shipping) |
|--------|-----------------------|--------------------|
| Signed-DLL-only loading policy | Blocked — per-model code cannot be signed | OK — only the EP DLL is loaded by the OS |
| EPContext tar portability | One artifact per OS | Single `.bc` works on Linux + Windows |
| Per-model artifact on disk treated as code | Yes (`.dll` / `.so`) | No (bitcode is data) |
| EP DLL binary size | Smaller (no LLVM ORC) | +5-10 MB statically linked ORC + codegen ([ClamAV example](https://groups.google.com/g/llvm-dev/c/InMxMlH5fXg)) |
| Compile-time optimizations | Full LLVM passes in compiler | Full LLVM passes in `hip-compiler.dll` (`opt_level` default 2, run before bitcode emit) |
| Session-creation cost | mmap + relocations | LLVM ORC codegen on already-optimized IR |
| Per-`Compute()` cost | Function pointer call | Function pointer call |
| Steady-state memory | mmap'd image (single mapping) | JITted code page + retained `llvm::Module` (bitcode `MemoryBuffer` released after `parseBitcodeFile`) |
| GPU portability | Same — arch-specific paths via side-by-side `custom_kernels_<arch>.{dll,so}` loaded at JIT init | Same |
| Per-OS work in the EP build | Per-OS native codegen pipeline | Per-OS `runtime.bc` only (Clang-built shim) |

Callouts on the table:

- **"Full LLVM passes" on both rows is intentional.** Optimization passes run in `hip-compiler.dll` *before* bitcode is written. The JIT side does codegen only. The common framing "JIT loses optimizations to time budget" does not apply here.
- **Per-`Compute()` cost is identical.** Both produce a function pointer that ORT calls per inference; LLVM is not on that path.
- **Both EP DLLs are OS-specific.** The "OS-portable" property is of the *per-model artifact*, not the EP DLL. Native DLL adds per-OS native codegen in the compiler; IR adds a per-OS `runtime.bc` shim.

---

## What the Native DLL path would look like (for reference)

If the signed-DLL constraint did not apply:

1. `hip-compiler.dll` would run platform codegen, link to `.dll`/`.so`, and write the artifact to the EPContext via `morphizen::FileSystem` (no analogue of the current `output_mode` schema field would exist; the artifact format would be implicit in the consumer pipeline).
2. The EP DLL embeds an in-memory PE loader on Windows ([MemoryModule](https://github.com/fancycode/MemoryModule), MPL 2.0) and an in-memory ELF loader on Linux (no canonical library; `memfd_create` + `dlopen` from `/proc/self/fd/N` is the common shape). At session creation the EP maps the appropriate artifact and resolves the five-symbol contract documented in [design/compiler-runtime-contract.md](design/compiler-runtime-contract.md).
3. CRT and runtime symbols are resolved through the loader's import table against host libraries; the embedded `runtime.bc` shim disappears.

The "memory-load vs disk file" sub-decision covered in the hip-original is moot here: both sub-paths require the OS loader (or a bypass of it) to map unsigned per-model code, and both are blocked by the same policy.

---

## Hybrid (implemented)

The hybrid that stores a format tag in the EPContext and chooses Native vs IR
per artifact is now implemented:

- **Tag:** `mlir_metadata::Metadata.artifact_format` (`"LLVM_IR"` | `"NATIVE"`),
  written by the compiler in `pass_main.cpp` alongside `artifact_filename` and
  read by `MlirCustomOp` before the artifact is opened. The compiler always
  records it; an empty/unknown value is fatal at load. (The standalone tools,
  which see a bare file with no metadata, detect format by file extension.)
- **Selection:** a single compile option, `artifact_format` (EP provider option)
  → `CompilationOptions.output_mode` (FlatBuffers enum `LLVM_IR` | `NATIVE`).
- **Default:** `LLVM_IR`. `NATIVE` is opt-in — the Native branch is still
  blocked by the signed-DLL policy for production deployment, so it exists for
  internal benchmarking against the JIT path. Both branches build and run on
  Windows and Linux.

---

## Open questions specific to this project

1. **`llvm::Module` retention after JIT codegen.** After `addIRModule`, ORC LLJIT owns the Module and currently keeps it alive for the JIT's lifetime. Whether dropping the Module post-codegen (via a custom `IRTransformLayer` or explicit unload) would meaningfully reduce steady-state memory is unmeasured. Suspect small — JITted code dominates.
2. **Session-creation latency on the largest shipping models.** Need wall-clock measurements on the Qwen-vision sized workloads (see [design/qwen-vision-ops.md](design/qwen-vision-ops.md)) before declaring "bounded" with confidence.
3. **EP DLL binary size budget.** The ~5-10 MB cost of statically linked LLVM ORC + the per-OS `runtime.bc` is accepted today. A future slimming pass (strip unused LLVM targets, drop dead codegen passes) is plausible but not planned.

---

## References

Project documents:

- [design/morphizen-ep-integration.md](design/morphizen-ep-integration.md) — DLL boundary contracts, `FileSystem`, JIT loader integration.
- [design/compiler-runtime-contract.md](design/compiler-runtime-contract.md) — `model_metadata` schema; five-symbol C ABI between the per-model bitcode and the EP DLL.
- [design/compilation-options.md](design/compilation-options.md) — compilation options (opt level, externalized constants).
- [`LlvmIrJit.cpp`](../backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.cpp) — in-process JIT implementation referenced throughout this doc.

External:

- [ONNX Runtime EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
- [LLVM ORCv2](https://llvm.org/docs/ORCv2.html)
- [MemoryModule](https://github.com/fancycode/MemoryModule) — Windows in-memory PE loader. Referenced as the Windows half of the rejected Native DLL path; not used in this repo.
- [ClamAV LLVM JIT footprint discussion](https://groups.google.com/g/llvm-dev/c/InMxMlH5fXg) — source for the ~5-10 MB static-link estimate.
