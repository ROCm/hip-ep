<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Release Signing Gates

**Date:** 2026-05-14
**Document Type:** Operational guidance
**Status:** Draft
**Related:** [design/compilation-options.md](design/compilation-options.md), [design/compiler-runtime-contract.md](design/compiler-runtime-contract.md)

---

## Background

The per-model artifact is now binary LLVM bitcode (`.bc`), not a native
DLL. The artifact is treated as data; the EP DLL JITs it in-process via
`BitcodeJIT` (`backend-mlir-compiler/custom-op-mlir/src/BitcodeJIT.h`).

Because the bitcode is data, no per-model code-signing is required.
Instead, every executable that the runtime loads must be signed up-front.
This document is the operational contract for the AMD release pipeline.

---

## Signing requirements

The following binaries MUST be signed by AMD before they ship:

| Binary | Source repo / target | Why |
|--------|----------------------|-----|
| `onnxruntime_morphizen_ep.dll` | this repo, target `onnxruntime_morphizen_ep.dll` | Hosts the JIT loader, the custom HIP kernel launchers, the EP runtime, and the ORT plugin entry points. The only EP-side binary loaded by `onnxruntime.dll`. |
| `hip-compiler.dll` | this repo, target `hip-compiler` | Loaded by `onnxruntime_morphizen_ep.dll` at compile time (session creation). |

The ROCm DLLs (`MIOpen.dll`, `libhipblaslt.dll`, `hipdnn_backend.dll`,
`libamdhip64.dll`, ...) are signed by their respective vendors and
shipped via the ROCm distribution; this repo does not own their signing.

Per-model bitcode artifacts (the `model_compiled` blob inside the
EPContext tar) are **not** signed. They are integrity-protected by the
content-addressed SHA-256 blobs that already live inside the EPContext
tar.

---

## CI gates

The release pipeline MUST enforce the following gates before tagging a
release build:

### Gate 1 — signed binary check

For each binary in the signing-requirements table above, the gate fails
unless **all** of the following hold:

- `signtool verify /pa /v <path>` returns success.
- The signing certificate's subject matches the AMD code-signing
  certificate the release pipeline owns.
- The certificate is not in the revocation list at gate-evaluation time.

### Gate 2 — bitcode reproducibility

Bitcode emission must be byte-identical across rebuilds of the same
commit. The gate runs the compiler twice on a fixed reference model and
fails if `cmp` reports any difference:

```bash
hip-compiler ref_model.mlir -o build1.bc
hip-compiler ref_model.mlir -o build2.bc
cmp build1.bc build2.bc  # must succeed
```

If this gate fails, look for nondeterministic inputs to the compiler:
absolute paths baked into metadata, embedded timestamps, hash-set
iteration order, etc.

### Gate 3 — no unsigned-PE deployment surface

The runtime path must never write an unsigned PE to disk and load it.
The gate greps the EP DLL's import table for `LoadLibrary*` /
`LoadLibraryEx*` calls and cross-references the static configuration of
DLLs the EP is allowed to load (only ROCm DLLs from
`BitcodeJIT::create`'s explicit list). Any other `LoadLibrary` site is
a regression.

---

## What changed vs. the pre-bitcode design

| Concern | Before (native DLL) | After (bitcode) |
|---------|---------------------|-----------------|
| Per-model `.dll` signing | impractical — signing on end-user machine isn't an option | not needed, artifact is data |
| WDAC strict | blocked the temp-DLL load | no temp DLL; clean |
| EDR reflective-DLL heuristics | flagged the temp-DLL load | no temp DLL; clean |
| Signed binaries on disk | 2 (EP DLL + compiler DLL) + one per-model `.dll` per cached model | 2 (EP DLL + compiler DLL) only |
