<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Plugin Extension API for Vendor-Specific Backends

**Date:** 2026-05-26
**Document Type:** Design
**Status:** Proposal — under review
**Related:** [`include/hip/Compiler/PluginAPI.h`](../../include/hip/Compiler/PluginAPI.h),
[`include/hip/Compiler/PluginRegistry.h`](../../include/hip/Compiler/PluginRegistry.h),
[`include/hip/Compiler/PluginLoader.h`](../../include/hip/Compiler/PluginLoader.h)

> **Status note (2026-05-26).** This document describes a multi-PR
> plan; PRs 1–5 of the rollout have landed in private review, plus a
> PR-6 cleanup pass driven by the design self-review (silent-failure
> fixes, doc corrections, defensive idempotency, host-owned bitcode
> buffers, slot bounds-checks, exception containment). PR 1
> introduced the C ABI header and the loader skeleton with no hook
> sites; PR 2 wired the pipeline-slot dispatch and the vtable
> registry; PR 3 added LLVM bitcode contribution with override-from-
> source semantics; PR 4 added external library + library-path
> contribution; PR 5 finalised the documentation; PR 6 hardened the
> implementation against the issues found in self-review (see the
> "Implementation status" section below for the per-issue list). The
> API surface is **not yet frozen** — vendor-team review and the
> shared-MLIR work in Open Question 6 are still outstanding — and
> should not be depended on by external consumers until the doc
> Status flips to "Stable".

---

This design lets a confidential AMD-internal team ship vendor-specific
ONNX ops, custom kernels, and MLIR passes on top of `onnx-hipdnn-ep`
without forking the public repo or contributing proprietary code
upstream. The shape is field-for-field aligned with the LLVM/MLIR
upstream plugin pattern (see Appendix C).

The downstream (confidential) repo does not yet exist — the vendor
team has not started their side of the work. Practical implication:
the sample plugin in `test/plugin/sample_plugin/` is the **only**
consumer of the plugin ABI for the foreseeable future. Any ABI we
commit to in PRs 1–5 must be defensible on its own merits; we cannot
lean on "the vendor team is already using it" to justify decisions.

This affects sequencing — see section 6 and section 8.

---

## 1. Problem Statement

The internal team needs to:

1. Add new ONNX ops (or new lowerings of existing ops) that are not, and
   should not be, in the public `onnx-hipdnn-ep` repo.
2. Use their own HIP / assembly kernels in place of the in-tree
   `hip_custom_kernels` for some ops.
3. Optionally, add MLIR passes that run inside the existing pipeline
   (e.g., target-specific fusion, layout massaging) without modifying
   `lib/Dialect/Transforms/Pipelines.cpp`.
4. Stay in sync with the public repo without merge pain — i.e., they want
   the public repo to be a clean dependency, not a fork.

The current code does not provide a clean extension point for any of
these. The realistic options today amount to "rebuild
`hip_custom_kernels` to override symbols" or "add new symbols and
rebuild the runtime bitcode." Both require modifying the public
source tree.

### 1.A Plugin Path vs. Upstream Contribution Path

**The plugin model is only for the confidential subset.** It is not a
substitute for contributing upstream — it is a complement to it.

Decision rule:

| Vendor change | Right path |
| --- | --- |
| Generic op support, performance fix, bug fix, refactor, new MLIR pass that is not target-specific, infra improvement, build-system fix | **Upstream PR** to `onnx-hipdnn-ep` following `CONTRIBUTING.md`. Goes through the normal review + CI process. Benefits the whole ecosystem. |
| New ONNX op lowering whose kernel implementation is non-confidential | **Upstream PR**. The kernel goes into `3rd-party/custom_kernels/` if appropriate. |
| Fixes or improvements to the plugin ABI itself (PluginAPI.h, registry, loader) | **Upstream PR**. The plugin ABI is public infrastructure even if some plugins are not. |
| Confidential ops, confidential kernels, vendor-specific lowerings or fusion strategies that cannot be disclosed | **Plugin DLL** in the vendor's confidential repo. |
| A change that has a confidential **and** a non-confidential half (e.g., a generic improvement that enables a confidential kernel) | **Split**. The non-confidential half goes upstream; the confidential half lives in the plugin. |

In other words: the vendor team should default to upstream contribution
and only fall back to the plugin DLL when there is an actual disclosure
constraint. The plugin model exists to keep proprietary code out of the
public repo — not to keep the vendor team's day-to-day improvements out
of the public repo.

This is also the correct posture for ABI hygiene: the more eyeballs
that exercise the public ABI, the faster bugs in it surface. If the
vendor team finds a rough edge in `PluginAPI.h` or `HipEpPluginRegistry`,
that fix should land upstream where everyone (including future plugin
authors) benefits.

The CONTRIBUTING.md flow already in tree applies as-is to vendor team
PRs against the public repo — no special exemptions, no special review
path. They are just contributors.

## 2. Why This Is Achievable in This Codebase

The public pipeline already has three boundaries that map cleanly onto
plugin extension points:

| Boundary | What it currently is | Why it is a natural plugin seam |
| --- | --- | --- |
| Pass list in `buildOnnxToHipPipeline` / `buildOnnxToHipPipelineTail` (`lib/Dialect/Transforms/Pipelines.cpp`) | An ordered sequence of `pm.addPass(...)` calls. Passes are registered with MLIR's global registry via `registerHipPipelines()` | Plugins can register additional passes with the global registry and the public pipeline can call them by name at well-defined slots — exactly how upstream MLIR pass plugins work |
| `discoverLibraries` in `lib/Compiler/CompilerDriver.cpp` | Resolves `amdhip64`, `MIOpen`, `hipblaslt`, `hip_custom_kernels`, `hipdnn_*` from `THEROCK_DIST` and env vars | Adding vendor library paths and library names is one more block of the same shape (extension beyond upstream) |
| `wrap_*` functions in `lib/Runtime/real/*.cpp`, embedded as bitcode | C++ → LLVM bitcode → embedded byte array → linked into `model.dll` at compile time | Vendor `wrap_*` bitcode can be appended to the linked module the same way (extension beyond upstream) |

The DLL boundary is also already three-tier: ORT → `hip-compiler.dll`
→ generated `model.dll`. Adding a fourth tier — a
`vendor-extension.dll` loaded by `hip-compiler.dll` — keeps proprietary
code on the vendor side of a stable C ABI.

## 3. Proposed Architecture: One Plugin, One Callback, One Registry

The proposal follows the upstream pattern: **one** entry point, **one**
registration callback, **one** registry that the plugin uses to add
passes, runtime bitcode, and external libraries.

```
┌──────────────────────────────────────────────────────────────────────┐
│                 vendor_extension.dll  (vendor-private)               │
│                                                                      │
│  hipEpGetPluginInfo() returns HipEpPluginLibraryInfo {               │
│      apiVersion, pluginName, pluginVersion,                          │
│      registerCallbacks(HipEpPluginRegistry &)                        │
│  }                                                                   │
│                                                                      │
│  Inside registerCallbacks the plugin uses the registry to:           │
│   • register passes with MLIR's global pass registry                 │
│   • request pass insertion at named pipeline slots                   │
│   • contribute LLVM bitcode for wrap_* runtime functions             │
│   • contribute library paths + library names to lld-link             │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │  loaded by hip-compiler.dll via
                              │  HipEpPluginLoader::Load(path)
                              ▼
              ┌────────────────────────────────┐
              │       hip-compiler.dll         │
              │  (public, unchanged ABI)       │
              └────────────────────────────────┘
```

### 3.A The Public ABI Header

`include/hip/Compiler/PluginAPI.h` (new public header). The struct shape
mirrors `mlir::PassPluginLibraryInfo` and `llvm::PassPluginLibraryInfo`
field-for-field.

```cpp
#ifndef HIP_COMPILER_PLUGIN_API_H
#define HIP_COMPILER_PLUGIN_API_H

#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace hip::compiler {
class HipEpPluginRegistry;
}

namespace hip::compiler {

/// Identifies the API version understood by this plugin.
///
/// The supported version is incremented for ANY ABI-breaking change to
/// the HipEpPluginLibraryInfo struct (callbacks added, removed, or
/// reordered). This matches the upstream LLVM/MLIR convention; we do
/// not use a major/minor split. Drivers reject mismatched versions.
#define HIP_EP_PLUGIN_API_VERSION 1

extern "C" {

struct HipEpPluginLibraryInfo {
  /// The API version understood by this plugin, usually
  /// HIP_EP_PLUGIN_API_VERSION.
  uint32_t APIVersion;
  /// A meaningful name of the plugin (logged on load). E.g.
  /// "AMDInternalAcceleratorPlugin".
  const char *PluginName;
  /// The version of the plugin, vendor's own scheme. E.g. "1.2.3".
  const char *PluginVersion;

  /// The single registration callback. The plugin uses the supplied
  /// registry to register passes, request pipeline-slot insertions,
  /// contribute runtime bitcode, and contribute external libraries.
  /// Called once at plugin load.
  void (*RegisterCallbacks)(HipEpPluginRegistry &) = nullptr;
};

}  // extern "C"

}  // namespace hip::compiler

/// The public entry point for a hip-compiler plugin. The driver calls
/// this to obtain the plugin info struct.
///
/// LLVM_ATTRIBUTE_WEAK is intentional: the same source can be statically
/// linked into a tool (where this symbol is resolved at link time) or
/// dynamically loaded (where the loader looks up the symbol by name).
extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo();

#endif  // HIP_COMPILER_PLUGIN_API_H
```

### 3.B The Registry (where capabilities are exposed)

The registry is the C++ class that the plugin's single callback uses.
The C ABI struct never changes when capabilities are added; the
**registry interface** evolves through normal C++ ABI rules (same
build-of-MLIR requirement applies, same as upstream MLIR plugins).

```cpp
// include/hip/Compiler/PluginRegistry.h (new public header)

namespace hip::compiler {

/// Well-defined slots in the public pipeline at which plugin passes
/// can be inserted. Append-only across versions.
enum class PipelineSlot {
  AfterSimplifyOnnx,
  AfterOnnxLoopOutline,
  AfterConvertOnnxToHip,
  BeforeBufferization,
  AfterPoolAllocs,
  BeforeConvertHipToLLVM,
  AfterGenerateInterface,
};

/// Public registry passed to plugins' RegisterCallbacks.
class HipEpPluginRegistry {
public:
  // ---------- MLIR passes (upstream-shaped) -------------------------
  /// Equivalent of mlir::PassRegistration<Pass>. The plugin's pass is
  /// added to the MLIR global pass registry; the public pipeline
  /// instantiates it by name at the requested PipelineSlot.
  template <typename PassT>
  void registerPass();

  /// Request that a registered pass run at a named pipeline slot.
  void requestPipelineSlot(PipelineSlot slot, llvm::StringRef passName);

  // ---------- Extensions beyond upstream ----------------------------
  /// Contribute LLVM bitcode that will be linked into model.dll via
  /// llvm::Linker AFTER the in-tree runtime_bc_data is linked.
  void addRuntimeBitcode(const void *data, size_t sizeBytes);

  /// Contribute one library search path to lld-link.
  void addLibraryPath(llvm::StringRef path);

  /// Contribute one library name (or full path to a .lib) to lld-link.
  void addLibrary(llvm::StringRef nameOrFullPath);
};

}  // namespace hip::compiler
```

Why a registry rather than five callbacks: this matches LLVM
(`PassBuilder &`), MLIR pass plugins (no-arg + global registry), and
MLIR dialect plugins (`DialectRegistry *`). The C struct stays at
exactly one callback; new capabilities are added by extending the
registry **class**, not the C struct. Old C structs continue to load.

### 3.C Loader Side

`hip-compiler.dll` grows a small loader modeled after
`mlir::PassPlugin::load` and `llvm::PassPlugin::Load`. It uses
`llvm::sys::DynamicLibrary` so we get the same cross-platform behavior
upstream gets.

```cpp
// include/hip/Compiler/PluginLoader.h

namespace hip::compiler {

class HipEpPluginLoader {
public:
  /// Attempts to load a plugin DLL/so from a given path. Mirrors
  /// mlir::PassPlugin::load.
  static llvm::Expected<HipEpPluginLoader> Load(const std::string &filename);

  llvm::StringRef getFilename()      const { return filename_; }
  llvm::StringRef getPluginName()    const { return info_.PluginName; }
  llvm::StringRef getPluginVersion() const { return info_.PluginVersion; }
  uint32_t        getAPIVersion()    const { return info_.APIVersion; }

  void registerCallbacks(HipEpPluginRegistry &R) const {
    if (info_.RegisterCallbacks)
      info_.RegisterCallbacks(R);
  }

private:
  // ... see header for full definition.
};

/// Loads all plugins in the HIP_EP_PLUGINS environment variable
/// (semicolon-separated) the first time it is called. Idempotent.
const std::vector<HipEpPluginLoader> &loadPluginsOnce();

}  // namespace hip::compiler
```

`HipEpPluginLoader::Load` is the direct analog of `PassPlugin::Load`:
opens the library, looks up `hipEpGetPluginInfo` by name, calls it,
checks `APIVersion`, returns the loaded plugin or an `llvm::Error`.

Hook sites in the public repo are **three** (added incrementally by
PRs 2/3/4):

- `lib/Dialect/Transforms/Pipelines.cpp` — at each of the
  `PipelineSlot` enum values (currently scattered across
  `buildOnnxToHipPipeline` and `buildOnnxToHipPipelineTail`), consult
  the registry for any plugin-requested pass at that slot and add
  them via the public `pm.addPass(...)` interface.
- `lib/Target/LLVM/LLVMBackend.cpp:linkRuntimeModule` — after the
  in-tree `runtime_bc_data` is linked, walk plugin-contributed bitcode
  buffers and `linker.linkInModule` each.
- `lib/Compiler/CompilerDriver.cpp:discoverLibraries` — append
  plugin-contributed paths and libraries.

All three hooks are <15 lines each and no-op when no plugin is loaded.

### 3.D Vendor-Side Kernel Override (Concrete Walkthrough)

Vendor wants their own `hip_elementwise_gelu`, `hip_rope_forward`, etc.
There are two pre-existing choices today:

- **Path A**: rebuild `hip_custom_kernels` with vendor's `.hip` sources
  keeping the same symbol names. Works today, no public-repo changes
  needed, but requires a private fork of `3rd-party/custom_kernels/`.
- **Path B**: ship a sibling library and point `HIP_CUSTOM_KERNELS_DIR`
  at it.

This proposal adds **Path C** — drop a vendor DLL into `HIP_EP_PLUGINS`:

```cpp
// In vendor_extension.dll, inside RegisterCallbacks:
[](HipEpPluginRegistry &R) {
  R.addLibraryPath("C:/vendor/install/lib");
  R.addLibrary("vendor_kernels");           // vendor_kernels.lib
  R.addRuntimeBitcode(vendor_wrap_bc,       // bitcode that defines
                      vendor_wrap_bc_size); // wrap_gelu etc., calling
                                            // vendor_gelu_v3 symbols
}
```

For `wrap_*` symbols where the vendor wants their version to **replace**
the in-tree one, vendor's bitcode is linked **after** the in-tree
bitcode and uses `Linker::Flags::OverrideFromSrc` (`llvm::Linker`
flag — analogous to `--allow-multiple-definition` with last-one-wins).
This is documented behavior; see Open Question 1.

For new symbols the vendor introduces (`wrap_vendor_decompress`, etc.),
there is no collision and standard linking applies.

Vendor's public-repo footprint stays the env var `HIP_EP_PLUGINS=...`
and nothing else.

### 3.E Vendor-Side New Op (Concrete Walkthrough)

Vendor adds `onnx.Custom("VendorFusedAttention")`:

1. In `vendor_extension.dll`, vendor implements an MLIR pass —
   `VendorFusedAttentionLoweringPass` — derived from
   `mlir::OperationPass<mlir::func::FuncOp>` (or a module pass) whose
   `runOnOperation` matches the `onnx.Custom` op by name and replaces
   it with `llvm.call @wrap_vendor_fused_attention` (or with a
   vendor-private op that a sibling vendor pass lowers to LLVM later
   in the pipeline).
2. Inside `RegisterCallbacks`, vendor calls
   `R.registerPass<VendorFusedAttentionLoweringPass>()` to put the
   pass in MLIR's global registry, then
   `R.requestPipelineSlot(PipelineSlot::AfterConvertOnnxToHip,
   "vendor-fused-attention-lowering")` to schedule it.
3. `R.addRuntimeBitcode(...)` contributes bitcode for
   `wrap_vendor_fused_attention`, which dispatches to vendor kernel
   symbols.
4. `R.addLibraryPath(...)` + `R.addLibrary("vendor_kernels")` add the
   kernel library.

In-tree code is touched at zero of the nine usual op-add touchpoints.
All nine equivalent steps live in the plugin DLL, where vendor controls
visibility. The vendor pass runs after `ConvertOnnxToHipPass` so any
ONNX op the in-tree pass leaves untouched is still on the table for
the vendor pass to rewrite.

### 3.F Why "Register a Pass" Instead of "Register Patterns"

Upstream MLIR plugins register passes, not patterns. We follow that
convention because:

- Patterns belong inside a pass — they are an implementation detail
  of how a pass transforms IR. The pass is the schedulable unit.
- A vendor pass can encapsulate vendor-specific patterns, cost models,
  attribute handling, or even multi-step rewriting. A bare
  `RewritePatternSet` extension cannot.
- It keeps the plugin's MLIR ABI surface symmetric with upstream.
  Vendor code is portable to other MLIR-based tools that follow the
  same convention.
- The plugin loader does not need to know about `RewritePatternSet`
  at all — only `mlir::Pass`.

## 4. What Has To Change in the Public Repo

This is intentionally small. The plan is:

1. New header `include/hip/Compiler/PluginAPI.h` — the
   `HipEpPluginLibraryInfo` C struct + `hipEpGetPluginInfo` weak entry
   declaration. Public.
2. New header `include/hip/Compiler/PluginRegistry.h` — the
   `HipEpPluginRegistry` C++ class and `PipelineSlot` enum. Public.
3. New header `include/hip/Compiler/PluginLoader.h` + implementation
   `lib/Compiler/PluginLoader.cpp` — the `HipEpPluginLoader` class
   and `loadPluginsOnce()`. Private to `LibHipCompiler`.
4. Three hook sites in `Pipelines.cpp`, `LLVMBackend.cpp`,
   `CompilerDriver.cpp`. Each ~5–15 lines. All guarded so they no-op
   when no plugins are loaded.
5. CMake: link `LibHipCompiler` against the LLVM Linker library if it
   is not already (it is, transitively via `HipTargetLLVM`). No new
   build-time dependencies.
6. Tests (in-tree, with a minimal **public** sample plugin under
   `test/plugin/sample_plugin/`):
   - Unit test that loads the sample, verifies the plugin name is
     logged.
   - LIT test that runs an `onnx.Custom("PrintAndPassthrough")`-only
     model through a sample-plugin pass and checks IR after the
     plugin slot has the plugin's hip-dialect op.
   - E2E test that builds a `model.dll` linking sample-plugin bitcode
     and runs it via `hip-test-dll`.
7. CONTRIBUTING / README mention: "Plugins are loaded via
   `HIP_EP_PLUGINS`. ABI is at
   `include/hip/Compiler/PluginAPI.h`. See the sample under
   `test/plugin/sample_plugin/`."

The public sample plugin is critical so the API is exercised in CI;
without it the loader hooks atrophy on every refactor.

## 5. What Has To Change on the Vendor Side

A new repo (or directory in the vendor's monorepo) with:

- `vendor_extension/` — code that builds `vendor_extension.dll`.
  Includes a CMake target that links **against the public** `hip-compiler`
  for headers (specifically `PluginAPI.h` and the MLIR / HIP-dialect
  headers it transitively needs).
- `vendor_extension/runtime/` — `.cpp` files compiled with `clang
  -emit-llvm` exactly the way `lib/Runtime/CMakeLists.txt` does it.
  The output `.bc` is embedded into `vendor_extension.dll` the same
  way `runtime_ir_data.cpp` embeds `runtime.bc` (xxd-style byte
  array).
- `vendor_extension/kernels/` — vendor's HIP kernels, built with
  `hipcc` into `vendor_kernels.lib`. Same shape as
  `3rd-party/custom_kernels/CMakeLists.txt`.

Vendor consumes the public `hip-compiler` as a binary dependency
(via the public install tree under `THEROCK_DIST` style or a similar
`HIPDNN_EP_DIST`), exactly the way the public EP consumes TheRock today.
No fork of the public repo is required.

## 6. Multi-PR Implementation Plan

This is intended to land via small, reviewable PRs (the way the
contribution guide requires). Each PR is independently mergeable. The
sequencing matches what upstream plugin support looks like: ABI
first, pass-registration second, runtime extensions last.

### PR 1 — Public Plugin ABI Header + Loader Skeleton

Scope:

- Add `include/hip/Compiler/PluginAPI.h` (the
  `HipEpPluginLibraryInfo` struct + `hipEpGetPluginInfo` weak
  declaration).
- Add `include/hip/Compiler/PluginRegistry.h` (the
  `HipEpPluginRegistry` class + `PipelineSlot` enum, **registry
  methods are stubs that no-op** in this PR — they fill in across
  later PRs).
- Add `lib/Compiler/PluginLoader.{cpp,h}` (the `HipEpPluginLoader`
  class + `loadPluginsOnce()`). Reads `HIP_EP_PLUGINS`. **No hook
  sites yet** — loader only loads and logs plugin name + version.
- CONTRIBUTING update: short paragraph on plugin loading semantics
  pointing at the sample.
- Unit test: build a tiny sample plugin in `test/plugin/`, set
  `HIP_EP_PLUGINS=$<TARGET_FILE:sample_plugin>`, assert loader logs
  the plugin name.

Risk: low. Zero behavior change unless env var is set. No registry
methods do anything yet, so plugins can be loaded but cannot affect
compilation — exactly what we want for an ABI-establishing PR.
Size budget: ~300 LOC of production code, ~600 LOC including the
public design doc and the sample-plugin scaffolding.

### PR 2 — Pass Registration + Pipeline Slots

This is the **first** PR that lets a plugin actually change
compilation.

Scope:

- Implement `HipEpPluginRegistry::registerPass<PassT>()` (delegates
  to MLIR's global pass registry — `mlir::PassRegistration<PassT>`).
- Implement `HipEpPluginRegistry::requestPipelineSlot(slot, name)`.
- Wire `lib/Dialect/Transforms/Pipelines.cpp` to consult the registry
  at each `PipelineSlot` enum value and to add plugin-requested
  passes by name.
- Sample plugin contributes a no-op pass (e.g.,
  `SamplePrintFunctionsPass` that just emits a remark per `func.func`)
  and asks for it at `PipelineSlot::AfterConvertOnnxToHip`.
- LIT test verifies the remark appears in the diagnostic output of a
  small model.

Risk: medium. Slot enum values become a stable surface; subsequent
additions must be append-only.

### PR 3 — Runtime Bitcode Contribution **(landed 2026-05-26)**

Scope as landed:

- `HipEpPluginRegistry::addRuntimeBitcode(data, size)` is now wired
  into the vtable + per-process registry storage; the registered
  buffers are exposed by the new `pluginBitcodeBuffers()` accessor.
  Plugins are required to back their pointers with static storage
  (typically the DLL's read-only data segment), since the registry
  records (data, size) by reference and never copies.
- `lib/Target/LLVM/LLVMBackend.cpp::linkRuntimeModule` was
  refactored around a `linkBitcodeBuffer` helper, then extended to
  walk `pluginBitcodeBuffers()` after the in-tree `runtime_bc_data`
  is linked. Plugin buffers are linked with
  `Linker::Flags::OverrideFromSrc` so vendor symbols cleanly shadow
  in-tree ones (resolves Open Question 1 in the affirmative for the
  override side).
- The sample plugin in `test/plugin/sample_plugin/` now compiles
  `sample_plugin_runtime.cpp` to LLVM bitcode at build time (clang
  `-emit-llvm`) and embeds the result via the existing `xxd.py`
  pipeline. The plugin's `RegisterCallbacks` calls
  `addRuntimeBitcode(kSamplePluginBitcode, kSamplePluginBitcodeSize)`.
  In a degraded build without clang/python at configure time, the
  embedded buffer is empty (size 0) and the plugin skips the call;
  the unit test detects the degraded mode and skips the round-trip
  assertion with an explanatory message.
- Unit test (`test/plugin/test_plugin_loader.cpp`) verifies the
  bitcode round-trip: after `RegisterCallbacks` runs,
  `pluginBitcodeBuffers()` grows by exactly one entry, the entry's
  size is non-trivial (>= 4 bytes), and the first four bytes are
  the LLVM bitcode magic `0x42 0x43 0xC0 0xDE` (i.e. `BC\xc0\xde`).
  This proves the buffer survived the C ABI boundary intact and is
  parseable by `llvm::parseBitcodeFile`.

Tests deferred to a follow-up:

- An E2E test that exercises override semantics (a vendor `wrap_*`
  shadowing an in-tree definition that the model actually calls).
  Doing this cleanly requires picking an in-tree symbol whose
  observable behaviour can be checked from the dev environment, and
  the dev environment currently has a missing-`amdhip64.lib`
  problem unrelated to this work that prevents `E2E_Compile`/
  `E2E_Execute` from running locally. The `linkRuntimeModule` code
  path itself is exercised on every model compile, so the
  integration is exercised in the broad sense; the explicit
  override-vs-in-tree assertion will land once the dev-env block
  clears.

Risk as landed: low for the new-symbol path (round-trip + magic-
bytes check are tight); medium for the override path until a
follow-up adds the E2E override assertion described above.

### PR 4 — External Library Contribution **(landed 2026-05-26)**

Scope as landed:

- `HipEpPluginRegistry::addLibraryPath` / `addLibrary` are wired
  into the vtable + per-process registry storage; the registered
  paths and library names are exposed by the new
  `pluginLibraryPaths()` and `pluginLibraries()` accessors. Both
  copy the plugin's `StringRef` into a per-process `std::string`
  to decouple the recorded value's lifetime from the plugin's
  argument storage.
- `CompilerDriver::discoverLibraries` was split into
  `discoverInTreeLibraries` (the existing THEROCK_DIST + custom-
  kernels + hipDNN-graph-runtime body) and a new top-level
  `discoverLibraries` that calls the in-tree helper, then appends
  plugin paths and library names. Critically, the early-return on
  missing `THEROCK_DIST` was lifted so plugin contributions flow
  through even on a CPU-only smoke build that loads a vendor
  plugin.
- Plugin contributions are appended *after* the in-tree libraries.
  This keeps in-tree-defined symbols winning under lld-link's
  command-line search; vendors who need to override an in-tree
  symbol use the bitcode mechanism (PR 3) with
  `Linker::Flags::OverrideFromSrc` instead. The doc comment on
  `pluginLibraries()` documents this contract.
- The sample plugin grew a sibling `hip_ep_sample_lib` static
  library (compiled from a one-function `sample_lib.cpp`) and
  hands its directory to `addLibraryPath` and its bare name
  `hip_ep_sample_lib` to `addLibrary`. Both calls are
  unconditional in the sample, so the unit test asserts "grew by
  exactly 1" rather than "grew by 0 or 1". CMake plumbing in
  `test/plugin/sample_plugin/CMakeLists.txt` propagates the
  static lib's directory into the plugin source via two compile-
  time defines (`HIP_EP_SAMPLE_LIB_DIR`, `HIP_EP_SAMPLE_LIB_NAME`).
- Unit test (`test/plugin/test_plugin_loader.cpp`) asserts the
  round-trip: after `RegisterCallbacks` runs,
  `pluginLibraryPaths()` and `pluginLibraries()` each grew by
  exactly one entry; the new library entry equals the literal
  `hip_ep_sample_lib`; the new path entry is non-empty. Plugin
  version bumped 0.3.0 → 0.4.0 so the literal-match assertion
  catches stale builds.

Tests deferred to a follow-up:
  An E2E test that invokes a model whose generated code calls a
  symbol defined in the plugin static library. Same dev-env
  blocker as PR 3: this dev machine cannot run E2E_Compile or
  E2E_Execute because of a missing `amdhip64.lib`. The
  `discoverLibraries` code path is exercised on every model
  compile via `linkToDLL`, so the integration is exercised in
  the broad sense; the explicit "plugin-defined symbol resolved
  through plugin lib" assertion lands once the dev-env block
  clears.

Risk as landed: low. The change is symmetric to the existing
`HIP_CUSTOM_KERNELS_DIR` path; the only meaningful behaviour
change for non-plugin users is that `discoverLibraries` no longer
early-returns on missing `THEROCK_DIST`. Functional behaviour is
identical: when no plugins are loaded, the post-`THEROCK_DIST`
section runs with empty plugin accessors and produces the same
`libraries` / `library_paths` vectors as before.

### PR 5 — CONTRIBUTING + README + Vendor Documentation **(landed 2026-05-26)**

Scope as landed:

- `README.md` gained a "Plugin extension API" section pointing at
  the design doc, the authoring guide, and the in-tree sample,
  with an explicit note that the ABI is in proposal status.
- `CONTRIBUTING.md`'s plugin section was expanded from the PR 1
  framing ("PR 1 in flight") to "PRs 1–4 landed; freeze pending".
  Added a third bullet documenting the rules for adding new methods
  to `HipEpPluginRegistry` (vtable extension, host wiring, unit
  test, no removals/reorders).
- New `docs/plugin_authoring.md` is a practical companion to this
  design doc. It covers: mental model; the in-tree sample as the
  worked example; quickstart for a minimal plugin; pattern for each
  of the five contribution points (`registerPass`,
  `requestPipelineSlot`, `addRuntimeBitcode`, `addLibraryPath`,
  `addLibrary`); a distribution checklist.

Risk as landed: zero. Docs only.

### PR 6 — Self-Review Cleanup **(landed 2026-05-26)**

PR 6 is a hardening pass driven by a careful self-review of the
PR 1–5 surface. No new capability; every fix is mechanical or
documentation, motivated by a concrete misuse mode that reading
the LLVM source confirmed could bite a real plugin author.

Scope as landed:

- **Loader behaviour.** Bad plugin paths in `HIP_EP_PLUGINS` now
  emit a single-line `[plugin-loader] WARNING:` to stderr instead
  of being silenced behind `HIPDNN_EP_DEBUG`. Duplicate paths
  (`foo.dll;foo.dll`) are deduplicated so `RegisterCallbacks` is
  invoked exactly once per unique entry. A throwing
  `RegisterCallbacks` is contained by a `try { ... } catch
  (std::exception &) { ... } catch (...) { ... }` block that
  warns and continues with the next plugin, bounding the blast
  radius of CRT / libstdc++ mismatches across the plugin DLL
  boundary.
- **Registry behaviour.** Out-of-range `PipelineSlot` casts in
  `requestPipelineSlot` are bounds-checked against the V1 enum
  size, warned, and dropped (rather than recorded silently and
  later mismatched). `addRuntimeBitcode` now copies the plugin's
  bytes into host-owned storage and skips empty buffers with a
  warning (instead of either trusting plugin lifetime or letting
  a 0-byte call propagate as a cryptic
  `llvm::parseBitcodeFile` error). `pluginLibraryPaths` and
  `pluginLibraries` return owning `std::string` copies rather
  than `StringRef` views, eliminating the StringRef-stability
  hazard entirely.
- **Defensive idempotency.** Each accessor (`pluginPassesForSlot`,
  `pluginBitcodeBuffers`, `pluginLibraryPaths`, `pluginLibraries`)
  now calls `dispatchPluginRegistrationsOnce` itself, so a future
  tool that bypasses `CompilerDriver::compile` / `hip-mlir-opt`
  still observes plugin-contributed state. The guard is
  `std::call_once`, so it is essentially free after the first
  call.
- **Doc + assertion fixes.** The `registerPass<>()` docstring no
  longer claims the pass joins the host's MLIR registry (it
  doesn't, per Open Question 6 — the plugin DLL writes into its
  own copy of `mlir::passRegistry` because both host and plugin
  link MLIR statically). Appendix D.2 of this doc no longer
  claims plugins link `HipCInterface.lib` (they don't — the
  vtable is exactly what avoids needing an import lib). The
  authoring guide gained a "Symbol naming and override
  semantics" section explaining that
  `Linker::Flags::OverrideFromSrc` is unconditional and
  all-or-nothing per `llvm/lib/Linker/LinkModules.cpp`. A
  `static_assert` on `sizeof(VTable)` is a tripwire forcing
  maintainers to bump `HIP_EP_PLUGIN_API_VERSION` whenever the
  vtable layout changes.

What we did **not** do in PR 6:

- The cross-DLL `mlir::PassRegistration` problem (Open Question 6)
  is unchanged; the docstring is honest about it but the fix is
  a separate PR. Reading LLVM's `PassPluginLibraryInfo` shows the
  fix path: route `registerPass<>()` through the vtable so the
  host TU calls `mlir::registerPass(allocator)`, mirroring LLVM's
  `void (*RegisterPassBuilderCallbacks)(PassBuilder &)` design.
- The `OverrideFromSrc` semantic itself is unchanged. Whether to
  switch to per-symbol opt-in is a design call that needs vendor
  input, and warrants its own PR + discussion. The authoring
  guide warns about it loudly in the meantime.

Risk as landed: low. Every change is either documentation, a
defensive no-op (idempotent dispatch), a stricter check that
turns silent failures into loud warnings (slot bounds, zero-byte
bitcode, throwing callbacks), or a contract tightening that
makes the existing tests clearer (StringRef → std::string,
host-owned bitcode). No behaviour change for the sample plugin
or for any well-formed vendor plugin.

### Vendor-Side Bring-Up (Vendor Repo, Not Public)

**Prerequisite (not yet met)**: the confidential vendor repo must
exist and be set up to consume the public install tree. As of
2026-05 it does not.

Scope (vendor work, **not a public-repo PR**):

- Vendor sets up their (new) confidential repo with a CMake target
  that consumes the public `hip-compiler` install tree's headers.
  Per Appendix D.2: no host import library is needed; the registry
  vtable is filled by the host and the plugin links only against
  the headers + MLIR for `mlir::PassRegistration<T>`'s definition.
- Vendor builds `vendor_extension.dll` and smoke tests by loading
  their plugin into the public `hip-compiler` and reproducing the
  public sample-plugin test, then layering in their own ops.

This work does not land in the public repo. PRs 1–6 do not depend
on it; they can land in any order relative to vendor-side work.

## 7. Open Questions

1. **Symbol collision policy for runtime bitcode.** If a vendor plugin
   defines `wrap_gelu` and the in-tree `runtime_activation.bc` also
   defines `wrap_gelu`, what does `llvm::Linker` do? Default behavior
   errors on duplicate definition. We propose using
   `Linker::Flags::OverrideFromSrc` for plugin-contributed bitcode so
   vendor symbols cleanly shadow in-tree ones, and documenting that
   "first-loaded plugin with a given symbol wins" if multiple plugins
   contribute the same symbol. PR 3 commits to this; needs a test for
   the override and a test for the new-symbol case.

2. **MLIR / C++ ABI surface in the registry.** `HipEpPluginRegistry`
   exposes templated and class-typed methods (e.g.
   `registerPass<PassT>`, references to MLIR types). Same as upstream
   MLIR plugins, this is not a pure C ABI; both sides must be built
   against the same MLIR. We document this constraint explicitly in
   the header and in CONTRIBUTING. Plugins must be rebuilt when the
   public repo updates LLVM. (This matches the constraint upstream
   MLIR plugins live with.)

   **PR-2 update.** In implementing PR 2 we discovered that the
   `requestPipelineSlot` / `addRuntimeBitcode` / `addLibraryPath` /
   `addLibrary` methods cannot be normal C++ method calls across
   the DLL boundary because hip-compiler ships as a static library,
   not a DLL — so plugin DLLs have nothing to import from. We solved
   this by making the registry a thin C-vtable: the plugin sees
   inline thunks that dispatch through function pointers stored on
   the registry instance, populated by the host. See PluginRegistry.h
   for details. This is the same pattern COM and the V8 embedder API
   use; it preserves the upstream "registry passed by reference"
   feel without requiring a shared-library hip-compiler.

3. **Pass options for plugin passes.** MLIR passes are typically
   parameterized via a generated options struct. Plugin passes that
   want options need to use the same MLIR mechanism. The registry
   does not need to know about options — they go through MLIR's
   normal pass-options surface. No public-API changes needed; just
   needs documentation.

4. **`onnx.Custom` namespace ownership.** When vendor adds an
   `onnx.Custom("VendorOp")`-shaped op, what guards against a future
   in-tree pass also wanting to handle `onnx.Custom` ops? Today the
   in-tree pipeline does not match `onnx.Custom` generically — each
   op-name is matched explicitly (e.g., `MultiHeadAttention`,
   `GroupQueryAttention`). We propose documenting a vendor-prefix
   convention (e.g., `onnx.Custom("amdvendor.FusedAttention")`) so
   the in-tree side never collides.

5. **Test isolation.** The sample plugin must build in CI even on
   machines without `HIP_EP_PLUGINS` set in the environment. CMake
   should always build it; the test runner sets the env var
   per-test. No design impact, but PR 1 needs to get this right.

6. **Shared MLIR requirement for cross-DLL pass instances.**
   *(New in PR 2 — needs vendor-team / build-system review.)* MLIR
   carries process-global state for its pass registry, dialect
   registry, and TypeIDs. When the plugin DLL statically links the
   `MLIR*.lib` static libs from `prebuilt-local/`, that state is
   duplicated per DLL: the plugin's
   `mlir::PassRegistration<MyPass>()` writes into the plugin's copy
   of the pass registry, not the host's. As a result, a plugin pass
   "registered" through the public ABI is not visible to the host's
   `parsePassPipeline` lookup, so `Pipelines.cpp` slots cannot
   actually instantiate plugin passes today.

   **Status.** PR 2 lands the host-side machinery (vtable-based
   registry, slot-request storage, pipeline-slot wiring, idempotent
   plugin dispatch). The PR-2 LIT test
   `test/lit/Plugin/sample_plugin_pass.mlir` is committed with
   `XFAIL: *` precisely so this gap is reviewable in code rather
   than only in this doc.

   **Resolution path.** This is the same problem upstream LLVM
   solves with `libLLVM.so` — a single shared library that both the
   host and the plugin link against, so `mlir::PassRegistry`,
   `mlir::DialectRegistry`, and `mlir::TypeID` live in one place.
   Adopting that for `onnx-hipdnn-ep` requires either:

   - Switching `prebuilt-local/` to ship `libMLIR.dll` /
     `libMLIR.so` instead of the per-component static libs (the
     "whole-libMLIR shared lib" build mode upstream supports), or
   - Building hip-compiler itself as a shared library, exporting
     the MLIR symbols it uses, and having plugins delay-load
     against it (Windows-specific; brittle across compilers).

   **Recommendation.** Defer this to a focused build-system PR
   after vendor review confirms the public ABI is the right shape.
   Until then, plugins can still contribute LLVM bitcode (PR 3) and
   linker entries (PR 4) without any MLIR state crossing the DLL
   boundary, so the broader design is not blocked.

### Resolved by adopting the upstream pattern

- *Versioning scheme*: dropped major/minor in favor of single
  `APIVersion` per upstream's documented "any callback change is
  breaking" rule.
- *Number of callbacks in the C struct*: collapsed from five to one;
  capabilities live on the registry class instead.
- *Pattern-set extension surface*: removed. Plugins register passes,
  not patterns. Patterns are private to the plugin's pass.
- *Constant externalization for plugin patterns*: also removed as a
  question — plugins own their own passes, so they own their own
  constant externalization. The in-tree
  `ConvertOnnxToHipPass`'s externalization helper is not exposed to
  plugins.

## 8. Confidence Assessment

The design is rated at **high** confidence overall after a 2026-05-26
spike that exercised the proposed ABI shape end-to-end on Windows MSVC.
Per-axis breakdown:

| Axis | Confidence | Why |
| --- | --- | --- |
| Plugin ABI shape matches upstream | **High** | Field-for-field aligned with `mlir::PassPluginLibraryInfo` and `llvm::PassPluginLibraryInfo`. Verified against upstream sources cited in Appendix C. |
| ABI works on Windows MSVC | **High** | Spike on 2026-05-26 built + ran on cl.exe (MSVC 19.44) clean. Entry-point export, struct-by-value return, and callback-with-reference all behave correctly. See "Spike findings" below. |
| Hook sites are the right ones | **High** | Three hooks anchor to single, named, narrow functions: pipeline-slot in `Pipelines.cpp`, `linkRuntimeModule`, `discoverLibraries`. No speculative additions. |
| Registry covers needed capabilities | **High** for passes; **Medium-High** for bitcode/library | Pass registration uses MLIR's existing global registry, so the upstream pattern carries directly. Bitcode and library injection are extensions beyond upstream — concept is sound but only public-tree validation will close out the residual risk. |
| ABI versioning is workable | **High** | Single `APIVersion` matches upstream discipline; rebuild-on-change is the simple, well-understood model. |
| Multi-PR sequencing is mergeable | **High** | Each PR only fills in registry methods one capability at a time; the previous PR's tests still pass without later capabilities. |
| Vendor-side build works against the public install tree | **Medium** | Symmetric to how the public EP consumes TheRock today; validated end-to-end only in PR 6. |
| No proprietary code leaks | **High** | All vendor code is in the closed DLL; the only public footprint is `HIP_EP_PLUGINS` env var. |

What would still push the overall design to **high+**:

1. ~~A ~30-minute prototype of PR 1...~~ **DONE 2026-05-26.** Spike
   built and ran on Windows MSVC 19.44 (VS 2022). All five validation
   checks passed (see "Spike findings" below). Confidence on the
   loader/ABI mechanism is now **high**.
2. A spike on PR 2 (plugin pass registered via the global MLIR
   registry, requested at a slot in `Pipelines.cpp`) to confirm the
   slot model fits cleanly into the existing pipeline construction
   without disrupting the existing pass-ordering invariants.
3. **Vendor-team review of this doc** before PR 1 lands. Because the
   vendor repo doesn't exist yet, the vendor team can shape the ABI
   for free right now — once PR 1 lands the ABI gets a public
   commitment and changing it costs more.

### Spike findings (2026-05-26)

A standalone CMake project outside the public repo exercised the
proposed ABI shape:

- `plugin_api.h` — the `HipEpPluginLibraryInfo` struct and
  `hipEpGetPluginInfo` declaration, with `__declspec(dllexport)` on
  the entry point (since `LLVM_ATTRIBUTE_WEAK` is a no-op on `_WIN32`
  per upstream `llvm/Support/Compiler.h`).
- `sample_plugin.cpp` → `sample_plugin.dll` (MSVC, static CRT).
- `spike_host.cpp` → `spike_host.exe` mimicking what
  `HipEpPluginLoader::Load` does inside `hip-compiler.dll`:
  `LoadLibraryA` + `GetProcAddress` for `hipEpGetPluginInfo`, then
  call the entry point and validate the result.

Build was clean (no warnings on the entry-point definition or
declaration; no name-mangling complaints; no struct-by-value warnings).
`dumpbin /exports` confirmed `hipEpGetPluginInfo` is exported under
the unmangled C name. The host then ran all five validation checks:

```
[spike] CHECK 1 PASS: DLL loaded
[spike] CHECK 2 PASS: symbol hipEpGetPluginInfo resolved
[spike] CHECK 3 PASS: struct returned by value (APIVersion=1,
                     PluginName='SpikeSamplePlugin', PluginVersion='0.1')
[spike] CHECK 4 PASS: RegisterCallbacks pointer present
[spike] CHECK 5 PASS: callback fired, registry.sentinel = 0xC0FFEE
[spike] ALL CHECKS PASS
```

Implication for the design:

- **Windows weak-symbol concern is moot for our use case.** Upstream
  `LLVM_ATTRIBUTE_WEAK` is a no-op on `_WIN32`. We only do dynamic
  loading (not the static-link-into-tool mode upstream sometimes
  uses), so `__declspec(dllexport)` on the plugin side is sufficient.
  The header should still spell `LLVM_ATTRIBUTE_WEAK` for symmetry
  with upstream and to leave the door open for non-Windows builds.
- **Struct-by-value across the C ABI works cleanly with MSVC.** No
  ABI surprise, no calling-convention mismatch.
- **Function-pointer callback fires correctly across the DLL boundary
  with a passed-by-reference C++ class** (`HipEpPluginRegistry &`).
  The reference identity is preserved (the sentinel set inside the
  plugin is observed by the host).
- The toolchain (cl.exe + Ninja + CMake) is the same one the public
  repo uses, so the spike result transfers directly.

### Implementation readiness

The technical readiness gates are met. The non-technical gate that is
**not** met: the vendor team has not yet reviewed the design.

Recommended sequencing given the current state:

1. Share this doc with the vendor team for design review.
2. Incorporate any feedback (capability set, naming, slot enum
   contents).
3. Run a spike on PR 2 (plugin pass registered via the global MLIR
   registry, requested at a slot in `Pipelines.cpp`) to confirm the
   slot model fits cleanly.
4. Land PR 1 with a frozen ABI.
5. Land PRs 2–5 incrementally; the sample plugin in `test/plugin/`
   keeps the API exercised in CI even before the vendor team is
   building against it.
6. Vendor team starts PR 6 (their confidential repo) when they are
   ready. Public-side PRs do not block on this.

If the vendor team needs more time to review, holding PR 1 on draft
status is **fine** — there is no production cost to keeping this in
design review. The cost of landing PR 1 prematurely is committing to
ABI choices the vendor team would have shaped differently.

---

## Appendix A — Why Not Simpler Alternatives

| Alternative | Why rejected |
| --- | --- |
| Vendor forks the public repo | Merge pain on every public release. Forces vendor's proprietary code to live in a vendor-only branch of a public-shaped repo. |
| Vendor only contributes via Path A (override `hip_custom_kernels`) | Works for capability C but not for new ops or new MLIR passes. |
| Vendor contributes everything upstream | Confidentiality constraint. Some kernels and lowerings cannot be open-sourced. |
| Build-time plugin (CMake `add_subdirectory(vendor)`) | Couples the build of the public repo to the presence of a private repo. Ours-and-theirs CI breaks. |
| Out-of-process plugin (separate executable / RPC) | Overkill. The compile boundary is already a DLL load; a sibling DLL load is the smallest delta. |

## Appendix B — File-Level Touch List for the Public Repo

These are the files that **would** change across PRs 1–5. Listed for
sizing only — none of them are touched by this design doc itself.

| File | PR | What changes |
| --- | --- | --- |
| `include/hip/Compiler/PluginAPI.h` (new) | 1 | Add `HipEpPluginLibraryInfo` + `hipEpGetPluginInfo` weak decl. |
| `include/hip/Compiler/PluginRegistry.h` (new) | 1 | Add `HipEpPluginRegistry` class + `PipelineSlot` enum (methods stubbed). |
| `include/hip/Compiler/PluginLoader.h` (new) | 1 | Add `HipEpPluginLoader` class + `loadPluginsOnce` decl. |
| `lib/Compiler/PluginLoader.cpp` (new) | 1 | Loader implementation. |
| `lib/Compiler/CMakeLists.txt` | 1 | Add `PluginLoader.cpp` to `LibHipCompiler`. |
| `lib/Compiler/PluginRegistry.cpp` (new) | 2 | Implement `registerPass` + `requestPipelineSlot`. |
| `lib/Dialect/Transforms/Pipelines.cpp` | 2 | Consult registry at each `PipelineSlot`. |
| `lib/Target/LLVM/LLVMBackend.cpp` | 3 | Link plugin bitcode after in-tree runtime. |
| `lib/Compiler/CompilerDriver.cpp` | 4 | Append plugin libraries in `discoverLibraries`. |
| `test/plugin/sample_plugin/*` (new) | 1–4 | Sample plugin source + CMake + LIT/E2E tests. |
| `CONTRIBUTING.md` | 5 | Plugin section. |
| `README.md` | 5 | One-liner pointer. |

Total expected diff across PRs 1–5: roughly 1000–1400 lines, of which
the sample plugin and tests are the bulk. Production code added to
`LibHipCompiler` is well under 400 lines.

## Appendix C — Upstream Plugin API Alignment Matrix

This is the per-field comparison between our `HipEpPluginLibraryInfo`
and the upstream headers. Sources verified:

- `llvm/include/llvm/Plugins/PassPlugin.h` (`LLVM_PLUGIN_API_VERSION = 2`)
- `mlir/include/mlir/Tools/Plugins/PassPlugin.h` (`MLIR_PLUGIN_API_VERSION = 1`)
- `mlir/include/mlir/Tools/Plugins/DialectPlugin.h`
- `llvm/examples/Bye/Bye.cpp` (the in-tree pass-plugin example)

| Concept | Upstream | Ours | Compliant? |
| --- | --- | --- | --- |
| API version field | Single `APIVersion` (uint32_t) | Single `APIVersion` (uint32_t) | YES |
| Versioning policy | "Any callback add/remove/reorder bumps the version" | Same | YES |
| Plugin name field | `PluginName` (`const char *`) | `PluginName` (`const char *`) | YES |
| Plugin version field | `PluginVersion` (`const char *`) | `PluginVersion` (`const char *`) | YES |
| Number of callbacks in struct | 1 (LLVM has 1 primary + 1 codegen aux; MLIR has 1) | 1 | YES |
| Callback signature shape | Takes registry by reference (`PassBuilder &` / `DialectRegistry *`) | Takes `HipEpPluginRegistry &` | YES |
| Entry point name | `llvmGetPassPluginInfo` / `mlirGetPassPluginInfo` / `mlirGetDialectPluginInfo` | `hipEpGetPluginInfo` | YES (same convention) |
| Entry point return | Struct **by value** | Struct **by value** | YES |
| Entry point linkage | `extern "C" ::ns::T LLVM_ATTRIBUTE_WEAK` | `extern "C" ::ns::T LLVM_ATTRIBUTE_WEAK` | YES |
| Loader class shape | `static Expected<Plugin> Load(filename)` using `llvm::sys::DynamicLibrary` | Same shape (`HipEpPluginLoader::Load`) | YES |
| Unit of MLIR extension | A pass (or a dialect) registered with global registry | A pass registered with global registry | YES |
| Pattern-set extension | Not exposed | Not exposed | YES |
| Bitcode injection | Not present | Present (extension beyond upstream) | EXTENSION |
| External library injection | Not present | Present (extension beyond upstream) | EXTENSION |
| Pre-codegen callback | LLVM has it (used by `llc -plugin`); MLIR does not | We do not — no native-code generation in plugin scope | INTENTIONAL OMISSION |

**Net**: every field of the C struct is upstream-compliant. The two
extensions (bitcode + libraries) are deliberate additions for a code
generator that produces a DLL, not a transformation tool that consumes
and emits IR. They live on the registry (a class), not the C struct, so
the C ABI surface stays minimal.

## Appendix D — Repository Topology

Three views: who owns what, build-time, and runtime composition.

### D.1 Side-by-side: who owns what

```
┌─────────────────────────────────────────┬────────────────────────────────────────┐
│ PUBLIC: onnx-hipdnn-ep                  │ PRIVATE: vendor-extension              │
│ (GitHub, MIT)                           │ (vendor confidential repo)             │
├─────────────────────────────────────────┼────────────────────────────────────────┤
│ Existing tree (unchanged):              │ vendor_extension/                      │
│   lib/Compiler/CompilerDriver.cpp       │ ├── CMakeLists.txt                     │
│   lib/Conversion/OnnxToHip/...          │ │     (consumes public install tree)   │
│   lib/Conversion/HipToLLVM/...          │ │                                      │
│   lib/Dialect/Transforms/Pipelines.cpp  │ ├── src/                               │
│   lib/Runtime/...                       │ │   └── plugin_main.cpp                │
│   lib/Target/LLVM/...                   │ │       (hipEpGetPluginInfo + cb)      │
│   3rd-party/custom_kernels/             │ │                                      │
│   lib/HipDNNGraph(Runtime)/             │ ├── passes/                            │
│                                         │ │   ├── VendorFusedAttentionPass.cpp   │
│ NEW (PRs 1-5):                          │ │   ├── VendorLayoutPass.cpp           │
│   include/hip/Compiler/PluginAPI.h      │ │   └── ...                            │
│   include/hip/Compiler/PluginRegistry.h │ │       (private MLIR passes)          │
│   include/hip/Compiler/PluginLoader.h   │ │                                      │
│   lib/Compiler/PluginLoader.cpp         │ ├── runtime/                           │
│   lib/Compiler/PluginRegistry.cpp       │ │   ├── wrap_vendor_*.cpp              │
│   3 hook sites in:                      │ │   │   (compiled to .bc, embedded)    │
│     Pipelines.cpp                       │ │   └── runtime_ir_data_vendor.cpp     │
│     LLVMBackend.cpp                     │ │       (vendor's xxd-style blob)      │
│     CompilerDriver.cpp                  │ │                                      │
│   test/plugin/sample_plugin/            │ ├── kernels/                           │
│   CONTRIBUTING.md (plugin section)      │ │   ├── vendor_gelu.hip                │
│                                         │ │   ├── vendor_attention.hip           │
│ License: MIT                            │ │   └── ...                            │
│ Visibility: open source                 │ │       (built to vendor_kernels.lib)  │
│                                         │ │                                      │
│ Vendor team contributes here for        │ └── tests/                             │
│ non-confidential changes via the        │     └── (vendor's own E2E)             │
│ normal CONTRIBUTING.md flow             │                                        │
│ (see section 1.A).                      │ License: vendor's own / proprietary    │
│                                         │ Visibility: never published            │
└─────────────────────────────────────────┴────────────────────────────────────────┘
```

### D.2 Build-time dependency

The vendor DLL needs **headers only** from the public install tree
(plus MLIR transitively, for `mlir::PassRegistration<T>` in
`registerPass<>()`). It does **not** link against any
`hip-compiler` library: the `HipEpPluginRegistry` thunks dispatch
through a vtable populated by the host at load time, so the plugin
has no host-side symbols to resolve. This is why the host can ship
as a static library (`LibHipCompiler.lib`) embedded in the EP DLL
without forcing every host to also export an import lib.

```
┌──────────────────────────────────────────┐
│  PUBLIC repo (cloned by vendor team)     │
│  c:\vendor\src\onnx-hipdnn-ep\           │
│                                          │
│  cmake --preset windows-release          │
│  cmake --build build/                    │
│  cmake --install build/ --prefix=PREFIX  │
│                                          │
│  Install tree at PREFIX:                 │
│    PREFIX/include/hip/Compiler/PluginAPI.h         │
│    PREFIX/include/hip/Compiler/PluginRegistry.h    │
│    PREFIX/include/hip/Dialect/IR/HipOps.h.inc      │
│    PREFIX/include/...  (MLIR headers, transitive)  │
└──────────────────────────────────────────┘
                  │
                  │  no fork; vendor pins to a specific
                  │  public release tag (e.g. v0.7.2)
                  │
                  ▼
┌──────────────────────────────────────────┐
│  PRIVATE repo                            │
│  c:\vendor\src\vendor-extension\         │
│                                          │
│  cmake -DHIPDNN_EP_DIST=PREFIX           │
│        -B build                          │
│  cmake --build build                     │
│                                          │
│  Build links against:                    │
│    PREFIX/include/hip/Compiler/PluginAPI.h         │
│    PREFIX/include/hip/Compiler/PluginRegistry.h    │
│    PREFIX/include/...  (MLIR pass APIs)  │
│    (MLIR libs from the same tree, for    │
│     mlir::PassRegistration<T>'s defn.)   │
│                                          │
│  Does NOT link against any hip-compiler  │
│  static or import lib: the plugin's      │
│  HipEpPluginRegistry calls go through a  │
│  host-supplied vtable, no host symbols   │
│  cross the DLL boundary.                 │
│                                          │
│  Build outputs:                          │
│    build/vendor_extension.dll            │
│    build/vendor_kernels.lib              │
│    build/runtime_ir_data_vendor.obj      │
└──────────────────────────────────────────┘
```

### D.3 Runtime composition (what gets deployed)

```
┌─────────────────────────────────────────────────────────────────┐
│  Deployment directory                                           │
│  (e.g., vendor's customer site, or vendor QA test rig)          │
│                                                                 │
│  From PUBLIC install tree:                                      │
│    onnxruntime_morphizen_ep.dll       (loaded by ORT)           │
│    hip-compiler.dll                    (loaded by EP)           │
│    hipdnn_backend.dll, hipdnn_graph_runtime.dll                 │
│    MIOpen.dll, hipblaslt.dll, amdhip64.dll                      │
│                                                                 │
│  From PRIVATE build:                                            │
│    vendor_extension.dll                (loaded by hip-compiler  │
│                                         when HIP_EP_PLUGINS is  │
│                                         set)                    │
│    NOTE: vendor_kernels.lib is NOT a runtime file —             │
│      it gets statically linked into model.dll at compile time   │
│                                                                 │
│  Environment:                                                   │
│    HIP_EP_PLUGINS=C:\vendor\dist\vendor_extension.dll           │
│    THEROCK_DIST=C:\vendor\dist\therock                          │
└─────────────────────────────────────────────────────────────────┘
```

### D.4 Contribution decision tree

```
                Vendor team has a change
                          │
                          ▼
                Is it confidential?
                          │
            ┌─────────────┴─────────────┐
            │                           │
           YES                          NO
            │                           │
            ▼                           ▼
   Lands in PRIVATE repo        Upstream PR to PUBLIC repo
   (vendor-extension/)          via CONTRIBUTING.md flow
            │                           │
            ▼                           ▼
   Built into                   Lands in onnx-hipdnn-ep
   vendor_extension.dll;        Vendor team bumps their pin
   never published              ("public install version")
                                in their private repo to pick
                                up the change
```

The "public install version" pin is the only versioned coupling
between the two repos. Vendor team controls when to update it; public
repo's pace of change is decoupled from vendor's release cycle.
