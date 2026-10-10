---
title: Overview
description: This page introduces HIP EP, explains how it executes an ONNX graph on an AMD GPU, and points you to the next steps.
---

HIP EP is an **ONNX Runtime Execution Provider (EP)** for AMD GPUs. It runs as
part of ONNX Runtime rather than as a standalone inference server or API. Once
registered, ONNX Runtime delegates supported parts of the graph to HIP EP for
GPU execution.

This site documents **{{ site.hip_ep_version }}**. Release-specific links on
this page point to the **{{ site.hip_ep_version }}** tag rather than `main`, so
the documentation remains tied to that release.

If your application already uses ONNX Runtime, no code changes are required
beyond registering HIP EP. The same application can then run the supported graph
operations on the GPU.

## How a graph gets executed

HIP EP is an LLM inference backend rather than a standalone operator library.
When ONNX Runtime assigns a subgraph to HIP EP, the subgraph goes through an
MLIR-based compilation pipeline:

1. **ONNX → HIP dialect.** ONNX operations are converted to a custom MLIR
   `hip` dialect that explicitly represents GPU memory, kernels, and library calls.
2. **HIP dialect → LLVM IR.** Shape inference, memory planning and buffer
   pooling are performed before the dialect is lowered to LLVM IR.
3. **Execution.** The compiled result schedules the workload, using
   [hipBLASLt](https://github.com/ROCm/hipBLASLt) for matrix operations and
   custom HIP kernels for other operations.

{% comment %}
The old wording here was "produces machine code for your specific GPU", which
is wrong and was corrected on a hip-ep developer's review. The GPU kernels are
compiled by hipcc at build time into `custom_kernels_*.dll` --
docs/design/custom_kernel_design.md states it directly: "the per-model bitcode
never carries kernel code". What the per-model compile produces is the
host-side code that allocates, schedules and launches. Do not put device
codegen back into this paragraph.

The DLL naming is not stable and the text above no longer depends on it.
v0.4.0 shipped one `custom_kernels_gfx<arch>.dll` per architecture; v0.5.1
ships a single `gfx11-generic` build covering the whole family, plus one for an
architecture this site does not document. The claim that survives both is the
one that matters: device code is prebuilt and shipped, not generated on the
target machine.
{% endcomment %}
The compiler output drives GPU execution but does not contain the GPU kernels
themselves. The kernels are built ahead of time and shipped in the package as
`custom_kernels_*`. As a result, a single package can support multiple
architectures without requiring a GPU compiler on the target machine.

By default, the per-model artifact is **OS-portable LLVM bitcode**. HIP EP
JIT-loads this bitcode in-process together with the embedded runtime bitcode.
Native `.dll`/`.so` model artifacts are also available as an opt-in mode.

The first inference is slower because the model is compiled at runtime. This
compilation happens once per process. Benchmarks should therefore include a
warm-up phase; otherwise, the results include compilation overhead.

## Supported hardware

The `{{ site.hip_ep_version }}` Windows package covers the following RDNA 3.5
GPUs. The performance data on this site was collected on Strix Halo
(`gfx1151`); the other entries indicate package coverage and should not be
interpreted as equivalent benchmark results.

| GPU | Architecture |
|---|---|
| Ryzen AI Max ("Strix Halo") | `gfx1151` |
| Ryzen AI ("Strix Point") | `gfx1150` |
| Ryzen AI ("Krackan Point") | `gfx1152` |

{% comment %}
This used to say the package "includes GPU kernels for all three RDNA 3.5
parts", which described v0.4.0's three per-architecture DLLs. v0.5.1 covers the
same three GPUs with one `gfx11-generic` code object instead. The reader-facing
consequence is identical -- one download, no architecture to choose -- so the
sentence is written to that consequence and not to the file layout, which has
now changed once and may change again.
{% endcomment %}
The Windows release provides a single package containing code objects for all
three parts, with no architecture-specific variants. For other AMD GPU
families, build HIP EP from source and specify the target with `--hip_arch`.

## Version pinning

HIP EP uses specific versions of its upstream dependencies. Using a different
ONNX Runtime version is not supported because the EP is loaded as a plugin
against a specific ABI.

| Component | Version |
|---|---|
| HIP EP | `{{ site.hip_ep_version }}` |
| ONNX Runtime | `1.27.0` |
| ONNX Runtime GenAI (OGA) | `0.14.0` + AMDGPU integration [PR 2194](https://github.com/microsoft/onnxruntime-genai/pull/2194) |

The full dependency set — including LLVM/MLIR/LLD, protobuf, flatbuffers, ONNX Runtime,
TheRock ROCm — is pinned in
[`cmake/deps.txt`]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/cmake/deps.txt).
{% comment %}
`OGA_VERSION` and `OGA_PR_PATCHES` used to live in windows-build.yml and were
linked there. At v0.5.1 that file is only an orchestrator -- it calls
windows-deps.yml, windows-build-real.yml, windows-build-mock.yml and
windows-gpu-test.yml -- and the two settings moved to windows-deps.yml
(lines 33-34 at this tag), while the actual package staging moved to
windows-build-real.yml. Both values are unchanged: 0.14.0 and PR 2194.
Re-check which file holds them on the next bump; this has already moved once.
{% endcomment %}
OGA packaging is defined in
[`windows-deps.yml`]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/.github/workflows/windows-deps.yml),
including the `OGA_VERSION` and `OGA_PR_PATCHES` settings.
These files are the source of truth if the table above differs from the repository.

## Official sources

| Source | What to use it for |
|---|---|
| [{{ site.hip_ep_version }} release]({{ site.repo_url }}/releases/tag/{{ site.hip_ep_version }}) | Download the release and Python packages documented on this site |
| [`cmake/deps.txt`]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/cmake/deps.txt) | Check pinned dependency versions |
| [`windows-build-real.yml`]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/.github/workflows/windows-build-real.yml) | See which binaries, libraries, and wheels are staged into the Windows packages |
| [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) | Learn about the tokenizer, KV-cache and decode loop used for LLM inference |
| [Supported operations]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/docs/supported-operations.md) | See which ONNX operations are supported by the compile pipeline |

## Where to go next

<div class="card-grid" markdown="0">
  <a class="card card--link" href="{{ '/docs/get-started/binary-package/' | relative_url }}">
    <p class="card__title">Run a model</p>
    <p class="card__body">Extract the release archive, add
      <code>bin</code> to <code>PATH</code>, and run an inference on the GPU. No
      compiler required.</p>
  </a>
  <a class="card card--link" href="{{ '/docs/get-started/python-package/' | relative_url }}">
    <p class="card__title">Use the Python package</p>
    <p class="card__body">Use the Python package to serve an LLM, verify that it is running on the GPU, run benchmarks, and try your own model.</p>
  </a>
</div>

The
[model matrix]({{ '/docs/models/' | relative_url }}) lists the models included in the {{ site.hip_ep_version }} snapshot, while the
[benchmarks]({{ '/docs/benchmarks/' | relative_url }}) page explains how to interpret the results. For implementation details, see
the [design documentation]({{ site.repo_url }}/tree/{{ site.hip_ep_version }}/docs/design)
which covers pass ordering, the compiler/runtime ABI and memory planning.
