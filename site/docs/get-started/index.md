---
title: Get Started
description: "There are two ways to run HIP EP on an AMD GPU on Windows: the binary package and the Python package."
---

Both paths assume a Windows system with a supported AMD GPU and a current AMD
graphics driver. Each path takes you through installation and ends with a model
running on the GPU. The main difference is how HIP EP is installed and what
tools are available afterward. Choose the path that matches how you plan to use
HIP EP; the two paths are independent.

{% comment %}
There used to be a third path here, and a /docs/get-started/source-build/ page
behind it. Both were removed: a source build tracks the repository's main
branch while this site is pinned to a release, so the page was guaranteed to
drift, and it was never run end to end on a machine. Build instructions belong
next to the code that changes them. Every source-build reference on this site
now resolves to the one link below -- the repository's Windows quick start --
and nowhere else. Do not restore a local page for it.
{% endcomment %}
Building HIP EP from source is not covered here. It is documented in the
repository, in the
[Windows quick start]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/docs/quick_start.md).

<div class="card-grid card-grid--fill" markdown="0">
  <div class="card">
    <p class="card__title">1 · Binary package</p>
    <p class="card__body">Extract the release zip, add <code>bin</code> to
      <code>PATH</code>, and run provided tools. The archive contains the required runtime libraries and binaries, so no separate Python or Visual Studio installation is required.</p>
  </div>
  <div class="card">
    <p class="card__title">2 · Python package</p>
    <p class="card__body">Install the required wheels into a Python 3.14 environment. The installation order matters, and pip also installs the required ROCm runtime packages. The Python package provides benchmarks and example scripts that you can inspect and modify rather than a set of pre-built executables.</p>
  </div>
</div>

## Which one

| If you want to | Read |
|---|---|
| Run the shipped models and benchmark them with minimal setup | [Run models with the binary package]({{ '/docs/get-started/binary-package/' | relative_url }}) |
| Use ONNX Runtime and OGA from your own Python code | [Run models from Python]({{ '/docs/get-started/python-package/' | relative_url }}) |

## Check that your GPU is covered

The Windows package includes kernels for three RDNA 3.5 integrated GPUs, so
there is no architecture-specific download to choose. Your GPU must be one of
the supported targets, however.
The benchmark data on this site was collected on Strix Halo
(`gfx1151`); Strix Point and Krackan Point are also included in the package, but
you should complete the GPU verification step on the corresponding installation
page before relying on the package for your system.

```powershell
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, AdapterCompatibility
```

| Adapter name contains | Architecture | Product family |
|---|---|---|
| Radeon 8060S / 8050S | `gfx1151` | Ryzen AI Max 300 ("Strix Halo") |
| Radeon 890M / 880M | `gfx1150` | Ryzen AI 300 ("Strix Point") |
| Radeon 860M / 840M | `gfx1152` | Ryzen AI 300 ("Krackan Point") |

If your adapter is not listed — for example, a discrete Radeon GPU, an older
integrated GPU, or an Instinct GPU — do not assume that the Windows release
package supports it. The three architectures listed above are the targets
covered by the validation on this site. Other architectures have to be built
from source with `--hip_arch`, which the repository's
[Windows quick start]({{ site.repo_url }}/blob/{{ site.hip_ep_version }}/docs/quick_start.md)
covers.

Also check that your AMD graphics driver is up to date. HIP EP uses the bundled
HIP runtime to communicate with the kernel-mode driver, and an outdated driver
can cause launch failures that are otherwise difficult to diagnose. If necessary,
install the current [AMD Adrenalin driver](https://www.amd.com/en/support).

## Before you start

**The Windows binary package does not require a separate ROCm installation.**
The archive includes the HIP runtime, the code-object manager, hipBLASLt,
rocBLAS and MIOpen required by the EP. A current AMD graphics driver and the
release download are sufficient.

{% comment %}
The download and extracted sizes used to be here and on binary-package.md's
requirements table. Dropped on a hip-ep developer's review: a few hundred MB is
not a number anyone plans around, and the page never states the far larger
figure that does matter -- the model. The ROCm download below was kept at that
point, on the grounds that it is not a disk budget but a surprise -- pip pulls
it from a third-party index mid-install and people read the pause as a hang --
and its size was dropped later, by the page owner, in the rewrite of this
section. The surprise is still described; only the figure is gone.
{% endcomment %}
The Python package is different. Its EP wheel declares the ROCm runtime as a
dependency instead of bundling it, so pip downloads the required ROCm packages
from an AMD package index during installation. See
[that page]({{ '/docs/get-started/python-package/' | relative_url }}) for the
index URL and installation details.

## What you get

The three routes do not carry the same tools, and the differences are worth
knowing before you pick one:

| Tool | What it is for | Where it comes from |
|---|---|---|
| `hip-onnx-runner` | Run an ONNX model through the EP; dump outputs; compare against CPU results | binary package only |
| `hip-compiler`, `hip-mlir-opt`, `hip-inspect` | Compile and inspect the MLIR pipeline directly | binary package only |
| `onnxruntime_perf_test` | Measure steady-state inference latency for a single graph | binary package only |
| `model_benchmark` | Run an end-to-end generative benchmark, including prefill and decode, through OGA | binary package only |
| `model_mm`, `benchmark_multimodal.py` | Run the corresponding benchmarks for vision-language models | binary package only |
| `run_onnx.py`, `benchmark_e2e.py`, `vlm_benchmark.py` | Run text and vision-language benchmarks as editable Python scripts | Python package only |

The binary package provides the most complete set of pre-built tools. The
Python package provides the benchmark logic as Python scripts rather than as
standalone executables.

The binary package does not install anything into a system directory or register
global components. To remove it, delete the extracted directory.

The Python package is installed through pip, so it can be installed in a virtual
environment and removed with the environment when no longer needed.
