<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quick Start Guide (Linux)

Pick one entry path to populate `bin/` and `lib/`, point `$ROOT` at it, then
follow the shared **Testing & Benchmarking** section — every tool runs the same
way regardless of which entry path you chose.

- **Build from source, native** (developers). `python build.py` builds
  directly on the host (no Docker) into `<workspace>/install/`.
- **Build from source, Docker** (developers who prefer isolation).
  `./docker/run.sh build` runs the same `build.py` inside a container.
- **Use the prebuilt package** (testers). `gh run download
  linux-gpu-test-package` lands the binaries under
  `<workspace>/prebuilt/<run-id>/`; no rebuild needed.

All C++ dependencies (LLVM/MLIR/LLD, protobuf, flatbuffers, ONNX Runtime) and
the TheRock ROCm SDK are resolved automatically by `cmake/deps.cmake`: LLVM is
built from source, so the host needs no system LLVM. The cold from-source LLVM
build is the long pole (multi-hour); it lands under `<workspace>/build/` and is
reused across rebuilds.

See [quick_start.md](quick_start.md) for the Windows flow.

## Host prerequisites

| Tool | Purpose |
|------|---------|
| **cmake ≥ 3.29, ninja, git, a C++ compiler, python3** | Native build path |
| **Docker** 26+ | Docker build path (optional) |
| **AMD gfx1151 GPU** + `/dev/kfd` + `/dev/dri/renderD*` | HIP runtime |
| **`gh` CLI** (`gh auth login`) | Download the prebuilt package (that path only) |

For the Docker path, the current user must be in the host **`docker`** group. The user does
**not** need to be in `render` / `video` — GPU passthrough is handled by
docker's `--device=/dev/kfd`, and the container entrypoint
([`docker/entrypoint.sh`](../docker/entrypoint.sh)) detects the host GID
on `/dev/kfd` and adds the in-container user to that group automatically.

All commands below assume you are in `<workspace>/hip-ep/` (the
project root). The build writes to sibling directories under `<workspace>/`
(`build/`, `install/`); the Docker path auto-mounts the whole workspace. The
TheRock ROCm SDK is auto-downloaded into the build tree at
`build/hip-ep/_therock/`.

## Build from source (developers)

```bash
git clone https://github.com/ROCm/hip-ep.git
cd hip-ep
```

### Native (no Docker)

`build.py` is a plain cross-platform driver: it
checks the toolchain, auto-detects the GPU arch (from `/sys/class/kfd`), then
runs the cmake configure/build/install into `<workspace>/install/`. All
dependencies (incl. from-source LLVM) are resolved by `cmake/deps.cmake`.

```bash
python3 build.py
#   --hip_arch gfx1151        override target GPU (else auto-detected; falls
#                             back to --mock if no AMD GPU is present)
#   --mock                    mock runtime (no GPU/HIP/TheRock)
#   --config RelWithDebInfo   build type (default Release)
#   --skip_tests              skip the LIT tests (run by default after install)
#   --clean                   remove build/ and install/
```

### Docker (optional, same driver inside a container)

```bash
./docker/run.sh image        # build the image once (first time only)
./docker/run.sh build        # runs build.py inside the container
#   HIP_ARCHITECTURES=gfx942 ./docker/run.sh build   # override target GPU
```

The result tree under `<workspace>/install/`:

- `bin/hip-onnx-runner`, `bin/hip-compiler`, `bin/hip-mlir-opt`, `bin/hip-test`
- `lib/libhip-compiler.so`, `lib/libhipgpu.so`

A locally-built `install/` is **not** fully self-contained: `libonnxruntime.so`
lives in the ONNX Runtime prefix and the ROCm libs in TheRock, so run with
`LD_LIBRARY_PATH=<workspace>/install/lib:<ort-prefix>/lib:$THEROCK_DIST/lib`.
(The prebuilt package below already bundles these into `install/lib`, so
`install/lib:$THEROCK_DIST/lib` is enough.)

> **ONNX Runtime** is downloaded automatically (Microsoft's official prebuilt
> release) — you don't fetch it yourself. `model_benchmark` (OGA) is not part
> of the local build; use the prebuilt package below if you need it.

Skip to the [Open a container shell](#open-a-container-shell-and-set-root)
section once `install/bin/` is populated.

## Use the prebuilt package (testers, no compile)

A ready-to-run package is published for each green build. It contains every
`.so` / binary the host needs (`libhipgpu`,
`hip-onnx-runner`, `onnxruntime_perf_test`, `model_benchmark`,
`libonnxruntime`, `libonnxruntime-genai`) plus a `clang`/`lld` toolchain in
`bin/` (next to the other tools). TheRock is **not** included — install ROCm on
the host. Download it with `gh` (no compile needed):

```bash
git clone https://github.com/ROCm/hip-ep.git
cd hip-ep
./docker/run.sh image    # skip if the image is already built

# Pick the latest green run from
# https://github.com/ROCm/hip-ep/actions/workflows/linux-build.yml
RUN_ID=<paste-id>
mkdir -p ../prebuilt/$RUN_ID
( cd ../prebuilt/$RUN_ID && \
    gh run download $RUN_ID --repo ROCm/hip-ep \
        --name linux-gpu-test-package --dir . && \
    chmod +x bin/* )
```

After extraction, `<workspace>/prebuilt/$RUN_ID/` matches the
build-from-source `install/` layout (`bin/` — which also carries the bundled
clang/lld — `lib/`, and `etc/`).

## Open a container shell and set `$ROOT`

If you built/ran natively (no Docker), skip the container and just set
`ROOT`/`THEROCK_DIST`/`LD_LIBRARY_PATH` in your own shell using the explicit
paths below (replace `$WORKSPACE` with your `<workspace>` dir). Otherwise,
`./docker/run.sh shell` opens a long-lived container with GPU passthrough and
UID alignment:

```bash
./docker/run.sh shell

# Inside the container — pick ONE depending on which entry path you used:
export ROOT="$WORKSPACE/install"            # built from source
# export ROOT="$WORKSPACE/prebuilt/$RUN_ID"  # downloaded prebuilt package

# THEROCK_DIST points the in-process tooling at the TheRock SDK that
# `cmake/deps.cmake` auto-downloaded into the build tree at
# $WORKSPACE/build/hip-ep/_therock during the build. It is
# required at runtime even though the prebuilt package bundles transitive .so
# files into $ROOT/lib, because:
#   1. CompilerDriver::discoverLibraries() reads $THEROCK_DIST to
#      construct `-L<therock>/lib` paths it passes to ld.lld when
#      compiling per-model DLLs (otherwise the link errors with
#      "undefined symbol: hipGetDeviceCount" etc.).
#   2. libamd_comgr_loader.so.1 (shipped in $ROOT/lib) is a stub that
#      dlopen()s the real libamd_comgr.so.3 at runtime; the real lib
#      lives only in $THEROCK_DIST/lib and is not in ldd's view of
#      transitive deps, so it has to be on LD_LIBRARY_PATH explicitly.
#
# LIBRARY_PATH is GCC/clang's build-time linker search-path env;
# `hip-compiler` keeps it on the search path for any future offline-link
# step that needs it.
#
# Per-arch GPU kernels ship as `libcustom_kernels_gfx<arch>.so` next to
# the EP `.so` under `$ROOT/lib`. The EP `.so` is built with
# `RPATH=$ORIGIN`, so LlvmIrJit's `dlopen` of the matching variant
# resolves through the loader without needing it on `LD_LIBRARY_PATH`.
export THEROCK_DIST="$WORKSPACE/therock-dist"
export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"
export LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"
# clang/lld for the per-model DLL link (the prebuilt package ships it in
# $ROOT/bin, alongside the other tools).
export PATH="$ROOT/bin:$PATH"

# Sanity check
ldd "$ROOT/lib/libhipgpu.so" | grep "not found"   # expect empty
"$ROOT/bin/hip-onnx-runner" --help | head -5
```

`$WORKSPACE` is set automatically by
[`docker/entrypoint.sh`](../docker/entrypoint.sh) to the bind-mounted
host workspace path. Every command below uses `$ROOT/bin/<tool>`
exclusively, so the snippets are identical regardless of entry path.

## Testing & Benchmarking

### Model Inference with hip-onnx-runner

`hip-onnx-runner` runs a single ONNX model through hipgpu EP and reports
timing. It is built by `build.py` and also ships in the prebuilt
package.

> `hip-onnx-runner` runs with random inputs by default. LLM models need
> in-range `input_ids` (< vocab size); use
> `tools/hip-onnx-runner/gen_hip_onnx_runner_inputs.py` to produce a
> `gen_inputs/` directory and pass it via `-i`:
>
> ```bash
> python tools/hip-onnx-runner/gen_hip_onnx_runner_inputs.py \
>   -o gen_inputs /path/to/MODEL.onnx
> ```

```bash
# Run with hipgpu EP (default), on your model directly (dynamic shape)
$ROOT/bin/hip-onnx-runner -m /path/to/model.onnx -i gen_inputs

# Resolve symbolic input dims at runtime (the EP still compiles the dynamic graph)
$ROOT/bin/hip-onnx-runner -m /path/to/model.onnx -i gen_inputs -f sequence_length:128

# Run with CPU only (no EP)
$ROOT/bin/hip-onnx-runner -m /path/to/model.onnx -n -i gen_inputs

# Dump outputs for comparison
$ROOT/bin/hip-onnx-runner -m /path/to/model.onnx -i gen_inputs -d 2

# L2-norm compare EP vs CPU outputs
$ROOT/bin/hip-onnx-runner -L ep_o_dump,cpu_o_dump
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-m` | Path to `.onnx` model |
| `-n` | CPU only, skip EP registration |
| `-d 0\|1\|2\|3` | Dump level: 0=off, 1=inputs, 2=outputs, 3=both |
| `-i <dir>` | Load inputs from directory instead of random |
| `-f <name>:<val>` | Resolve a symbolic input dim at runtime (repeatable/comma-separated); EP still compiles the dynamic graph, unmatched symbolic dims default to 1 |
| `-L dir1,dir2` | L2-norm comparison of two output directories |

### Latency Benchmarking with onnxruntime_perf_test

`onnxruntime_perf_test` benchmarks inference latency. It ships in the prebuilt
package; a local build may not include it. The examples below compare hipgpu
EP against the CPU EP baseline (DML is Windows-only).

```bash
# CPU baseline (no EP; useful to size the EP speedup)
$ROOT/bin/onnxruntime_perf_test -e cpu -t 30 -c 1 -s /path/to/MODEL.onnx

# hipgpu EP
$ROOT/bin/onnxruntime_perf_test \
  --plugin_ep_libs "hipgpu|$ROOT/lib/libhipgpu.so" \
  --plugin_eps     "hipgpu" \
  -C "session.disable_cpu_ep_fallback|1" \
  -t 60 -c 1 -s -I \
  /path/to/MODEL.onnx
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-t 60` | Run for 60 seconds |
| `-c 1` | 1 concurrent thread |
| `-s` | Show per-iteration latency statistics |
| `-I` | Use sequential inputs (do not randomize) |

> The first iteration triggers hipgpu's HIP kernel JIT compile
> (multi-minute); bump `-t` so steady-state samples dominate the average.

### OGA End-to-End Benchmarking with model_benchmark

`model_benchmark` benchmarks the full generative pipeline (prefill +
decode token generation). It is not part of the local build, so use the
prebuilt package to get it.

The EP is selected by the model's `genai_config.json` `provider_options` and
auto-discovered next to the OGA runtime lib -- do NOT pass `--ep_library`
(upstream `model_benchmark` rejects it). With the upstream OGA (0.15.1 + PR2376)
the EP is the AMD GPU umbrella (`provider_options [{ "AMDGPU": {"profile": "hip"} }]`);
the prebuilt package bundles the umbrella libs.

```bash
# Auto-generated prompt (128 tokens, generate 32)
$ROOT/bin/model_benchmark \
  -i /path/to/oga-model-dir \
  -l 128 -g 32 -ml -1 \
  -r 5 -w 1

# Prompt from file
$ROOT/bin/model_benchmark \
  -i /path/to/oga-model-dir \
  --prompt_file /path/to/prompt.txt -g 128 -ml -1 \
  -r 5 -w 1
```

> **Note on `-ml -1`**: required when the model uses a fixed-shape pipeline
> (`prefill_*.onnx` + `decode_*.onnx` with a baked-in `max_length` such as 256).
> Without it, model_benchmark overrides `genai_config.json`'s `search.max_length`
> with `prompt_length + generation_length`, causing
> `Got invalid dimensions for input: attention_mask  Got: <l+g>  Expected: <max_length>`
> at decode time. Pass `-ml -1` to keep the config's value.

**Key flags:**

| Flag | Description |
|------|-------------|
| `-i <path>` | Path to OGA model directory (with `genai_config.json`) |
| `-l <n>` | Length of auto-generated prompt (exclusive with `--prompt_file`) |
| `--prompt_file <path>` | Load prompt text from a file (exclusive with `-l`) |
| `-g <n>` | Max number of tokens to generate |
| `-r <n>` | Number of benchmark repetitions |
| `-w <n>` | Number of warmup runs |

## Runtime requirements (deploy host)

The `bin/hip-compiler` invokes `clang++` at runtime to link the per-model DLL
(driver-mediated link via `ld.lld`). The prebuilt package **includes a
`clang`/`lld` toolchain** in `$ROOT/bin/` (next to the other tools), so the host
needs no separate clang install — just put `$ROOT/bin` on `$PATH`:

```bash
export PATH="$ROOT/bin:$PATH"
```

No other LLVM/GCC tools are needed at runtime — the clang driver handles crt
selection, sysroot, multiarch -L paths, and the libstdc++ link line internally
(the host's system libstdc++/gcc supplies crt). A local build instead uses the
`clang++` it was configured against.

## Troubleshooting

**`Got invalid dimensions for input: attention_mask  Got: X  Expected: Y`**

Either:

1. `model_benchmark -l <prompt_length>` doesn't match the OGA model
   directory's `genai_config.json` → `model.decoder.fixed_prompt_length`.
   Pass the same value to `-l`, or pick a different `genai_config_*.json`
   that matches your intended length.
2. `-ml` (max_length) flag is missing — see the note in
   [6.3 OGA End-to-End Benchmarking](#oga-end-to-end-benchmarking-with-model_benchmark).
   Without `-ml -1`, model_benchmark overrides the config's `search.max_length`
   with `prompt_length + generation_length`, breaking the static mask shape
   that decode_*.onnx expects.

**`Failed to get HIP device count or no devices available` on bare-metal host**

Most common cause: the host user is not in the `render` group, so opening
`/dev/kfd` fails silently and `hipGetDeviceCount` returns 0. Check:

```bash
ls -la /dev/kfd          # expect group=render
groups                   # expect 'render' in the list
```

Fix: `sudo usermod -aG render $USER && newgrp render` (or relogin).
Inside `./docker/run.sh shell` this is handled automatically — the
container entrypoint reads the host GID off `/dev/kfd` and adds the
in-container user to it, so you don't need host `render` membership.

**`EP library not found: libhipgpu.so` from `hip-onnx-runner`**

The runner's search order is `$MORPHIZEN_EP_LIB` (full path) → cwd →
`<exe-dir>/libhipgpu.so` → `<exe-dir>/../lib/libhipgpu.so`.
If you're running out of `install/bin/`, no env vars are needed — the
sibling `install/lib/` is auto-discovered. If you've copied the binary
elsewhere, set `MORPHIZEN_EP_LIB=/full/path/to/libhipgpu.so`.

**`clang++ not found on PATH and HIPDNN_CLANG_PATH is unset or stale` from `hip-compiler`**

The per-model DLL link calls `clang++ -shared` as a subprocess; the
driver handles crt / sysroot / multiarch / libstdc++ for us. The path
is baked at configure time (`HIPDNN_CLANG_PATH`) with `findProgramByName`
as the runtime fallback, so this error only fires when **both** the
baked path is invalid (e.g. the package deployed to a host where the
build-time clang path no longer exists) **and** `clang++` is absent from
PATH.

Fix on the deploy host: put the bundled clang on PATH —
`export PATH="$ROOT/bin:$PATH"` (the prebuilt package ships
`clang++`/`ld.lld` in `$ROOT/bin`, alongside the other tools).
