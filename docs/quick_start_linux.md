<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quick Start Guide (Linux)

Linux is **Docker-only**. Pick one entry path to populate `bin/` and `lib/`,
point `$ROOT` at it, then follow the shared **Testing & Benchmarking**
section — every tool runs the same way regardless of which entry path
you chose.

- **Build from source** (developers). `./docker/run.sh build` produces
  a self-contained `<workspace>/install/`.
- **Use the prebuilt CI artifact** (testers). `gh run download
  linux-gpu-test-package` lands the binaries under
  `<workspace>/prebuilt/<run-id>/`; no rebuild needed.

See [quick_start.md](quick_start.md) for the Windows flow.

## Host prerequisites

| Tool | Purpose |
|------|---------|
| **Docker** 26+ | Build + run environment |
| **AMD gfx1151 GPU** + `/dev/kfd` + `/dev/dri/renderD*` | HIP runtime |
| **`gh` CLI** (`gh auth login`) | Download CI artifacts (artifact path only) |

The current user must be in the host **`docker`** group. The user does
**not** need to be in `render` / `video` — GPU passthrough is handled by
docker's `--device=/dev/kfd`, and the container entrypoint
([`docker/entrypoint.sh`](../docker/entrypoint.sh)) detects the host GID
on `/dev/kfd` and adds the in-container user to that group automatically.

All commands below assume you are in `<workspace>/onnx-hipdnn-ep/` (the
project root). Sibling directories (`onnxruntime/`, `onnxruntime-genai/`,
`prebuilt-local/`, `therock-dist/`, `build/`, `install/`, `prebuilt/`)
are auto-mounted into the container by `docker/run.sh`.

## Build from source (developers)

```bash
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
cd onnx-hipdnn-ep
git submodule update --init --recursive

# Step 1 — Build the image (first time only;
# hipdnn-ep-build:llvm22-noble, ~3 GB)
./docker/run.sh image

# Step 2 — Build everything (TheRock + ORT + protobuf + flatbuffers + this
# project) inside the container. Idempotent: skipped steps print [skip].
./docker/run.sh build
#   BUILD_OGA=1            also build model_benchmark + libonnxruntime-genai.so
#   HIP_ARCHITECTURES=...  override target GPU (default gfx1151)
#   SKIP_LIT=1             skip in-build LIT tests
#   FORCE_RECONFIGURE=1    re-run cmake after editing options
```

The result tree under `<workspace>/install/` is self-contained:

- `bin/hip-onnx-runner`, `bin/hip-compiler`, `bin/hip-mlir-opt`,
  `bin/hip-test-dll` (and `bin/model_benchmark` with `BUILD_OGA=1`)
- `lib/libhip-compiler.so`, `lib/libonnxruntime_morphizen_ep.so`,
  `lib/libonnxruntime.so*`, `lib/libamdhip64.so*` + TheRock transitive
  (so `LD_LIBRARY_PATH=<workspace>/install/lib` is enough — see
  [docker/build.sh](../docker/build.sh) A.7b stage step)

> **ORT / OGA source layout**: if you already have a checkout of
> [`onnxruntime`](https://github.com/Microsoft/onnxruntime) or
> [`onnxruntime-genai`](https://github.com/AMDmoore/onnxruntime-genai) under
> `<workspace>/onnxruntime/` or `<workspace>/onnxruntime-genai/`, the
> build skips the clone step and uses your checkout as-is (no `fetch`,
> no `checkout`, no `submodule update`).

Skip to the [Open a container shell](#open-a-container-shell-and-set-root)
section once `install/bin/` is populated.

## Use the prebuilt CI artifact (testers, no compile)

The CI workflow
[`.github/workflows/linux-build.yml`](../.github/workflows/linux-build.yml)
builds and bundles every `.so` / binary the host needs (libamdhip64,
libLLVM 22, libonnxruntime, libonnxruntime_morphizen_ep, hip-onnx-runner,
onnxruntime_perf_test, model_benchmark, libonnxruntime-genai). The
artifact path reuses the same Docker image as the build-from-source
flow — no `./docker/run.sh build` needed:

```bash
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
cd onnx-hipdnn-ep
./docker/run.sh image    # skip if the image is already built

# Pick the latest green run from
# https://github.com/ROCm/onnx-hipdnn-ep/actions/workflows/linux-build.yml
RUN_ID=<paste-id>
mkdir -p ../prebuilt/$RUN_ID
( cd ../prebuilt/$RUN_ID && \
    gh run download $RUN_ID --repo ROCm/onnx-hipdnn-ep \
        --name linux-gpu-test-package --dir . && \
    chmod +x bin/* )
```

After extraction, `<workspace>/prebuilt/$RUN_ID/` matches the
build-from-source `install/` layout (`bin/`, `lib/`, `etc/`).

## If you want to test for another Architecture
if you want to build for gfx1201 ( rx 9070xt), you need modify below place before you build
1. in  .github/workflows/linux-build.yml, change value of THEROCK_VERSION & HIP_ARCHITECTURES, as comments specified.
2. in docker/build.sh, change value of THEROCK_VERSION & HIP_ARCHITECTURES
3. in docker/run.sh, change value of HIP_ARCHITECTURES

## Open a container shell and set `$ROOT`

Both paths converge here. `./docker/run.sh shell` opens a long-lived
container with GPU passthrough and UID alignment:

```bash
./docker/run.sh shell

# Inside the container — pick ONE depending on which entry path you used:
export ROOT="$WORKSPACE/install"            # built from source
# export ROOT="$WORKSPACE/prebuilt/$RUN_ID"  # downloaded CI artifact

# THEROCK_DIST points the in-process tooling at the TheRock SDK that
# `docker/build.sh` A.4 already downloaded into the workspace. It is
# required at runtime even though the artifact bundles transitive .so
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
# clang++ -shared (driving the per-model DLL link inside hip-compiler)
# prepends every entry as `-L<dir>`, so the in-tree
# `libhip_custom_kernels.a` resolves via the level-3 name-only
# `-lhip_custom_kernels` fallback.
export THEROCK_DIST="$WORKSPACE/therock-dist"
export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"
export LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib"

# Sanity check
ldd "$ROOT/lib/libonnxruntime_morphizen_ep.so" | grep "not found"   # expect empty
"$ROOT/bin/hip-onnx-runner" --help | head -5
```

`$WORKSPACE` is set automatically by
[`docker/entrypoint.sh`](../docker/entrypoint.sh) to the bind-mounted
host workspace path. Every command below uses `$ROOT/bin/<tool>`
exclusively, so the snippets are identical regardless of entry path.

## Model Preparation

Same as the Windows guide — see
[quick_start.md "Model Preparation"](quick_start.md#model-preparation).
Run from the project root inside the container.

## Testing & Benchmarking

### Model Inference with hip-onnx-runner

`hip-onnx-runner` runs a single ONNX model through MorphiZen EP and reports
timing. It is built automatically when `BUILD_HIP_TOOLS=ON` (default) and
also ships in the CI artifact.

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
# Run with MorphiZen EP (default), using a fixed-shape model from Model Preparation
$ROOT/bin/hip-onnx-runner -m /path/to/output/prefill_p512m16384.onnx -i gen_inputs

# Run with CPU only (no EP)
$ROOT/bin/hip-onnx-runner -m /path/to/output/prefill_p512m16384.onnx -n -i gen_inputs

# Dump outputs for comparison
$ROOT/bin/hip-onnx-runner -m /path/to/output/prefill_p512m16384.onnx -i gen_inputs -d 2

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
| `-L dir1,dir2` | L2-norm comparison of two output directories |

### Latency Benchmarking with onnxruntime_perf_test

`onnxruntime_perf_test` benchmarks inference latency. Both entry paths
ship it next to `hip-onnx-runner`. The examples below compare
MorphiZen EP against the CPU EP baseline (DML is Windows-only).

```bash
# CPU baseline (no EP; useful to size the EP speedup)
$ROOT/bin/onnxruntime_perf_test -e cpu -t 30 -c 1 -s /path/to/MODEL.onnx

# MorphiZen EP
$ROOT/bin/onnxruntime_perf_test \
  --plugin_ep_libs "MorphiZenEP|$ROOT/lib/libonnxruntime_morphizen_ep.so" \
  --plugin_eps     "MorphiZenEP" \
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

> The first iteration triggers MorphiZen's HIP kernel JIT compile
> (multi-minute); bump `-t` so steady-state samples dominate the average.

### OGA End-to-End Benchmarking with model_benchmark

`model_benchmark` benchmarks the full generative pipeline (prefill +
decode token generation). The build-from-source path requires
`BUILD_OGA=1` to produce it; the CI artifact bundles it automatically.

```bash
# Auto-generated prompt (128 tokens, generate 32)
$ROOT/bin/model_benchmark \
  -i /path/to/oga-model-dir \
  --ep_library MorphiZenEP $ROOT/lib/libonnxruntime_morphizen_ep.so \
  -l 128 -g 32 -ml -1 \
  -r 5 -w 1

# Prompt from file
$ROOT/bin/model_benchmark \
  -i /path/to/oga-model-dir \
  --ep_library MorphiZenEP $ROOT/lib/libonnxruntime_morphizen_ep.so \
  --prompt_file /path/to/prompt.txt -g 128 -ml -1 \
  -r 5 -w 1
```

> **Note on `-ml -1`**: required when the model uses a fixed-shape pipeline
> (`prefill_*.onnx` + `decode_*.onnx` with a baked-in `max_length` such as 256).
> Without it, model_benchmark overrides `genai_config.json`'s `search.max_length`
> with `prompt_length + generation_length`, causing
> `Got invalid dimensions for input: attention_mask  Got: <l+g>  Expected: <max_length>`
> at decode time. Pass `-ml -1` to keep the config's value (mirrors what the
> Windows CI does in [`.github/workflows/windows-build.yml`](../.github/workflows/windows-build.yml)).

**Key flags:**

| Flag | Description |
|------|-------------|
| `-i <path>` | Path to OGA model directory (with `genai_config.json`) |
| `--ep_library <name> <path>` | Register a custom EP library |
| `-l <n>` | Length of auto-generated prompt (exclusive with `--prompt_file`) |
| `--prompt_file <path>` | Load prompt text from a file (exclusive with `-l`) |
| `-g <n>` | Max number of tokens to generate |
| `-r <n>` | Number of benchmark repetitions |
| `-w <n>` | Number of warmup runs |

## Advanced: bare-metal source build (no Docker)

If you need to build outside Docker (running CI locally on a custom
kernel, etc.), the canonical sources of truth are:

- [`.github/workflows/linux-build.yml`](../.github/workflows/linux-build.yml)
  — the GitHub Actions pipeline (apt deps + cmake recipe per step)
- [`docker/build.sh`](../docker/build.sh) — the same A.4–A.9 step
  sequence the Docker path runs
- [`docker/Dockerfile`](../docker/Dockerfile) — exact apt package list
  (`llvm-22-dev`, `libpolly-22-dev`, `libmlir-22-dev`, `lld-22`,
  `clang-22`, `python3-dev`, ...)

## Runtime requirements (deploy host)

The artifact's `bin/hip-compiler` invokes `clang++` at runtime to link
the per-model DLL (driver-mediated link via `ld.lld`). The path is
baked at configure time and PATH-lookup is the fallback, so deploy
hosts need **either** the same `/usr/lib/llvm-22/bin/clang++` baked
path **or** `clang++` somewhere on `$PATH`. One apt invocation covers
both:

```bash
sudo apt install clang-22
```

No other LLVM/GCC tools are needed at runtime — the clang driver
handles crt selection, sysroot, multiarch -L paths, and the
libstdc++ link line internally. (The container image installs
`clang-22` in [`docker/Dockerfile`](../docker/Dockerfile) Layer 2 as
part of the `apt.llvm.org` block, so users on the Docker path don't
need any host-side install.)

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

**`EP library not found: libonnxruntime_morphizen_ep.so` from `hip-onnx-runner`**

The runner's search order is `$MORPHIZEN_EP_LIB` (full path) → cwd →
`<exe-dir>/libonnxruntime_morphizen_ep.so` → `<exe-dir>/../lib/libonnxruntime_morphizen_ep.so`.
If you're running out of `install/bin/`, no env vars are needed — the
sibling `install/lib/` is auto-discovered. If you've copied the binary
elsewhere, set `MORPHIZEN_EP_LIB=/full/path/to/libonnxruntime_morphizen_ep.so`.

**`clang++ not found on PATH and HIPDNN_CLANG_PATH is unset or stale` from `hip-compiler`**

The per-model DLL link calls `clang++ -shared` as a subprocess; the
driver handles crt / sysroot / multiarch / libstdc++ for us. The path
is baked at configure time (`HIPDNN_CLANG_PATH`) with `findProgramByName`
as the runtime fallback, so this error only fires when **both** the
baked path is invalid (e.g. CI artifact deployed to a host that doesn't
have llvm-22 in the same location) **and** `clang++` is absent from
PATH.

Fix on the deploy host: `sudo apt install clang-22` (or the matching
LLVM version package). The container image installs this in Layer 2 of
[`docker/Dockerfile`](../docker/Dockerfile).
