<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Numeric Verification Tests

Per-operator correctness tests for an ORT execution provider. Each
test builds (or loads) a small ONNX model, runs it through a pluggable
backend, and compares the output against an ORT CPU reference --
catching silent kernel bugs (all-zero outputs, broken broadcasts,
wrong tile mapping) that wouldn't surface in LIT tests.

The framework is EP-agnostic: every EP knob is supplied at the
command line, so any ORT execution provider can be plugged in. See
[Example: MorphiZen EP](#example-morphizen-ep) below for the
canonical wiring used by this repo's build.

The suite is independent of the perf tests in [test/python/](../python/):
different goal, different runner, different fixtures.

## Install

The suite has four Python dependencies: `numpy`, `onnx`,
`onnxruntime`, and `pytest`. Two paths:

- **Already on the repo's canonical conda env** (set up via
  `conda env create -f environment.yml && conda activate hipdnn-ep`,
  see [CLAUDE.md](../../CLAUDE.md)) -- nothing to install,
  `environment.yml` already pulls in all four (as a superset, with
  `onnxruntime-directml` instead of plain `onnxruntime`).
- **Fresh venv / CI worker / outside this repo** --
  `pip install -r test/numeric/requirements.txt`.

## Quick Start

The framework has no repo-relative defaults. To actually exercise an
EP you supply `--ep-dll` (plus any `--ep-option KEY=VALUE` entries the
EP needs); without `--ep-dll` every test SKIPs cleanly, so the suite
is harmless to run on machines that haven't built the EP. Skeleton:

```bash
# Put the EP's runtime DLL dependencies on the loader search path --
# shell-layer concern, not the framework's job. Set any debug/profile
# env vars the EP recognises here too.
export PATH=/path/to/ep/runtime/deps:$PATH

cd test/numeric
pytest --backend ort_ep \
       --ep-name   YourExecutionProviderName \
       --ep-dll    /path/to/ep.dll \
       --ep-option some_key=some_value \
       --ep-option other_key=other_value \
       -s
```

For a concrete, runnable Windows example using this repo's build of
the MorphiZen EP, jump to
[Example: MorphiZen EP](#example-morphizen-ep). Most contributors
wrap the example into a local one-liner (shell function, `.bat`,
Makefile target, ...) to suit their setup.

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--backend <name>` | `ort_ep` | Inference backend to test. |
| `--output-dir <path>` | `python_test_output/` | Single root for all test output (see [Output Layout](#output-layout)). The two subdirs below derive from this when not set explicitly. |
| `--work-dir <path>` | `<output-dir>/intermediate/` | Per-test artifact directory (model + inputs + actual & expected outputs). Deleted on pass, retained on fail for postmortem. |
| `--keep-artifacts` | off | Keep all per-test artifacts even when the test passes. |
| `--cache-dir <path>` | `<output-dir>/cache/` | Persistent reference-output cache. |
| `--no-cache` | off | Disable the cache; always run ORT CPU as the reference. |
| `--refresh-cache` | off | Invalidate matching entries and re-run CPU (writes the new value back). |
| `--ep-dll <path>` | -- | EP DLL path. Required; tests SKIP if absent. |
| `--ep-name <name>` | `ExecutionProvider` | Alias the EP DLL is registered under; ORT advertises the device with this exact name. Any non-empty string works -- pick a meaningful one for log clarity. |
| `--ep-option KEY=VALUE` | (empty) | Repeatable. Pass-through into ORT's per-provider `provider_options` dict. Both key and value are EP-specific (see your EP's docs). Examples: `device_id=0` (CUDA/DirectML), `config_file=/path/to/conf.json` (MorphiZen/VitisAI). |
| `-s`, `-x`, `-k`, etc. | -- | All standard pytest flags work. |

## Output Layout

Everything the framework writes at runtime lives under one root
(`--output-dir`, default `python_test_output/` next to this README).
The whole tree is gitignored.

```
test/numeric/python_test_output/
├── intermediate/                          # ephemeral; deleted on pass
│   └── <NNN>_<sample-name>/               # one dir per run_sample() call
│       ├── model.onnx                     # exactly what the backend saw
│       ├── inputs/in_0.npy ...
│       ├── outputs_actual/out_0.npy ...   # what the backend produced
│       └── outputs_expected/out_0.npy ... # reference (cache / cpu)
└── cache/                                 # persistent across runs
    └── <sample-name>/                     # one dir per sanitised node id
        ├── manifest.json                  # { content_hash, shapes, ... }
        ├── model.onnx
        ├── inputs/in_0.npy ...
        └── outputs/out_0.npy ...
```

### `intermediate/` -- per-test postmortem snapshot

- Subdir name is `<3-digit counter>_<sanitised pytest node id>` so the
  on-disk ordering matches the run order, even across parametrisations.
- Every `run_sample()` call writes the model, the inputs the backend
  received, the backend's actual outputs, and the reference's expected
  outputs into the same subdir.
- On test PASS the whole subdir is deleted (override with
  `--keep-artifacts`).
- On test FAIL the subdir is retained, giving you a self-contained
  reproduction: `cd python_test_output/intermediate/NNN_<name>/` and
  every byte the test saw is right there next to the model.

### `cache/` -- persistent reference-output cache

- `<sample-name>` is the sanitised pytest node id (e.g. the
  parametrised test `test_matmul_qo_proj_llama_shape[128]` becomes
  `test_matmul_qo_proj_llama_shape_128`), so the directory tells you
  which test owns it.
- The sha256 of `(model_bytes + each input as contiguous bytes)` is
  stored *inside* `manifest.json` -- never in the path. On every cache
  read the runner recomputes the hash and rebuilds the entry if it no
  longer matches, so changing a seed / shape / scale automatically
  invalidates the cache without any manual bump.
- One directory per test name, ever -- disk usage is bounded; no
  `--prune-cache` ritual.
- If two distinct samples happen to sanitise to the same name they
  will repeatedly invalidate each other; the runner emits a `WARNING
  [cache] thrashing detected on '<name>'` log line when that happens.
  Fix: pass an explicit `name="..."` to one of the `run_sample` calls.

## Backend Configuration

### `ort_ep` -- ORT execution-provider backend (default)

The backend draws a clean line between **Python-consumed knobs** (CLI
flags) and **EP-DLL-consumed knobs** (environment variables). The
framework never reads env for its own knobs, and never sets env on
the DLL's behalf -- each layer owns one job. See
[the example wiring](#example-morphizen-ep) at the end of this
section for what a concrete EP looks like through this lens.

#### Python-consumed knobs (CLI flags)

These are values the Python framework itself reads (passed to
`register_execution_provider_library` and `provider_options`). There
is no env-var fallback: the suite is expected to run against installed
artefacts whose layout varies across CI tarballs, system installs, and
arbitrary working directories, so repo-relative guessing inside the
backend would do more harm than good. Pass them explicitly on every
pytest invocation.

| Flag | Required? | Purpose |
|---|---|---|
| `--ep-dll <path>` | Yes -- tests SKIP if absent | Path to the EP DLL. Registered with ORT at session start. |
| `--ep-name <name>` | No (defaults to `ExecutionProvider`) | Alias passed as the first argument to `register_execution_provider_library`; ORT then advertises the registered EP under this exact name in `get_ep_devices()`. The framework filters on it after registration, so any non-empty value is accepted -- pick a meaningful one for log clarity. |
| `--ep-option KEY=VALUE` | No -- repeatable | Pass-through entry into ORT's per-provider `provider_options` dict (the `Dict[str, str]` ORT hands to the EP's `OrtEpFactory::CreateEp`). Both KEY and VALUE are EP-specific -- the framework has no knowledge of any particular key. Only the first `=` separates key from value, so values containing `=` are preserved verbatim. Malformed entries, empty keys, and duplicate keys raise a clean SKIP rather than being silently dropped. |

The directory containing `--ep-dll` is automatically prepended to
`PATH` for any small co-located dependencies the EP needs at
registration time (e.g. a compiler DLL shipped next to the EP DLL).
This is an invariant of EP packaging, not a user choice.

#### EP-DLL-consumed knobs (environment variables)

These are values the EP DLL reads directly via Win32
`GetEnvironmentVariableA` (or that affect the Windows DLL loader, in
the case of `PATH`). The framework intentionally does not expose CLI
flags for them: a flag whose only job is to copy a value into
`os.environ` for the DLL to read would just be useless indirection
over `set X=...`, and the same env vars usually drive the EP across
*every* consumer (test runner, model loader, benchmark tool) -- one
surface, everywhere.

The framework only documents this surface; it does not enforce or
validate any particular variable. Refer to your EP's own documentation
for the env vars it recognises.

#### Typical workflows

- **Local development** -- copy the
  [Example: MorphiZen EP](#example-morphizen-ep) recipe into your
  shell or a personal one-liner (`.bat`, shell function, Makefile
  target). Re-use it for every invocation.
- **CI / shared shell** -- export the EP's runtime env vars (any
  dependent-DLL `PATH` additions, debug toggles, etc.) in your job
  setup, then invoke pytest with `--ep-dll` (plus any `--ep-option`
  flags your EP needs) pointing at your install. The pytest line
  itself is identical to the local case.
- **One-off experiment with a different DLL** -- just change the
  `--ep-dll` value on the command line (or append a second `--ep-dll`
  to your wrapper; pytest's argparse uses last-occurrence for
  repeated options).
- **Turn on an EP-specific debug/profile mode for one test** -- set
  the EP's env var in your shell (e.g. `set MY_EP_DEBUG=1`), run
  pytest with `-k <pattern> -s`, then unset it. The framework
  intentionally has no flag for this -- the env var is the canonical
  surface used everywhere else too.

Each EP session runs with `session.disable_cpu_ep_fallback=1`. Silent
CPU fallback would defeat the suite's purpose, so any unsupported op
fails the session creation rather than producing a misleading "pass".

#### Example: MorphiZen EP

A complete recipe for this repo's build of the MorphiZen EP,
runnable as-is from the repo root after `python build.py`. If you're
targeting a different EP, substitute the filenames and env vars
accordingly.

**Windows (cmd):**

```cmd
rem One-time per shell: activate conda + VS, expose ROCm runtime DLLs.
call "%USERPROFILE%\miniforge3\condabin\conda.bat" activate base
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "THEROCK_DIST=%CD%\install\therock"
set "PATH=%THEROCK_DIST%\bin;%PATH%"

rem Run the suite.
cd test\numeric
pytest --backend ort_ep ^
       --ep-name      MorphiZenExecutionProvider ^
       --ep-dll       "..\..\install\dist\bin\onnxruntime_morphizen_ep.dll" ^
       --ep-option    config_file=..\..\install\dist\bin\morphizen_config.json
```

**Linux (bash):**

```bash
export THEROCK_DIST=$PWD/install/therock
export PATH=$THEROCK_DIST/bin:$PATH

cd test/numeric
pytest --backend ort_ep \
       --ep-name      MorphiZenExecutionProvider \
       --ep-dll       ../../install/dist/bin/libonnxruntime_morphizen_ep.so \
       --ep-option    config_file=../../install/dist/bin/morphizen_config.json
```

What each piece does:

- **`THEROCK_DIST` + `PATH`** -- the EP's compiled-model DLLs link
  against ROCm runtime libs (`amdhip64_7.dll`, `MIOpen.dll`, ...) at
  session-create time; the Windows loader needs `%THEROCK_DIST%\bin`
  on `PATH` to find them. This is the *only* env var the EP requires
  at runtime; everything else is a CLI flag or optional toggle.
- **`--ep-name MorphiZenExecutionProvider`** -- matches the string
  the EP's C++ side passes to `OrtEpFactory::CreateEpFactories`.
- **`--ep-dll`** -- the registered EP DLL. The framework resolves
  the path to absolute before handing it to ORT (because
  `register_execution_provider_library` resolves relative paths
  against the onnxruntime *package* directory, not cwd -- so the
  relative recipe above would otherwise need to live inside the ORT
  install) and auto-prepends the resolved parent directory to `PATH`
  so co-located dependencies (`hip-compiler.dll`) are found at
  registration time.
- **`--ep-option config_file=...`** -- forwards the key/value into
  ORT's `provider_options` dict for this EP. **`config_file` is
  MorphiZen's own convention** (the EP's
  [`config_reader.cpp`](../../morphizen/morphizen-core/src/binary/config_reader.cpp)
  reads this key, treats the value as a path to a protobuf-JSON
  config, and uses it to drive its pass pipeline). Other EPs use
  different keys -- e.g. CUDA/DirectML expose `device_id`, TensorRT
  uses `trt_engine_cache_path`. Consult your EP's docs.

The MorphiZen EP additionally honours these env vars when set in the
shell before invoking pytest:

| Env var | Effect |
|---|---|
| `HIPDNN_EP_DEBUG=1` | Verbose debug logging from the EP DLL. |
| `HIPDNN_EP_PERF=1` | Per-op GPU/CPU profiling output. **Do not use for tok/s measurements** -- adds ~58% overhead (see CLAUDE.md). |
| `HIPDNN_EP_GQA_*` | Various GQA path toggles (`HIPDNN_EP_GQA_FLASH_DECODE`, `HIPDNN_EP_GQA_CACHE_SEQLENS`, ...). |

These are read by the DLL directly; the framework neither sets them
nor knows about them. Set them in your shell exactly as you would
when running `hip-onnx-runner.exe` or `model_benchmark.exe`.

### Adding a new backend

1. Implement `framework/my_backend.py`:
   ```python
   from framework.backend import Backend

   class MyBackend(Backend):
       @property
       def name(self) -> str:
           return "MyBackend"

       def run(self, model_path, inputs):
           ...  # return list of numpy arrays

   def create(pytest_config) -> MyBackend:
       # pytest_config exposes config.getoption("--my-flag") for any
       # backend-specific CLI options you added in pytest_addoption.
       ...
   ```
2. Register in [conftest.py](conftest.py):
   ```python
   _BACKENDS["my_backend"] = lambda config: __import__(
       "framework.my_backend", fromlist=["create"]
   ).create(config)
   ```
3. Add any backend-specific CLI options in the same `pytest_addoption`
   function (e.g. `parser.addoption("--my-dll", ...)`) so they show up
   under `pytest --help`.
4. Run: `pytest --backend my_backend`.

## Writing a New Test

```python
import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


class TestMyOp:
    @pytest.mark.parametrize("shape", [[1, 16], [4, 256]])
    def test_my_op(self, model_runner, shape):
        tp = np_to_onnx_type(np.float16)
        X = helper.make_tensor_value_info("X", tp, shape)
        Y = helper.make_tensor_value_info("Y", tp, shape)
        node = helper.make_node("MyOp", ["X"], ["Y"])
        model = make_model_from_nodes([node], [X], [Y])

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)
```

- `model_runner` is the per-test fixture; `run_sample(...)` accepts
  `bytes | onnx.ModelProto | Path | str` for `model` and a list of
  `np.ndarray | Path | str` for `inputs`.
- `reference` defaults to `"cache"`. Pass `reference="cpu"` for cheap
  ops where caching is not worth the disk traffic.

## Bring Your Own ONNX

For samples that come from a real model trace (or any other source)
the framework happily reads them straight off disk:

```python
from pathlib import Path

def test_external_op(model_runner):
    actual, expected = model_runner.run_sample(
        model=Path("D:/captures/my_op/model.onnx"),
        inputs=[Path("D:/captures/my_op/in_0.npy")],
    )
    compare_outputs(actual, expected, atol=1e-3)
```

The model and any number of `.npy` input files are loaded straight
from disk; the reference is still resolved through the cache (with an
implicit ORT CPU run on first miss), so an externally-captured sample
gets the same drift-tripwire and zero-cost replay as an in-suite one.

## Memory Strategy

Each test follows the same sequence:
1. Persist the model bytes to disk and release the Python buffer.
2. Run the EP backend, destroy its session.
3. Run CPU (if needed), destroy its session.
4. Snapshot inputs + actual/expected outputs into the work dir (free
   on pass -- the conftest fixture deletes the whole subdir; retained
   on fail for postmortem).
5. Compare.

Only one ORT session lives at a time, which keeps peak memory low for
models with large weight initialisers (e.g. the Llama gate/up
projection holds a ~115 MB fp16 weight).

## Project Layout

```
test/numeric/
├── README.md               this file
├── pytest.ini
├── requirements.txt        pip install -r ...  (numpy, onnx, onnxruntime, pytest)
├── conftest.py             --backend / --output-dir / --no-cache / ... + fixtures
├── framework/
│   ├── backend.py          Backend ABC
│   ├── ort_cpu_backend.py  ORT CPU reference (implicit, internal)
│   ├── ort_ep_backend.py   ORT EP backend
│   ├── comparator.py       allclose + cosine
│   ├── onnx_utils.py       model construction helpers
│   ├── reference_cache.py  cache I/O + drift tripwire
│   └── model_runner.py     run_sample(...) orchestrator
├── tests/
│   ├── test_sigmoid.py     unary fp16, small + Llama MLP gate shape
│   └── test_matmul.py      MatMul fp16, small + Llama Q/O & gate/up shapes
└── python_test_output/     auto-created, gitignored (see Output Layout above)
```
