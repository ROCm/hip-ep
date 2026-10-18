<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TurboQuant development checks

This is an experimental, explicit-opt-in MSE-only QR codec for the Qwen
24-query-head / 4-KV-head / dimension-256 geometry. Read
[`turboquant-kv-cache.md`](../../docs/design/turboquant-kv-cache.md) and the
[validation results](../../docs/design/turboquant-validation.md) before use.
The strict encoder gate and some full-model comparisons currently fail; this
is not a production-quality accuracy claim.

## CPU reference

Install NumPy, SciPy and pytest, then run from the repository root:

```console
python -m pytest tools/turboquant/test_reference.py -q
```

The checked-in `fixtures/constants.npz` freezes the Windows validation rotation
and K/V centroids. It contains mathematical codec constants, not model weights.
The standalone reference source revision was
`a6218a9b99ecabd3028da670fb0f85bb1114e552`. Regenerating a QR rotation on another
platform can change signs/rounding; do not silently replace these constants
when comparing recorded results. Serialize the exact constants with:

```console
python tools/turboquant/write_codec.py tools/turboquant/fixtures/constants.npz codec.bin
```

The resulting codec SHA-256 must be
`a2a312f645ba0a7970bda404fb4f1e1f15a01c2dd18c257b6a4c83397e806a3c`.

## GPU checks

The scripts use an external work directory with this layout:

```text
work/
  hip-ep/                       this repository
  reference/codec/constants.npz copy of the frozen fixture
  reference/codec/codec.bin     serialized codec
  build/                       profiling DLL and build outputs
  results/                     test receipts
  install/turboquant/           Ring-KV build including TurboQuant
```

Set `TQ_WORK_ROOT` to the absolute work directory, `TQ_HIP_RUNTIME` to the HIP
runtime library, and `TQ_TEST_LIBRARY` to the installed gfx1151 custom-kernel
library. On Windows, initialize the ROCm SDK DLL search paths first (the local
validation environment imports its ROCm PyTorch package before the direct
ctypes checks). Run `check_hip.py` for strict code bytes, guarded allocations,
decoder oracle and negative workspace/bounds checks. Its 4096-vector case is
expected to expose the documented two-byte encoder discrepancy; do not weaken
the assertion. Run `check_hip_extended.py` with `TQ_RING_LIBRARY` set to a
Ring-enabled custom-kernel library for append/long-history checks and direct
kernel timing. These timings are not full-model token throughput.

`check_genai_windows.py` checks GenAI shared-cache allocation and invalid-codec
error propagation with a small generated ONNX graph. It expects the Windows
SDK and work layout described above. `benchmark/check_long_windows.py` checks
the decoder against an FP64 oracle at 128K, 128K+128 and 256K, including guards.

## Reproduce Windows full-model measurements

The `benchmark/` files are the development harness used for the tables. Copy
them into `work/scripts/` (their parent directory defines the work root), and
use `work/venv/Scripts/python.exe`. They expect ROCm SDK packages under that
interpreter's base environment. Build three separate installs with the normal
repository build procedure, changing `HIP_EP_GQA_IMPL`:

| Selection | Install directory |
| --- | --- |
| `existing` | `install/existing` |
| `ring_q_resident` | `install/ring_q_resident` |
| `ring_kv_outer` | `install/turboquant` |

The last install supplies both dense Ring-KV and TurboQuant; provider options
select the packed cache. Record the source revision, dirty diff and library
hashes for every build. The reported baseline selects production GQA in the
same integration source tree, rather than claiming a pristine upstream binary.

Use the model `amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, revision
`9b37cf56a001720ba7ce98ff80ea651b02e5931b`, in `work/models/qwen-original`.
Verify downloaded files before creating `results/windows-qwen-download.json`
with `{"status":"passed","verified":[{"file":"relative-file-name"}, ...]}`.
The profiler hard-links these verified model files into per-case directories.
Do not include model files or credentials in a source commit.

Create the `build/` and `results/` directories, then build the allocation sidecar from an MSVC x64 developer console:

```console
cl /LD /EHsc /std:c++17 /O2 scripts/windows_memory_counter.cpp /Fe:build/windows_memory_counter.dll
```

Run the observer in a separate terminal before each case, then the case itself:

```console
venv\Scripts\python scripts\windows_memory_observer.py --variant existing --length 4096
venv\Scripts\python scripts\windows_profile_run.py existing 4096 --repetitions 3
```

Repeat serially for `existing`, `ring_q`, `ring`, `tq4`, at 128, 256, 512, 1024,
2048, 4096, 8192, 16384, 32768, 65536 and 131072. Use one measured request at
131072. Generate each dense Ring reference before its corresponding `tq4`
case. For 256K use `synthetic-existing`, `synthetic-ring_q`, `synthetic-ring`
and `synthetic-tq4` with length 262144; the observer variant omits `synthetic-`.
Run `consolidate_comparison.py` after all receipts exist. Do not overlap GPU
tests or compiler builds with timing runs.

If dense 128K cannot complete, the historical TQ profiler requires
`TQ_ARCHIVED_RING_128K` pointing to an explicitly labeled archived reference
directory containing `probe.json` and logits arrays. This reference is for
numerical diagnostics only; the consolidator never substitutes its Linux
performance or memory data into Windows rows. No archived reference is bundled.

For the three long direct-kernel checks, use `python scripts/windows_profile_run.py kernel-long` after installing the kernels and preparing the frozen constants.

The launcher bounds each case to one hour and stops its own child process tree
if available system memory remains below 1 GiB for 20 seconds. Retain failed
receipts rather than converting partial timings into completed results.
WDDM totals are process GPU commitments sampled approximately every 20 ms,
not physical residency. The sidecar's separate event peaks measure live EP
allocation requests. The two counters have different scopes; only WDDM totals
are decomposed into weights, KV and residual other in the published table.

Source formatting and replacing the private archived-reference path with an
environment variable do not retroactively change recorded harness hashes.
The metrics JSON retains the original measured binary and harness provenance.
