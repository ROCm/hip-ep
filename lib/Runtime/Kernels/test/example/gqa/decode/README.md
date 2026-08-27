# GQA Decode Test Suite

Tests and latency probes for `hip_gqa_flash_decode()`, the FA-2 split-K decode
(`sq == 1`) that serves MHA and GQA, fp16 and INT8 KV caches, with optional
sliding window, per-head attention sink and smooth softmax.

Every program here is standalone: it compiles `gqa_kernel.hip` directly and
calls the kernel entry, so nothing needs the EP or a model to be built first.

## Prerequisites

- **HIP SDK** (ROCm for Windows or Linux) with `hipcc` on `PATH`
- A supported AMD GPU; the commands below target `gfx1151`, change
  `--offload-arch` to match yours

All commands are run from this directory:

```bash
cd lib/Runtime/Kernels/test/example/gqa/decode
```

`../../../..` is the `Kernels` root, which is where the kernel source and its
headers live. The snippets below spell that out rather than using a variable so
each one can be pasted on its own.

## Quick start

Build and run the fp16 correctness suite:

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_decode.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_decode.exe
./test_gqa_decode.exe
```

The last line is the verdict:

```text
ALL PASS (0 failing case(s))
```

## Programs

| Program | What it does |
|---|---|
| `test_gqa_decode.cpp` | fp16 correctness and latency over the full geometry matrix |
| `test_gqa_decode_i8.cpp` | The same for an INT8 KV cache, plus the fp16/INT8 ratio |
| `test_gqa_decode_bias.cpp` | Correctness with an additive attention mask, on linear and ring caches |

The first two cover correctness and latency; flags select how much of each you
get. `test_gqa_decode_bias` is correctness only and takes no flags.

### `test_gqa_decode_bias` (additive mask)

The mask is the one input the kernel indexes by absolute KV *position* rather
than by buffer slot, so it is the only place a right-sized sliding-window cache
("ring") is visible: the slots arrive rotated and every entry lands on the wrong
key. That failure is silent — the attention still sums over the right set of
keys, so the output is plausible rather than NaN — which is why half the cases
here run on a ring, and why they use a random mask. A window mask cannot detect
a rotation, because a ring of exactly `window` cells holds only in-window
positions and all of its live entries are zero.

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_decode_bias.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_decode_bias.exe
./test_gqa_decode_bias.exe
```

Each line reports the geometry, whether the cache is linear or a ring (with the
rotation), and `relL2` against an fp32 CPU reference. Healthy fp16 values are
around `2e-04`; a case passes under `5e-03`. Removing the de-rotation from the
kernel puts every ring case at `~1e+00` and leaves every linear case passing,
which is the check that the ring cases are load-bearing rather than decorative.

### `test_gqa_decode` (fp16)

Sweeps MHA and GQA at HpG 1/2/3/4/5/8/16, head_dim 64/128/256/512 and
`len` 512..32768, including sliding-window, head-sink and smooth-softmax
variants. Each case is checked against an fp32 CPU reference and against the
other kernel implementation (WMMA versus scalar), then timed.

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_decode.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_decode.exe
./test_gqa_decode.exe
```

Per case it prints the geometry, `relL2` against the CPU reference, the A/B
difference between the two kernel implementations, the latency of each fixed
configuration next to the autotuned one, and `PASS` or `FAIL`. A case passes on
`relL2 < 2e-2`; healthy fp16 values are around `3e-04`.

| Option | Default | Meaning |
|---|---|---|
| `--iters <n>` | 200 | Timed calls per configuration |
| `--only <substring>` | all | Run only the models whose name contains the substring |
| `--prod-only` | off | Time only the autotuned configuration |
| `--md` | off | Emit a Markdown table instead of per-case text |
| `--fused` | off | Also time the one-block-per-head decode, where the shape allows it |
| `--seed <n>` | 1234 | Input RNG seed |
| `--verbose` | off | Dump the first mismatching elements when a case fails |
| `--b/--h/--g/--d/--max-seq/--total/--window/--sink/--smooth` | — | Run one custom shape instead of the matrix |

### `test_gqa_decode_i8` (INT8 KV)

Same idea for a symmetric per-channel INT8 KV cache. Each case is checked
against an INT8 reference computed on the host, and against the fp16 result to
show the quantization error separately from any kernel error.

```bash
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-deprecated-declarations \
  test_gqa_decode_i8.cpp ../../../../hip/gqa_kernel.hip \
  -I../../../../include -o test_gqa_decode_i8.exe
./test_gqa_decode_i8.exe
```

`relL2 kern/i8ref` is the kernel error and should sit near `2e-04`;
`quant(i8/fp16)` is the quantization error and is expected to be much larger
(around `4e-03`).

## Selecting a kernel configuration by hand

The decode entry autotunes the implementation, split count and KV tile height
on the first call per shape and caches the winner. These variables override that
and bypass the autotuner, which is what you want when profiling one specific
configuration:

| Variable | Values | Effect |
|---|---|---|
| `HIPDNN_GQA_DECODE_SCALAR` | `1` | Force the scalar split-K producer |
| `HIPDNN_GQA_DECODE_WMMA` | `1` | Force the WMMA split-K producer |
| `HIPDNN_GQA_DECODE_SPLITS` | `1`..`64` | Force the split count |
| `HIPDNN_GQA_DECODE_BKV` | `16`, `32` | Force the WMMA KV tile height (head_dim 64 only) |
| `HIPDNN_GQA_DECODE_NOAUTOTUNE` | `1` | Keep the heuristic default, skip tuning |
| `HIPDNN_EP_DEBUG` | `1` | Print every autotune candidate's score and the winner |

```bash
HIPDNN_EP_DEBUG=1 ./test_gqa_decode.exe --iters 100 --only 'gpt_oss-20b full'
```

On Windows PowerShell, set them with `$env:HIPDNN_EP_DEBUG = '1'` and clear them
with `Remove-Item Env:\HIPDNN_EP_DEBUG`.

Note that `test_gqa_decode` sets these itself to time its fixed-configuration
columns, so it overwrites whatever you export. `--prod-only` leaves them alone.

## Comparing two builds

Correctness is a single verdict line, but latency is not.

```bash
./test_gqa_decode.exe --prod-only --only 'gpt_oss-20b full' --iters 500
```

`--prod-only` times only the configuration the kernel would actually pick, so
the fixed-configuration runs cannot sit between the timed ones and change the
clock state around them. `--only` exists for the same reason at a coarser
grain: the shapes run in a fixed order inside one process, so a shape inherits
the state of whatever ran before it, and a shape that got faster makes the next
one look slower. Give each model its own process.

Run-to-run spread on this path is a few percent, and changing the number of
instantiated kernels shifts code-object layout enough to move even untouched
kernels by about a microsecond. A comparison worth reporting therefore needs
several hundred timed calls per shape, alternating which build runs first
across repeated process pairs, and a median rather than a mean.
