# KV-Pareto kernel unit test

This leaf checks the per-token `k8v8_g32_planar_v1` codec directly against the
production kernel library. Q, new K/V and output are FP16. Cached K/V use INT8
payload, BF16 scales and INT8 zero points. It supports B=1, H=16 or 24, G=4
and D=256 on gfx1151. Other devices print `SKIP`. A gfx1151 package with the
codec disabled fails.

The encoder must match the host reference quantizer byte for byte. Attention
uses an FP64 reference over dequantized history and direct FP16 new K/V. Every
checked row must satisfy absolute/relative element tolerances and relative L2
below 0.005. The entire cache is compared after each case, including preserved
history and unused capacity. Short queries check every row. Long queries
check the first, middle and last rows.

## Packaged invocation

From the package's `bin` directory:

```text
hipdnn-kernel-ut.exe --leaf gqa/kv_pareto/Xfp16_Wi8_Yfp16 --coverage 1 --timing screen
hipdnn-kernel-ut.exe --leaf gqa/kv_pareto/Xfp16_Wi8_Yfp16 --timing standard -- --heads 24 --new-tokens 1 --past 4095 --capacity 4133
```

`ut_interface.json` records all custom parameters and runnable decode/append
examples. `common/ut_interface.py` prints and runs their controller commands.
Any custom flag selects exactly one attention case. Unsupported geometry,
invalid numbers and insufficient capacity fail before GPU allocation.

## Standalone build and custom shape

The Makefile links an existing hip-ep package built with
`HIP_EP_ENABLE_KV_PARETO=ON`. It does not embed production kernels or LUTs.
Pass `HIP_SDK` and `HIP_EP_PACKAGE`. Windows uses the import library in `lib`
and DLLs in `bin`. WSL can use that Windows SDK and package. Native Linux
uses the shared library in `lib`.

```text
make test HIP_SDK=<sdk> HIP_EP_PACKAGE=<package> COVERAGE=1 TIMING=screen
make test_custom HIP_SDK=<sdk> HIP_EP_PACKAGE=<package> H=24 SQ=1 PAST=4095 CAPACITY=4133
make build HIP_SDK=<sdk> HIP_EP_PACKAGE=<package> PKG_DIR=<destination> PKG_NAME=<executable-name>
make clean
```

| Make variable | Driver flag | Default | Constraint |
| --- | --- | ---: | --- |
| B | --batch | 1 | 1 |
| H | --heads | 24 | 16 or 24 |
| G | --kv-heads | 4 | 4 |
| D | --head-dim | 256 | 256 |
| SQ | --new-tokens | 17 | 1–262144 |
| PAST | --past | 129 | Nonnegative |
| CAPACITY | --capacity | 4096 | 1–262272, at least PAST + SQ |
| SEED | --seed | 1234 | 1–2147483647 |
| SCALE | --scale | 0.0625 | Finite and positive |

`COVERAGE=1|2|3` defaults to 1 and is passed as `--coverage`. Direct executable
invocation also reads `HIPDNN_UT_COVERAGE`. Tier 1 retains the 27 encoding,
contract, H16/H24 prefill, append and decode-boundary checks. Tier 2 adds 64K
and 128K decode and a 4K append. Tier 3 adds maximum-capacity decode/append
and single-call 256K prefill. These long cases are optional.

`OFFLOAD` defaults to `--offload-arch=gfx11-generic` and selects the test's
stream-gate ISA. `KERNEL_ARCH` selects the packaged kernel library and defaults
to that target. Use `OFFLOAD=--offload-arch=gfx1151` with a chip package. A
generic test executable can link that same chip package with
`KERNEL_ARCH=gfx1151`. Builds for gfx1170/gfx1171 skip at runtime.

## Timing and configuration

`TIMING=screen|standard|confirm` defaults to standard and sets
`HIPDNN_UT_TIMING`. Optional `ADLX_INCLUDE` points to the ADLX SDK headers
for Windows GPU frequency sampling. The shared adaptive timer measures the complete encoder or
attention call, emits `UT_TIMING`, and binds each checked result with
`UT_TIMING_BINDING`. Contract/rejection cases have no timing result.

These correctness timings reuse one operand/cache allocation and explicitly
report warm caches. Repeated encoding is idempotent and preserves the same
history. They do not replace the separate model and attention performance
campaigns.

`MODE=lookup|autotune` defaults to lookup. Prefill uses the production GQA
resolver and embedded table in lookup mode, including its heuristic fallback.
Autotune mode lets the production prefill kernel choose its configuration.
The controller and Makefile capture its stderr diagnostic from process startup
so each row records the configuration that ran. Direct online-prefill invocation
must also redirect stderr and set `HIPDNN_KERNEL_UT_STDERR` to that log path.
Decode and encoding have fixed codec configurations. There is no independent
KV-Pareto LUT or ranking interface.

When `HIPDNN_RESULTS_CSV` is set, each case appends a result row. The packaged
controller supplies that path and adds the suite summary. The standalone
Makefile resets `out/results.csv` and captures stderr in
`out/kernel.stderr.log`.
