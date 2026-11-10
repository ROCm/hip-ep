# layer_norm / Xfp32_Wfp32_Yfp32

Calls the production `hip_layer_norm` entry; inputs and CPU reference are generated
in process with a fixed seed. No model-specific kernel or replacement timer.

```
make test HIP_SDK=<sdk> COVERAGE=3
make test_custom HIP_SDK=<sdk> ARGS="--rows R --norm N [--bias 0|1] [--mean 0|1] [--inv-std 0|1] [--stash fp16|fp32] [--eps E]"
make build HIP_SDK=<sdk>
make clean HIP_SDK=<sdk>
```

`MODE=lookup|autotune` is accepted by the uniform API; this fixed launcher has no
LUT/tuner. `COVERAGE=1|2|3` selects 1/2/3 typical shape families while retaining
every categorical case. Bias on/off, Mean and InvStd output independently on/off, each supported stash dtype, epsilon=0/1e-5 at every tier; row/norm sizes include odd tails, width below/above a block and real width 1152. The FP64 in-process reference checks every output and every requested statistic.

Arguments: `--rows R --norm N [--bias 0|1] [--mean 0|1] [--inv-std 0|1] [--stash fp16|fp32] [--eps E]`. `--iters` defaults to 20, matching the owning main u4 UT;
profiling runs one process and consumes its mean directly. Unknown, duplicate,
malformed and incomplete arguments fail with exit 2. A failed launch, any output
mismatch or invalid timing fails the case. CSV uses the standard controller schema.

This main revision has no native-v2 common timer. The unchanged owning ColdCopies
and rotatedMeanMs algorithm is copied inline from
`matmul_nbits/Xfp16_Wu4_Yfp16/test_matmul_nbits.cpp` at main
`a30a147d3c02937c044cb761996c5f5d95c7e675`; original timing-section SHA256:
`ac4d749bd9cc183a6d5e03ddde6b0f6f03d2c402e3e52295d171caf1136d6c4e`. Only flushed reported-call markers were added outside the event
interval. This is the current main protocol, not native-v2 evidence. The default
kernel build uses release `-O2` and `gfx11-generic`; packaging links the production DLL.

All input/scale/bias read operands participate in the owning cold-copy rotation. Optional Mean/InvStd outputs are included in the complete measured call.


`HIPDNN_UT_COVERAGE=1|2|3` selects the sweep tier when `--coverage` is absent;
the CLI value takes precedence. Semantic single-case flags require the complete
geometry (`--axis-size` for Gather, `--rows` and `--norm` for LayerNormalization).
They never silently turn into a default sweep. The requested results CSV must
open and flush successfully or the process fails. LayerNorm case identifiers
retain epsilon with enough digits to distinguish FP32 values.

The normal Windows CI build creates these drivers through
`unit_test/CMakeLists.txt`, links `custom_kernels_<arch>.dll`, and installs them
under `bin/kernel-tests/<arch>/`. The unchanged `hipdnn-kernel-ut.exe` entry point
runs each registered leaf with `--coverage 1`; every case appends the standard CSV
row and the controller appends its `shape=suite` summary. Standalone `make test`
and `test_custom` compile driver/kernel as separate objects, like existing leaves.
