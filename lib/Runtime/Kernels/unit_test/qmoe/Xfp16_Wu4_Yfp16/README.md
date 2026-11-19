# QMoE FP16/uint4 complete-wrapper unit test

The self-contained driver includes the original `real/qmoe.cpp` wrapper.
It supplies only stream/preallocated scratch services and the owning packed
zero-point cache API. Routing, bucketing, gathering, both expert projections,
SwiGLU, per-call activation quantization and weighted reduction run through
the production dispatch chain. The leaf covers FP16 activations/scales/bias,
packed uint4 weights/optional zero points, fused SwiGLU and softmax top-k
routing. It does not claim ReLU2, sigmoid-bias routing or an unfused FC3 path.

```text
make build HIP_SDK=<sdk> [OFFLOAD=--offload-arch=<target>] FLATC=<flatc> FLATBUFFERS_INC=<include>
make test HIP_SDK=<sdk> MODE=lookup COVERAGE=3 FLATC=<flatc> FLATBUFFERS_INC=<include>
make test_custom HIP_SDK=<sdk> ARGS="--tokens 1 --hidden 3072 --inter 1024 --experts 256 --topk 8 --block-size 32 --fc1-bias 1 --fc2-bias 1 --zero-points 0 --normalize 1 --router-weights 0 --alpha 1 --beta 0"
make clean HIP_SDK=<sdk>
```

Every tier retains block sizes 32/64/128, all four FC1/FC2 bias combinations,
zero points on/off, routing normalization on/off, external router weights
on/off, and balanced/skewed/tied routing. Tied external weights include the
zero-sum case. The explicit shape matrix round-robins decode, short bursts,
and prefill around the default fused cutoff (1/2/8/9/17/33 tokens); tiers
thin only that geometry. Projection/expert/top-k families are
(H,I,E,k)=(128,128,8,1), (256,128,8,2), (128,256,16,4),
(256,256,16,8), with 288 categorical cases per tier. Skewed/tied prefill tests empty experts and both
small and big row tiles. Custom shapes require all six geometry arguments,
32-aligned projection sizes and block-size divisibility. Unknown, duplicate,
incomplete or unsupported arguments fail with exit 2.

The seeded CPU reference checks every output, every selected expert index
and all routing weights. Decode follows the actual per-group int8 activation
quantization with nearest-even rounding and uint4 dot products. Prefill
rounds dequantized weights, FC1, scaled SwiGLU and FC2 storage to FP16 at
their actual boundaries. The reference uses independent double dot products,
threaded over experts; it neither samples outputs nor calls a GPU reference.
The prefill reduction follows ascending expert ID (stable routing-slot order
for ties), rounding the running sum to FP16 after every contribution, as the
production sorted-pair reduction does. Decode retains its single final store.
Each finite decode output must be within `0.0005 + 0.005*abs(reference)`, full
relative L2 at most 0.005; routing weights tolerate one FP16 rounding unit.
When a correctness check fails, stderr reports bounded QMOE_CHECK examples
with token/channel or token/top-k-slot coordinates, actual/reference values and
the applicable error limit. Its summary separates output, expert-index and routing-
weight failures and includes maximum errors and aggregate relative L2. A small
relative L2 alone does not establish a pass.

For prefill, the per-element limit is
max(0.0005 + 0.005*abs(reference), 2^-10*sum(abs(expert_contribution))).
Each contribution includes its routing weight. The additional floor is one
relative FP16 ULP of the contribution magnitude, so cancellation between experts
does not turn normal intermediate FP16 rounding into an arbitrarily large
relative error at a near-zero final output. This is a precision-policy change
for cancellation; it does not change the absolute/relative coefficients or
the full relative-L2 limit of 0.005, and it does not apply to decode.
Expert indices and routing weights retain their separate checks.

A targeted regression can be run with:
```text
make test_custom HIP_SDK=<sdk> MODE=lookup TIMING=standard ARGS="--tokens 128 --hidden 2880 --inter 2880 --experts 128 --topk 4 --block-size 32 --fc1-bias 1 --fc2-bias 1 --zero-points 1 --normalize 1 --router-weights 0 --routing-pattern balanced --alpha 1.7020000219345093 --beta 1 --limit 7"
```
Supply FLATC and FLATBUFFERS_INC as for the build command above.

The default infinite SwiGLU limit is preserved; positive finite limits
are also exercised. The default dispatch environment is consumed as in production. Inputs are
synthetic and standalone source/runtime equivalence must be reported separately.

Normal timing uses the existing `unit_test/common/hip_adaptive_timing.h`
`ut-adaptive-v3` policy, selected with `TIMING=screen|standard|confirm`
(default `standard`) or `HIPDNN_UT_TIMING`. Legacy `--iters` does not control
sampling. CPU-checked CSV rows bind their exact `UT_TIMING` record.
All weights/scales/optional zero points/bias rotate together. The actual
production `real/qmoe.cpp` wrapper and every per-call kernel remain inside
timing; CPU reference, scratch allocation and input preparation are untimed.
`COVERAGE=1|2|3` is independent of the timer preset. `MODE=lookup|autotune`
uses the existing MatMulNBits resolver/table or its documented fallback;
QMoE itself has no LUT. Existing MatMulNBits/GQA JSON refresh commands remain
unchanged. Source staging/build invalidation must include the quoted runtime
wrapper and its transitive headers as well as this leaf and the common timer.

`HIPDNN_UT_CAPTURE_ONLY=1` requires one explicit shape, executes one complete
call and full CPU check, and skips rotation/events/host hold gates. The zero
CSV time is diagnostic only; the exact success marker is
`UT_CAPTURE_ONLY complete_calls=1 timing=disabled stream_gate=disabled`.
