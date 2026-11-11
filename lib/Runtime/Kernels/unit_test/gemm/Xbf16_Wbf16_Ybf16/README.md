# hip_gemm Xbf16_Wbf16_Ybf16

This leaf compiles and runs only `Xbf16_Wbf16_Ybf16`. Its self-contained
`test_gemm.cpp` uses a leaf-local fixed `hip_gemm` dtype code and host/device
storage: fp16=0, bf16=3, fp32=1. It has no runtime dtype selector.

Timing uses [`unit_test/common`](../../common/README.md) adaptive complete-call
policy (`HIPDNN_UT_TIMING=screen|standard|confirm`). The default `standard`
adapts to at least one second / ten complete calls; pass `TIMING=confirm` for
formal performance numbers or `TIMING=screen` for short LUT ranking.

```
make test HIP_SDK=<HIP SDK root> OFFLOAD=--offload-arch=gfx11-generic MODE=lookup COVERAGE=1 TIMING=screen
make test_custom HIP_SDK=<HIP SDK root> M=32 N=512 K=512 TA=0 TB=1
make clean
```

`MODE=lookup` is the default. Coverage tiers thin the typical-shape list the
same way as the other unit_test leaves. Results go to `out/results.csv` with
`UT_TIMING` JSON evidence on stdout.
