# hip_gemm Xbf16_Wbf16_Ybf16

This leaf compiles and runs only `Xbf16_Wbf16_Ybf16`. Its self-contained `test_gemm.cpp`
uses a leaf-local fixed `hip_gemm` dtype code and host/device storage: fp16=0,
bf16=3, fp32=1. It has no runtime dtype selector.

```
make test OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<HIP SDK root> MODE=lookup COVERAGE=1
make test_custom OFFLOAD=--offload-arch=gfx1151 HIP_SDK=<HIP SDK root> M=32 N=512 K=512 TA=0 TB=1
make clean
```

`MODE=lookup` is the default. Tier 1 has four quick representative cases: the
GemvNt, GemvNn, and Wmma phases, plus a
bias route. It writes the leaf identity through `HIPDNN_RESULTS_LEAF` and uses
the schema-v3 gfx1151 LUT when linked.
