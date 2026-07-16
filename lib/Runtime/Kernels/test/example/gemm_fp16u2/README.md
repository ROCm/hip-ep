# fp16 x u2 MatMulNBits — Test & Benchmark vs fp16 x u4

Benchmarks and verifies the `bits=2` (uint2, 2-bit packed) quantized-weight
kernel in `matmul_nbits_kernel.hip`, and compares it head-to-head against the
existing `bits=4` (uint4, nibble-packed) kernel on **identical**
(M, K, N, group_size) shapes. Both are ONNX `MatMulNBits` conventions.

This is a sibling of `../gemm_fp16u4` and `../gemm_fp16u3` — same build style,
same Makefile conventions — but a single test binary loads both a
u2-quantized and a u4-quantized copy of the same-shaped weight matrix (sharing
the same `A`) and runs both kernels back-to-back, printing a side-by-side
comparison.

## Format background

- **u4** (ONNX `MatMulNBits`): nibble-packed, `B_packed` is `[N, K/2]` uint8,
  low nibble = value `2k`, high nibble = value `2k+1`. Default zero point 8.
- **u2** (ONNX `MatMulNBits`): 2-bit packed, 4 values per byte, LSB-first —
  `B` is `[N, K/4]` uint8, value `k` occupies bits `[2k, 2k+2)`. Because 2
  divides 8 evenly this is simultaneously the ONNX blockwise convention
  (`blob_size = block_size/4` bytes) and a plain continuous 2-bit stream, so
  the two are byte-identical when `block_size % 4 == 0`. Row stride is `K/4`
  bytes — half of u4's `K/2` — so u2 moves ~50% of u4's weight traffic.
  Default zero point 2. Three dispatch paths mirror bits=3/bits=8:
  1. **WMMA fast path** (M >= 16, K % 32 == 0, group_size % 32 == 0 and
     >= 32): the 2-bit stream is decoded inline in the WMMA K-loop, no
     separate dequant buffer.
  2. **GEMV** (decode/small-M, block_size power of two >= 32, K % 32 == 0):
     vectorized loads, many weights per transaction.
  3. **Naive fallback**: anything meeting neither gate above.

## Zero-point handling (two conventions, both validated)

ONNX packs `zero_points` at `bits` bits, so a real 2-bit model ships them
**4-per-byte packed** (`[N, ceil(num_groups_k/4)]`), while the u2 kernels
index one byte per group. The runtime wrapper
(`lib/Runtime/real/matmul_nbits.cpp`) closes that gap: for `bits=2` asym it
unpacks the packed blob to one-byte-per-group via
`hip_matmul_nbits_unpack_zp_u8_2bit()` (pointer-keyed cache, once per
`zero_points`) and passes it back as `pre_unpacked_zp_u8`. This test exercises
both layouts:

1. **one-byte-per-group** (`*_zeros_u8`) passed directly as `zero_points`
   with `pre_unpacked_zp_u8 = null` — the plain direct-call convention used by
   the benchmark loop.
2. **ONNX-packed** (`*_zeros_packed`) unpacked once via the 2-bit unpack
   kernel and fed back as `pre_unpacked_zp_u8` — the real runtime integration
   path. A dedicated correctness check (`u2 packed-zp real-model path`, not
   benchmarked) confirms the unpack reproduces `*_zeros_u8` byte-for-byte and
   that the resulting GEMM matches the reference.

See `gen_matmul_nbits_u2_data.py`'s module docstring for the exact file
layout.

## Usage

```bash
# Single shape (default group_size=128, with zero points)
make test SIZE=1x2880x2880 GS=128

# Decode shape (M=1), no zero points
make test SIZE=1x2880x5120 GS=128 NO_ZEROS=1

# A handful of preset MoE-ish shapes (prefill + decode)
make test_all

# Decode-focused benchmark across a few K/N pairs
make bench_decode

# Full model shape sweep (every M x (K,N) pair in the JSON)
make test_model MODEL_JSON=test_model/gpt_oss_20b.json GS=128 NO_ZEROS=1
```

Each shape prints:

```
  --- u2 ---
  Pre-warmup ... Warmup ... Benchmarking ...
  Median: X.XXXXXX ms, Y GFLOPS, Z GB/s
  Verify: N/N OK, ... PASS

  --- u4 ---
  ... (same fields)

  === u2 vs u4 comparison ===
  Kernel     Median(ms)      GFLOPS      GB/s    B size(KB)    Verify
  u2         ...             ...         ...     ...           PASS
  u4         ...             ...         ...     ...           PASS
  u2/u4 weight size: 50.0%   u2 vs u4 speed: 1.1x (u2 faster)

  --- u2 packed-zp real-model path ---     (only with zero points)
  Unpack check: N/N zero_points match one-byte-per-group   PASS
  GEMM (packed-zp): N/N OK, ...   PASS
```

Manual steps (equivalent to what `make test` does):

```bash
python gen_matmul_nbits_u2_data.py 128x2880x5120 --group-size 128 --dir data
make direct
./build/test_direct.exe 128x2880x5120 128 data
```

## Files

- `gen_matmul_nbits_u2_data.py` — generates a shared FP16 `A` plus
  independently-quantized u2 and u4 copies of `B` (same shape), scales,
  optional zero points (both one-byte-per-group and ONNX-packed for u2), and
  NumPy references for both.
- `gen_model_data_u2.py` — calls the above once per shape in a model config
  JSON (mirrors `../gemm_fp16u4/gen_model_data.py`); skips shapes where
  `K % 32 != 0`.
- `test_matmul_nbits_u2.cpp` — loads both quantized copies, benchmarks +
  verifies each via `hip_matmul_nbits()`, prints the comparison table, and
  validates the packed-zp real-model path. Supports single-shape (default)
  and `--model <json>` sweep modes.
- `Makefile` — same direct-compile-and-link style as `../gemm_fp16u4`
  (compiles `matmul_nbits_kernel.hip` and the test driver as two TUs, links
  them — no prebuilt `.lib` needed). Defaults to `--offload-arch=gfx1150`
  and `HIP_SDK=C:\AMD\Rocm\7.1`; override on the command line, e.g.
  `make test OFFLOAD=--offload-arch=gfx1151 HIP_SDK=C:\path\to\rocm`.
