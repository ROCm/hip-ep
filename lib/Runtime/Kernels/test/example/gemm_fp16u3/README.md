# fp16 x u3 MatMulNBits — Test & Benchmark vs fp16 x u4

Benchmarks and verifies the exploratory `bits=3` (uint3, continuous-bitstream
packed) quantized-weight kernel in `matmul_nbits_kernel.hip`, and compares it
head-to-head against the existing `bits=4` (uint4, nibble-packed, ONNX
`MatMulNBits` convention) kernel on **identical** (M, K, N, group_size)
shapes.

This is a sibling of `../gemm_fp16u4` — same build style, same Makefile
conventions — but a single test binary loads both a u3-quantized and a
u4-quantized copy of the same-shaped weight matrix (sharing the same `A`)
and runs both kernels back-to-back, printing a side-by-side comparison.

## Format background

- **u4** (existing, ONNX `MatMulNBits` convention): nibble-packed, `B_packed`
  is `[N, K/2]` uint8, low nibble = value `2k`, high nibble = value `2k+1`.
- **u3** (new, *not* an ONNX convention — exploratory format defined in
  `matmul_nbits_kernel.hip` Section 1c/1d): a continuous per-row 3-bit
  bitstream. Value `k` occupies bits `[3k, 3k+3)` of row `n`'s byte buffer,
  LSB-first. Row stride is `ceil(K*3/8)` bytes — 37.5% smaller than u4's
  `K/2` bytes for the same `K`, i.e. u3 should need to move ~62.5% of the
  memory traffic u4 does for the weight matrix. Three dispatch paths,
  mirroring bits=8 (see Section 2d for the WMMA kernel and its alignment
  proof):
  1. **WMMA fast path** (M >= 16, K % 32 == 0, group_size % 32 == 0 and
     >= 32): the 3-bit stream is decoded *inline* in the WMMA K-loop, no
     separate dequant buffer — every 8-value chunk packs into exactly 3
     bytes (24 bits), so the decode is a single 32-bit register op with no
     cross-byte-boundary shifting.
  2. **GEMV** (decode/small-M, block_size power of two >= 32, K % 32 == 0):
     vectorized 96-bit (`uint3`) loads, 32 weights per transaction.
  3. **Naive fallback**: anything meeting neither gate above.

Because the two formats have incompatible zero-point plumbing inside
`hip_matmul_nbits()` (u4's asym path expects pre-unpacked uint8 *and* fp16
zero-point buffers to correctly hit all of WMMA / GEMV / naive dispatch),
the generator emits each alongside a shared `A`, and the test binary wires
each kernel's call according to its own zp contract — see
`gen_matmul_nbits_u3_data.py`'s module docstring for the exact file layout.

### Zero-point handling for u3 (two conventions, both validated)

ONNX packs `zero_points` at `bits` bits, so a real 3-bit model ships them as a
**continuous per-row 3-bit stream** (`[N, ceil(num_groups_k*3/8)]`), while the
u3 kernels index one byte per group. The runtime wrapper
(`lib/Runtime/real/matmul_nbits.cpp`) closes that gap: for `bits=3` asym it
unpacks the packed stream to one-byte-per-group via
`hip_matmul_nbits_unpack_zp_u8_3bit()` (pointer-keyed cache, once per
`zero_points`) and passes it back as `pre_unpacked_zp_u8`. This test exercises
both layouts:

1. **one-byte-per-group** (`*_zeros`) passed directly as `zero_points` with
   `pre_unpacked_zp_u8 = null` — the plain direct-call convention used by the
   benchmark loop.
2. **ONNX-packed** (`*_zeros_packed`) unpacked once via the 3-bit unpack
   kernel and fed back as `pre_unpacked_zp_u8` — the real runtime integration
   path. A dedicated check (`u3 packed-zp real-model path`, not benchmarked)
   confirms the unpack reproduces `*_zeros` byte-for-byte and that the GEMM
   result matches the reference.

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
  --- u3 ---
  Pre-warmup ... Warmup ... Benchmarking ...
  Median: X.XXXXXX ms, Y GFLOPS, Z GB/s
  Verify: N/N OK, ... PASS

  --- u4 ---
  ... (same fields)

  === u3 vs u4 comparison ===
  Kernel     Median(ms)      GFLOPS      GB/s    B size(KB)    Verify
  u3         ...             ...         ...     ...           PASS
  u4         ...             ...         ...     ...           PASS
  u3/u4 weight size: 62.5%   u3 vs u4 speed: 1.1x (u3 faster)
```

Manual steps (equivalent to what `make test` does):

```bash
python gen_matmul_nbits_u3_data.py 128x2880x5120 --group-size 128 --dir data
make direct
./build/test_direct.exe 128x2880x5120 128 data
```

## Files

- `gen_matmul_nbits_u3_data.py` — generates a shared FP16 `A` plus
  independently-quantized u3 and u4 copies of `B` (same shape), scales,
  optional zero points, and NumPy references for both.
- `gen_model_data_u3.py` — calls the above once per shape in a model config
  JSON (mirrors `../gemm_fp16u4/gen_model_data.py`); skips shapes where
  `K % 32 != 0`.
- `test_matmul_nbits_u3.cpp` — loads both quantized copies, benchmarks +
  verifies each via `hip_matmul_nbits()`, prints the comparison table.
  Supports single-shape (default) and `--model <json>` sweep modes.
- `Makefile` — same direct-compile-and-link style as `../gemm_fp16u4`
  (compiles `matmul_nbits_kernel.hip` and the test driver as two TUs, links
  them — no prebuilt `.lib` needed). Defaults to `--offload-arch=gfx1150`
  and `HIP_SDK=C:\AMD\Rocm\7.1`; override on the command line, e.g.
  `make test OFFLOAD=--offload-arch=gfx1151 HIP_SDK=C:\path\to\rocm`.
