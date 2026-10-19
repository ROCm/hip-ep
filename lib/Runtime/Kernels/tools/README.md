# Kernel resource check

`check_kernel_resources.py` compiles every custom-kernel source (the
`_kernel_sources` list in `../CMakeLists.txt`) with the product flags
(`-O2`, `--offload-arch=gfx11-generic` and `gfx1170`, as in
`../cmake/hip_utils.cmake`) and compares each kernel against the baselines next
to it, `kernel_resources_<arch>.json`: VGPRs, scratch bytes per lane, spilled
VGPRs and occupancy from the compiler's resource remarks, plus the instruction
count and the number and total width of the LDS loads in its ISA. It needs only
the HIP SDK; no CMake build and no GPU.

## When to run it

- **Every compiler bump** (the ROCm / TheRock version in `cmake/deps.txt`):
  run it with the new SDK before anything else. A new clang can need more
  registers for the same source, and a kernel whose register budget is close to
  that need starts spilling; or it can stop assuming something the source never
  stated and emit narrower or more instructions. Neither shows up in
  correctness tests or in online autotune (which simply stops picking the
  kernel), only in the offline LUT path, which keeps picking it. ROCm 7.14 ->
  7.16 turned one MatMulNBits prefill tile from 3 spilled values into 138
  (3.5-4x slower on every model whose LUT picks it), and stopped giving
  `__shared__` arrays 16-byte alignment they never declared, which split every
  vector LDS read in two (QDQ Gemm decode 1.3x slower, fp32 Conv 1.4x).
- After changing a kernel's tile shape, unrolling, accumulator layout or
  `__launch_bounds__` / `amdgpu_waves_per_eu`.

```
python lib/Runtime/Kernels/tools/check_kernel_resources.py --rocm <HIP SDK root>
python lib/Runtime/Kernels/tools/check_kernel_resources.py --rocm <HIP SDK root> --squeeze 112
python lib/Runtime/Kernels/tools/check_kernel_resources.py --rocm <HIP SDK root> --squeeze 232 --squeeze-max 256
```

`--rocm` defaults to `$HIP_PATH`. Takes a few minutes with the default job
count (half the cores); `--sources gqa_kernel,matmul_nbits_kernel` limits it.

## Reading the report

| Line | Meaning | Action |
|---|---|---|
| `ERROR NEW SCRATCH` / `MORE SCRATCH` | a kernel spills (or uses a private array) where the baseline did not, or more | find the kernel's launch path; if a LUT or heuristic can pick it, fix it (see below) before the bump |
| `ERROR NEW KERNEL WITH SCRATCH` | an instantiation that is not in the baseline uses scratch | as above, or allowlist it with a reason |
| `ERROR NARROWER LDS LOADS` | more than 10% more LDS load instructions moving no more bytes | usually a `__shared__` array read as vectors without a declared alignment: add `__align__(16)` |
| `WARN OCCUPANCY DROP` | more VGPRs crossed an allocation step | usually harmless; matters for VGPR-limited kernels on 1024-VGPR parts |
| `WARN MORE INSTRUCTIONS` | over 10% more instructions (`--insn-growth`) | compare the kernel's hot loop in `-S` output of both compilers; often index math the new compiler no longer simplifies |
| `TIGHT` (with `--squeeze N`) | spill-free today with N to `--squeeze-max` VGPRs (default 128), spills when only N are available | less than (budget - N) registers of margin; the next compiler may tip it over. Use 112 / 128 for 1024-thread or `waves_per_eu(8)` kernels, 232 / 256 for the ones at the architectural limit |

Exit status is 1 on any error. `--strict` turns the warnings into errors.

What it cannot see: the instruction count is per kernel, so a regression in
the hot loop of a long kernel can stay under the threshold (ROCm 7.16 added 32
useless VALU per KV step to the GQA d128 prefill loop, +7% of the loop but ~2%
of the kernel). It also cannot tell whether a kernel is reachable: an error on
an autotune candidate nobody picks matters less than one on a LUT pick. Treat
the report as the list of kernels to look at, and still benchmark the LUT
picks (A/B against the previous compiler) before taking the new SDK.
`--squeeze` needs `-mllvm -amdgpu-stress-vgpr`, which clang 24 (ROCm 7.16) has.

## Updating the baseline

Once the report is understood (regressions fixed, or accepted with a reason),
rewrite the baselines with the new compiler:

```
python lib/Runtime/Kernels/tools/check_kernel_resources.py --rocm <HIP SDK root> --update
```

Kernels that use scratch in the new baseline are added to its `allow` map with
a generated reason; replace it with the real one (pre-existing debt, a tuning
candidate the tuner never picks, ...). Commit the JSON together with the
compiler bump so the next bump compares against it.

## What usually fixes a regression

- **Declare the real block size.** A kernel without `__launch_bounds__` is
  compiled for 1024-thread workgroups: 8 waves per SIMD, so a 128-VGPR cap on
  gfx11-generic / gfx1150 / gfx1152 / gfx1170 (192 on gfx1151) even if it is
  only ever launched with 64 threads. An `amdgpu_waves_per_eu` request below
  that implied minimum is silently ignored.
- **Size occupancy requests for the 1024-VGPR file.** `amdgpu_waves_per_eu(N)`
  means 1024 / N VGPRs in the product binary (gfx11-generic), not the 1536 / N
  of gfx1151.
- **Keep a margin.** A kernel that needs about as many VGPRs as its budget
  spills as soon as a compiler is slightly less frugal. Shrink the live set
  (hold the smaller operand, recompute addresses) rather than relying on it.
- **No data-dependent `break` in loops over register arrays.** It makes the
  trip count unknown, the array index a runtime value, and the array private
  memory (scratch without a single spilled register).
- **Declare the alignment of `__shared__` arrays read as vectors.** Write
  `__shared__ __align__(16) float tile[...]`; the element alignment is all the
  language promises, and it is what clang 24 uses.
