# elementwise_bcast Xany/Wany/Yany standalone test

Standalone `hipcc`-only test for `hip_elementwise_binary_bcast`: Add, Mul, Min
and Max of two 4-D operands with per-axis broadcasting, in fp32, fp16 and
bf16. It does not configure or build hip-ep, ONNX Runtime, LLVM, or MLIR.
`make` compiles `elementwise_binary_kernel.hip` into the leaf; the CMake
package build links `custom_kernels_<arch>.dll` instead. Inputs and the CPU
reference are generated in-process -- no python, no data files.

One leaf covers all three dtypes (`Xany_Wany_Yany`, as for `cast/` and
`conv/`): the launcher is one function with one dispatch on the shapes, the
alignment and the volume, and the dtype only selects the kernel
instantiation. Both operands and the output always have the same dtype, so
the three combinations share every case, and a separate leaf per dtype would
repeat the same driver three times.

```
make test        HIP_SDK=%HIP_SDK% [OFFLOAD=--offload-arch=gfx11-generic] [COVERAGE=1|2|3] [TIMING=standard]
make test_custom HIP_SDK=%HIP_SDK% ARGS="--n 8 --c 256 --h 56 --w 56 [--lhs full|n|c|h|w|scalar] [--rhs full|n|c|h|w|scalar] [--op add|mul|min|max] [--dtype fp32|fp16|bf16] [--align aligned|dense|output]"
make clean
```

| Target | Meaning |
|---|---|
| `test` | Coverage-tiered sweep -- see "Shape coverage" below. |
| `test_custom` | Runs one given case through the same check and timer: `--n --c --h --w` (the output shape) are required. `--lhs` (default `full`) and `--rhs` (default `c`) give each operand's shape: `full` is the output's shape, `n`, `c`, `h` or `w` keeps only that axis (a per-axis bias or scale), `scalar` has one element. `--op` (default `add`), `--dtype` (default `fp16`) and `--align` (default `aligned`) are optional. An axis that neither operand keeps must have the extent 1. Unknown, repeated, incomplete or unsupported arguments exit 2. |
| `clean` | Removes `out/`. |

## What it checks

Against a CPU reference that applies the operation in fp32, as the kernel
does, and rounds to the output type: every comparison is bit-exact, and a NaN
matches any NaN. Min and Max are the kernel's own (`a < b ? a : b`,
`a > b ? a : b`), which does not propagate a NaN the way a NaN-propagating
minimum does, so the inputs hold a NaN only for Add and Mul.

The kernel picks a path from the operand shapes, the buffers' alignment and
the output volume: both operands with the output's shape; one operand with
the output's shape and the other varying along one axis, with a power-of-two
extent below that axis; the same with any extent below the axis that is a
whole number of 16-byte vectors (8 elements of fp16 and bf16, 4 of fp32); and
the general kernel. The shapes cover each of them for each dtype, including
extents just on either side of a vector (7x7, 14x14, 28x28, 56x56 maps, an
extent of exactly 8 and of exactly 4) and a channel axis other than axis 1.

Every shape runs:

- in both operand orders (the dense operand on either side);
- for every op;
- with aligned buffers, with the dense operand one element off a 16-byte
  boundary, and with the output one element off;
- with inputs near zero and with values spread over a few units, and with
  every seventh element replaced by an extreme (+-0, +-1, the smallest normal,
  the largest finite value, +-Inf, and a NaN for Add and Mul).

The output has guard regions on both sides that must stay untouched, and a
fill pattern that fails any element left unwritten. Three contract cases run
per dtype: an empty output returns success without touching memory, an output
beyond 32-bit indexing returns -2 before any access, and an unsupported dtype
returns -1.

## Shape coverage (3 tiers)

Every tier runs all dtypes, ops, operand orders and alignment / input
variants on each selected shape (the two large shapes run only the aligned
case with the spread values). `COVERAGE` only thins the shape list: tier1
covers the per-channel forms on 28x28, 14x14 and 7x7 maps, a rank-3 operand
and the flat form; tier2 adds the 56x56 map, other channel axes, extents of
exactly 8 and 4, a scalar, both operands broadcasting, and a 16 x 128 x 28 x
28 shape that is timed; tier3 (default) adds a power-of-two extent, a channel
on the last axis, and an 8 x 256 x 56 x 56 shape that is timed. The program
prints the number of shapes it runs, and a successful run prints one `PASS
relL2=...` line per case and ends with `ALL PASS`.

## `MODE` and `out/results.csv`

This kernel has no autotune table and no tuner: `MODE` is accepted for
consistency with other leaves and only fills the `mode` column.

Each case appends one row to `out/results.csv` (see `unit_test/README.md`),
`shape` like `2x24x28x28+1x24x1x1` (the two operand shapes) and `config` like
`fixed:fp16:add:dense_lhs:aligned:random` (the op has one fixed path per
shape). A case that passes is timed after its check with the shared timer
(`common/hip_adaptive_timing.h`) on rotated copies of its inputs, and its row
carries that `time_ms`; `TIMING` selects the timer preset. A shape below 2^20
elements is checked but not timed (`time_ms` 0): a call takes only a few
microseconds, too short for the event timer on some HIP runtimes.

`OFFLOAD` defaults to `--offload-arch=gfx11-generic` (gfx115x); pass
`--offload-arch=gfx1170` or `gfx1171` for those chips.
