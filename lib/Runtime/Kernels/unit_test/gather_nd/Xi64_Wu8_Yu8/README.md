# gather_nd / Xi64_Wu8_Yu8

Unit test for `hip_gather_nd` (`hip/gather_nd_kernel.hip`) with int64 indices
(`X`), uint8 data (`W`) and uint8 output (`Y`). uint8 is also the storage of a
bool tensor: the runtime gives every bool element one byte, so a bool
GatherND (e.g. rows of an attention mask) runs this path.

## What it checks

Every output is compared **bit-exactly** with a CPU GatherND reference
written in the test (ONNX semantics, including `batch_dims` and negative
indices). Half of each data tensor holds 0/1 bool values, the other half
arbitrary bytes, so a kernel that reads or writes the wrong element width
shows up as a mismatch.

Categorical situations, each run at every size of the tier:

| Case | Data | Indices | Meaning |
|---|---|---|---|
| `rows` | `[R, C]` | `[M, 1]` | pick whole rows |
| `elements` | `[R, C]` | `[M, 2]` | pick single elements |
| `batched` | `[2, R, C]` | `[2, M, 1]`, `batch_dims = 1` | rows per batch |
| `planes` | `[2, R, C]` | `[M, 2]` | pick rows of a rank-3 tensor |

Every index list starts with the first and the last position of each indexed
dim, once as a positive and once as the equivalent negative index; the other
rows are random, half of them negative.

- Every element is written and nothing outside the output: each case runs
  twice with two different fill bytes and 64 guard bytes on each side.
- An unsupported dtype (u16) returns -1 and leaves the output untouched. The
  kernel prints an "unsupported dtype" line for it; that is expected.

## Coverage

`--coverage N` (or `HIPDNN_UT_COVERAGE`, default 3) picks the sizes
`R x C, M index rows`; the unsupported case runs at every tier.

| Tier | Sizes |
|---|---|
| 1 | 7 x 33 (M = 5), 512 x 512 (M = 300, a 512-token causal mask) |
| 2 | + 64 x 256 (M = 64) |
| 3 | + 2048 x 2048 (M = 2048) |

## Targets

```
make test        HIP_SDK=<sdk> [OFFLOAD=--offload-arch=<gfx>] [COVERAGE=1|2|3]
make test_custom HIP_SDK=<sdk> ARGS="--coverage 1"
make build ...
make clean
```

The op has no autotune table and no tuner; `MODE` only labels
`out/results.csv`, and every row's `config` is `fixed:gather_nd`.
