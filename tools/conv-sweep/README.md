# conv-sweep

A scratch tool, not for merging: it times every tile configuration of hip-ep's conv kernel on a list of real
convolution shapes and compares them with what `hip_conv` picks by default.

It includes `conv_kernel.hip`, forces each candidate configuration, checks the result against the one-thread-per-output
reference kernel in the same file, and times it with hipEvents (median over interleaved rounds). The `default` row is
`hip_conv` itself, so it measures whatever dispatch the kernel file you build against has.

## Run

Windows, Git Bash, TheRock `hipcc`, a gfx1151 GPU (`ARCH=` for another target).

```
THEROCK=/path/to/_therock bash tools/conv-sweep/build.sh [path/to/conv_kernel.hip]
tools/conv-sweep/sweep.exe tools/conv-sweep/shapes.txt out.csv --batches 1,8 --dtype f16
python tools/conv-sweep/sweepan.py tools/conv-sweep/shapes.txt out.csv
```

Other options: `--configs a,b,...`, `--iters 20`, `--dtype f16|f32|bf16`. Run it on an idle GPU.
`shapes.txt` lines are `id cin h w cout g k0 k1 s0 s1 p0 p1 d0 d1 oh ow`. It holds 45 non-depthwise conv shapes of
11 vision models, ids by kind (d = dense, s = strided 1x1, g = grouped, m = channel multiplier).
`shapes_guard.txt` holds large shapes the original dispatch was calibrated on; a change should not make them slower.

Output columns: `shape,dtype,B,cfg,blocks,us,relerr,status`.

## Caveats

- `sweep.hip` forces configurations by instantiating the kernel's templates, so it needs small adaptations when the
  templates or `hip_conv`'s signature change. If a build fails, that is the place to look.
- With `hip_conv_ws` in the kernel file, `-DSWEEP_SPLITK` adds split factors as extra configurations. Another
  workspace API needs its own hook there.
- Tuned and tested on gfx1151 only.
