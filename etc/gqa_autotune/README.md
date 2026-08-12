# GQA offline autotune LUTs

`gfx1151.json` is the reviewable source and `gfx1151.fb` is the FlatBuffer
consumed by the runtime. The data was generated from the `Autotune-Best` sheet
in `GQA_Shapes_Autotune.xlsx` and expanded through each row's `src_rows`.

Compatibility:

- GPU architecture: `gfx1151`
- HIP runtime version: `70151803` (`hipcc 7.1.51803`)
- GQA kernel ABI: `gqa-v1`
- KV-cache format: FP16

The current table contains 66 exact entries and 58 conservative bucket
entries:

- Decode: 33
- Prefill v5: 4
- Prefill v7: 15
- Prefill v8: 14

Bucket entries use inclusive power-of-two upper bounds for `seq_q` and
`seq_kv`, while preserving the measured static geometry and `max_seq`:

- Decode: 28 full-attention entries; sliding-window decode is excluded.
- Prefill v5: 4 entries for `seq_q` buckets 128 and 512.
- Prefill v7: 15 entries preserving the measured `NW=2/4` transitions.
- Prefill v8: 11 deduplicated entries using `ND=4`, `MT=1`, and `BKV=32`.

These are conservative first-pass buckets derived from measured winners. They
do not wildcard `max_seq` or extrapolate to unmeasured head geometries.
Sliding-window decode needs additional lower/middle/upper measurements before
bucket entries can be added safely.

`max_seq` remains part of the exact key. The measurements show that two
sliding-window decode shapes with the same effective KV length can select
different winners when their backing cache strides differ.

To package this LUT into EPContext, set the MorphiZen provider option:

```text
gqa_autotune_lut=/path/to/etc/gqa_autotune/gfx1151.fb
```

To regenerate the binary with `flatc`:

```bash
flatc --binary --strict-json -o /tmp \
  schemas/gqa_autotune.fbs etc/gqa_autotune/gfx1151.json
cp /tmp/gfx1151.bin etc/gqa_autotune/gfx1151.fb
```
