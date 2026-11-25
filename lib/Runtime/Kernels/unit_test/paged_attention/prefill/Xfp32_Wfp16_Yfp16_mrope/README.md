# Paged MRoPE prefill

FP32 Q/K and cosine/sine, three-stream int64 positions, FP16 V/pools/output.
The production fused entry performs non-interleaved MRoPE (partial or full D),
FP16 conversion and paged append in one prepare kernel, then shared attention.
Positions select rotation rows independently from past lengths and page tables.
Out-of-cache positions copy the affected pair, matching standalone MRoPE.

`make test COVERAGE=1|2|3 TIMING=screen|standard|confirm MODE=lookup|autotune`
retains both layouts, partial/full dimensions, all position patterns, page maps,
ragged/inactive batches and rejection categories in every tier. Higher tiers
add longer contexts and page sizes. CPU FP32 attention precision thresholds
remain 0.002 prefill / 0.02 decode; pools and actual unfused-chain outputs must
match exactly, inputs and guards remain intact.

`make test_custom MROPE_LAYOUT=1 MROPE_SECTIONS=11,11,10 ROTARY_DIM=64
MROPE_SCALE=1 POSITION_MODE=distinct COMPARE=1` compares complete fused and
two production MRoPE + two Cast + Page calls, using the same selected attention
config and coherent rotating pools. No whole-model execution is involved.
Other geometry, map, window and error arguments follow `ut_interface.json`.
`POSITION_MODE=same|distinct|out-of-cache`; ROTARY and INTERLEAVED stay zero.

`RANK=1` enables existing UT_RANK candidate input/ranking and LUT refresh;
external `LUT_FB` is accepted, missing entries invoke the real production tuner.
Canonical shape keys include FP32 leaf identity, MRoPE layout/sections/scale
and positions. Attention configuration/schema remains Page v2; no fabricated
measurements or automatic copying of old FP16 timings. Release targets use
the existing OFFLOAD selection (`gfx11-generic`, `gfx11-7-generic`).

`--coverage N --list-cases 1` lists the actual suite keys without allocating or
launching GPU work. Every tier retains the explicit RoPE/window categorical
matrix: full context (-1 and 0), window one, a window crossing physical pages,
and a window larger than the context. FP16 covers RoPE off/on; the MRoPE leaf
crosses both layouts and partial/full rotation. D256 positive-window prefill
remains a rejection case; decode D256 positive windows are normal cases.
The native descriptor's reserved SOFTCAP field is a legacy ABI check only;
the HIP op and v2 runtime wrappers have no softcap parameter.
