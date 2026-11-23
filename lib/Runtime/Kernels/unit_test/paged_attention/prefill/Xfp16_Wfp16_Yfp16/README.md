# FP16 paged attention prefill

This native leaf calls `hip_paged_attention_f16` from the public header. It checks
FP32 attention (including split-half full-D RoPE and FP16 cache storage), readonly
inputs/metadata, complete pool contents and allocation guards. CPU references
are computed in process with up to eight threads. Errors return nonzero.

`make test HIP_SDK=<sdk> COVERAGE=1|2|3 TIMING=screen|standard|confirm MODE=lookup|autotune`
uses the owning shared v3 timer; defaults are COVERAGE=3, TIMING=standard,
MODE=lookup and OFFLOAD=--offload-arch=gfx11-generic. Pass OFFLOAD for another
release target registered by Kernels/CMakeLists.txt, or gfx1170/gfx1171. Build
with `make build`; clean only your private BUILD_DIR with `make clean`.
`ADLX_INCLUDE` enables optional telemetry. Allocations, reference and input uploads
are outside the timer. The timed callback includes validation, Q/new-K RoPE,
append, attention and decode reduce. Each call reuses readonly past; repeated
append writes the same tokens, without accumulating history. Operand rotation
uses separate pool copies with the same immutable Q/K/V/metadata. The actual
working bytes/copies are logged; the 512 MiB allocation cap can limit rotation,
so small cases are not certified cold. HIPDNN_KERNEL_UT_WARM=1 requests one copy.
CPU checking and timing use separate optional GpuExecutionScope phases.

The complete size-checked entry `hip_paged_attention_f16_configured_v2` runs
metadata validation, Q/new-K RoPE, append, attention and reduce. Its final geometry
is `splits=N;bkv=N;hb=N;qw=N;mt=N;nd=N`. Decode accepts explicit power-of-two
splits from 1 through 64, or splits=0 for device-length adaptive selection; its
other fields are zero. Prefill candidates expose the legal HB/QW/MT or HB/ND
layouts of the shared GQA bodies, independently of physical page size. Native
candidate enumeration and production validation govern the supported tuples.
Workspace queries cover the maximum split stride. The no-config public entry
uses the current geometry heuristic and device-length adaptive decode; the
legacy two-int configured API still rejects explicit decode splits=0.

MODE=lookup resolves an exact full native case key from the compiled Page FB.
Table op=paged_attention, PSHD/Fp16, schema2, ABI `paged_fp16_pshd_v2`, actual
chip arch and legal geometry must agree. Versioned winners use `configs_v2`;
legacy arrays remain preserved for migration and are ignored by the v2 loader.
Absent/miss/incompatible/corrupt table
or invalid config is logged as fallback, then runs complete-entry online selection.
MODE=autotune bypasses the table and selects among actual Page launch candidates
with the owning shared screen timer. Each candidate passes the same CPU reference
and storage checks; the selected complete entry then uses requested TIMING.
Online screen timing is selection evidence, separate from final timing.
`HIPDNN_PAGED_ATTENTION_CFG=default` selects actual no-LUT default; `id=N` or
`splits=N;bkv=N;hb=N;qw=N;mt=N;nd=N` pins final v2 geometry, logs fixed:pin
and conflicts with ranking. Legacy two-field requests are converted and validated.
Exact keys preserve q/past arrays, page geometry/mapping, RoPE/position/options
and seed. There is no nearest matching across incompatible ragged cases.

Expected rejection and all-empty cases have no timing and retain a CSV row.
The output threshold inherits the dense leaf: decode relL2<2e-2, prefill <2e-3.
All FP16 cache writes and untouched inputs/storage are checked bit-exactly.

Every coverage tier retains HPG/D geometries, all mapping/rotary/window situations,
B=1/multiple/ragged/empty requests, append edges and all error categories. Tiers
thin only the explicit contexts {31,513,2049}; S={17,31,96} independently of BKV.
Prefill query rows include 3,17,33 and chunked history. Decode includes 16 fixed
splits, with empty splits on short/window-one cases.

`make test_custom` runs one case through the same correctness/timer path:

```
make test_custom HIP_SDK=<sdk> B=3 H=8 G=2 D=128 P=64 S=17 M=8 QLENS=3,0,17 PAST=16,17,18
```

Parameters match ut_interface.json: B=3, H=8, G=2, D=128; P=64 physical pages,
S=17 slots per page, M=8 table columns; QLENS=3,0,17,
PAST=16,17,18 (comma-separated arrays of B nonnegative integers); ROTARY=0,
ROTARY_DIM=0 (driver expands 0 to D), INTERLEAVED=0, MAX_POSITION=4096;
SCALE=0 auto or finite positive, WINDOW=-1 (nonpositive full causal), SOFTCAP=0;
MAP=permuted (ordered/permuted/shared), SEED=1119, ERROR=none, PHASE=prefill.
The phase must match this leaf. H is a positive multiple of G. D is
64/96/128/256, decode H/G is 1/2/3/4/5/6/8/16 plus 9 only at D128.
Shared mapping requires at least one readonly prefix page on active requests;
append pages are private. P/M must hold the chosen mappings. No allocator/COW
is implemented. T=sum(QLENS), cum and post lengths are generated in process.

ERROR selects a reproducible invalid descriptor or device metadata, while
keeping input allocation/reference safe: cum_start/cum_end/cum_order,
negative_past/decode_length/post_overflow/table_small/bad_page/position,
workspace/workspace_alignment/alias/pool_alias, softcap/negative_scale/nan_scale/
inf_scale/nan_softcap/interleaved/partial_rotary/d256_window/geometry/
write_conflict/null_rotary/size_overflow/tensor_alignment. d256_window is a prefill rejection only.
The public header defines asynchronous status and page ownership; the test
checks host return, stream completion and the workspace device status separately.

The host validator runs before allocation/reference generation. write_conflict
needs two active (q>0) requests, including legal B2; it lowers the larger past
and maps its append-start page to the smaller-past request. The injected post
never increases. Metadata/storage errors require T>0; decode_length requires
T>=2 and valid decode geometry, table_small requires M*S<INT32_MAX, position
requires an active post>1. position/null_rotary normalize the injected RoPE
fields to full-D split-half. Inapplicable inputs return exit2 with the reason.
Host descriptor errors can still test T=0; ordinary B1/B2/empty requests remain
legal. Decode rejects ERROR=d256_window early; ordinary D256+window is supported.

External LUT / native ranking (no installed Page table required):

```sh
make build RANK=1 HIP_SDK=<sdk> FLATC=<matching-flatc> FLATBUFFERS_INC=<include> BUILD_DIR=out_rank
# Write seed.csv: leaf,shape,priority,config; one default row per full native
# shape printed by this driver. Use CSV quoting where necessary.
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py rank-catalog --candidates seed.csv --leaf-root <unit_test> --build-dir out_rank --out catalog.log
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py rank-candidates --candidates seed.csv --catalog catalog.log --out candidates.csv
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py rank-measure --candidates candidates.csv --leaf-root <unit_test> --build-dir out_rank --out rank.log
# Empty JSON is the reproducible base for a new table; copy it privately and
# set gpu_arch to the actual measured chip. Only representative points qualify.
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py rank-build --base private-base.json --rank-log rank.log --out private.json
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py rank-report --from-lut private.json --rank-log rank.log --out report.csv --md report.md
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py compile --out private.json --fb-out private.fb --flatc <matching-flatc>
make test_custom MODE=lookup LUT_FB=<absolute-private.fb> LUT_ARCH=<actual-chip> HIP_SDK=<sdk> FLATC=<matching-flatc> FLATBUFFERS_INC=<include> BUILD_DIR=out_lookup <same-shape-args>
# Existing JSON -> default + old final geometry, preserving all unrelated keys:
python ../../../../hip/autotune/paged_attention/scripts/update_lut.py rank-seed --from-lut private.json --out refresh.csv
```

The CSV literal default line for this leaf's default custom case is:

```csv
leaf,shape,priority,config
paged_attention/prefill/Xfp16_Wfp16_Yfp16,prefill_B3_H8_G2_D128_P64_S17_M8_q3+0+17_past16+17+18_rope0_rD0_inter0_pos4096_scale0_win-1_cap0_mappermuted_shared0_seed1119_errnone,0,default
```

RANK=1 emits UT_CANDIDATES for HIPDNN_UT_RANK_CATALOG=1 and executes the existing
ut-rank-v3 screen/standard/conditional-confirm engine for HIPDNN_UT_RANK=<csv>.
Each candidate passes CPU/storage checks before timing; native ID0 is the real
no-table default. HIPDNN_UT_RANK_REFERENCE=1 selects independent sequential
confirm. Shared UT emits config-bound timing and UT_RANK selection records;
ordinary CSV is not used to mislabel multi-candidate rank work. Default is
required, invalid IDs/geometry/CSV fail, duplicate final geometry is removed.
The optional --launcher-json prefix of the common rank tools wraps the real
foreground executable for managed GPU execution; CPU catalog needs no lease.
JSON/FB parsing and embed happen at preparation/build, never in timed launches.
These commands write the selected output table; they do not install it globally.


`SHARED_PREFIX=0` / `--shared-prefix 0` uses one readonly page for MAP=shared;
a positive count shares that many tokens, must be page-aligned and readonly
for every active request. Canonical Q/newKV/history generation is independent
of the physical page mapping. Shared history has identical logical bytes.

Optional original-Flash diagnostic build (single case, not a second leaf):

```sh
make build COMPARE=1 FLASH_SOURCE=<exact-original-gqa.hip> HIP_SDK=<sdk> FLATC=<flatc> FLATBUFFERS_INC=<include> BUILD_DIR=out_compare
out_compare/test_paged_attention.exe --compare 1 <all-custom-shape-arguments>
```

The UT adapter isolates original production symbols and exposes its actual
in-process config cache, without modifying math. The original public Flash
entry performs its maintained no-LUT online tuning in untimed preparation.
The Page comparison's fallback is explicitly the original no-table default;
set HIPDNN_PAGED_ATTENTION_CFG=default for a pinned no-table comparison, or
supply a compatible FB for a real hit. Each side uses the same logical FP16
Q/newKV/history, lengths, FP16 cos/sin and scale. Logical byte identity is
recorded as FNV1a64; source/build evidence uses SHA256. Dense decode compacts
only active requests and stays batched; ragged prefill uses one stream/request
loop. Complete dense includes original RoPE and both append launches.
Original RoPE storage differences versus common FP32-then-FP16 rounding are
counted separately, while append is checked bit-exactly against the original
helper's actual output; readonly history/inputs/guards remain exact. Both
complete outputs use the same independent CPU reference and unchanged limits.
Five native timing entries distinguish complete Page/dense, original bare
Flash, and both cores on common post-RoPE FP16 operands; no subtraction of
noisy timings estimates stages. Same rotation count per pair is bounded by
512MiB per side; logs expose actual working/physical bytes and limitations.
Only named shared presets are used. COMPARE_RESULT/COMPARE_INPUT and native
CSV/UT_TIMING_BINDING provide per-entry identity, accuracy and timings.

`--compare-order 0|1` selects Page-first (0, default) or dense-first (1) complete timing order; alternate this diagnostic across pairs.

The public entry uses its geometry default, with device-length adaptive decode
splits. `MODE=autotune` explicitly calls production `preparePagedAttention` before
complete-call validation/timing. Preparation uses the shared GQA production tuner
on the validated buffers and permits synchronization; configured hot calls do not
tune or read metadata back. `HIPDNN_PAGED_ATTENTION_PRODUCTION_PREPARE=1` also
selects that route for the original Flash comparison (conflicts with public/pin).
The C `hip_paged_attention_f16_prepare_v2` API returns a size-checked config for
caller-owned reuse. V2 decode splits=0 denotes adaptive selection; the original
C two-int configured ABI continues to reject explicit splits=0.

For a complete-entry comparison, set HIPDNN_UT_COMPARE_COMPLETE_ONLY=1 to
skip the bare/core diagnostic timings. Both complete timings, CPU precision
and storage/readonly/guard checks remain; omitted timings are JSON null.

In a COMPARE=1 RANK=1 build, HIPDNN_UT_COMPARE_RANK=page|flash ranks the selected
side's complete callback through the existing native screen/standard/conditional
confirm engine. Page enumerates its own production candidates and default.
Flash enumerates its original production candidates, including supported WMMA
decode and D96 routes; its actual online selection is frozen before timing,
including each active ragged request. UT_RANK contains every candidate and
the measured winner. Ranking conflicts with public/preparation/config pins.

A qualified original Flash winner can be frozen with HIPDNN_UT_FLASH_NATIVE_CFG:
wmma=0|1;splits=N;bkv=N;hb=N;qw=N;mt=N;nd=N. For a per-request prefill selection,
separate each active request's descriptor with a pipe. Native pins must match
the original candidate space and conflict with HIPDNN_UT_FLASH_CFG. Final Page
winners use the existing HIPDNN_PAGED_ATTENTION_CFG. Config queries and search
remain outside complete-call event intervals.
