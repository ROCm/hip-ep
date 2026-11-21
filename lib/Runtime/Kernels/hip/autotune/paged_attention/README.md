# PageAttention native LUT

The host resolver prepares a validated final geometry before calling the pure-C
`hip_paged_attention_f16_configured` complete public API. The original
`hip_paged_attention_f16` keeps its no-config default. The native leaves own
lookup/online preparation exactly as the GQA native wrapper does; no ONNX/EP
integration is included. Resolver loading/JSON compile/embed is never performed
inside timed launches. Native online selection uses the existing shared screen
timer and final calls use the requested preset. Ranking uses ut-rank-v3.

`lut/empty.json` is the reproducible new-table base. Copy it privately and set
`gpu_arch` to the actual measured device. The Page table uses schema2,
`paged_fp16_pshd_v2`, PSHD layout and Fp16 identity. Full native shape strings
include phase, B/H/G/D, P/S/M, every q/past, RoPE options/max position, scale,
window, softcap, mapping/shared prefix and seed. Exact matching preserves ragged
and shared-page semantics; no nearest-neighbour claim is made. Configs contain
final `splits`/`bkv`/`hb`/`qw`/`mt`/`nd` in `configs_v2`, independent of native
IDs. Legacy `configs` and `fallbacks` remain preserved by migration; the v2
loader rejects schema1 and reads only the versioned config pool. Untuned/missing keys remain
computable through safe no-table selection. Dense GQA tables are incompatible.

Use the existing common command family in `scripts/update_lut.py`: rank-catalog,
rank-candidates, rank-seed (JSON only), rank-measure, rank-report, rank-build,
rank-reference-candidates and compile. The two native leaf READMEs give full
commands and default CSV rows. rank-build preserves headers, weights, fallbacks,
config indices/order and unrelated points. `make -C unit_test/common test`
includes exact Page ragged-key/duplicate-index roundtrip checks. Structural
roundtrips do not establish device correctness or performance qualification.

Make's RANK=1 builds the UT-only production adapter and empty payload. Ordinary
LUT_FB overrides are explicitly checked and embedded via the existing native
#embed pattern. The production CMake family registry uses the existing
_emit_lut_registry/file(READ HEX) path. No table is installed by a private UT.

V2 decode `splits=0;bkv=0;hb=0;qw=0;mt=0;nd=0` is the production device-length
adaptive default with a device-selected launch bound and fixed partial stride. Positive splits preserve explicit geometry, including 32/64 splits. Legacy two-field
configs and schema-v1 migration keep the positive-split contract.
`preparePagedAttention` returns exact lookup or explicitly calls the C
`hip_paged_attention_f16_prepare_v2` online preparation API on a miss. Preparation
may synchronize and mutate valid buffers; reuse its config through configured_v2
for hot calls. Invalid device metadata prevents tuning and all visible writes.
