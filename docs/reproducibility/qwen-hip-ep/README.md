<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Native QSA audit and paper handoff

<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->
## Latest eight-selector performance

This packet's identities, performance, accuracy and memory files remain the historical four-selector evidence; the eight-selector evidence is a separate packet.

**Decode throughput improved over four selectors at every measured length.** Prefill is slower at 4K/16K and slightly faster at 32K. All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change.

| Context | Eight vs four: prefill | Eight vs four: decode |
|---|---:|---:|
| 4K | -5.21% | +0.71% |
| 16K | -6.31% | +1.94% |
| 32K | +0.92% | +5.31% |

Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. Both prefill and decode use QSA at all eight selected layers.

The fixed 50-case BFCL component pilot scores Eight 29/50 versus reused Original 26/50; no new Original generation was run. This is not the official overall BFCL score. Eight GSM8K and application-memory comparisons remain unmeasured. The source checkpoint still misses its fixed 32K PPL gate; this pilot does not establish accepted long-context quality. See the [eight-selector report](../../experiments/qwen-qsa-eight-selector-results.md) and [raw-data paper handoff](../qwen-eight-hip-ep/README.md).
<!-- QWEN_EIGHT_PERFORMANCE_END -->

This packet records the Qwen/Qwen3.8-27B port to hip-ep on **Radeon 8060S,
gfx1151, Windows**. It includes the frozen selector identity, raw measurements,
protocols, validation receipts and explicit missing results. The branch is local;
neither these results nor the code have been pushed or published by this audit.

**Audit status:** the merged candidate passes 736 compiler tests, eight CPU unit
tests and the dedicated kernel/operator/full-model controls. The selected memory
comparison is batch1/4K, with both variants complete. The user canceled the
remaining memory campaign. Four completed cells are retained in total; the
interrupted batch8/4K default probe and all unrun cells are marked canceled.

Start with [AUDIT.md](AUDIT.md) for review findings and
[PR_DESCRIPTION.md](PR_DESCRIPTION.md) for the proposed pull request. The stacked
change also contains the preceding OLMo implementation and its experiments;
those results remain separate in the
[OLMo report](../../experiments/olmo-qsa-native-gqa-results.md) and
[agent evaluation report](../../experiments/olmo-qsa-agent-accuracy.md).

The review branch is `codex/qwen-native-qsa-pr`, checked out locally at
`_local/qwen-qsa/pr-worktree`. This packet is also kept in the main `hip-ep`
checkout so the paper agent can read it here. The preceding measured branch,
`codex/qwen-native-qsa`, retains the historical experiment stack. Unrelated
Markdown and the sibling source project are left unchanged.

## What changed

Four full-attention nodes, at zero-based layers **27/31/35/39**, become
`com.amd::QwenQsaGroupQueryAttention`. The compiler lowers each to one native
runtime call with node-owned state. The complete model remains one ONNX session
and one EP dispatch per forward, as in the default. Internal selector and
attention kernel launches are included in that dispatch and in measured time.

QKV/O projections, all 48 Gated DeltaNet blocks, Q/K normalization, native RoPE,
sigmoid output gate and all other attention nodes retain the same graph and
backbone bytes. Both hip-ep variants use fresh RTN W4A16 group128 projections,
FP16 activations/full KV and FP32 recurrent state. The selector weights and
features use BF16 with FP32 reductions. No training, calibration or recovery
was performed in this port.

The selector projects width5120 to four128-wide query heads and one128-wide key.
It applies three learned zero-centered norms, normalizes token keys before
four-token pooling, and uses partial RoPE at each group's start. Stable top512
historical groups plus the causal current block give at most2052 attended tokens.
The first actual prune is query position2052. Attention reads selected addresses
directly from full native KV buffers. See the
[mathematical and state contract](../../design/qwen-qsa-native-gqa.md).

Runtime scope is gfx1151/wave32, uniform unpadded batches1 through8, full prefill
then single-token in-place decode, canonical text positions and capacity at most
262144. These bounds are validation guards, **not full-model qualification at
every shape**. Full-model performance was measured only at batch1,4K/16K/32K.
Kernel controls at longer positions do not establish full-model memory capacity.

## Completed Qwen results

These are preserved measured-binary results. The merged candidate has separate
correctness checks and has not inherited a performance or accuracy campaign.

Throughput is tokens/second; positive change means faster. Two warmup pairs and
six balanced measured pairs per context used64 common teacher-forced decode
inputs, synchronized phase timers, final-position head and unchanged session/cache
addresses. Intervals are paired bootstrap95% intervals within this campaign,
not independent-run or cross-machine uncertainty.

| Prefix | Default prefill | QSA prefill | Change | Default decode | QSA decode | Change (95% interval) |
|---|---:|---:|---:|---:|---:|---:|
| 4096 | 438.0643 | 419.9710 | -4.1303% | 10.68362 | 10.73705 | +0.5001% [+0.1388,+0.9186] |
| 16384 | 415.9430 | 397.3432 | -4.4717% | 10.12564 | 10.25026 | +1.2308% [+0.6583,+1.7997] |
| 32768 | 384.8495 | 350.8747 | -8.8281% | 9.35624 | 9.50601 | +1.6007% [+0.2243,+2.9866] |

| Native W4A16 quality | Default | QSA |
|---|---:|---:|
| PPL,64 windows of4096 tokens,262080 targets | 5.96868807 | 5.97127044 |
| GSM8K,128 questions | 111/128 (86.71875%) | 111/128 (86.71875%) |
| GSM8K generation caps | 18 | 18 |

PPL increases0.0432652%. All128 GSM8K generated token sequences are identical;
these requests reach at most666 tokens and **never prune**. GSM8K therefore checks
the full-support path, not long-context sparse answer quality. The4K PPL test
does exercise pruning. Default-only natural long-document quality is available;
matched QSA natural-long quality and Qwen SWE-bench are unmeasured.
The separate50-case Qwen BFCLv4 pilot is described below.
The earlier OLMo BFCL result is a128-item pilot, not the full BFCLv4 score.

<!-- QWEN_BFCL_PILOT_STATUS_BEGIN -->
The [paired50-case Qwen BFCLv4 component pilot](../../experiments/qwen-qsa-bfcl-v4-50.md) is complete: **default26/50 (52.0%) versus QSA24/50 (48.0%)**. The fixed selection covers26 single-turn and24 multi-turn cases across17 components. It is not the official overall BFCLv4 score.

QSA reaches sparse pruning in24/50 cases. Default/QSA have0/0 capped responses across294/283 generation calls. The report and raw JSON/CSV retain prompt/response distributions, declared versus observed turns, generation steps, cap flags and pruning coverage. All100 variant rows were independently rescored with official checkers, and every request was independently retokenized. This diagnostic remains nongating.
<!-- QWEN_BFCL_PILOT_STATUS_END -->

The pilot uses BFCL's prompting interface: official Python-call system
instructions and Qwen's actual chat template, with thinking disabled. Official
AST, relevance and multi-turn checkers score the raw responses. Outputs receive
the official parser's processing, with no manual format repair. Multi-turn
success uses the official trajectory/state checks. Selection and prompts were
frozen before inference.
BFCL's `MAXIMUM_STEP_LIMIT=20` setting and pinned handler stopping semantics
are retained. Forced-quit events remain in the raw inference logs, separately
from the1,024-token response-cap flags.
The pinned handler checks `count > 20` after execution, so a stopped turn can
contain21 model calls; the per-turn CSV preserves the actual count.

## Memory interpretation and coverage

Use [memory.json](memory.json) or [memory.csv](memory.csv) for every combination
of batches1/4/8, prefixes4096/16384/32768 and both variants. Every cell has an
explicit status. Canceled and bounded/failed attempts must not be treated as
successful peaks. The human-readable
[memory report](../../experiments/qwen-qsa-memory.md) shows the captured matrix.
For a peak-comparison table, filter CSV rows to `phase=overall` and
`status=complete`, and compare a pair only when both variants completed the same
batch/context. Other phases and incomplete lower bounds remain available for
diagnostics. The JSON `status_counts` field summarizes coverage at export time.

The campaign is stopped, with no remaining memory probes scheduled. Its scope
decision is retained in `memory.json` under `campaign` and in the raw archive.
Use batch1/4K for the selected comparison. The already completed batch4/4K pair
remains available as supplementary evidence; the interrupted batch8/4K probe
contains only observed lower bounds, not a completed configuration peak.

These are memory-only probes: one fresh model/session process per variant, one
prefill, eight fixed decode calls, capacity=prefix+64, no performance repetitions.
The CPU-only external sampler targets100ms and retains phase events, process
tree, RSS/private commitment, WDDM dedicated/shared/total commitment and available
system RAM. Sampled peaks can miss short spikes. WDDM commitment is **not proven
physical GPU residency**; never add it to RSS or available/used system RAM.

QSA retains the full KV cache and adds node-owned projection/features/pools,
ranking scratch and attention partials. At batch1/4K, four selectors account for
200.8013MiB of non-BLAS scratch plus25.00293MiB of weights, before allocator and
BLAS overhead. The completed isolated pair reports19.3376GiB default versus
19.5849GiB QSA peak WDDM total commitment. This is consistent with extra QSA
workspace; it is not evidence of a KV memory saving. Workspace sharing is a
possible future optimization, not part of the measured implementation.

The64K co-resident performance attempt stopped on the RAM guard, produced no valid
timing and was canceled by the user. It is preserved as a failed attempt and is
excluded from the memory matrix. Do not rerun64K as part of reproducing this audit.
The batch4/4K default memory probe completed after1551.55s; that one-time duration
is retained in the raw record and is not a throughput measurement or a failure.

## Files for the paper's main thread

| File | Contents |
|---|---|
| [identity.json](identity.json) | Model revision, payload/tensor/data hashes, frozen layers and representations |
| [environment.json](environment.json) | Historical workstation inventory, software versions and memory-accounting caveats |
| [performance.json](performance.json), [performance.csv](performance.csv) | Native raw-summary metrics, intervals, MI355X reference and comparison restrictions |
| [quality.json](quality.json), [quality.csv](quality.csv) | Native PPL/GSM denominators and answers; separate BFCL50 summary after audit |
| [memory.json](memory.json), [memory.csv](memory.csv) | All18 cells, byte units, phase peaks and incomplete status |
| [provenance.json](provenance.json) | Measured binary/source relationship, local references and historical errata |
| [candidate-validation.json](candidate-validation.json) | Merged candidate source/binary hashes, commands, logs and passing correctness controls |
| [manifest.json](manifest.json) | SHA256 and byte count for every generated packet file |
| [archive.json](archive.json) | Local raw ZIP path, checksum, size and member count |
| `raw/` | Original quality rows (gzip), timing rows/protocols/summaries and numerical receipts |

The raw ZIP includes memory traces/events/source snapshots, failure records,
ORT dispatch profiles, build/validation receipts, OLMo evidence, MI355X source
records and measured/candidate source overlays. It also preserves the frozen BF16
selector, locked test/GSM8K inputs, tokenizer/config metadata, the small measured
ONNX graphs and the export topology template. `MANIFEST.json` inside it hashes
every member. Large backbone weights and compiled caches are excluded; their
hashes and local origins are recorded. The ZIP stays local and is outside Git;
copy it together with this packet when handing data to another machine.
The packet's `.gitattributes` preserves original evidence bytes across Git
checkouts, including Windows protocol line endings used by recorded hashes.
Portable BFCL rows use checksum-indexed gzip byte chunks under
`raw/native/bfcl-v4-50/row-parts/`; `rows-parts.json` lists their order.
Decompress and concatenate them to restore the exact original `rows.jsonl`.
The verifier checks every part and the reconstructed file. The local raw archive
also retains the original unchunked JSONL and all request/response records.

The measured build began at dirty HEAD2a43690; its implementation was subsequently
committed in7a71b72. Binary hashes and historical execution protocols identify the
measured stack. The final PR merge build is separate; **do not attribute old
timings to its new binary hashes**. Historical receipts are preserved, including
the reconnaissance inspector's unused`gqa_back.cpp` filename; the inspector now
records the compiled`real/gqa.cpp` implementation.
Historical generic stage receipts also recorded the OLMo selector hash for Qwen
commands. Those fields are retained as an erratum, not used as Qwen identity;
use `identity.json`, the Qwen export receipts and per-execution protocols instead.
The stage runner now selects the correct payload and snapshots Qwen scripts.

## Joining with MI355X

The source packet is from sparse-attn commit
`3b0c30ea4976085798a36d22cefa4089bbef6543`, performance job303195, trained
selector job303042 step128, changed quality303045 and matched original303046.
Join on exact model revision, raw selector tensor hashes, dataset/document IDs,
batch, prefix length, generation count and phase. The PT and safetensors payload
hashes differ by container format; raw tensor hashes establish identity.

MI355X uses the source BF16 backbone, a different execution stack, greedy argmax
and potentially different generated streams per variant. This hip-ep packet uses
W4A16 and fixed teacher-forced streams. The source reports median paired gains;
hip-ep reports a ratio of mean phase times. Keep separate platform/protocol columns
and do not average the two gains or infer hardware speedups from absolute TPS.
The native timing inputs concatenate known4K test windows; they are not a blind
natural-long holdout. Source quality must retain its own representation label.

The separate MI355X memory study is job303626, source
`7a9739e32f19588a1879e2b6545053fff469b4da`. Its original summary, raw rows and
protocol hashes are included under `raw/mi355x/`. It uses one warmup and two
64-decode requests per shape, PyTorch allocation/reservation high-water counters
and20ms device-HBM polling. The native cold8-decode probes use100ms WDDM commitment.
These memory gauges and workloads require separate labels in any combined figure.

The BFCL runner uses the WSL user's `~/.cache/hip-ep-qsa-bfcl/bin/python` by
default. Set `QWEN_BFCL_CPU_PYTHON` to select another isolated Linux CPU
environment with the recorded dependencies. Existing prompt profiles are
checked for equality rather than overwritten when preparing a resumed run.

## Reproduce or validate

Packet verification is CPU-only and uses the Python standard library:

```powershell
python scripts/qwen_qsa/package_evidence.py --verify
```

To regenerate after all probe workers are stopped, use
`python scripts/qwen_qsa/package_evidence.py`. For a separate PR checkout pass
`--evidence-root <measured-checkout>` and `--source-root <sparse-attn-checkout>`.
This exports evidence only and performs no GPU work. The measured checkout and
source payloads must exist; it will not fetch weights or alter the source project.

For native validation use the local environment and isolated build helper
`scripts/qsa/build_native_gqa.py --work-root <new-audit-directory>`. Its
`--toolchain-root`, `--ort-sdk` and `--lit` flags make local dependencies explicit;
`--plan` prints the recipe. Never install over the measured binary directory.
`validate_kernels.py --install <candidate-install> --output <new-receipt.json>`
allows primitive checks without replacing historical receipts. The audit records
the exact candidate build and checks separately. Performance/quality scripts
remain available for an explicitly requested new campaign.

## Remaining work

The50-case BFCLv4 pilot, selected memory comparison and local audit are complete.
No hip-ep jobs remain queued. Matched QSA
natural-long quality, Qwen SWE-bench and a new merged-candidate
performance/accuracy campaign remain unmeasured and are not scheduled. The
canceled memory shapes and64K attempt are not outstanding work.

Paper collation with MI355X and publication remain with the main thread. This
packet captures the pinned source jobs listed above; it does not track later
MI355X experiments. The local branch is ready for review, with push and PR
publication deferred at the user's request.
