<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen hip-ep: eight-selector performance and evaluation planning

**Current state:** the clean eight-selector performance campaign is complete at B1/4K,16K,32K after the140W charger change. It uses **one warmup and three measured requests per shape**, as requested. The Eight-only BFCL50 pilot is complete: **Eight 29/50 versus reused Original 26/50**. GSM8K remains unstarted. No64K or batched shapes were run.

## Data coverage: four and eight selectors

Selector weights are now public: [pinned Hugging Face assets and loading commands](../../design/qsa-selector-assets.md).
All60 BF16 tensor payloads and the three native graph/selector-bin re-exports
match their measured counterparts. [Migration verification](hf-selector-migration.json)
keeps new Hub container hashes separate from historical export identities;
historical timing and accuracy rows are unchanged.

Both variants' evidence is retained locally and in the portable packets below. Historical `Changed` labels in the four-selector packet mean **Four**, not Eight.

| Measurement | Four selectors | Eight selectors |
|---|---|---|
| B1 performance, 4K/16K/32K | Complete: [JSON](../qwen-hip-ep/performance.json), [report](../../experiments/qwen-qsa-native-gqa-results.md) | Complete: [JSON](eight-140w-performance.json), [report](../../experiments/qwen-qsa-eight-selector-results.md) |
| Same fixed BFCL50 cases | 24/50; Original 26/50: [report and raw-data links](../../experiments/qwen-qsa-bfcl-v4-50.md) | 29/50; Original 26/50 reused: [JSON](bfcl-eight-summary.json), [report and raw-data links](../../experiments/qwen-qsa-eight-bfcl-v4-50.md) |
| Native PPL, 64 documents | Original 5.968688; Four 5.971270: [quality JSON](../qwen-hip-ep/quality.json) | Not measured on hip-ep; source checkpoint still misses its fixed32K PPL gate |
| GSM8K, 128 cases | Original and Four 111/128; no pruning at these short contexts: [quality JSON](../qwen-hip-ep/quality.json) | Not run |
| Application memory | Selected B1/4K comparison complete; WDDM commitment, not physical residency: [memory JSON](../qwen-hip-ep/memory.json), [report](../../experiments/qwen-qsa-memory.md) | Not measured; host RAM telemetry is diagnostic only |

The performance comparisons reuse historical Original/Four controls across runs; the protocols differ as detailed below. The BFCL pilots are component scores, not official overall BFCL scores. Neither packet establishes accepted natural-long quality. No valid64K timing or Qwen SWE-bench result is available; no additional run is queued.

**Status-file guide:** [bfcl-eight-status.json](bfcl-eight-status.json) and [bfcl-eight-summary.json](bfcl-eight-summary.json) are the completed BFCL result; [eight-140w-performance.json](eight-140w-performance.json) is the accepted charged-run timing record. The packet's `status.json`, `control-status.json`, `original-control-status.json`, pause receipts and `raw/` status files are preserved snapshots of their named earlier campaigns. Their paused/canceled/intermediate states are historical evidence, not active jobs. [evaluation-time-estimate.json](evaluation-time-estimate.json) is a pre-evaluation planning snapshot; its no-accuracy-runs label describes when the estimate was made. The completed BFCL result above supersedes that planning status.

## Eight selectors: completed charged-power performance

All eight updated BF16 selectors at zero-based layers3/7/23/27/31/35/39/63 come from the frozen checkpoint640 export; the original four positions also use its updated weights. The backbone remains native RTN W4A16 group128, with FP16 activations/KV and FP32 recurrent state. No training,learned calibration,kernel/compiler rebuild or model update occurred during this campaign. Both prefill and decode use sparse QSA. The earlier eight-node/cache/finite/dispatch validation is reused with exact graph,selector and binary identities.

| Context | Original prefill tok/s (historical) | Four prefill tok/s (historical) | Eight prefill tok/s | Original decode tok/s (historical) | Four decode tok/s (historical) | Eight decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 438.06 | 419.97 | 398.08 | 10.68 | 10.74 | 10.81 |
| 16384 | 415.94 | 397.34 | 372.26 | 10.13 | 10.25 | 10.45 |
| 32768 | 384.85 | 350.87 | 354.10 | 9.36 | 9.51 | 10.01 |

These are **across-run** comparisons. Historical Original/four use2warmups and6requests in a paired two-resident-session run sharing cache buffers; Eight uses the user-requested1warmup and3requests with one resident full-model session. The common64 teacher-forced decode inputs,backbone,binaries,capacity and phase timers match. No paired confidence intervals or isolated layer-count causal claim is made. The locked prefixes concatenate known4K test windows; these timing streams are not a natural-long quality holdout. The separate charger Original4K check below recovered within1.45% prefill and1.01% decode of its archived baseline; Original16K/32K were not remeasured after the charger change.

Each new shape retains all four rows (`trial=-1,0,1,2`),with only trials0–2 aggregated. TPS is token count divided by mean elapsed phase seconds across **all three** measured requests. Independent HIP-synchronized CPU-wall timers cover full-session prefill and64 full-session decode calls; allocation,binding,recurrent zeroing and cold load are outside phase timers. Output finite checks,exact replay,stable cache pointers and unchanged sessions pass at every shape.

[Performance JSON](eight-140w-performance.json), [CSV](eight-140w-performance.csv), raw rows and saved aggregates: [4K rows](raw/eight-140w/4096/rows.jsonl) / [summary](raw/eight-140w/4096/summary.json), [16K rows](raw/eight-140w/16384/rows.jsonl) / [summary](raw/eight-140w/16384/summary.json), [32K rows](raw/eight-140w/32768/rows.jsonl) / [summary](raw/eight-140w/32768/summary.json). Protocols and reused historical rows/summaries are beside each. Source aggregation is `scripts/qwen_qsa/measure_eight.py`; exact as-run sources,commands and stop/RAM/time guards are in the local incremental ZIP below.

The charger rating is user-reported. AC state was checked about once per second and sampled host evidence every5seconds; every recorded AC state stayed connected. CPU/available-RAM telemetry is diagnostic host data,not GPU-residency measurement. Preflight observed only desktop/System GPU activity,with no competing model workload. The preflight scope proposal was superseded before launch; [protocol-amendment.json](raw/eight-140w/protocol-amendment.json) documents the final1/3 request.

<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->
## Four versus eight selectors

**Decode throughput improved over four selectors at every measured length.** Prefill is slower at 4K/16K and slightly faster at 32K. All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change.

| Context | Eight vs four: prefill | Eight vs four: decode |
|---|---:|---:|
| 4K | -5.21% | +0.71% |
| 16K | -6.31% | +1.94% |
| 32K | +0.92% | +5.31% |

Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. Both prefill and decode use QSA at all eight selected layers. See the [standalone performance report](../../experiments/qwen-qsa-eight-selector-results.md).
<!-- QWEN_EIGHT_PERFORMANCE_END -->

<!-- EIGHT_BFCL_RESULT_BEGIN -->
## Eight-selector BFCL result

**Eight 29/50 (58.0%)**, reused Original **26/50 (52.0%)**, historical Four **24/50**. Only Eight was generated. The fixed50-case component pilot is nongating,not the official overall BFCL score. The [full BFCL report](../../experiments/qwen-qsa-eight-bfcl-v4-50.md), [summary JSON](bfcl-eight-summary.json), and `raw/bfcl-eight-v4-50/` retain component scores,prompt/output/call/turn distributions,cap/context/step stops,pruning exposure,and official/token audits. [Local BFCL archive](bfcl-eight-archive.json) binds complete generated texts/prompts/token IDs and exact source/attempt snapshots.
<!-- EIGHT_BFCL_RESULT_END -->

## Estimated evaluation durations — planning reference

BFCL has now completed: Eight 29/50 versus reused Original 26/50. The completed campaign took 130.8 minutes including native load,generation,official scoring and independent audits. The tables below retain the prior planning estimates; differing output/tool trajectories are not a fixed-work speed comparison.

These are planning estimates derived **after** measured performance. Each actual historical BFCL prompt and generated token is costed separately using the new prefill/decode curves,then calibrated to historical generation time and actual tool/queue overhead. Both previous trajectories are used:294/283 generation calls,2,217,075/2,117,348 processed prompt tokens,and14,333/13,899 output tokens. The50 fixed cases include26 single-turn and24 multi-turn cases,1–7 declared turns,cap1024,thinking disabled. GSM128 uses its historical11,167 prompt and43,494 output tokens,cap512,thinking disabled. This is not a single decode-TPS multiplier.

| Evaluation | Eight only,reuse Original scores | Planning range | Fresh Original + Eight | Planning range |
|---|---:|---:|---:|---:|
| BFCL v4,50 cases | 123 min | 86–197 min | 238 min | 190–332 min |
| GSM8K,128 cases | 71 min | 57–91 min | 143 min | 122–175 min |

[Estimation JSON](evaluation-time-estimate.json) includes both trajectory scenarios,phase components,startup costs,formulas and limitations. `raw/eight-140w/bfcl-estimate-inputs.csv` and `gsm-estimate-inputs.csv` retain per-call/case workload lengths. Startup includes the maximum observed new Eight session load/compile plus15seconds for CPU/import/tokenizer setup; it may reuse OS/compiler caches. One resident session per variant is assumed; retaining four32-case GSM processes adds three startups per variant.

Ranges are planning envelopes,not confidence intervals or guaranteed upper bounds. BFCL can change tool paths,call counts,turn stops and lengths with the new selectors. Sub-4K phase costs are calibrated extrapolations,and historical GSM token totals are workload assumptions,not output or accuracy predictions. Fresh Original estimates reuse its historical evaluation time; only its4K charged baseline was freshly checked. **Checkpoint640 still misses the source fixed32K PPL gate. The completed BFCL pilot does not establish accepted long-context accuracy.**

## Repeat after the140W charger change

Same Original model, native binaries, input stream and independent synchronized phase timers; two warmups and two measured requests, with64 forced decode inputs each. AC power was detected. The140W rating is user-reported; wattage and actual device power limits were not independently measured. Preflight CPU load was2.8%, with only desktop/System GPU activity and no competing model workload observed.

| Phase | Earlier quiet control tok/s |140W repeat tok/s | Archived Original tok/s | Repeat vs archive |
|---|---:|---:|---:|---:|
| prefill | 306.25 | 431.71 | 438.06 | -1.45% |
| decode | 10.17 | 10.79 | 10.68 | +1.01% |

The recovered performance is consistent with power delivery/power limits contributing to the earlier drop. This before/after observation does not prove an exclusive cause. Both current controls contain two measured requests; the archive contains six, so no paired interval is claimed. All aggregates use token count divided by the mean elapsed phase seconds across **both** measured requests. Finite outputs, exact replay, stable sessions/cache addresses and raw hashes pass.

[Comparison JSON](original-140w-4k-comparison.json), [raw rows](raw/original-140w/4096/rows.jsonl), [saved aggregate](raw/original-140w/4096/summary.json), [protocol](raw/original-140w/4096/protocol.json), [power/load receipt](raw/original-140w/preflight/receipt.json) and [incremental local ZIP receipt](original-140w-archive.json) retain this repeat. Its protocol correctly labels native default attention, with no QSA nodes. The previous raw protocol and correction receipt remain unchanged. No kernel, model or compiler modification and no accuracy evaluation occurred. The worker has exited and the GPU-stage lock is released.

## Earlier Original4K control before the charger change

Two exact-shape warmups and **two measured requests**, each with4096 prefill tokens and64 common teacher-forced decode inputs. Prefill and decode have separate HIP-synchronized CPU-wall timers. Same preserved Original W4A16 group128 graph/backbone and native binaries; FP16 activation/KV and FP32 recurrent state. No export, kernel change or QSA model session was used by this control. One resident Original session.

| Phase | Fresh Original tok/s | Archived Original tok/s | Throughput change |
|---|---:|---:|---:|
| prefill | 306.25 | 438.06 | -30.09% |
| decode | 10.17 | 10.68 | -4.77% |

That earlier control did **not** recover the prior Original baseline under the sampled quiet conditions. The later charger repeat above recovers close to that baseline. The earlier data alone did not establish contention as the cause or isolate a QSA effect. The comparison is across runs, two fresh versus six archived measured requests; the historical paired run had two resident sessions sharing KV/state buffers. No paired confidence interval or acceptance of long-context accuracy is inferred.

Use [original-4k-comparison.json](original-4k-comparison.json), [all four raw requests](raw/original-control/4096/rows.jsonl), [saved aggregate](raw/original-control/4096/summary.json) and [protocol](raw/original-control/4096/protocol.json). TPS equals phase token count divided by **mean elapsed seconds over all measured requests** (`trial >= 0`), not the mean of per-request TPS. Exact aggregation is in `scripts/qwen_qsa/measure_eight_control.py`; the2-request option changes only the repeat count. The full finalization independently recomputed the displayed aggregate from both measured rows.

## Original metadata correction

The immutable raw protocol inherited the shared helper’s `phase_policy: sparse prefill and sparse decode` label. The [correction receipt](original-control-metadata-correction.json) records the accurate policy: **native default attention, with zero QSA nodes**. The executed Original graph exactly matches the archived Original validation hash; all four native binary hashes match the archived performance protocol. The graph contains16 native GQA and48 Gated DeltaNet nodes. No raw protocol bytes, model, binaries or measurements were changed, and no rerun occurred.

## Quiet-workload evidence

The final [eligibility receipt](raw/original-control/preflight/eligibility.json) records1.8% sampled CPU load, idle llmfc_mcp services at0% sampled CPU, about41GiB available RAM and no active GPU entries above0.1% across four GPU counter samples. This follows explicit forecast-chat completion evidence and actual Windows/WSL process inspection. No competing model process was observed. WSL was initially stopped; inspection briefly started its system services. No Python/model job existed there, and WSL startup activity was allowed to settle before timing. The coordinating documentation audit also ended before timing. Only owned benchmark processes were stopped; unrelated services and llm-forecast were untouched. These finite samples establish the checked startup conditions, not proof of zero interference at every instant.

## Preserved eight-selector run — not accepted as a clean comparison

The eight-selector transfer and native correctness controls passed. All eight updated BF16 selectors at zero-based layers3/7/23/27/31/35/39/63 match the frozen export and source tensor digests. The unchanged measured build runs sparse prefill and sparse decode. No training, learned calibration, kernel rebuild or sibling-repository write occurred. The checkpoint remains a development candidate that misses its source fixed32K PPL gate; native controls are not benchmark accuracy results.

The earlier eight-selector performance run is labeled **potentially contention-affected**, with no confirmed contention cause. Its rows remain unchanged:4K and16K each have two warmups and six measured requests;32K has two warmups and zero measured requests. No32K aggregate exists. Raw evidence is under `raw/potentially-contended/`. The [pause receipt](pause-receipt.json) lists exact processes stopped and confirms the queued default controller had started zero stages.

| Stopped role | Actual Python PID | Venv launcher PID |
|---|---:|---:|
|32K eight-selector worker|40628|27412|
|Eight performance supervisor|36584|34692|
|Queued fresh-default supervisor|42700|41824|

The worker then had no in-band boundary stop, so its in-flight request was terminated without producing a completed row. Subsequent workers support `--stop-file` at request boundaries. The Original4K control and all owned controller/worker processes are now absent, with the shared GPU-stage lock released.

## Handoff and remaining scope

The charger Original4K check and the newly authorized Eight B1/4K,16K,32K campaign are complete. The original paired4K pilot was superseded by that request. Evaluation-duration estimates are now saved above; The authorized Eight-only BFCL50 pilot is complete (29/50 versus reused Original 26/50). GSM8K remains unstarted; no64K or batched run is scheduled.

[validation.json](validation.json) preserves eight-node dispatch/cache/finite/replay controls; [transfer-provenance.json](transfer-provenance.json) distinguishes the handoff commit from the sibling repository’s later observed HEAD. [original-control-status.json](original-control-status.json) captures the narrowed scope. Combined source branch: [`users/rajeevp/qsa`](https://github.com/ROCm/hip-ep/tree/users/rajeevp/qsa); see the [OLMo/Qwen support scope](../../design/qsa-support.md). The [historical four-selector packet](../qwen-hip-ep/README.md), old results and measured installation are unchanged. New source snapshots/logs remain in `_local/qwen-qsa/eight-qsa-v1/` and `_local/qwen-qsa/original-short-control-v1/`. Large local ZIPs, model weights and compiled caches remain local; the combined source branch contains the portable evidence.

The [immutable local ZIP receipt](archive.json) binds all raw source snapshots/logs, the transferred BF16 selector bytes, ONNX graph, paused-run rows and Original-control evidence. Every member checksum was verified; large backbone weights and compiled caches are excluded. The ZIP stays local. [eight-identity.json](eight-identity.json) records the corrected eight-selector identity; its raw export receipt retains inherited baseline metadata as an explicit erratum. [manifest.json](manifest.json) hashes the portable packet.

[Eight charged-run incremental archive](eight-140w-archive.json) binds the exact as-run sources,commands,host samples,all timing rows,and CPU-only estimation/audit code. Every ZIP member checksum passed.
