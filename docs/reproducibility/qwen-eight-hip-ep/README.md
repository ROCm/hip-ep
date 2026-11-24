<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen hip-ep: Original 4K control and paused eight-selector transfer

**Current state:** the same Original-only B1/4K diagnostic was repeated after the user connected a140W charger. It recovered close to the archived Original baseline. Eight/QSA timing, longer shapes and BFCL/GSM8K remain paused; this repeat ran only4K.

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

The Original4K check and its charger repeat are complete. Await review before further timing. The previously proposed fresh Original/Eight4K pilot, the full eight-selector matrix and the evaluation-duration estimates have not been completed or accepted. BFCL50 and GSM8K128 must not start under the current request. No unmeasured estimate or benchmark accuracy is reported.

[validation.json](validation.json) preserves eight-node dispatch/cache/finite/replay controls; [transfer-provenance.json](transfer-provenance.json) distinguishes the handoff commit from the sibling repository’s later observed HEAD. [original-control-status.json](original-control-status.json) captures the narrowed scope. Current local branch: `codex/qwen-eight-qsa-perf`. The [historical four-selector packet](../qwen-hip-ep/README.md), old results and measured installation are unchanged. New source snapshots/logs remain in `_local/qwen-qsa/eight-qsa-v1/` and `_local/qwen-qsa/original-short-control-v1/`. Nothing is pushed.

The [immutable local ZIP receipt](archive.json) binds all raw source snapshots/logs, the transferred BF16 selector bytes, ONNX graph, paused-run rows and Original-control evidence. Every member checksum was verified; large backbone weights and compiled caches are excluded. The ZIP stays local. [eight-identity.json](eight-identity.json) records the corrected eight-selector identity; its raw export receipt retains inherited baseline metadata as an explicit erratum. [manifest.json](manifest.json) hashes the portable packet.
