#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Publishable four-way Windows metrics, with failures and provenance retained."""

from pathlib import Path
import hashlib
import json
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/profile"
OUT = ROOT / "results/comparison"
lengths = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
variants = ("existing", "ring_q", "ring", "tq4")
names = {
    "existing": "Baseline",
    "ring_q": "Ring-Q",
    "ring": "Ring-KV",
    "tq4": "TurboQuant",
}
samples = []
for path in D.glob("wddm-*.json"):
    samples.extend(json.loads(path.read_text()).get("peaks", {}).values())
rows = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(x, y):
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    cosine = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    l2 = float(np.linalg.norm(x - y) / np.linalg.norm(y))
    return {
        "cosine": cosine,
        "relative_l2": l2,
        "passed": cosine >= 0.999 and l2 <= 0.05,
    }


for n in lengths:
    suffix = str(n) + "-synthetic" if n == 262144 else str(n)
    control_path = D / "existing" / suffix / "receipt.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    for variant in variants:
        path = D / variant / suffix / "receipt.json"
        row = {
            "context": n,
            "variant": variant,
            "name": names[variant],
            "protocol": "synthetic" if n == 262144 else "full_request",
        }
        if not path.exists():
            row["status"] = "missing"
            rows.append(row)
            continue
        r = json.loads(path.read_text())
        row.update(status=r["status"], receipt_sha256=sha(path), error=r.get("error"))
        row["library_sha256"] = {
            Path(k).name: v for k, v in r.get("loaded_libraries", {}).items()
        }
        row["harness_sha256"] = r.get("harness_sha256")
        if "weights_bytes" in r:
            row["weights_gib"] = r["weights_bytes"] / 2**30
        if "kv_allocation" in r:
            row["kv_gib"] = r["kv_allocation"]["total_bytes"] / 2**30
            payload = r["kv_allocation"].get(
                "payload_bytes",
                32 * 4 * r["capacity"] * (132 if variant == "tq4" else 512),
            )
            assert payload <= r["kv_allocation"]["total_bytes"]
            row["kv_payload_gib"] = payload / 2**30
            assert r["kv_allocation"]["buffers"] == 32
        if r["status"] != "passed":
            rows.append(row)
            continue
        if n == 262144:
            row.update(
                prefill_seconds=None,
                ttft_seconds=None,
                decode_tps=r["decode_tokens_per_second"],
                measured_requests=1,
            )
            row["timing_samples"] = [
                {"decode_seconds": r["decode_seconds"], "generated_steps": 128}
            ]
            row["source_history_sha256"] = r["source_history_sha256"]
            row["ep_decode_peak_gib"] = r["decode_memory"]["peak"] / 2**30
        else:
            measured = [x for x in r["records"] if x["phase"] == "measurement"]
            assert measured
            for metric in (
                "prefill_seconds",
                "ttft_seconds",
                "decode_tokens_per_second",
            ):
                row[
                    "decode_tps" if metric == "decode_tokens_per_second" else metric
                ] = statistics.median(x[metric] for x in measured)
            row["measured_requests"] = len(measured)
            row["timing_samples"] = [
                {
                    k: x[k]
                    for k in (
                        "prefill_seconds",
                        "ttft_seconds",
                        "decode_seconds",
                        "decode_tokens_per_second",
                    )
                }
                for x in measured
            ]
            row["prompt_sha256"] = r["prompt_sha256"]
            for phase in ("prefill", "decode"):
                row["ep_" + phase + "_peak_gib"] = (
                    max(x[phase + "_memory"]["peak"] for x in measured) / 2**30
                )
        row["decode_ms_per_token"] = 1000 / row["decode_tps"]
        for phase in ("prefill", "decode"):
            eligible = []
            for s in samples:
                if (
                    s["variant"] != variant
                    or s["context"] != n
                    or s.get("pid") != r["pid"]
                ):
                    continue
                rep, label = s["phase"].split(":", 1)
                if (n == 262144 and phase == "decode" and label == "measuring") or (
                    n != 262144 and int(rep) > 0 and label == phase
                ):
                    eligible.append(s["committed"])
            total = max(eligible) / 2**30 if eligible else None
            row[phase + "_peak_gib"] = total
            row[phase + "_other_gib"] = (
                None if total is None else total - row["weights_gib"] - row["kv_gib"]
            )
            if total is not None:
                assert row[phase + "_other_gib"] >= 0
        if control and control["status"] == "passed":
            if n == 262144:
                assert control["source_history_sha256"] == r["source_history_sha256"]
                row["first_decode_vs_baseline"] = compare(
                    np.load(path.parent / "first-logits.npy"),
                    np.load(control_path.parent / "first-logits.npy"),
                )
            else:
                assert control["prompt_sha256"] == r["prompt_sha256"]
                row["prefill_vs_baseline"] = compare(
                    np.load(path.parent / "prefill-logits.npy"),
                    np.load(control_path.parent / "prefill-logits.npy"),
                )
                x = r["records"][0]["generated_token_ids"]
                y = control["records"][0]["generated_token_ids"]
                row["matching_tokens_vs_baseline"] = sum(a == b for a, b in zip(x, y))
                row["first_divergent_token_1based"] = next(
                    (i + 1 for i, (a, b) in enumerate(zip(x, y)) if a != b), None
                )
                if x[:1] == y[:1]:
                    row["first_decode_vs_baseline"] = compare(
                        np.load(path.parent / "first-decode-logits.npy"),
                        np.load(control_path.parent / "first-decode-logits.npy"),
                    )
                else:
                    row["first_decode_vs_baseline"] = {
                        "status": "not_comparable",
                        "reason": "First generated token differs; decode prefixes are not aligned",
                    }
        else:
            row["baseline_quality_reference"] = "unavailable"
        if variant == "tq4":
            ring_path = D / "ring" / suffix / "receipt.json"
            ring_ref = json.loads(ring_path.read_text()) if ring_path.exists() else None
            if ring_ref and ring_ref["status"] == "passed":
                if n == 262144:
                    assert (
                        ring_ref["source_history_sha256"] == r["source_history_sha256"]
                    )
                    row["first_decode_vs_ring_kv"] = compare(
                        np.load(path.parent / "first-logits.npy"),
                        np.load(ring_path.parent / "first-logits.npy"),
                    )
                else:
                    assert ring_ref["prompt_sha256"] == r["prompt_sha256"]
                    row["prefill_vs_ring_kv"] = compare(
                        np.load(path.parent / "prefill-logits.npy"),
                        np.load(ring_path.parent / "prefill-logits.npy"),
                    )
                    x = r["records"][0]["generated_token_ids"]
                    y = ring_ref["records"][0]["generated_token_ids"]
                    row["matching_tokens_vs_ring_kv"] = sum(
                        a == b for a, b in zip(x, y)
                    )
                    row["first_divergent_token_vs_ring_kv_1based"] = next(
                        (i + 1 for i, (a, b) in enumerate(zip(x, y)) if a != b), None
                    )
                    if x[:1] == y[:1]:
                        row["first_decode_vs_ring_kv"] = compare(
                            np.load(path.parent / "first-decode-logits.npy"),
                            np.load(ring_path.parent / "first-decode-logits.npy"),
                        )
                    else:
                        row["first_decode_vs_ring_kv"] = {
                            "status": "not_comparable",
                            "reason": "First generated token differs",
                        }
            else:
                row["current_ring_kv_quality_reference"] = "unavailable"
        row["previous_quality_checks"] = r.get("quality", r.get("first_step_quality"))
        rows.append(row)
OUT.mkdir(exist_ok=True)
summary = {
    "platform": "Windows Radeon 8060S gfx1151",
    "model": "amd/Qwen3.8-27B_gpuep_1.0_gpu_128",
    "model_revision": "9b37cf56a001720ba7ce98ff80ea651b02e5931b",
    "ring_dependency": "https://github.com/ROCm/hip-ep/pull/1252",
    "ring_parent": "63a00f5e35e54aff1d423525a9c8992fd8dea927",
    "upstream_baseline": "a4f9826b809c9483fe2353eea876c4c2257a5ad4",
    "rows": rows,
}
download = json.loads((ROOT / "results/windows-qwen-download.json").read_text())
summary["model_files"] = download["verified"]
manifest = ROOT / "reference/codec/manifest.json"
codec = json.loads(manifest.read_text()) if manifest.exists() else {}
summary["codec"] = {
    k: codec[k]
    for k in (
        "codec",
        "seed",
        "head_dim",
        "row_bytes",
        "transform",
        "normalization",
        "tie_break",
        "torch",
    )
    if k in codec
}
summary["codec"]["sha256"] = sha(ROOT / "reference/codec/codec.bin")
summary["codec"]["constants_sha256"] = sha(ROOT / "reference/codec/constants.npz")
if "source" in codec:
    summary["codec"]["standalone_source"] = {
        k: codec["source"][k] for k in ("commit", "files")
    }
if (D / "provenance.json").exists():
    provenance = json.loads((D / "provenance.json").read_text())
    summary["measured_source_provenance"] = {
        k: provenance[k]
        for k in (
            "base",
            "uncommitted_experimental_changes",
            "source_sha256",
            "allocation_counter_sha256",
        )
        if k in provenance
    }
summary["measurement_note"] = (
    "For this recorded campaign, the Ring parent advanced from 1fd4568 to 63a00f5 by documentation-only commits before baseline/Ring-Q builds. Subsequent PR preparation restored Unicode comments/log text and applied formatting; GPU arithmetic was not changed. Binary hashes identify the measured installs."
)
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def label(n):
    return str(n) if n < 1024 else str(n // 1024) + "K"


lines = [
    "# Qwen four-way attention comparison",
    "",
    "Windows 11 Enterprise 10.0.26200; Ryzen AI MAX+ PRO 395 / Radeon 8060S gfx1151; 128 GiB installed UMA, 63.776 GiB OS-managed after firmware reservation; driver 32.0.22050.4002. ROCm HIP compiler 7.15.26333, ORT 1.30.0, GenAI 0.17.0. Qwen3.8-27B has INT4 group-128 weights, FP16 activations, 16 full-attention layers and 48 linear/recurrent layers. Batch one; CPU decoder fallback disabled.",
    "",
    "Baseline is the production `existing` GQA selection on upstream a4f9826 plus the Ring/TurboQuant integration branch, with quantization disabled. Ring-Q and Ring-KV select their respective prefill schedules and share the same dense decoder. TurboQuant uses Ring-KV FP16 prefill, then the MSE-only 4-bit QR cache and natural-exponential compressed decoder. The installed binaries are separately identified in the accompanying metrics JSON.",
    "",
    "These Windows measurements share one model revision and harness. Historical lx01/Linux Ring PR figures are not mixed into these tables: OS, GenAI version, UMA reservation and memory-accounting scope differ.",
    "",
    "## Prefill and decode",
    "",
    "Medians exclude the diagnostic/warmup request. Model/generator construction, cache allocation and prompt construction are outside the timers; TTFT starts at `append_tokens`. Full-prefill prompts use random token IDs from 100 through 19,999 with NumPy seed 20261014. Three measured requests per full-prefill case through 64K; one at 128K. Generate 128 tokens with a fresh cache/state per request; decode times the 127 steps after first-token selection. At 256K, skip prefill, initialize identical seeded random KV/recurrent history, warm up eight steps, and measure 128 full-model decode steps including logits readback and greedy selection. Sequential blocks are not an alternating paired speedup experiment.",
    "",
    "| Prompt/context | Variant | Prefill s | TTFT s | Decode ms/token | Decode tok/s | Measured requests |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    if r["status"] != "passed":
        lines.append(
            f"| {label(r['context'])} | {r['name']} | {r['status']} | — | — | — | 0 |"
        )
    else:
        lines.append(
            f"| {label(r['context'])}{'*' if r['protocol'] == 'synthetic' else ''} | {r['name']} | {fmt(r['prefill_seconds'])} | {fmt(r['ttft_seconds'])} | {fmt(r['decode_ms_per_token'])} | {fmt(r['decode_tps'])} | {r['measured_requests']} |"
        )
lines += [
    "",
    "*256K is a synthetic performance experiment with no prefill or TTFT. The recorded TurboQuant-versus-Ring-KV first-step numerical screen fails for this workload; see the accuracy section. It is not a semantic long-context accuracy result.",
    "",
    "## Memory decomposition",
    "",
    "GiB. Phase peaks are per-process Windows GPU Total Committed sampled at a target 20 ms interval, not physical residency or whole-system RAM. The prefill memory phase includes first-token selection; decode covers the remaining 127 steps. Weights are the two verified constant allocations. KV is the verified allocation of 32 mapped-host cache buffers, including allocator padding, counted once. Other = phase peak − weights − KV and includes activations, recurrent state, scratch, allocator pools and runtime/driver allocations. Decode retains prefill allocations. The metrics JSON separately records exact EP allocation-event peaks.",
    "",
    "| Prompt/context | Variant | Weights | KV | Other prefill | Prefill peak | Other decode | Decode peak |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    vals = " | ".join(
        fmt(r.get(k), 4)
        for k in (
            "weights_gib",
            "kv_gib",
            "prefill_other_gib",
            "prefill_peak_gib",
            "decode_other_gib",
            "decode_peak_gib",
        )
    )
    lines.append(
        f"| {label(r['context'])}{'*' if r['protocol'] == 'synthetic' else ''} | {r['name']} | {vals} |"
    )
lines += [
    "",
    "Full-request cache capacity is prompt length + 128; synthetic capacity is 262,144. TurboQuant stores 128 code bytes plus one FP32 norm per 256-wide vector: 512/132 = 3.8788× payload reduction. Actual reserved-buffer savings vary with size-class padding at shorter contexts. The 256K total is smaller because the prefill activation pools were never allocated.",
    "",
    "## Accuracy and incomplete measurements",
    "",
    "Performance completion is separate from accuracy. The matching-prefix numerical screen is cosine ≥ 0.999 and relative L2 ≤ 0.05. Generated token agreement is evaluated independently; tolerance-passing logit differences can change greedy sequences. This single seeded prompt per length is not broad task-accuracy or perplexity qualification. Exact per-variant comparisons against the current Windows baseline are retained in the metrics JSON.",
    "",
]
for r in rows:
    if r["status"] != "passed":
        lines.append(
            f"- {r['name']} {label(r['context'])}: {r.get('error') or r['status']}. No completed latency or memory peak is substituted."
        )
    elif r["variant"] != "existing":
        prefill = r.get("prefill_vs_baseline", {})
        if prefill.get("passed") is False:
            lines.append(
                f"- {r['name']} {label(r['context'])} prefill numerical gate fails against baseline: cosine {prefill['cosine']:.6f}, relative L2 {prefill['relative_l2']:.6f}."
            )
        q = r.get("first_decode_vs_baseline", {})
        if q.get("passed") is False:
            lines.append(
                f"- {r['name']} {label(r['context'])} first-decode numerical gate fails against baseline: cosine {q['cosine']:.6f}, relative L2 {q['relative_l2']:.6f}."
            )
        if q.get("status") == "not_comparable":
            lines.append(
                f"- {r['name']} {label(r['context'])}: first-decode logits are not compared because the first generated token differs."
            )
        if r.get("first_divergent_token_1based"):
            lines.append(
                f"- {r['name']} {label(r['context'])}: {r['matching_tokens_vs_baseline']}/128 output positions match baseline; first divergence at generated token {r['first_divergent_token_1based']}."
            )
lines += [
    "",
    "The earlier TurboQuant-vs-Ring-KV checks also observed sequence divergence at 256, 512, 1K and 16K despite passing first-step logits, and a failed 256K synthetic numerical gate. The original 128K TurboQuant diagnostic used archived lx01 Ring-KV logits after the earlier Windows Ring-KV attempt stopped. This consolidation additionally compares against current Windows baseline and Ring-KV controls wherever they completed. Strict encoder parity still has two code-byte boundary differences at 4K, and Windows codec constants differ from the frozen lx01 export. These findings require accuracy qualification and keep this contribution in draft.",
    "",
    "The Windows synthetic source history is byte-identical across compared variants, but differs from the archived Linux source despite the same seed/distribution. No 256K prefill or semantic 256K accuracy claim is made.",
    "",
    "Depends on [Ring attention PR #1252](https://github.com/ROCm/hip-ep/pull/1252). Merge that PR first, then rebase this contribution onto main. The numerical limitations above must be resolved or explicitly scoped before release.",
]
(OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "rows": len(rows),
            "completed": sum(r["status"] == "passed" for r in rows),
            "missing": sum(r["status"] == "missing" for r in rows),
            "failed": sum(r["status"] == "failed" for r in rows),
        }
    )
)
