#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Synchronize Qwen performance documentation from the immutable charged-run data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs/reproducibility/qwen-eight-hip-ep"
BEGIN = "<!-- QWEN_EIGHT_PERFORMANCE_BEGIN -->"
END = "<!-- QWEN_EIGHT_PERFORMANCE_END -->"
REPORT = ROOT / "docs/experiments/qwen-qsa-eight-selector-results.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(source, target):
    return Path(os.path.relpath(target, source.parent)).as_posix()


def measured_results():
    path = PACKET / "eight-140w-performance.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["state"] == "complete"
    assert result["selector_layers"] == [3, 7, 23, 27, 31, 35, 39, 63]
    for row in result["contexts"]:
        folder = PACKET / "raw/eight-140w" / str(row["context"])
        summary = json.loads((folder / "summary.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        assert summary["rows_sha256"] == digest(folder / "rows.jsonl")
        assert summary["protocol_sha256"] == digest(folder / "protocol.json")
        assert protocol["warmups"] == 1 and protocol["measured_requests"] == 3
        trials = [
            json.loads(s) for s in (folder / "rows.jsonl").read_text().splitlines()
        ]
        assert [r["trial"] for r in trials] == [-1, 0, 1, 2]
        for phase, count in [("prefill", row["context"]), ("decode", 64)]:
            elapsed = statistics.mean(
                r[phase + "_seconds"] for r in trials if not r["warmup"]
            )
            expected = row[phase]["eight_tokens_s"]
            assert abs(count / elapsed - expected) < 1e-9
            four = row[phase]["four_historical_tokens_s"]
            original = row[phase]["original_historical_tokens_s"]
            assert (
                abs(100 * (expected / four - 1) - row[phase]["eight_vs_four_percent"])
                < 1e-9
            )
            assert (
                abs(
                    100 * (expected / original - 1)
                    - row[phase]["eight_vs_original_percent"]
                )
                < 1e-9
            )
    return result


def delta_table(rows):
    lines = [
        "| Context | Eight vs four: prefill | Eight vs four: decode |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['context'] // 1024}K | {r['prefill']['eight_vs_four_percent']:+.2f}% | {r['decode']['eight_vs_four_percent']:+.2f}% |"
        )
    return "\n".join(lines)


def full_table(rows):
    lines = [
        "| Context | Phase | Default tok/s (reused) | Four QSA tok/s (reused) | Eight QSA tok/s | Eight vs default | Eight vs four |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        for phase in ["prefill", "decode"]:
            d = r[phase]
            lines.append(
                f"| {r['context'] // 1024}K | {phase} | {d['original_historical_tokens_s']:.2f} | {d['four_historical_tokens_s']:.2f} | {d['eight_tokens_s']:.2f} | {d['eight_vs_original_percent']:+.2f}% | {d['eight_vs_four_percent']:+.2f}% |"
            )
    return "\n".join(lines)


def put_block(path, body, anchor=None):
    text = path.read_text(encoding="utf-8")
    block = BEGIN + "\n" + body.strip() + "\n" + END
    if BEGIN in text:
        text, count = re.subn(
            re.escape(BEGIN) + r".*?" + re.escape(END),
            lambda _: block,
            text,
            flags=re.S,
        )
        assert count == 1
    elif anchor:
        assert anchor in text, (path, anchor)
        text = text.replace(anchor, block + "\n\n" + anchor, 1)
    else:
        match = re.search(r"^# .+\n", text, re.M)
        assert match, path
        text = text[: match.end()] + "\n" + block + "\n" + text[match.end() :]
    if path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def update_documents():
    result = measured_results()
    if result is None:
        return []
    rows = result["contexts"]
    limits = (
        "Eight uses one warmup and three measured requests at batch one, with 64 common teacher-forced decode inputs. "
        "Default/four controls are reused across runs (two warmups, six requests, two resident sessions sharing cache buffers); "
        "Eight has one resident session. No paired confidence interval or isolated layer-count causal claim applies. "
        "Both prefill and decode use QSA at all eight selected layers."
    )
    conclusion = (
        "**Decode throughput improved over four selectors at every measured length.** "
        "Prefill is slower at 4K/16K and slightly faster at 32K. "
        "All eight weights, including the overlapping four positions, come from checkpoint 640; this is not a layer-count-only change."
    )
    quality = (
        "Eight-selector BFCL/GSM8K accuracy and application-memory comparisons have not been run on hip-ep. "
        "The source checkpoint still misses its fixed 32K PPL gate. Historical four-selector accuracy, memory and paired intervals do not transfer to Eight."
    )
    bfcl_summary_path = PACKET / "bfcl-eight-summary.json"
    bfcl_status_path = PACKET / "bfcl-eight-status.json"
    bfcl_report = ROOT / "docs/experiments/qwen-qsa-eight-bfcl-v4-50.md"
    if bfcl_summary_path.exists():
        bfcl = json.loads(bfcl_summary_path.read_text(encoding="utf-8"))
        assert bfcl["state"] == "complete"
        quality = (
            f"The fixed 50-case BFCL component pilot scores Eight {bfcl['variants']['Eight']['correct']}/50 "
            f"versus reused Original {bfcl['variants']['Original']['correct']}/50; no new Original generation was run. "
            "This is not the official overall BFCL score. Eight GSM8K and application-memory comparisons remain unmeasured. "
            "The source checkpoint still misses its fixed 32K PPL gate; this pilot does not establish accepted long-context quality."
        )
    elif bfcl_status_path.exists():
        quality = (
            "The user-authorized Eight-only BFCL50 pilot is running with the prior Original baseline reused. "
            "No final Eight accuracy result is available yet. Eight GSM8K and application-memory comparisons remain unmeasured. "
            "The source checkpoint still misses its fixed 32K PPL gate."
        )
    header = "<!--\nCopyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.\nLicensed under the MIT License.\n-->\n"
    report = (
        header + "# Qwen eight-selector QSA: performance through 32K\n\n"
        "Qwen/Qwen3.8-27B on Radeon 8060S/gfx1151, Windows. The unchanged native W4A16 group128 backbone uses FP16 activation/KV and FP32 recurrent state; selectors are BF16. "
        "Zero-based QSA layers are **3, 7, 23, 27, 31, 35, 39, 63**. No training, learned calibration or kernel/compiler rebuild occurred.\n\n"
        + conclusion
        + "\n\n"
        + full_table(rows)
        + "\n\n"
        + limits
        + "\n\n"
        "Each Eight value is phase token count divided by mean synchronized CPU-wall seconds over all three measured requests, not mean per-request TPS. "
        "The raw rows include one warmup (`trial=-1`) and three measured requests (`0,1,2`). Allocation, binding, recurrent zeroing and load/compile are outside phase timers. "
        "Finite outputs, exact replay, stable cache addresses and unchanged sessions pass at every shape.\n\n"
        "The 140W charger rating is user-reported. All sampled AC states stayed connected. The separate Original 4K charger control recovered to 431.71 prefill / 10.79 decode tok/s, "
        "within 1.45% / 1.01% of the archive. Only the 4K Original was freshly checked after the charger change. "
        "The earlier pre-charger/suspected-contention run remains preserved and excluded from this clean comparison; a specific cause is not proven by the before/after observation.\n\n"
        "The locked timing prefixes concatenate known 4K test windows; they are not a natural-long quality holdout. 64K and batched measurements remain excluded.\n\n"
        + quality
        + "\n\n"
        f"See the [paper handoff]({link(REPORT, PACKET / 'README.md')}), [performance JSON]({link(REPORT, PACKET / 'eight-140w-performance.json')}), "
        f"[CSV]({link(REPORT, PACKET / 'eight-140w-performance.csv')}) and [runtime-estimate JSON]({link(REPORT, PACKET / 'evaluation-time-estimate.json')}). "
        "The handoff links all raw rows, saved aggregates, protocols, exact identities and local immutable ZIP receipts. Runtime estimates are planning ranges; the current BFCL status is stated above.\n\n"
        "Regenerate the linked performance documentation with `python scripts/qwen_qsa/performance_docs.py` from the repository root. It verifies the saved raw-row arithmetic and hashes and uses no GPU.\n"
    )
    REPORT.write_text(report, encoding="utf-8")
    changed = [REPORT]
    eight_readme = PACKET / "README.md"
    put_block(
        eight_readme,
        "## Four versus eight selectors\n\n"
        + conclusion
        + "\n\n"
        + delta_table(rows)
        + "\n\n"
        + limits
        + f" See the [standalone performance report]({link(eight_readme, REPORT)}).",
        "## Estimated evaluation durations",
    )
    changed.append(eight_readme)
    legacy = {
        "docs/experiments/qwen-qsa-native-gqa-results.md": "The historical four-selector campaign and its quality tables remain below.",
        "docs/reproducibility/qwen-hip-ep/README.md": "This packet's identities, performance, accuracy and memory files remain the historical four-selector evidence; the eight-selector evidence is a separate packet.",
        "docs/reproducibility/qwen-hip-ep/AUDIT.md": "The audit below describes the historical implementation/merge checks. Eight reuses the measured native build; no new merged-candidate throughput campaign is claimed.",
        "docs/reproducibility/qwen-hip-ep/PR_DESCRIPTION.md": "This performance addendum belongs to local branch `codex/qwen-eight-qsa-perf`; the historical four-selector merge-candidate validation below remains separately identified.",
        "docs/design/qwen-qsa-native-gqa.md": "The same native operator/math/cache contract now has four- and eight-selector exports. The eight-selector variant reuses the measured kernels and compiler.",
        "docs/experiments/qwen-qsa-bfcl-v4-50.md": "All accuracy results below belong to the historical four-selector checkpoint. The Eight-only pilot is tracked separately in the latest paper handoff.",
        "docs/experiments/qwen-qsa-memory.md": "All memory measurements below belong to four selectors. Host RAM guards in the eight-selector performance run are not a new GPU-memory comparison.",
    }
    for name, note in legacy.items():
        path = ROOT / name
        body = (
            "## Latest eight-selector performance\n\n"
            + note
            + "\n\n"
            + conclusion
            + "\n\n"
            + delta_table(rows)
        )
        body += "\n\n" + limits + "\n\n" + quality
        body += f" See the [eight-selector report]({link(path, REPORT)}) and [raw-data paper handoff]({link(path, eight_readme)})."
        put_block(path, body)
        changed.append(path)
    root_readme = ROOT / "README.md"
    put_block(
        root_readme,
        f"- [Qwen eight-selector QSA: latest 4K/16K/32K performance]({link(root_readme, REPORT)})\n- [Qwen eight-selector paper packet and raw measurements]({link(root_readme, eight_readme)})",
        "- [Qwen isolated default/QSA memory-only matrix]",
    )
    changed.append(root_readme)
    if bfcl_report.exists():
        changed.append(bfcl_report)
    registry = dict(
        source_performance_sha256=digest(PACKET / "eight-140w-performance.json"),
        generator_sha256=digest(Path(__file__)),
        document_hash_normalization="UTF-8 text with LF line endings",
        documents={
            p.relative_to(ROOT).as_posix(): dict(
                normalized_lf_sha256=hashlib.sha256(
                    p.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest(),
                normalized_utf8_bytes=len(
                    p.read_text(encoding="utf-8").encode("utf-8")
                ),
            )
            for p in changed
        },
    )
    if bfcl_summary_path.exists():
        registry["bfcl_eight_summary_sha256"] = digest(bfcl_summary_path)
    if bfcl_status_path.exists():
        registry["bfcl_eight_status_sha256"] = digest(bfcl_status_path)
    (PACKET / "documentation.json").write_text(json.dumps(registry, indent=2) + "\n")
    manifest_path = PACKET / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state"] = (
        "eight_charged_performance_and_bfcl_complete"
        if bfcl_summary_path.exists()
        else "eight_charged_performance_complete_estimates_only"
    )
    manifest["files"] = {
        p.relative_to(PACKET).as_posix(): dict(sha256=digest(p), bytes=p.stat().st_size)
        for p in sorted(PACKET.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return changed


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for path in update_documents():
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
