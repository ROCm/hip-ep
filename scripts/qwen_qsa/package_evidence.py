#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Export portable paper evidence without running inference or modifying sources.

Use --verify to check a previously exported packet without the model or ROCm.
Large raw traces live in a local ZIP; small raw timing/quality rows are also
included in the versioned packet. Missing matrix cells remain explicit.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path, rows):
    assert rows
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_memory_case(folder, row):
    protocol = (folder / "protocol.json").read_bytes()
    samples = (folder / "samples.jsonl").read_bytes()
    assert digest(protocol) == row["protocol_sha256"]
    assert digest(samples) == row["samples_sha256"]
    for name, expected in row["source_sha256"].items():
        assert digest((folder / "source" / name).read_bytes()) == expected
    frames = [json.loads(line) for line in samples.splitlines()]
    assert frames
    mapping = {
        "peak_rss_bytes": "rss_bytes",
        "peak_private_commit_bytes": "private_commit_bytes",
        "worker_lifetime_peak_rss_bytes": "worker_lifetime_peak_rss_bytes",
        "worker_lifetime_peak_private_commit_bytes": "worker_lifetime_peak_private_commit_bytes",
        "peak_gpu_dedicated_bytes": "gpu_dedicated_bytes",
        "peak_gpu_shared_bytes": "gpu_shared_bytes",
        "peak_gpu_committed_bytes": "gpu_committed_bytes",
    }
    for phase, reported in row["memory"].items():
        selected = (
            frames
            if phase == "overall"
            else [frame for frame in frames if frame["phase"] == phase]
        )
        assert len(selected) == reported["samples"]
        for metric, field in mapping.items():
            values = [
                frame[field] for frame in selected if frame.get(field) is not None
            ]
            assert reported[metric] == (max(values) if values else None), (
                folder,
                phase,
                metric,
            )
        assert reported["minimum_system_ram_available_bytes"] == min(
            frame["system_ram_available_bytes"] for frame in selected
        )
    if row["status"] == "complete":
        worker = row["worker_result"]
        assert (
            worker["status"] == "complete"
            and worker["finite"]
            and worker["cache_pointers_unchanged"]
        )
        assert worker["prefill_calls"] == 1 and worker["decode_calls"] == 8
        assert worker["context"] == row["context"] and worker["batch"] == row["batch"]
        assert worker["capacity"] == row["context"] + 64


def verify(folder):
    manifest = load(folder / "manifest.json")
    for name, expected in manifest["files"].items():
        data = (folder / name).read_bytes()
        assert len(data) == expected["bytes"] and digest(data) == expected["sha256"], (
            name
        )
    for variant in ["Original", "Changed"]:
        base = folder / "raw/native/quality" / variant
        audit = load(base / "audit.json")
        for task in ["ppl", "gsm8k"]:
            data = gzip.decompress((base / (task + ".jsonl.gz")).read_bytes())
            assert digest(data) == audit["row_files_sha256"][task]
            rows = [json.loads(line) for line in data.splitlines()]
            assert len(rows) == (64 if task == "ppl" else 128)
            if task == "ppl":
                targets = sum(row["targets"] for row in rows)
                ppl = math.exp(sum(row["nll_sum"] for row in rows) / targets)
                assert targets == audit["targets"] and math.isclose(
                    ppl, audit["ppl"], rel_tol=1e-12
                )
            else:
                assert sum(row["correct"] for row in rows) == audit["gsm_correct"]
    for case in load(folder / "performance.json")["native"]:
        base = folder / "raw/native/performance" / f"{case['context']}-b{case['batch']}"
        summary, protocol = load(base / "summary.json"), load(base / "protocol.json")
        raw = (base / "rows.jsonl").read_bytes()
        assert digest(raw) == summary["rows_sha256"]
        assert (
            digest((base / "protocol.json").read_bytes()) == summary["protocol_sha256"]
        )
        rows = [json.loads(line) for line in raw.splitlines()]
        measured = [row for row in rows if not row.get("warmup", False)]
        for phase in ["prefill", "decode"]:
            means = {}
            for variant in ["Original", "Changed"]:
                selected = [row for row in measured if row["variant"] == variant]
                assert len(selected) == 6
                means[variant] = sum(row[phase + "_seconds"] for row in selected) / 6
                tokens = protocol["batch"] * (
                    protocol["context"] if phase == "prefill" else 64
                )
                assert math.isclose(
                    tokens / means[variant],
                    summary["phases"][phase][variant]["tokens_s"],
                    rel_tol=1e-12,
                )
            gain = 100 * (means["Original"] / means["Changed"] - 1)
            assert math.isclose(
                gain,
                summary["phases"][phase]["throughput_change_percent"],
                abs_tol=1e-10,
            )
    memory = load(folder / "memory.json")
    assert len(memory["cases"]) == 18
    assert (
        len({(x["batch"], x["context"], x["variant"]) for x in memory["cases"]}) == 18
    )
    descriptor = load(folder / "archive.json")
    archive_path = Path(descriptor["path"])
    if archive_path.exists():
        data = archive_path.read_bytes()
        assert len(data) == descriptor["bytes"] and digest(data) == descriptor["sha256"]
        with zipfile.ZipFile(archive_path) as archive:
            members = json.loads(archive.read("MANIFEST.json"))
            assert set(archive.namelist()) == set(members) | {"MANIFEST.json"}
            for name, expected in members.items():
                raw = archive.read(name)
                assert (
                    len(raw) == expected["bytes"] and digest(raw) == expected["sha256"]
                ), name
        print("Local raw archive and every member checksum verified.")
    else:
        print(
            "Local raw archive unavailable on this machine; portable files checked, archive checksum retained in archive.json."
        )
    print(
        "Packet hashes, raw quality totals, timing arithmetic and 18-cell memory coverage verified."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "sparse-attn")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/reproducibility/qwen-hip-ep"
    )
    parser.add_argument(
        "--archive",
        type=Path,
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
        return
    root, source, out = (
        args.evidence_root.resolve(),
        args.source_root.resolve(),
        args.output.resolve(),
    )
    work = root / "_local/qwen-qsa"
    if args.archive is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.archive = work / "paper-handoff-v1" / ("raw-evidence-" + stamp + ".zip")
    assert not args.archive.exists(), (
        "Preserve prior archives; choose a fresh --archive path"
    )
    assert not (root / "_local/qsa/gpu-stage.lock").exists(), (
        "Pause probes at a case boundary before making an immutable evidence packet"
    )
    out.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    raw_files = {}

    def capture(path, name, portable=False, compressed=False):
        if not path.exists():
            return
        data = path.read_bytes()
        raw_files[name] = data
        if portable:
            destination = out / "raw" / (name + (".gz" if compressed else ""))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                gzip.compress(data, mtime=0) if compressed else data
            )

    for path in work.glob("*.json"):
        capture(path, "native/receipts/" + path.name, portable=True)
    for path in work.glob("*.log"):
        capture(path, "native/logs/" + path.name)
    for path in [
        source / "HIP_EP_QWEN_HANDOFF.md",
        source / "QWEN_METHODS.md",
        source / "QWEN_PERFORMANCE.md",
        source / "docs/reproducibility/qwen-hip-ep-manifest.json",
    ]:
        capture(path, "mi355x/source-documents/" + path.name)
    for pattern in [
        "native-gqa/*validation.json",
        "native-gqa/compiler-tests.log",
        "native-gqa/build-command.json",
        "native-gqa/build.cmd",
        "native-gqa/models/*/export.json",
        "native-gqa/ort-profiles/*.json",
        "native-gqa/paired-performance/*/*",
        "native-gqa/quality/*/*",
        "timing-v1/*",
        "long-quality-v1/*.json",
        "long-quality-default/**/*",
        "memory-v1/**/*",
        "memory-only-v1/**/*",
        "audit/**/*",
    ]:
        for path in sorted(work.glob(pattern)):
            if not path.is_file() or path.suffix not in [
                ".json",
                ".jsonl",
                ".py",
                ".log",
                ".txt",
                ".cmd",
            ]:
                continue
            name = "native/" + path.relative_to(work).as_posix()
            name = name.replace(
                "native/native-gqa/paired-performance/", "native/performance/"
            )
            name = name.replace("native/native-gqa/quality/", "native/quality/")
            portable = (
                "/performance/" in name
                or "/quality/" in name
                or path.name.endswith("validation.json")
            )
            if "/audit/" in name and path.suffix == ".json":
                portable = True
            if "/quality/" in name and path.suffix not in [".json", ".jsonl"]:
                portable = False
            capture(
                path,
                name,
                portable,
                compressed="/quality/" in name and path.suffix == ".jsonl",
            )
    # Preserve OLMo and shared-runtime evidence for the complete stacked PR.
    olmo = root / "_local/qsa"
    for path in sorted((olmo / "stages").glob("qwen-*/**/*")):
        if path.is_file() and path.suffix in [".json", ".jsonl", ".py", ".log", ".txt"]:
            capture(
                path, "native/stages/" + path.relative_to(olmo / "stages").as_posix()
            )
    host_inventory = load(olmo / "environment.json")
    capture(olmo / "environment.json", "olmo/environment.json", portable=True)
    support = load(work / "default-4-support.json")
    assert host_inventory["torch"] == support["torch"]
    assert host_inventory["torch_hip"] == support["hip"]
    put(
        out / "environment.json",
        {
            "schema_version": 1,
            "scope": "Historical host inventory captured for the preceding OLMo campaign on the same workstation; GPU, Torch and HIP agree with Qwen support receipts. Available RAM at receipt is not a workload peak.",
            "inventory": host_inventory,
            "qwen_support_receipt": "raw/native/receipts/default-4-support.json",
            "transformers": load(work / "native-support.json")["transformers"],
            "memory_warning": "Distinguish installed-memory inventory, OS-managed RAM, WDDM dedicated/shared gauges and driver commitment; do not infer physical residency or add overlapping gauges.",
        },
    )
    for pattern in [
        "native-gqa/*audit.json",
        "native-gqa/*validation.json",
        "native-gqa/build-receipt.json",
        "native-gqa/paired-performance/**/*",
        "native-gqa/paired-quality/**/*",
        "native-gqa/gsm8k/**/*",
        "native-gqa/bfcl-v4/**/*",
    ]:
        for path in sorted(olmo.glob(pattern)):
            if path.is_file() and path.suffix in [".json", ".jsonl", ".txt", ".log"]:
                capture(path, "olmo/" + path.relative_to(olmo).as_posix())
    source_packet = source / "docs/reproducibility/qwen-trained-performance.json"
    mi355 = load(source_packet)
    capture(source_packet, "mi355x/qwen-trained-performance.json", portable=True)
    source_memory = source / "docs/reproducibility/memory-summary.json"
    mi355_memory = load(source_memory)
    capture(source_memory, "mi355x/memory-summary.json", portable=True)
    for relative, expected in {
        **mi355_memory["raw_sha256"],
        **mi355_memory["protocol_sha256"],
    }.items():
        path = source / "runs/303626" / relative
        assert digest(path.read_bytes()) == expected
        capture(path, "mi355x/runs/303626/" + relative, portable=True)
    for job in ["303195", "303045", "303046", "303626"]:
        for path in sorted((source / "runs" / job).glob("*")):
            if path.is_file() and path.suffix in [".json", ".jsonl", ".txt", ".log"]:
                capture(path, "mi355x/runs/" + job + "/" + path.name)
    source_raw = source / "runs/303195/performance.jsonl"
    assert digest(source_raw.read_bytes()) == mi355["raw_sha256"]
    handoff = load(work / "handoff-verification.json")
    identity = {
        key: handoff[key]
        for key in [
            "model",
            "revision",
            "selector_sha256",
            "parameters",
            "tensors",
            "data_sha256",
        ]
    }
    identity.update(
        schema_version=1,
        layers=[27, 31, 35, 39],
        training_updates_in_hip_ep=0,
        native_representation="Fresh RTN W4A16 group128; FP16 activation/KV; FP32 recurrent state; BF16 selector",
        mi355x_representation="Source BF16 backbone; frozen BF16 selector",
        selector_pt_sha256=mi355["selector_sha256"],
    )
    put(out / "identity.json", identity)
    performance = [
        load(path)
        for path in sorted(
            (work / "native-gqa/paired-performance").glob("*-b*/summary.json")
        )
    ]
    put(
        out / "performance.json",
        {
            "schema_version": 1,
            "native": performance,
            "mi355x": mi355,
            "join_warning": "Join by model revision, raw selector tensors, dataset IDs, context and batch. Precision, runtime, greedy vs teacher-forced inputs and statistics differ; do not pool absolute TPS or paired gains.",
        },
    )
    perf_rows = []
    for case in performance:
        for phase, values in case["phases"].items():
            perf_rows.append(
                {
                    "platform": "Radeon 8060S gfx1151",
                    "batch": case["batch"],
                    "context": case["context"],
                    "phase": phase,
                    "original_tokens_per_second": values["Original"]["tokens_s"],
                    "qsa_tokens_per_second": values["Changed"]["tokens_s"],
                    "throughput_change_percent": values["throughput_change_percent"],
                    "paired_95_low_percent": values["paired_95_interval_percent"][0],
                    "paired_95_high_percent": values["paired_95_interval_percent"][1],
                    "pairs": case["pairs"],
                    "statistic": "ratio_of_mean_phase_times",
                    "decode_inputs": "common_teacher_forced",
                }
            )
    table(out / "performance.csv", perf_rows)
    quality = {
        variant: load(work / "native-gqa/quality" / variant / "audit.json")
        for variant in ["Original", "Changed"]
    }
    source_quality = {
        variant: {
            task: load(source / "runs" / job / (task + "-summary.json"))
            for task in ["ppl", "gsm8k"]
        }
        for variant, job in [("Original", "303046"), ("Changed", "303045")]
    }
    put(
        out / "quality.json",
        {
            "schema_version": 1,
            "native": quality,
            "mi355x_source_bf16": source_quality,
            "scope": "64 x 4096 PPL windows; 128 GSM8K questions. GSM requests <=666 tokens never reach first pruning at position2052. QSA natural-long quality, Qwen BFCL and Qwen SWE-bench not measured.",
        },
    )
    table(
        out / "quality.csv",
        [
            {
                "platform": "Radeon 8060S gfx1151",
                "variant": variant,
                **{
                    key: row[key]
                    for key in [
                        "state",
                        "ppl_documents",
                        "targets",
                        "nll",
                        "ppl",
                        "gsm_questions",
                        "gsm_correct",
                        "gsm_caps",
                    ]
                },
                "gsm_accuracy_percent": 100 * row["gsm_correct"] / row["gsm_questions"],
            }
            for variant, row in quality.items()
        ],
    )
    memory_cases, memory_rows = [], []
    for context in [4096, 16384, 32768]:
        for batch in [1, 4, 8]:
            for variant in ["Original", "Changed"]:
                case_path = (
                    work
                    / "memory-only-v1"
                    / f"b{batch}-s{context}-{variant}"
                    / "result.json"
                )
                row = (
                    load(case_path)
                    if case_path.exists()
                    else {
                        "variant": variant,
                        "batch": batch,
                        "context": context,
                        "status": "pending",
                    }
                )
                if case_path.exists():
                    audit_memory_case(case_path.parent, row)
                memory_cases.append(row)
                for phase, values in row.get("memory", {"overall": {}}).items():
                    memory_rows.append(
                        {
                            "batch": batch,
                            "context": context,
                            "capacity": context + 64,
                            "variant": variant,
                            "status": row["status"],
                            "phase": phase,
                            "complete_peak": row["status"] == "complete",
                            **values,
                        }
                    )
    put(
        out / "memory.json",
        {
            "schema_version": 1,
            "cases": memory_cases,
            "status_counts": {
                state: sum(case["status"] == state for case in memory_cases)
                for state in sorted({case["status"] for case in memory_cases})
            },
            "mi355x_reference": "raw/mi355x/memory-summary.json",
            "cross_platform_memory_warning": "MI355X job303626 uses one warmup and two64-decode requests, PyTorch allocation/reservation peaks and20ms device-HBM sampling. Native uses one cold8-decode request and100ms WDDM commitment. Do not compare these as the same memory metric.",
            "unit": "bytes",
            "scope": "One fresh session, one prefill, eight decode calls, 100ms target sampling. WDDM commitment is not physical residency; do not add it to process RSS or system RAM.",
            "excluded_context": 65536,
            "excluded_reason": "User canceled after co-resident performance attempt hit system RAM guard; no successful 64K timing.",
        },
    )
    table(out / "memory.csv", memory_rows)
    # Bind measured binaries to the source commit that introduced them, not to
    # the documentation/merge commit being prepared now. This is a reconstruction,
    # since the original build began with dirty HEAD 2a43690.
    native_paths = [
        name
        for name in git(root, "diff", "--name-only", "1c9a454...7a71b72").splitlines()
        if name.startswith(("lib/", "include/", "morphizen/"))
    ]
    source_map = {}
    for name in native_paths:
        data = subprocess.check_output(
            ["git", "-C", str(root), "show", "7a71b72:" + name]
        )
        source_map[name] = {
            "git_blob": git(root, "rev-parse", "7a71b72:" + name),
            "sha256": digest(data),
        }
        raw_files["measured-source/" + name] = data
    script_commit = "efe78ea"
    for name in git(
        root, "diff", "--name-only", "1c9a454..." + script_commit
    ).splitlines():
        if name.startswith("scripts/"):
            raw_files["historical-experiment-source/" + name] = subprocess.check_output(
                ["git", "-C", str(root), "show", script_commit + ":" + name]
            )
    put(
        out / "provenance.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_checkout_commit": git(root, "rev-parse", "HEAD"),
            "packet_tool_commit": git(ROOT, "rev-parse", "HEAD"),
            "local_origin_main": git(ROOT, "rev-parse", "origin/main"),
            "source_reference_commit": "3b0c30ea4976085798a36d22cefa4089bbef6543",
            "source_checkout_commit": git(source, "rev-parse", "HEAD"),
            "mi355x_memory_source_commit": mi355_memory["source_commit"],
            "packet_script_sha256": digest(Path(__file__).read_bytes()),
            "historical_experiment_script_commit": script_commit,
            "source_line_endings": "Git source reconstructions use repository-normalized bytes; measured Windows source hashes can include CRLF. Original raw protocols and per-probe source snapshots preserve actual execution bytes.",
            "measured_native_source_reconstruction": source_map,
            "build_provenance_limit": "Measured build began at dirty HEAD2a43690; native implementation subsequently committed in7a71b72. Binary hashes and execution receipts are authoritative; final PR merge build is separate and has no inherited performance claims.",
            "errata": [
                "Historical generic run_stage.py receipts for Qwen hashed the OLMo selector and only snapshotted scripts/qsa. Use the Qwen execution protocols/export/tensor hashes as model identity; the stage selector fields are not Qwen identity. Future runner selects the Qwen payload and snapshots both script directories.",
                "Historical native-support.json recorded unused gqa_back.cpp; compiled implementation is real/gqa.cpp. Inspector corrected without rewriting historical receipt.",
                "32K paired timing memory sampler attached after four measured pairs; 4K/16K timing had no sampler.",
                "Batch4/4K default memory probe completed after1551.55s. Stop attempt found parent already gone; retain as complete, not timing or failure.",
            ],
        },
    )
    with zipfile.ZipFile(
        args.archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, data in sorted(raw_files.items()):
            archive.writestr(name, data)
        archive.writestr(
            "MANIFEST.json",
            json.dumps(
                {
                    name: {"bytes": len(data), "sha256": digest(data)}
                    for name, data in sorted(raw_files.items())
                },
                indent=2,
            )
            + "\n",
        )
    put(
        out / "archive.json",
        {
            "path": str(args.archive.resolve()),
            "bytes": args.archive.stat().st_size,
            "sha256": digest(args.archive.read_bytes()),
            "members": len(raw_files),
            "scope": "Local immutable raw evidence; no model weights, no cache binaries. Every member has SHA256 in MANIFEST.json. Not uploaded.",
        },
    )
    names = [
        "environment.json",
        "identity.json",
        "performance.json",
        "performance.csv",
        "quality.json",
        "quality.csv",
        "memory.json",
        "memory.csv",
        "provenance.json",
        "archive.json",
    ]
    names.extend(
        path.relative_to(out).as_posix()
        for path in (out / "raw").rglob("*")
        if path.is_file()
    )
    if (out / "candidate-validation.json").exists():
        names.append("candidate-validation.json")
    put(
        out / "manifest.json",
        {
            "schema_version": 1,
            "files": {
                name: {
                    "bytes": (out / name).stat().st_size,
                    "sha256": digest((out / name).read_bytes()),
                }
                for name in sorted(names)
            },
        },
    )
    verify(out)
    print(f"Archived {len(raw_files)} source/evidence files to {args.archive}")


if __name__ == "__main__":
    main()
