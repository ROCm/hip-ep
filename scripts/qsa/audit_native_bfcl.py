#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Independently rescore every selected BFCL pilot row with official inputs."""

import copy
import argparse
import hashlib
import json
import subprocess

import bfcl_native_eval as bfcl


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompts(value):
    if isinstance(value, dict):
        result = [value["formatted_prompt"]] if "formatted_prompt" in value else []
        return result + [p for v in value.values() for p in prompts(v)]
    if isinstance(value, list):
        return [p for v in value for p in prompts(v)]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Audit completed pairs without emitting a completed pilot receipt",
    )
    args = parser.parse_args()
    task = bfcl.TASK
    manifest_path = task / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["bfcl_revision"] == bfcl.REVISION
    assert len(manifest["cases"]) == 128
    assert len({c["id"] for c in manifest["cases"]}) == 128
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=bfcl.SOURCE, text=True
        ).strip()
        == bfcl.REVISION
    )
    assert not subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "diff", "--name-only"],
        cwd=bfcl.SOURCE,
        text=True,
    ).strip()
    rows_by_key, entries, truth, hashes = {}, {}, {}, {}
    for category in sorted({c["category"] for c in manifest["cases"]}):
        path = task / category / "rows.jsonl"
        if args.partial and not path.exists():
            continue
        hashes[str(path.relative_to(task))] = sha(path)
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            key = row["variant"], row["id"]
            assert key not in rows_by_key, key
            rows_by_key[key] = row
        entries[category] = bfcl.load_dataset_entry(
            category, include_prereq=False, include_language_specific_hint=False
        )
        truth[category] = (
            [] if "relevance" in category else bfcl.load_ground_truth_entry(category)
        )
    models = {v: bfcl.handler(v) for v in ["Original", "Changed"]}
    selected, missing = [], []
    for case in manifest["cases"]:
        if args.partial and any((v, case["id"]) not in rows_by_key for v in models):
            missing.append(case["id"])
            continue
        pair = []
        category, index = case["category"], case["index"]
        for variant in models:
            row = rows_by_key[variant, case["id"]]
            assert row["index"] == index and row["bfcl_revision"] == bfcl.REVISION
            assert row["pair_index"] == case["pair_index"]
            assert row["order"] == (
                case["pair_index"] % 2
                if variant == "Original"
                else 1 - case["pair_index"] % 2
            )
            gold = (
                None
                if not truth[category]
                else copy.deepcopy(truth[category][index]["ground_truth"])
            )
            checked = bfcl.score(
                models[variant], entries[category][index], row["result"], gold, category
            )
            assert checked["valid"] == row["correct"], (variant, case["id"])
            pair.append(row)
            selected.append(row)
        original_prompts, changed_prompts = [prompts(r["metadata"]) for r in pair]
        assert original_prompts and changed_prompts, case["id"]
        assert original_prompts[0] == changed_prompts[0], case["id"]
    assert len(selected) > 0 and len(selected) % 2 == 0
    if not args.partial:
        assert len(selected) == 256
    receipt = {
        "status": "passed",
        "scope": "128-case stratified single-turn/multi-turn pilot; not official BFCL overall",
        "bfcl_revision": bfcl.REVISION,
        "cases": len(selected) // 2,
        "complete": len(selected) == 256,
        "missing_pairs": missing,
        "rows": len(selected),
        "categories": len(entries),
        "manifest_sha256": sha(manifest_path),
        "protocol_sha256": sha(task / "protocol.json"),
        "row_files_sha256": hashes,
        "first_prompts_identical": True,
        "official_scores_recomputed": True,
        "training_updates": 0,
    }
    bfcl.write_json(
        task / ("pilot-partial-audit.json" if args.partial else "pilot-audit.json"),
        receipt,
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
