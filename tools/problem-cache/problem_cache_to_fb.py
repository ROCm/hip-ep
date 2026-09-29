#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Convert a problem-cache JSON dump to a FlatBuffer.

The JSON is the file written by ``MIGRAPHX_PROBLEM_CACHE`` or by
``hip-rocmlir-compiler --dump-problem-cache``: an array of
``[device, entries]`` buckets. A legacy flat object whose keys are JSON
``{name, problem}`` strings is accepted too. Null solutions are skipped.

``problem`` and ``solution`` are stored as text. A string value is stored as
itself (a rocMLIR problem key or ``gemm:`` perfConfig). Any other JSON value
is stored as canonical JSON.

Pass ``--flatbuffer`` to merge into an existing table. Rows already present,
matched on device, operator name, and problem text, keep their place and take
the incoming solution. New rows are appended.

    python problem_cache_to_fb.py cache.json -o cache.fb
    python problem_cache_to_fb.py more.json -o cache.fb --flatbuffer cache.fb
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "problem_cache.fbs"


def canonical_text(value) -> str | None:
    """Store a cache value as the text the table keeps."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_entry_key(key) -> tuple[str, object]:
    if isinstance(key, str):
        key = json.loads(key)
    if not isinstance(key, dict) or "name" not in key or "problem" not in key:
        raise ValueError("problem-cache entry key must be {name, problem}")
    return str(key["name"]), key["problem"]


def device_of(obj: dict) -> dict:
    return {
        "device_name": str(obj.get("device_name") or ""),
        "gfx_name": str(obj.get("gfx_name") or ""),
        "cu_count": int(obj.get("cu_count") or 0),
        "wavefront_size": int(obj.get("wavefront_size") or 0),
    }


def empty_device() -> dict:
    return device_of({})


def iter_pairs(entries):
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("problem-cache entry must be a [key, solution] pair")
            yield item[0], item[1]
        return
    if isinstance(entries, dict):
        yield from entries.items()
        return
    raise ValueError("problem-cache entries must be a list of pairs or an object")


def make_record(device: dict, key, solution) -> dict | None:
    name, problem = parse_entry_key(key)
    problem_text = canonical_text(problem)
    solution_text = canonical_text(solution)
    if not name or problem_text is None or solution_text is None:
        return None
    return {
        "device_name": device["device_name"],
        "gfx_name": device["gfx_name"],
        "cu_count": device["cu_count"],
        "wavefront_size": device["wavefront_size"],
        "name": name,
        "problem": problem_text,
        "solution": solution_text,
    }


def load_problem_cache(path: Path) -> tuple[list[dict], int]:
    root = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    skipped = 0

    def add(device: dict, key, solution) -> None:
        nonlocal skipped
        record = make_record(device, key, solution)
        if record is None:
            skipped += 1
            return
        records.append(record)

    if isinstance(root, list):
        for bucket in root:
            if not isinstance(bucket, list) or len(bucket) != 2 or not isinstance(bucket[0], dict):
                raise ValueError(
                    "expected a device-keyed problem cache: [[device, entries], ...]"
                )
            device = device_of(bucket[0])
            for key, solution in iter_pairs(bucket[1]):
                add(device, key, solution)
    elif isinstance(root, dict):
        device = empty_device()
        for key, solution in root.items():
            add(device, key, solution)
    else:
        raise ValueError("problem cache JSON must be an array or an object")
    return records, skipped


def record_key(record: dict) -> tuple:
    return (
        record["device_name"],
        record["gfx_name"],
        int(record["cu_count"]),
        int(record["wavefront_size"]),
        record["name"],
        record["problem"],
    )


def merge_records(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int, int]:
    """Keep existing order. Incoming replaces a matching row; new rows append."""
    index = {record_key(record): i for i, record in enumerate(existing)}
    merged = list(existing)
    replaced = 0
    added = 0
    for record in incoming:
        key = record_key(record)
        slot = index.get(key)
        if slot is None:
            index[key] = len(merged)
            merged.append(record)
            added += 1
            continue
        if merged[slot]["solution"] != record["solution"]:
            replaced += 1
        merged[slot] = record
    return merged, added, replaced


def as_table(records: list[dict]) -> dict:
    entries = []
    for record in records:
        entries.append(
            {
                "device_name": record["device_name"],
                "gfx_name": record["gfx_name"],
                "cu_count": int(record["cu_count"]),
                "wavefront_size": int(record["wavefront_size"]),
                "name": record["name"],
                "problem": record["problem"],
                "solution": record["solution"],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "entries": entries}


def records_from_table(table: dict) -> list[dict]:
    if "entries" not in table and "ProblemCache" in table:
        table = table["ProblemCache"]
    version = int(table.get("schema_version") or SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported problem-cache flatbuffer schema version {version}")
    records = []
    for entry in table.get("entries") or []:
        records.append(
            {
                "device_name": str(entry.get("device_name") or ""),
                "gfx_name": str(entry.get("gfx_name") or ""),
                "cu_count": int(entry.get("cu_count") or 0),
                "wavefront_size": int(entry.get("wavefront_size") or 0),
                "name": str(entry.get("name") or ""),
                "problem": str(entry.get("problem") or ""),
                "solution": str(entry.get("solution") or ""),
            }
        )
    return records


def find_flatc(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("flatc")
    if found:
        return found
    therock = os.environ.get("THEROCK_DIST", "")
    if therock:
        candidate = Path(therock) / "bin" / "flatc.exe"
        if candidate.is_file():
            return str(candidate)
    raise SystemExit("flatc not found; pass --flatc or set THEROCK_DIST")


def run_flatc(flatc: str, args: list[str]) -> None:
    proc = subprocess.run([flatc, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"flatc failed: {detail}")


def decode_flatbuffer(flatc: str, path: Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run_flatc(
            flatc,
            [
                "--json",
                "--raw-binary",
                "--strict-json",
                "-o",
                str(tmp_path),
                str(SCHEMA),
                "--",
                str(path),
            ],
        )
        produced = list(tmp_path.glob("*.json"))
        if len(produced) != 1:
            raise SystemExit(f"flatc did not write a JSON table for {path}")
        table = json.loads(produced[0].read_text(encoding="utf-8"))
    return records_from_table(table)


def encode_flatbuffer(flatc: str, records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        table_path = tmp_path / "problem_cache.json"
        table_path.write_text(
            json.dumps(as_table(records), ensure_ascii=False),
            encoding="utf-8",
        )
        run_flatc(
            flatc,
            [
                "--binary",
                "--strict-json",
                "-o",
                str(tmp_path),
                str(SCHEMA),
                str(table_path),
            ],
        )
        produced = list(tmp_path.glob("*.bin")) + list(tmp_path.glob("*.mxpc"))
        if len(produced) != 1:
            raise SystemExit("flatc did not write a binary table")
        shutil.copyfile(produced[0], output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json",
        type=Path,
        help="problem-cache JSON from MIGRAPHX_PROBLEM_CACHE or "
        "hip-rocmlir-compiler --dump-problem-cache",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="FlatBuffer to write")
    parser.add_argument(
        "--flatbuffer",
        type=Path,
        default=None,
        help="Existing FlatBuffer to append this JSON onto",
    )
    parser.add_argument("--flatc", default=None, help="flatc executable")
    args = parser.parse_args(argv)

    if not args.json.is_file():
        raise SystemExit(f"JSON not found: {args.json}")
    if not SCHEMA.is_file():
        raise SystemExit(f"schema not found: {SCHEMA}")
    flatc = find_flatc(args.flatc)

    incoming, skipped = load_problem_cache(args.json)
    existing: list[dict] = []
    if args.flatbuffer is not None:
        if not args.flatbuffer.is_file():
            raise SystemExit(f"flatbuffer not found: {args.flatbuffer}")
        existing = decode_flatbuffer(flatc, args.flatbuffer)

    merged, added, replaced = merge_records(existing, incoming)
    encode_flatbuffer(flatc, merged, args.output)
    print(
        f"wrote {args.output} entries={len(merged)} "
        f"added={added} replaced={replaced} skipped_null={skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
