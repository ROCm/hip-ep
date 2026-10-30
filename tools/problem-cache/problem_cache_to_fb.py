#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Compile a rocMLIR LUT JSON file to a FlatBuffer.

The JSON is the document ``hip-rocmlir-compiler --dump-problem-cache`` writes.
It has the same shape as the MatMulNBits and GQA LUTs: ``schema_version``,
``gpu_arch``, ``kernel_abi``, distance weights, then ``configs``, ``points``,
and ``fallbacks``. ``flatc --binary --strict-json`` turns it into a ``.fb``.

    python problem_cache_to_fb.py gfx1151.json -o gfx1151.fb
    python problem_cache_to_fb.py more.json -o gfx1151.fb --flatbuffer gfx1151.fb

``--flatbuffer`` merges by problem string. A problem already in the table keeps
its place and takes the incoming perfConfig. New problems are appended. Configs
are de-duplicated again after the merge.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
KERNEL_ABI = "rocmlir-conv-v1"
HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "problem_cache.fbs"


def require_lut(doc: dict) -> dict:
    if not isinstance(doc, dict) or "configs" not in doc or "points" not in doc:
        raise ValueError(
            "expected a rocMLIR LUT JSON object with configs and points, "
            "not a MIGraphX problem-cache dump"
        )
    version = int(doc.get("schema_version") or 0)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}")
    abi = str(doc.get("kernel_abi") or "")
    if abi != KERNEL_ABI:
        raise ValueError(f"unsupported kernel_abi {abi!r}")
    configs = doc["configs"]
    points = doc["points"]
    fallbacks = doc.get("fallbacks") or []
    if not isinstance(configs, list) or not isinstance(points, list):
        raise ValueError("configs and points must be arrays")
    if not isinstance(fallbacks, list):
        raise ValueError("fallbacks must be an array")
    for config in configs:
        if not isinstance(config, dict) or not isinstance(config.get("perf_config"), str):
            raise ValueError("each config must be {perf_config: string}")
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("each point must be an object")
        if not isinstance(point.get("problem"), str):
            raise ValueError("each point needs a problem string")
        index = int(point["config"])
        if index < 0 or index >= len(configs):
            raise ValueError(f"point config index {index} is out of range")
        for dim in ("m", "n", "k"):
            int(point[dim])
    return doc


def perf_configs(doc: dict) -> list[str]:
    return [str(config["perf_config"]) for config in doc["configs"]]


def points_of(doc: dict) -> list[dict]:
    configs = perf_configs(doc)
    rows = []
    for point in doc["points"]:
        rows.append(
            {
                "problem": str(point["problem"]),
                "perf_config": configs[int(point["config"])],
                "m": int(point["m"]),
                "n": int(point["n"]),
                "k": int(point["k"]),
            }
        )
    return rows


def build_document(header: dict, rows: list[dict]) -> dict:
    pool: list[str] = []
    index: dict[str, int] = {}

    def config_index(perf_config: str) -> int:
        slot = index.get(perf_config)
        if slot is None:
            slot = len(pool)
            index[perf_config] = slot
            pool.append(perf_config)
        return slot

    point_rows = []
    for row in rows:
        point_rows.append(
            {
                "problem": row["problem"],
                "config": config_index(row["perf_config"]),
                "m": int(row["m"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "gpu_arch": str(header.get("gpu_arch") or ""),
        "rocm_version": int(header.get("rocm_version") or 0),
        "kernel_abi": KERNEL_ABI,
        "model_key": str(header.get("model_key") or ""),
        "weight_m": float(header.get("weight_m") or 1.0),
        "weight_n": float(header.get("weight_n") or 1.0),
        "weight_k": float(header.get("weight_k") or 1.0),
        "configs": [{"perf_config": perf} for perf in pool],
        "points": point_rows,
        "fallbacks": [],
    }


def merge_rows(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int, int]:
    """Keep existing order. Incoming replaces a matching problem; new rows append."""
    index = {row["problem"]: i for i, row in enumerate(existing)}
    merged = list(existing)
    replaced = 0
    added = 0
    for row in incoming:
        slot = index.get(row["problem"])
        if slot is None:
            index[row["problem"]] = len(merged)
            merged.append(row)
            added += 1
            continue
        if merged[slot]["perf_config"] != row["perf_config"]:
            replaced += 1
        merged[slot] = row
    return merged, added, replaced


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


def decode_flatbuffer(flatc: str, path: Path) -> dict:
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
        return require_lut(json.loads(produced[0].read_text(encoding="utf-8")))


def encode_flatbuffer(flatc: str, document: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        table_path = tmp_path / "rocmlir_lut.json"
        table_path.write_text(
            json.dumps(document, ensure_ascii=False),
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
        produced = list(tmp_path.glob("*.bin")) + list(tmp_path.glob("*.fb"))
        if len(produced) != 1:
            raise SystemExit("flatc did not write a binary table")
        shutil.copyfile(produced[0], output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json",
        type=Path,
        help="LUT JSON from hip-rocmlir-compiler --dump-problem-cache",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help=".fb to write")
    parser.add_argument(
        "--flatbuffer",
        type=Path,
        default=None,
        help="Existing .fb to merge this JSON into",
    )
    parser.add_argument("--flatc", default=None, help="flatc executable")
    args = parser.parse_args(argv)

    if not args.json.is_file():
        raise SystemExit(f"JSON not found: {args.json}")
    if not SCHEMA.is_file():
        raise SystemExit(f"schema not found: {SCHEMA}")
    flatc = find_flatc(args.flatc)

    incoming = require_lut(json.loads(args.json.read_text(encoding="utf-8")))
    header = incoming
    rows = points_of(incoming)
    if args.flatbuffer is not None:
        if not args.flatbuffer.is_file():
            raise SystemExit(f"flatbuffer not found: {args.flatbuffer}")
        existing = decode_flatbuffer(flatc, args.flatbuffer)
        header = existing
        rows, added, replaced = merge_rows(points_of(existing), rows)
    else:
        added = len(rows)
        replaced = 0

    document = build_document(header, rows)
    encode_flatbuffer(flatc, document, args.output)
    print(
        f"wrote {args.output} points={len(document['points'])} "
        f"configs={len(document['configs'])} added={added} replaced={replaced}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
