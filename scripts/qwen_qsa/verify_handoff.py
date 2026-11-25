#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify the read-only trained Qwen handoff without loading a model or using a GPU."""

import array
import argparse
import collections
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "scripts/qsa"))
from selector_assets import get_selector_asset  # noqa: E402

SOURCE = ROOT.parent / "sparse-attn"
WORK = ROOT / "_local/qwen-qsa"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PAYLOAD_SHA = "d7729599f421d80ace536f84145bafa9a6e1eaad22dde18ce42392014a58b5e4"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def header(path):
    with path.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < count < 16 << 20
        return json.loads(stream.read(count)), 8 + count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-backbone-bytes", action="store_true")
    parser.add_argument("--selectors-only", action="store_true")
    args = parser.parse_args()
    assert sys.byteorder == "little"
    asset = get_selector_asset("qwen-four")
    payload = asset.weights
    assert sha(payload) == PAYLOAD_SHA
    if args.selectors_only:
        print(json.dumps(asset.receipt(), indent=2))
        return
    manifest = {
        "tensors": {
            name: dict(
                dtype=item["dtype"],
                shape=item["shape"],
                raw_little_endian_sha256=item["sha256"],
            )
            for name, item in asset.provenance["tensor_records"].items()
        }
    }
    descriptors, start = header(payload)
    keys = set(descriptors) - {"__metadata__"}
    assert keys == set(manifest["tensors"]) and len(keys) == 16
    checked, parameter_count = {}, 0
    with payload.open("rb") as stream:
        for name in sorted(keys):
            info, expected = descriptors[name], manifest["tensors"][name]
            assert info["dtype"] == expected["dtype"] == "BF16"
            assert info["shape"] == expected["shape"]
            first, end = info["data_offsets"]
            count = math.prod(info["shape"])
            assert end - first == count * 2 and end <= payload.stat().st_size - start
            stream.seek(start + first)
            raw = stream.read(end - first)
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == expected["raw_little_endian_sha256"], name
            values = array.array("H")
            values.frombytes(raw)
            assert all((v & 0x7F80) != 0x7F80 for v in values), name
            parameter_count += count
            checked[name] = {
                "dtype": "BF16",
                "shape": info["shape"],
                "raw_sha256": digest,
            }
    assert parameter_count == 13108736
    data_hashes = {
        "train.json": "fda26c9657832e625f829702b8dbb81d956f4532d3c27eaefeb99ff6176fa064",
        "validation.json": "14161d9fa7bbe98725bd42a528e5ec1354b6045e09ae9b4cc97a8ab937541d97",
        "test.json": "255b0110c2d500438f489de52c424eb993bdcd0f2a13a20851cd6a878ac64227",
        "gsm8k.json": "81ff84e59c942977cc8306ae41ee83cd02225e103bbb3d3c012f072566a100c7",
    }
    for name, expected in data_hashes.items():
        assert sha(SOURCE / "data/qwen-selector-v1" / name) == expected, name
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots"
        / REVISION
    )
    backbone = {
        "snapshot": str(snapshot),
        "exists": snapshot.exists(),
        "weight_bytes_verified": False,
    }
    if snapshot.exists():
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        dtypes, tensors, total = collections.Counter(), {}, 0
        for shard in sorted(set(index["weight_map"].values())):
            path = snapshot / shard
            descriptors, first = header(path)
            for name, value in descriptors.items():
                if name == "__metadata__":
                    continue
                assert index["weight_map"][name] == shard
                assert name not in tensors
                assert value["data_offsets"][1] <= path.stat().st_size - first
                size = value["data_offsets"][1] - value["data_offsets"][0]
                dtypes[value["dtype"]] += size
                total += size
                tensors[name] = {"shape": value["shape"], "dtype": value["dtype"]}
        assert set(tensors) == set(index["weight_map"])
        assert total == index["metadata"]["total_size"]
        backbone.update(
            shards=len(set(index["weight_map"].values())),
            tensors=len(tensors),
            tensor_bytes=total,
            dtype_bytes=dict(dtypes),
            config=json.loads((snapshot / "config.json").read_text()),
            index_sha256=sha(snapshot / "model.safetensors.index.json"),
        )
        original = json.loads(
            (SOURCE / "runs/303002/original-manifest.json").read_text()
        )
        assert backbone["index_sha256"] == original["weight_index_sha256"]
        if args.verify_backbone_bytes:
            identities = original["cached_shard_identities"]
            assert set(identities) == set(index["weight_map"].values())
            for name, expected in identities.items():
                actual = sha(snapshot / name)
                assert actual == expected, name
                print("Verified backbone shard", name, flush=True)
            backbone.update(weight_bytes_verified=True, shard_sha256=identities)
    receipt = {
        "status": "passed",
        "scope": "CPU payload/data verification and backbone-header inventory; no inference",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip(),
        "model": "Qwen/Qwen3.8-27B",
        "revision": REVISION,
        "selector_sha256": PAYLOAD_SHA,
        "selector_asset": asset.receipt(),
        "parameters": parameter_count,
        "tensors": checked,
        "data_sha256": data_hashes,
        "backbone": backbone,
        "training_updates": 0,
        "gpu_allocations": 0,
        "baseline_established": False,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "handoff-verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "portable_tensors": len(checked),
                "parameters": parameter_count,
                "backbone_snapshot_exists": snapshot.exists(),
                "backbone_header_bytes": backbone.get("tensor_bytes"),
                "baseline_established": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
