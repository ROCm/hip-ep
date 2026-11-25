#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Fetch pinned, checksum-verified public QSA selectors; never run Hub code."""

import argparse
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = json.loads(Path(__file__).with_suffix(".json").read_text())["assets"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


@dataclass(frozen=True)
class SelectorAsset:
    key: str
    folder: Path
    spec: dict
    config: dict
    provenance: dict

    @property
    def weights(self):
        return self.folder / "selectors.safetensors"

    def receipt(self):
        return dict(
            repo_id=self.spec["repo_id"],
            revision=self.spec["revision"],
            filename="selectors.safetensors",
            sha256=self.spec["files"]["selectors.safetensors"],
            metadata_sha256={
                k: v
                for k, v in self.spec["files"].items()
                if k != "selectors.safetensors"
            },
            historical_export_identity_sha256=self.spec["compatibility_sha256"],
            tensor_count=len(self.provenance["tensor_records"]),
        )

    def tensors(self):
        from safetensors.torch import load_file

        return load_file(str(self.weights), device="cpu")


def get_selector_asset(key, *, cache_root=None, offline=False):
    """Download only when called, pin every file, and validate all tensor bytes.

    Imports do not download files or initialize a GPU. The historical identity
    is retained separately from the Hub container hash for old run receipts.
    """
    spec = ASSETS[key]
    cache = Path(
        cache_root
        or os.environ.get(
            "HIP_EP_QSA_SELECTOR_CACHE", ROOT / "_local/qsa/selector-assets"
        )
    )
    folder = cache / key
    for filename, expected in spec["files"].items():
        path = folder / filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Missing cached selector asset: {path}")
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                spec["repo_id"], filename, revision=spec["revision"], local_dir=folder
            )
        if sha256(path) != expected:
            raise ValueError(f"Selector checksum mismatch: {key}/{filename}")
    config = json.loads((folder / "selector_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    for field in [
        "base_model",
        "base_model_revision",
        "layers",
        "tensor_shapes_per_layer",
    ]:
        if config[field] != spec[field]:
            raise ValueError(f"Unexpected selector configuration: {key}/{field}")
    if config["dtype"] != "bfloat16" or config["layer_indexing"] != "zero_based":
        raise ValueError(f"Unsupported selector dtype/layer indexing: {key}")
    if provenance["weight_file_sha256"] != spec["files"]["selectors.safetensors"]:
        raise ValueError(f"Incorrect weight provenance: {key}")
    for field in ["source_export_sha256", "source_checkpoint_sha256"]:
        if provenance[field] != spec[field]:
            raise ValueError(f"Incorrect source provenance: {key}/{field}")
    expected_names = {
        f"selectors.{layer}.{name}": shape
        for layer in spec["layers"]
        for name, shape in spec["tensor_shapes_per_layer"].items()
    }
    with (folder / "selectors.safetensors").open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_length < 16 << 20:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(header_length))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if set(tensors) != set(expected_names) or set(tensors) != set(
            provenance["tensor_records"]
        ):
            raise ValueError(f"Unexpected selector tensor names: {key}")
        spans = []
        for name, shape in expected_names.items():
            tensor, record = tensors[name], provenance["tensor_records"][name]
            if (
                tensor["dtype"] != record["dtype"]
                or tensor["dtype"] != "BF16"
                or tensor["shape"] != shape
                or record["shape"] != shape
            ):
                raise ValueError(f"Unexpected selector tensor type/shape: {key}/{name}")
            begin, end = tensor["data_offsets"]
            elements = 1
            for dim in shape:
                elements *= dim
            if (
                begin < 0
                or end - begin != elements * 2
                or end
                > (folder / "selectors.safetensors").stat().st_size - 8 - header_length
            ):
                raise ValueError(f"Invalid selector tensor offsets: {key}/{name}")
            stream.seek(8 + header_length + begin)
            if hashlib.sha256(stream.read(end - begin)).hexdigest() != record["sha256"]:
                raise ValueError(f"Selector tensor checksum mismatch: {key}/{name}")
            spans.append((begin, end))
        if any(a[1] > b[0] for a, b in zip(sorted(spans), sorted(spans)[1:])):
            raise ValueError(f"Overlapping selector tensors: {key}")
    return SelectorAsset(key, folder, spec, config, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*ASSETS, "all"], default="all")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    selected = list(ASSETS) if args.variant == "all" else [args.variant]
    print(
        json.dumps(
            {
                key: get_selector_asset(
                    key, cache_root=args.cache_root, offline=args.offline
                ).receipt()
                for key in selected
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
