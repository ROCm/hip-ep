#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Download the pinned Original source and record immutable inference inputs."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from selector_assets import get_selector_asset

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "_local/qsa"
MODEL = "allenai/Olmo-3-1025-7B"
REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
REFERENCE = ROOT.parent / "sparse-attn"
# Retained for historical container-byte audits; active loaders use Hub assets.
SELECTOR = REFERENCE / "checkpoints/extended-selector/302354/step-4096/selector-bf16.pt"
SELECTOR_SHA = "decd3160f4321839c80fee2ba0b0241575fc06017676e38a04286a3ae3daa10b"


def selector_fingerprint():
    """Verify public weights while retaining the historical export identity."""
    return get_selector_asset("olmo-four").spec["compatibility_sha256"]


def original_folder(bits=4, layers=32):
    """Keep invalid all-YaRN prototypes separate from the faithful export."""
    name = f"original-w{bits}-rope-v2"
    if layers != 32:
        name += f"-{layers}-layer-diagnostic"
    return WORK / name


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    asset = get_selector_asset("olmo-four")
    assert asset.spec["compatibility_sha256"] == SELECTOR_SHA
    receipt = {
        "stage": "downloading_original",
        "model": MODEL,
        "revision": REVISION,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "initial_worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "platform": platform.platform(),
        "selector_source": str(asset.weights),
        "selector_sha256": SELECTOR_SHA,
        "selector_asset": asset.receipt(),
        "training_updates": 0,
    }
    out = WORK / "preparation.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    path = snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=WORK / "base-model",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        max_workers=3,
    )
    print("Download complete; hashing source files", flush=True)
    receipt["files"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(Path(path).iterdir())
        if p.is_file()
    }
    receipt["stage"] = "original_source_verified"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
