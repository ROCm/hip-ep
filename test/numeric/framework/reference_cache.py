#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""On-disk cache for ORT CPU reference outputs.

The cache lives at ``<cache_dir>/<sample-name>/`` -- one directory per
sanitized pytest node id, ever. ``<cache_dir>`` defaults to
``<output_dir>/cache/`` (configurable via ``--cache-dir``; see
``conftest.py``). The sha256 of ``(model_bytes ++ inputs.tobytes())`` is
stored *inside* ``manifest.json`` rather than in the path, so the layout
stays human-readable and disk usage is bounded by the number of
distinct test names. On every cache read the runner recomputes the
hash and invalidates the entry if it no longer matches -- this is the
drift tripwire that catches edits to the test model or the input
distribution without requiring the contributor to bump a version.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _contig_preserve_rank(arr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous view of *arr* that keeps a 0-d array 0-d.

    ``np.ascontiguousarray`` is documented to "convert an array to an
    array with at least one dimension", so calling it on a numpy scalar
    silently promotes ``shape=()`` to ``shape=(1,)``. That promotion
    then round-trips through ``np.save`` / ``np.load`` and rank-corrupts
    every on-disk artifact we hand back to the comparator -- which made
    ``compare_outputs`` flag a phantom shape mismatch the moment any
    test passed a rank-0 input or returned a rank-0 output (see the
    ``test_gather_axis0_scalar_index`` failure pattern). This helper
    keeps rank intact while still guaranteeing a contiguous buffer.
    """
    if arr.flags.c_contiguous:
        return arr
    if arr.ndim == 0:
        return arr.copy(order="C")
    return np.ascontiguousarray(arr)


_TAG = "[Cache]"
_MANIFEST = "manifest.json"
_MODEL = "model.onnx"
_INPUTS_DIR = "inputs"
_OUTPUTS_DIR = "outputs"


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------


def sanitize_name(name: str) -> str:
    """Make a pytest node id safe for use as a filesystem directory name.

    Also used by ``ModelRunner`` for per-test work-directory tagging, so
    both layouts stay aligned (one test name -> one sanitized form
    everywhere). Pytest parametrization tokens like
    ``test_matmul_qo_proj[128]`` become ``test_matmul_qo_proj_128``: the
    brackets are turned into underscores, runs of underscores collapse,
    and the leading/trailing underscores are stripped.
    """
    name = re.sub(r"[\[\](){}]", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def compute_content_hash(model_bytes: bytes, inputs: list[np.ndarray]) -> str:
    """Return ``sha256(model_bytes ++ each input's shape/dtype/bytes)`` as hex.

    The hash is used purely as a tripwire: it never appears in any
    filesystem path. Storing it inside the manifest makes the cache
    directory layout self-documenting while still detecting silent edits
    to the test model or input distribution.

    Note: rank promotion is invisible at the byte level -- ``np.array(1,
    dtype=int64).tobytes()`` and ``np.array([1], dtype=int64).tobytes()``
    produce the same 8 bytes. ORT then happily runs a rank-1 input
    against a rank-0 model input slot and produces a rank-1 output that
    silently disagrees with the model's declared output shape -- which
    poisons the cache. Including ``shape`` and ``dtype`` in the hash
    catches that drift on the very next test run.
    """
    h = hashlib.sha256()
    h.update(model_bytes)
    for arr in inputs:
        contig = _contig_preserve_rank(arr)
        h.update(repr(contig.shape).encode("utf-8"))
        h.update(str(contig.dtype).encode("utf-8"))
        h.update(contig.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    content_hash: str
    model_size: int
    input_shapes: list[list[int]]
    input_dtypes: list[str]
    output_shapes: list[list[int]]
    output_dtypes: list[str]
    created_at: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "content_hash": self.content_hash,
                "model_size": self.model_size,
                "input_shapes": self.input_shapes,
                "input_dtypes": self.input_dtypes,
                "output_shapes": self.output_shapes,
                "output_dtypes": self.output_dtypes,
                "created_at": self.created_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        d = json.loads(text)
        return cls(
            content_hash=d["content_hash"],
            model_size=d["model_size"],
            input_shapes=d["input_shapes"],
            input_dtypes=d["input_dtypes"],
            output_shapes=d.get("output_shapes", []),
            output_dtypes=d.get("output_dtypes", []),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


class ReferenceCache:
    """File-backed cache for CPU reference outputs.

    A single instance is created per pytest session in ``conftest.py`` and
    shared by every test. The ``no_cache`` / ``refresh`` flags map to the
    ``--no-cache`` / ``--refresh-cache`` CLI options.
    """

    def __init__(
        self,
        root: Path | str,
        no_cache: bool = False,
        refresh: bool = False,
    ) -> None:
        self.root = Path(root)
        self.no_cache = no_cache
        self.refresh = refresh
        if not self.no_cache:
            self.root.mkdir(parents=True, exist_ok=True)

    def entry_dir(self, name: str) -> Path:
        return self.root / sanitize_name(name)

    def load(
        self,
        name: str,
        model_bytes: bytes,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray] | None:
        """Return cached outputs for *name* if the manifest hash matches.

        Returns ``None`` when the cache is disabled, when no entry exists,
        when the manifest is unreadable, or when the content hash does
        not match (i.e. the test model or inputs have changed since the
        cache was written).
        """
        if self.no_cache:
            return None
        if self.refresh:
            self._clear(name)
            return None

        d = self.entry_dir(name)
        manifest_path = d / _MANIFEST
        if not manifest_path.exists():
            return None

        try:
            manifest = Manifest.from_json(manifest_path.read_text())
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"{_TAG} unreadable manifest at {manifest_path} ({e}); ignoring")
            return None

        current_hash = compute_content_hash(model_bytes, inputs)
        if current_hash != manifest.content_hash:
            # Drift tripwire fired. We log loudly so a contributor who just
            # changed a seed / shape / scale sees the cache is being
            # rebuilt -- and so that two distinct samples colliding on the
            # same sanitized name are obvious (they will thrash this path
            # on every run).
            print(
                f"{_TAG} invalidated: '{sanitize_name(name)}' content changed "
                f"(manifest={manifest.content_hash[:12]}, "
                f"current={current_hash[:12]}); will re-run CPU reference"
            )
            print(
                f"{_TAG} WARNING: if multiple tests share this cache directory, "
                f"they will repeatedly invalidate each other -- pass an explicit "
                f"name= to disambiguate."
            )
            return None

        outputs = self._load_outputs(d)
        if outputs is None:
            return None
        print(
            f"{_TAG} hit: '{sanitize_name(name)}' ({len(outputs)} output(s), "
            f"hash={current_hash[:12]})"
        )
        return outputs

    def store(
        self,
        name: str,
        model_bytes: bytes,
        inputs: list[np.ndarray],
        outputs: list[np.ndarray],
    ) -> None:
        """Persist a freshly-computed CPU reference to disk.

        No-op when the cache is disabled. Always overwrites any existing
        entry: writes are cheap relative to the CPU reference run that
        just produced them.
        """
        if self.no_cache:
            return

        d = self.entry_dir(name)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        (d / _INPUTS_DIR).mkdir(parents=True, exist_ok=True)
        (d / _OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

        (d / _MODEL).write_bytes(model_bytes)

        for i, arr in enumerate(inputs):
            np.save(d / _INPUTS_DIR / f"in_{i}.npy", _contig_preserve_rank(arr))

        for i, arr in enumerate(outputs):
            np.save(d / _OUTPUTS_DIR / f"out_{i}.npy", _contig_preserve_rank(arr))

        manifest = Manifest(
            content_hash=compute_content_hash(model_bytes, inputs),
            model_size=len(model_bytes),
            input_shapes=[list(a.shape) for a in inputs],
            input_dtypes=[str(a.dtype) for a in inputs],
            output_shapes=[list(a.shape) for a in outputs],
            output_dtypes=[str(a.dtype) for a in outputs],
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        )
        (d / _MANIFEST).write_text(manifest.to_json())
        print(
            f"{_TAG} stored: '{sanitize_name(name)}' at {d} "
            f"(hash={manifest.content_hash[:12]})"
        )

    def _clear(self, name: str) -> None:
        d = self.entry_dir(name)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"{_TAG} refreshed: cleared '{sanitize_name(name)}'")

    @staticmethod
    def _load_outputs(d: Path) -> list[np.ndarray] | None:
        outputs_dir = d / _OUTPUTS_DIR
        if not outputs_dir.exists():
            return None
        files = sorted(
            outputs_dir.glob("out_*.npy"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not files:
            return None
        return [np.load(p) for p in files]
