#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
DLL Cache: hash-based persistent cache for compiled GPU DLLs.

Avoids recompiling the same MLIR graph across runs. Cache keys are
derived from the MLIR text content + registry version + compiler hash.
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from . import op_registry

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("TEMP", os.environ.get("TMP", "/tmp")), "hip_dll_cache"
)


class DllCache:
    """Persistent hash-based cache for compiled GPU DLLs."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def compute_key(self, mlir_text: str) -> str:
        """Compute cache key from MLIR text + registry version.

        The key changes when:
        - The graph structure changes (different MLIR)
        - Op patterns change (registry version bump)
        """
        h = hashlib.sha256()
        h.update(mlir_text.encode("utf-8"))
        h.update(op_registry.REGISTRY_VERSION.encode("utf-8"))
        return h.hexdigest()[:16]

    def get(self, key: str) -> Optional[Path]:
        """Look up a cached DLL by key.

        Returns the DLL path if cached, None otherwise.
        """
        dll_path = self.cache_dir / key / "model.dll"
        if dll_path.exists():
            self._hits += 1
            log.debug(f"Cache hit: {key}")
            return dll_path
        self._misses += 1
        return None

    def put(self, key: str, src_dll: Path, metadata: Optional[dict] = None) -> Path:
        """Store a compiled DLL in the cache.

        Args:
            key: Cache key
            src_dll: Path to the compiled DLL
            metadata: Optional metadata to store alongside

        Returns:
            Path to the cached DLL
        """
        target_dir = self.cache_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / "model.dll"
        shutil.copy2(str(src_dll), str(dst))

        if metadata:
            meta_path = target_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        log.debug(f"Cached DLL: {key} ({dst.stat().st_size / 1024:.0f} KB)")
        return dst

    def clear(self):
        """Clear all cached DLLs."""
        if self.cache_dir.exists():
            shutil.rmtree(str(self.cache_dir))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        entries = (
            sum(1 for d in self.cache_dir.iterdir() if d.is_dir())
            if self.cache_dir.exists()
            else 0
        )
        total_size = (
            sum(f.stat().st_size for f in self.cache_dir.rglob("*.dll"))
            if self.cache_dir.exists()
            else 0
        )
        return {
            "cache_dir": str(self.cache_dir),
            "entries": entries,
            "total_size_kb": total_size / 1024,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
        }

    def __repr__(self):
        s = self.stats
        return (
            f"DllCache(entries={s['entries']}, "
            f"size={s['total_size_kb']:.0f}KB, "
            f"hits={s['hits']}, misses={s['misses']})"
        )
