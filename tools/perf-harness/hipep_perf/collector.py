#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Drain the trace shim's ring into a raw journal while the target runs.

The collector lives in the harness process, not the target, so every record
the shim committed is kept however the target exits. The journal is the raw
64-byte records, appended in ring order; ``trace.load`` turns it into tables.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .ring import RECORD_DTYPE, Ring


class Collector:
    def __init__(
        self,
        out_dir: Path,
        session: str | None = None,
        capacity: int = 1 << 20,
        poll_ms: float = 2.0,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or f"{os.getpid()}_{int(time.time() * 1000) % 10**9}"
        self.ring = Ring(self.session, capacity)
        self.journal_path = self.out_dir / "trace.records"
        self.meta_path = self.out_dir / "trace.meta.json"
        self._journal = self.journal_path.open("wb")
        self._poll = poll_ms / 1000.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="hipep-collector", daemon=True
        )
        self.records = 0
        self.peak_backlog = 0

    def env(self) -> dict[str, str]:
        """Variables that point the shim at this collector."""
        return {"HIPEP_TRACE": self.session}

    def start(self) -> Collector:
        self._thread.start()
        return self

    def _drain_once(self) -> int:
        backlog = self.ring.head - self.ring.tail
        self.peak_backlog = max(self.peak_backlog, backlog)
        recs = self.ring.drain()
        if len(recs):
            recs.tofile(self._journal)
            self.records += len(recs)
        return len(recs)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._drain_once() == 0:
                time.sleep(self._poll)

    def stop(self, extra_meta: dict | None = None) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        # A producer that reserved a slot just before exiting may never commit
        # it; stop when nothing more arrives rather than waiting for head.
        for _ in range(50):
            if self._drain_once() == 0 and self.ring.tail >= self.ring.head:
                break
            time.sleep(0.01)
        self._journal.close()
        meta = {
            "session": self.session,
            "records": self.records,
            "capacity": self.ring.capacity,
            "peak_backlog": self.peak_backlog,
            "uncommitted": max(0, self.ring.head - self.ring.tail),
            **self.ring.header(),
            **(extra_meta or {}),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))
        self.ring.close()
        return meta


def load_journal(out_dir: Path) -> tuple[np.ndarray, dict]:
    out_dir = Path(out_dir)
    recs = np.fromfile(out_dir / "trace.records", dtype=RECORD_DTYPE)
    meta = json.loads((out_dir / "trace.meta.json").read_text())
    return recs, meta
