#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""SqttData chunks: raw SQ Thread Trace token streams (one per shader engine).

Two responsibilities:

1. :func:`extract_blobs` - pull the raw (Zstd-inflated) SQTT byte streams out of the
   RDF container. This is straightforward and fully implemented.
2. :func:`decode_tokens` - turn those raw hardware token streams into structured
   dispatch / event / occupancy / realtime records. This is the pure-Python port
   of AMD's ``rocprof-trace-decoder`` gfx1x token parser (see
   :mod:`rgp_parser.struct.sqtt_decode`). :func:`load_records` can alternatively
   ingest a pre-decoded ``records.json`` for the same downstream pipeline
   (symbolize -> model -> formats).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .rdf import RdfFile


@dataclass
class DecodedRecords:
    """Raw, pre-symbolization decode output (dispatches / events / occupancy)."""

    rt_freq: int = 100_000_000
    rt0: int = 0
    wave_size: int = 32
    shader_engines: int = 0
    dispatches: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    occupancy: list[dict] = field(default_factory=list)


def extract_blobs(rdf: RdfFile) -> list[bytes]:
    """One raw SQTT byte stream per shader engine, in chunk order."""
    return [rdf.data(c) for c in rdf.by_id("SqttData")]


def load_records(path: str) -> DecodedRecords:
    """Load a pre-decoded records.json (optional external decode)."""
    with open(path, "r") as f:
        r = json.load(f)
    return DecodedRecords(
        rt_freq=r.get("rt_freq", 100_000_000),
        rt0=r.get("rt0", 0),
        wave_size=r.get("wave_size", 32),
        shader_engines=len({d.get("se", 0) for d in r.get("dispatches", [])}) or 0,
        dispatches=r.get("dispatches", []),
        events=r.get("events", []),
        occupancy=r.get("occupancy", []),
    )


def decode_tokens(blobs: list[bytes], *, wave_size: int = 32) -> DecodedRecords:
    """Pure-Python SQTT hardware-token decoder (gfx11 / RDNA3).

    Faithful port of AMD's ``rocprof-trace-decoder`` token parser plus the
    wave-attribution / wall-clock post-processing. See
    :mod:`rgp_parser.struct.sqtt_decode` for the implementation and its
    source-of-truth references. Produces the raw record set (dispatches / events /
    occupancy) the rest of the pipeline consumes.

    One raw SQTT byte stream per shader engine is expected (see
    :func:`extract_blobs`); dispatches / events / realtime are read from SE0
    (they are broadcast), occupancy from every SE.
    """
    from . import sqtt_decode

    r = sqtt_decode.decode(blobs, wave_size=wave_size)
    return DecodedRecords(
        rt_freq=r["rt_freq"],
        rt0=r["rt0"],
        wave_size=r["wave_size"],
        shader_engines=len(blobs),
        dispatches=r["dispatches"],
        events=r["events"],
        occupancy=r["occupancy"],
    )
