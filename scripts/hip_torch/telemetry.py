#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Telemetry: structured stats collection for the HIP backend.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class SubgraphInfo:
    """Info about one compiled/fallback subgraph."""

    subgraph_id: int
    total_ops: int
    supported_ops: int
    unsupported_ops: List[str]
    compiled: bool
    cache_hit: bool = False
    compile_time_ms: float = 0.0
    fallback_reason: str = ""


@dataclass
class BackendStats:
    """Accumulated statistics for the torch.compile backend."""

    total_subgraphs: int = 0
    compiled_subgraphs: int = 0
    fallback_subgraphs: int = 0
    total_ops: int = 0
    supported_ops: int = 0
    unsupported_ops: int = 0
    unsupported_op_names: Set[str] = field(default_factory=set)
    cache_hits: int = 0
    cache_misses: int = 0
    subgraphs: List[SubgraphInfo] = field(default_factory=list)

    def record_subgraph(self, info: SubgraphInfo):
        self.subgraphs.append(info)
        self.total_subgraphs += 1
        if info.compiled:
            self.compiled_subgraphs += 1
        else:
            self.fallback_subgraphs += 1
        self.total_ops += info.total_ops
        self.supported_ops += info.supported_ops
        self.unsupported_ops += len(info.unsupported_ops)
        self.unsupported_op_names.update(info.unsupported_ops)
        if info.cache_hit:
            self.cache_hits += 1
        elif info.compiled:
            self.cache_misses += 1

    def summary(self) -> str:
        lines = [
            f"Subgraphs: {self.total_subgraphs} "
            f"(compiled={self.compiled_subgraphs}, fallback={self.fallback_subgraphs})",
            f"Ops: {self.total_ops} "
            f"(supported={self.supported_ops}, unsupported={self.unsupported_ops})",
        ]
        if self.cache_hits or self.cache_misses:
            lines.append(f"Cache: hits={self.cache_hits}, misses={self.cache_misses}")
        if self.unsupported_op_names:
            lines.append(f"Unsupported: {sorted(self.unsupported_op_names)}")
        return "\n".join(lines)

    def reset(self):
        self.__init__()

    def to_dict(self) -> Dict:
        return {
            "total_subgraphs": self.total_subgraphs,
            "compiled_subgraphs": self.compiled_subgraphs,
            "fallback_subgraphs": self.fallback_subgraphs,
            "total_ops": self.total_ops,
            "supported_ops": self.supported_ops,
            "unsupported_ops": self.unsupported_ops,
            "unsupported_op_names": sorted(self.unsupported_op_names),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
