#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
torch.compile custom backend for the HIP MLIR compiler.

Uses the op_registry for support checking, dll_cache for artifact
caching, and telemetry for structured stats.

Usage:
    from hip_torch.backend import hip_backend, get_stats
    compiled = torch.compile(model, backend=hip_backend)
"""

import logging
from typing import List

import torch

from . import op_registry
from .telemetry import BackendStats, SubgraphInfo

log = logging.getLogger(__name__)

# Global stats
_stats = BackendStats()


def get_stats() -> BackendStats:
    return _stats


def reset_stats():
    _stats.reset()


def _get_op_name(node) -> str:
    """Extract aten op name from FX node."""
    target = node.target
    s = str(target)
    if hasattr(target, "__name__"):
        return target.__name__
    if "aten." in s:
        return s.split("torch.ops.")[-1] if "torch.ops." in s else s
    return s


def _check_support(gm: torch.fx.GraphModule):
    """Classify ops in a subgraph as supported or unsupported."""
    supported, unsupported = [], []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = _get_op_name(node)
        target_str = str(node.target)
        if op_registry.is_supported(name) or op_registry.is_supported(target_str):
            supported.append(name)
        else:
            unsupported.append(name)
    return supported, unsupported


def _hip_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """Core backend: classify subgraph and return eager execution."""
    supported, unsupported = _check_support(gm)
    total = len(supported) + len(unsupported)
    compiled = len(unsupported) == 0

    info = SubgraphInfo(
        subgraph_id=_stats.total_subgraphs + 1,
        total_ops=total,
        supported_ops=len(supported),
        unsupported_ops=unsupported,
        compiled=compiled,
        fallback_reason=f"unsupported: {set(unsupported)}" if unsupported else "",
    )
    _stats.record_subgraph(info)

    if compiled:
        log.info(f"Subgraph {info.subgraph_id}: HIP COMPILED ({total} ops)")
    else:
        log.info(
            f"Subgraph {info.subgraph_id}: FALLBACK "
            f"({len(supported)}/{total} supported)"
        )

    return gm.forward


def hip_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """torch.compile backend entry point."""
    return _hip_backend(gm, example_inputs)
