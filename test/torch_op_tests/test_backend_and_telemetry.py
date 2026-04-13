#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Tests for backend, telemetry, and registry gen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from hip_torch.telemetry import BackendStats, SubgraphInfo


def test_stats_record():
    stats = BackendStats()
    info = SubgraphInfo(1, 10, 8, ["neg", "cos"], compiled=False)
    stats.record_subgraph(info)
    assert stats.total_subgraphs == 1
    assert stats.fallback_subgraphs == 1
    assert stats.unsupported_ops == 2
    assert "neg" in stats.unsupported_op_names


def test_stats_compiled():
    stats = BackendStats()
    info = SubgraphInfo(1, 10, 10, [], compiled=True, cache_hit=True)
    stats.record_subgraph(info)
    assert stats.compiled_subgraphs == 1
    assert stats.cache_hits == 1


def test_stats_summary():
    stats = BackendStats()
    stats.record_subgraph(SubgraphInfo(1, 5, 5, [], True))
    stats.record_subgraph(SubgraphInfo(2, 3, 2, ["neg"], False))
    s = stats.summary()
    assert "Subgraphs: 2" in s
    assert "compiled=1" in s


def test_stats_reset():
    stats = BackendStats()
    stats.record_subgraph(SubgraphInfo(1, 5, 5, [], True))
    stats.reset()
    assert stats.total_subgraphs == 0


def test_stats_to_dict():
    stats = BackendStats()
    stats.record_subgraph(SubgraphInfo(1, 5, 5, [], True))
    d = stats.to_dict()
    assert d["total_subgraphs"] == 1
    assert isinstance(d["unsupported_op_names"], list)


# Test gen_ops if it was generated
def test_gen_ops_file():
    try:
        from hip_torch._gen_ops import GEN_OPS, GEN_VERSION

        assert len(GEN_OPS) > 20
        assert len(GEN_VERSION) == 12
        assert "torch.aten.add.Tensor" in GEN_OPS
        assert "torch.aten.silu" in GEN_OPS
        print(f"  _gen_ops.py: {len(GEN_OPS)} ops, version={GEN_VERSION}")
    except ImportError:
        print("  _gen_ops.py not found (run tools/hip-op-registry-gen.py)")


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
