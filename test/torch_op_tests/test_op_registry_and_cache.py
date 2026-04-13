#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Tests for Phase 1: op_registry, dll_cache, compiler."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from hip_torch.op_registry import (
    COMPUTE_OPS,
    FREE_OPS,
    OP_REGISTRY,
    REGISTRY_VERSION,
    SUPPORTED_OPS,
    get_default_values,
    get_info,
    get_ops_by_category,
    is_supported,
)
from hip_torch.dll_cache import DllCache


# ── Op Registry Tests ────────────────────────────────────────────────


def test_registry_has_entries():
    assert len(OP_REGISTRY) > 30, f"Expected 30+ ops, got {len(OP_REGISTRY)}"


def test_supported_ops_derived():
    assert SUPPORTED_OPS == frozenset(OP_REGISTRY.keys())


def test_free_ops_subset():
    assert FREE_OPS.issubset(SUPPORTED_OPS)


def test_compute_ops_disjoint():
    assert len(COMPUTE_OPS & FREE_OPS) == 0


def test_registry_version_stable():
    """Version should be deterministic."""
    assert len(REGISTRY_VERSION) == 12
    assert REGISTRY_VERSION.isalnum()


def test_is_supported_direct():
    assert is_supported("torch.aten.add.Tensor")
    assert is_supported("torch.aten.silu")
    assert is_supported("torch.aten.mm")


def test_is_supported_builtin():
    assert is_supported("getitem")
    assert is_supported("operator.getitem")


def test_is_supported_skip():
    assert is_supported("_log_api_usage_once")


def test_is_supported_unknown():
    assert not is_supported("torch.aten.nonexistent_op_xyz")


def test_get_info():
    info = get_info("torch.aten.add.Tensor")
    assert info is not None
    assert info.category == "elementwise"
    assert info.required_args == 3
    assert info.hip_op == "hip.add"


def test_get_info_none():
    assert get_info("nonexistent") is None


def test_get_ops_by_category():
    matmul_ops = get_ops_by_category("matmul")
    assert "torch.aten.mm" in matmul_ops
    assert "torch.aten.linear" in matmul_ops


def test_get_default_values():
    defaults = get_default_values("torch.aten.linear")
    assert defaults == ((None,),)


def test_all_ops_have_category():
    for name, info in OP_REGISTRY.items():
        assert info.category, f"{name} missing category"


def test_key_ops_present():
    """Verify critical ops for Qwen models are registered."""
    critical = [
        "torch.aten.rms_norm",
        "torch.aten.linear",
        "torch.aten.silu",
        "torch.aten.mul.Tensor",
        "torch.aten.add.Tensor",
        "torch.aten.scaled_dot_product_attention",
    ]
    for op in critical:
        assert op in SUPPORTED_OPS, f"Missing critical op: {op}"


# ── DLL Cache Tests ──────────────────────────────────────────────────


def test_cache_compute_key():
    cache = DllCache(tempfile.mkdtemp())
    key1 = cache.compute_key("mlir text 1")
    key2 = cache.compute_key("mlir text 2")
    key1b = cache.compute_key("mlir text 1")
    assert key1 != key2, "Different MLIR should give different keys"
    assert key1 == key1b, "Same MLIR should give same key"
    assert len(key1) == 16


def test_cache_miss():
    cache = DllCache(tempfile.mkdtemp())
    assert cache.get("nonexistent_key") is None
    assert cache.stats["misses"] == 1


def test_cache_put_and_get():
    cache_dir = tempfile.mkdtemp()
    cache = DllCache(cache_dir)

    # Create a fake DLL
    fake_dll = Path(tempfile.mktemp(suffix=".dll"))
    fake_dll.write_bytes(b"fake dll content")

    key = "test_key_12345678"
    cached = cache.put(key, fake_dll, metadata={"test": True})
    assert cached.exists()

    # Retrieve
    result = cache.get(key)
    assert result is not None
    assert result.exists()
    assert result.read_bytes() == b"fake dll content"
    assert cache.stats["hits"] == 1

    fake_dll.unlink()


def test_cache_clear():
    cache = DllCache(tempfile.mkdtemp())
    fake_dll = Path(tempfile.mktemp(suffix=".dll"))
    fake_dll.write_bytes(b"test")
    cache.put("key1", fake_dll)
    cache.clear()
    assert cache.get("key1") is None
    assert cache.stats["entries"] == 0
    fake_dll.unlink(missing_ok=True)


def test_cache_stats():
    cache = DllCache(tempfile.mkdtemp())
    stats = cache.stats
    assert "entries" in stats
    assert "total_size_kb" in stats
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats


# ── Run all tests ────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
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
