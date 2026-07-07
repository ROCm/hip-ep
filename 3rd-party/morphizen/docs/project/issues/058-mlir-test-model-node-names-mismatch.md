<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #058: MLIR Test Model Node Names Do Not Match ONNX Model

## Metadata
- **Type:** Bug
- **Priority:** HIGH
- **Status:** OPEN
- **Dependencies:** Issue #028
- **Discovered from:** Issue #028 test enablement

## Description

After enabling MLIR backend tests (removing GTEST_SKIP), many tests fail because they reference node names that exist in the original ONNX model (pt_resnet50.onnx) but do not exist in the MLIR model (pt_resnet50.onnx.mlir).

## Problem

**ONNX model node names used in tests:**
- `"blob.1"` - graph input
- `"1327"` - graph output
- `"1321"` - Gemm node
- `"287"` - Add node
- `"111"`, `"138"` - for Fuse tests
- `"module_10.weight"`, `"1261"` - constant initializers

**MLIR model node naming:**
- MLIR uses auto-generated names like `unnamed_1895010848000`
- ONNX node names are not preserved during ONNX→MLIR conversion

**Failing tests:**
- `GraphTest.NodeIndex` - cannot find node "1321"
- `GraphTest.FindConsumers` - cannot find node_arg "1321"
- `GraphTest.NodeArgFindProducer` - cannot find node_arg "1327"
- `GraphTest.FindNodeArgGraphInput` - cannot find node_arg "blob.1"
- `GraphTest.FindNodeArgGraphOutput` - cannot find node_arg "1327"
- `GraphTest.Fuse` - cannot find node "138"
- `GraphTest.TryFuse` - depends on nodes "111", "138"
- `GraphTest.VirtualFuse` - depends on nodes "111", "138"
- `PatternTest.CommutableNode` - cannot find node "287"
- `PatternTest.LoadSaveBinary` - cannot find node "287"

## Root Cause

The MLIR model is generated from the ONNX model but:
1. Node names are regenerated during conversion
2. Original ONNX node naming conventions are not preserved
3. Tests hardcode ONNX node names

## Proposed Solutions

### Option A: Use dynamic node discovery in tests
Instead of hardcoding node names, tests could:
1. Get first graph input/output by index
2. Find nodes by op_type rather than name
3. Use graph structure traversal

### Option B: Create MLIR-specific test data
Create a separate set of expected values for MLIR tests based on MLIR model's actual node names.

### Option C: Improve ONNX→MLIR conversion
Preserve original ONNX node names in MLIR conversion if possible.

**Recommended:** Option A (flexible) or Option B (quick fix)

## Affected Tests

Tests that hardcode ONNX node names (10+ tests):
- GraphTest: NodeIndex, FindConsumers, NodeArgFindProducer, FindNodeArgGraphInput, FindNodeArgGraphOutput, Fuse, TryFuse, VirtualFuse
- PatternTest: CommutableNode, LoadSaveBinary

## Evidence

```
D:\ROCm\MorphiZen\unit-test\morphizen\test_graph.cpp(184): error: Value of: node.has_value()
  Actual: false
Expected: true
# Trying to find node "1321" which doesn't exist in MLIR model

D:\ROCm\MorphiZen\unit-test\morphizen\test_graph.cpp(211): error: cannot find node_arg 1327
# Trying to find node_arg "1327" which doesn't exist in MLIR model

F20260204 01:02:44 node_builder.cpp:579] Check failed: node.has_value() cannot find node: 138
# Fatal error when trying to fuse nodes that don't exist
