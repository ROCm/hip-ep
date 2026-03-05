<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Canonicalization Pass

**Date:** 2026-03-02
**Document Type:** Implementation
**Status:** Draft
**Related:** [02-OnnxToHip.md](02-OnnxToHip.md), [02b-OneShotBufferize.md](02b-OneShotBufferize.md), [04-MemoryPooling.md](04-MemoryPooling.md)

## Overview

MLIR Canonicalizer running after [one-shot-bufferize](02b-OneShotBufferize.md) and `buffer-results-to-out-params`. Cleans up IR produced by bufferization before optional memory pooling.

## Role in Pipeline

```
one-shot-bufferize
      ↓
buffer-results-to-out-params
      ↓
canonicalize        ← this pass
      ↓
memory-pooling (optional)
      ↓
convert-hip-to-llvm
```

## Generic MLIR Patterns

Standard canonicalization patterns from `mlir::createCanonicalizerPass()`:

- Constant folding
- Dead code elimination (removes unused `memref.alloc` after bufferization)
- Algebraic simplifications
- `memref.copy` self-copy elimination (`memref.copy %x, %x` → removed)

## What Gets Cleaned Up

Bufferization may leave behind dead allocations when in-place bufferization succeeds (the buffer is reused, making the original `memref.alloc` unused). Canonicalization removes these.

**Example:**
```mlir
// After one-shot-bufferize (relu bufferized in-place)
%alloc = memref.alloc() : memref<1x64x224x224xf32, 1>  // dead — relu reuses input
hip.relu ins(%ctx, %input : ...) outs(%alloc : ...)

// After canonicalize
hip.relu ins(%ctx, %input : ...) outs(%input : ...)     // alloc gone
```

## Testing

LIT tests under `test/lit/Transforms/` cover:
- Dead `memref.alloc` elimination
- `memref.copy` self-copy removal
- Constant folding interactions

## Related Documents

- **[02b-OneShotBufferize.md](02b-OneShotBufferize.md)** - one-shot-bufferize (produces the IR this pass cleans)
- **[02-OnnxToHip.md](02-OnnxToHip.md)** - Upstream conversion pass
- **[04-MemoryPooling.md](04-MemoryPooling.md)** - Downstream pass (consumes clean memref.alloc ops)
- **[../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md)** - Operation definitions
- **[../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md)** - Pipeline overview
