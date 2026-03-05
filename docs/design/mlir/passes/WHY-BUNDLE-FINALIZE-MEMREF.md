<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Why finalize-memref-to-llvm Patterns Are Bundled Into ConvertHipToLLVMPass

**Date:** 2026-03-03
**Document Type:** Tech Note
**Status:** Draft
**Related:** [LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md), [05-HipToLLVM.md](05-HipToLLVM.md)

---

## Overview

Should `populateFinalizeMemRefToLLVMConversionPatterns` run inside `ConvertHipToLLVMPass` (bundled), or as a separate pipeline stage after it (split)?

The answer is bundled. This document explains why the split produces unnecessary IR artifacts and what the MLIR project guidance says.

---

## Analysis

### The IR chain

After Stage 5 (memory-pooling), the IR contains a chain where a HIP op, a memref op, and another HIP op are linked:

```mlir
%pool = hip.get_pool(%ctx)          : memref<?xi8, 1>
%buf  = memref.view %pool[0][] : memref<?xi8,1> to memref<128xf32, 1>
hip.conv(%ctx, %input, %weights, %buf, ...)
```

`hip.get_pool` and `hip.conv` are HIP dialect ops. `memref.view` is a memref dialect op sitting between them.

### What the split approach produces

When `ConvertHipToLLVMPass` marks `memref::MemRefDialect` as **legal** and runs `applyPartialConversion`:

- `hip.get_pool` is converted → its result changes from `memref<?xi8,1>` to `!llvm.struct<...>`.
- `memref.view` is **not converted** (legal). It still expects a `memref` input. The framework bridges the gap:

  ```
  Cast A: unrealized_conversion_cast(!llvm.struct → memref<?xi8,1>)
  ```

- `hip.conv` is converted. Its adaptor calls `typeConverter.materializeTargetConversion` on the `memref.view` result (still `memref` type, not yet converted). The framework bridges again:

  ```
  Cast B: unrealized_conversion_cast(memref<128xf32,1> → !llvm.struct<...>)
  ```

After a separate `finalize-memref-to-llvm` stage lowers `memref.view` to LLVM, both Cast A and Cast B remain — `finalize-memref-to-llvm` lowers memref **ops**, not `unrealized_conversion_cast` ops. A third stage, `reconcile-unrealized-casts`, is then required to eliminate them.

The casts are not a bug. The MLIR framework inserts them intentionally as a bridge mechanism for multi-pass lowering. But they are avoidable.

### What bundling does

When `populateFinalizeMemRefToLLVMConversionPatterns` is added to the **same** `RewritePatternSet` and `memref::MemRefDialect` is marked **illegal**, all three ops — `hip.get_pool`, `memref.view`, `hip.conv` — are converted in one `applyPartialConversion` call.

`applyPartialConversion` maintains a single **value mapping table** shared by every pattern in the set. When a pattern rewrites an op, it registers `old SSA value → new SSA value` in this table. When another pattern's `ConvertOpToLLVMPattern` adaptor needs an operand, it looks the operand up in the same table.

In the bundled case:

1. The finalize-memref pattern rewrites `memref.view` → registers
   `%buf : memref<128xf32,1>` → `%buf_llvm : !llvm.struct<...>` in the table.
2. The `hip.conv` pattern's adaptor looks up `%buf` → finds `%buf_llvm` directly.
3. No cast is inserted because the mapping is already present.

In the split case, when the `hip.conv` adaptor runs in a separate `applyPartialConversion` call, `%buf` has not yet been rewritten — it is not in the table. The framework inserts `unrealized_conversion_cast(memref → !llvm.struct)` as a placeholder and continues. That placeholder must be removed later by `reconcile-unrealized-casts`.

Result: no `unrealized_conversion_cast` ops, no cleanup pass required.

### MLIR project guidance

The MLIR `TargetLLVMIR` documentation states:

> "Many different dialects can be lowered to LLVM but are provided as different sets of patterns. However, this is primarily useful for testing and prototyping, and **using the collection of patterns together is highly recommended**."

A separate MLIR team announcement clarified the split approach: if multiple `convert-*-to-llvm` passes are run as separate pipeline stages, `reconcile-unrealized-casts` **must** run after all of them. This is the correct idiom for the split case — not an endorsement of splitting.

---

## Conclusion

| | Bundled | Split |
|-|---------|-------|
| `unrealized_conversion_cast` ops | None | Produced at every memref/HIP op boundary |
| Extra pipeline stage needed | No | Yes (`reconcile-unrealized-casts`) |
| MLIR guidance | Recommended | Supported, with mandatory cleanup |
| IR inspectable between memref and LLVM stages | No | Yes |

The only benefit of splitting is that intermediate IR (with memref ops still present after HIP ops are lowered) is inspectable between pipeline stages. This has debugging value but is not worth the structural cost in a production pipeline.

`populateFinalizeMemRefToLLVMConversionPatterns` is called explicitly inside `ConvertHipToLLVMPass` alongside the HIP op patterns. The call is visible in the pass implementation, so the intent is clear without needing a separate pipeline stage to make it explicit.

---

## Related Documents

- [LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md) — Pipeline stage table
- [05-HipToLLVM.md](05-HipToLLVM.md) — ConvertHipToLLVMPass implementation
