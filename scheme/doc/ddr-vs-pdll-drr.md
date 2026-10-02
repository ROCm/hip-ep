<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

**Date:** 2026-10-01
**Document Type:** Tech Note
**Status:** Draft
**Related:** [plugin-interface.md](../../docs/design/plugin-interface.md)

---

# DDR vs. MLIR DRR and PDLL

## Overview

This note compares the hip-ep DDR (Declarative Dialect Rewriting) against MLIR's two established pattern DSLs — DRR (TableGen-based) and PDLL — for the purpose of writing dialect conversion patterns. The comparison is scoped to the onnx→HipSR lowering pass.

---

## Analysis

### Capability: Dialect Conversion

PDLL documents this limitation explicitly:

> *"Planned but missing PDLL features include: Support for use in dialect conversion (no RFC yet)"*

DRR has the same limitation. Neither generates `ConversionPattern` subclasses.

The onnx→HipSR pass uses `applyFullConversion` with a `TypeConverter`. Every pattern receives type-converted operands via `ConversionPatternRewriter`. DDR's `SchemeConversionPattern` inherits from `mlir::ConversionPattern` and is the only DSL-based approach that supports this today.

### Edit-Test Loop

| Approach | Steps to test a pattern change |
|---|---|
| DRR | Edit `.td` → run `mlir-tblgen` → rebuild binary |
| PDLL | Edit `.pdll` → run `mlir-pdll` → include header → rebuild binary |
| DDR | Edit `.sls` → reload |

PDLL requires `mlir-pdll` installed and version-matched to the MLIR build. DDR requires no additional tools.

### Constraints and Rewrites

PDLL's `Constraint` and `Rewrite` declarations are backed by C++ `native` blocks — C++ code embedded inside the `.pdll` file. DDR's `:where` clause and `:rewrite` body accept any Scheme expression. A named Scheme function serves the same purpose without requiring C++:

```scheme
;; Named constraint — callable from any pattern:
(define (device-tensor? !type)
  (= 1 (mlir-type-is-device-tensor !type)))

;; Usage in match:
:where (device-tensor? (mlir-value-get-type %data))
```

DRR uses `NativeCodeCall` (C++ string escapes) for the same purpose.

### Turing-Complete Pattern Computation

DRR and PDLL require C++ escapes for complex pattern logic (axis normalization, shape broadcasting, resource attribute construction from memory addresses). DDR expresses this in Scheme, in the same file as the pattern, with no C++ required.

### Debugging

DDR provides `:debug-codegen` (shows the generated Scheme lambda) and `:debug-matching` (traces execution). DRR and PDLL require gdb or compile-time instrumentation.

### Deployment

| Property | DRR | PDLL | DDR |
|---|---|---|---|
| Patterns in binary | Yes | Yes | No — filesystem |
| Cold-start cost | Zero | Zero | ~100–200ms (first session) |
| Out-of-sync failure | Build error | Build error | Runtime failure |

Pattern `.sls` files must be co-deployed and version-matched with the binary. A mismatch produces a runtime failure rather than a build error.

### FFI Maintenance

Each MLIR API function used from Scheme requires a C++ registration (`Sregister_symbol`) and a Scheme `foreign-procedure` declaration. DRR and PDLL operate within MLIR's type system and have no equivalent cost. The `foreign-entry?` dynamic dispatch partially mitigates this for attribute types — new attrs are discoverable at runtime without Scheme changes.

### Summary

| Dimension | DRR | PDLL | DDR |
|---|---|---|---|
| Dialect conversion (ConversionPattern) | No | No | **Yes** |
| Edit→test loop | Minutes | Minutes | **Seconds** |
| Extra toolchain | mlir-tblgen | mlir-pdll + mlir-tblgen | **None** |
| Constraints/rewrites without C++ | No | No | **Yes** |
| Turing-complete pattern computation | Via C++ | Via C++ | **Native** |
| Interactive debugging | No | No | **Yes** |
| Filesystem deployment dependency | No | No | **Yes** |
| FFI maintenance burden | None | None | Real |

---

## Conclusion

DDR is the only DSL-based option for dialect conversion patterns today. PDLL and DRR cannot express `ConversionPattern` subclasses.

Within dialect conversion use cases, DDR's Scheme host provides measurable advantages over hypothetical future PDLL dialect-conversion support: no C++ required for constraints or rewrites, seconds-level edit-test cycle, and interactive debugging.

The two operational risks — filesystem coupling and FFI maintenance burden — are the primary cost of the DDR approach. Both are engineering discipline problems, not language design problems.

---

## Related Documents

- [plugin-interface.md](../../docs/design/plugin-interface.md) — Plugin slot and pass registration
- [MLIR PDLL documentation](https://mlir.llvm.org/docs/PDLL/) — PDLL language reference and known limitations
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) — ConversionPattern and TypeConverter
