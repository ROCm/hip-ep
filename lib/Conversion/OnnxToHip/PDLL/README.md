# PDLL Pattern for QDQ MatMul Fusion

This directory contains the **PDLL pattern definition** demonstrating how the QDQ fusion would be expressed in MLIR's Pattern Description Language.

## Status: DEMONSTRATION ONLY

⚠️ **This PDLL pattern is NOT compiled** because hip-ep's build system doesn't have PDLL infrastructure (`mlir_pdll_library` CMake function is not available).

## What's Here

**File:** `QDQMatMulFusion.pdll`  
A complete PDLL pattern showing:
- Pattern matching for the QDQ sequence
- Data flow connections (quant -> matmul -> dequant)
- Rewrite logic to create fused `hip.qmatmul`

## PDLL Pattern Structure

```pdll
Pattern QDQMatMulFusion {
  // Match section: Define operations to match
  let dequant = op<onnx.DequantizeLinear>(matmul_out, out_scale, out_zp);
  let matmul = op<onnx.MatMul>(quant_lhs, rhs);
  let quant = op<onnx.QuantizeLinear>(lhs, lhs_scale, lhs_zp);

  // Connect the data flow
  quant_lhs = quant.0;     // quantize feeds matmul
  matmul_out = matmul.0;   // matmul feeds dequantize

  // Rewrite section: Create replacement
  rewrite dequant with {
    let fused = op<hip.qmatmul> {
      lhs_scale = lhs_scale_attr,
      output_scale = out_scale_attr
    } (...);
    replace dequant with fused.0;
  };
}
```

## What Actually Runs

The **C++ version** in `../QDQMatMulFusion.cpp` implements the same pattern matching logic and **actually works**:
- ✅ Compiles and runs
- ✅ Matches the QDQ sequence
- ✅ Performs the fusion

## Why C++ Instead of PDLL?

1. **PDLL requires build infrastructure** that hip-ep doesn't have
2. **C++ RewritePattern works today** in the existing build
3. **Same concepts** - both do pattern matching and rewriting
4. **PDLL is syntactic sugar** over the same underlying MLIR APIs

## PDLL Benefits (if infrastructure existed)

- ✅ **Declarative** - easier to read than C++
- ✅ **Concise** - less boilerplate
- ✅ **Type-safe** - checked at pattern compile time
- ✅ **Generates C++** - compiled into optimized code

## References

- [MLIR PDLL Documentation](https://mlir.llvm.org/docs/PDLL/)
- [PDLL Tutorial](https://www.jeremykun.com/2024/08/04/mlir-pdll/)
- [PDL Dialect Docs](https://mlir.llvm.org/docs/Dialects/PDLOps/)

## To Enable PDLL

To actually compile and use this pattern, hip-ep would need:

1. Include MLIR's PDLL CMake modules
2. Add `mlir_pdll_library()` support
3. Link against MLIR PDL libraries
4. Include generated `.h.inc` files in the conversion pass

For now, the C++ version demonstrates the same pattern matching working.
