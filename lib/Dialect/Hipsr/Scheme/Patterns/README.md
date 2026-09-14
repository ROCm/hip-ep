# ONNX to HipSR Conversion Patterns (Scheme)

This directory contains Scheme implementations of ONNX to HipSR dialect conversion patterns.

## Structure

Each pattern file exports:
1. Pattern definitions using `define-conversion-pattern` macro
2. `populate-*-patterns` function for pattern registration

Example:
```scheme
(library (patterns cast)
  (export populate-cast-patterns
          onnx-cast->hipsr)
  
  (define-conversion-pattern onnx-cast->hipsr ...)
  
  (define (populate-cast-patterns converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr)))
```

## Pattern DSL

See `tech/design/2026-09-14-mlir-hipsr-pattern-dsl.md` for complete syntax.

Basic structure:
```scheme
(define-conversion-pattern pattern-name
  :if-match
    %result = "dialect.op" (%operand ...) (:attr $value) ... :type (!t ...) -> !t
  :rewrite %result
    (mlir-build
      %new-op = "target.Op" (%operand) :type (!t) -> !t))
```

## C++ Equivalents

Each `.sls` pattern file replaces a C++ `*Conversion.cpp` file:

| Scheme | C++ |
|--------|-----|
| `cast.sls` | `CastConversion.cpp` |
| `reshape.sls` | `ReshapeConversion.cpp` |
| More to be migrated... | |

## Testing

Patterns are tested remotely on xcoengvm226019:

```bash
ssh -p 23762 xcoengvm226019
cd /workspace/hip-ep/build/hip-ep-1
ninja hip-mlir-opt

bin/hip-mlir-opt \
  /workspace/hip-ep/hip-ep-1/test/lit/Conversion/onnx-to-hipsr/simple.mlir \
  --convert-onnx-to-hipsr
```

See `tech/guides/HIP-EP-DEV-ENV.md` for build instructions.
