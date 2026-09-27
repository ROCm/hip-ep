#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Shape — placeholder for future Scheme DSL port.
;;
;; Blocked by: hipsr.compute requires operandSegmentSizes attribute that
;; the generic mlir-build-op-with-regions does not set. Need special support
;; in the builder FFI before this can be expressed in the DSL.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns shape)
  (export populate-shape-patterns)
  (import (rnrs (6))
          (mlir ffi))

  ;; Delegate to C++ for now — see blocker above.
  (define (populate-shape-patterns type-converter patterns ctx)
    (mlir-populate-shape-conversion-patterns type-converter patterns ctx))

) ;; end library (patterns shape)
