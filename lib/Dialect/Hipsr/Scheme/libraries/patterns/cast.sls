#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Cast Pattern - Using define-conversion-pattern Macro
;;
;;===----------------------------------------------------------------------===;;

(library (patterns cast)
  (export populate-cast-patterns
          onnx-cast->hipsr)
  (import (rnrs (6))
          (mlir ffi)
          (mlir pattern-macro))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern - MLIR Operation Syntax
  ;;
  ;; Uses operation syntax: (%v = "op.name" (operands) ((attrs)) ...)
  ;; The macro recognizes "hipsr.placeholder" and "hipsr.cast" and translates
  ;; to specialized FFI calls: mlir-create-placeholder-op, mlir-create-cast-op
  ;;===--------------------------------------------------------------------===;;

  (define-conversion-pattern onnx-cast->hipsr
    :match 
    (%output = "onnx.Cast" (%input) ((to = !to_type)) : (!input-type) -> !output-type)
    :rewrite %output :with
    (%ctx = (mlir-get-hipsr-context-arg op))
    (%placeholder = "hipsr.placeholder" (%ctx %input !output-type)
                    ((placeholder_type 0))
                    : (!input-type) -> !output-type)
    (%cast = "hipsr.cast" (%ctx %input %placeholder !output-type)
             ((cast_attrs))
             : (!input-type !output-type) -> !output-type))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter))

) ;; end library (patterns cast)
