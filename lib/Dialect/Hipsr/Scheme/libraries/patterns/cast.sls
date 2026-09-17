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
  ;; Cast Pattern - Using Macro
  ;;===--------------------------------------------------------------------===;;

  (define-conversion-pattern onnx-cast->hipsr
    :match 
    (%output = "onnx.Cast" (%input) ((to = !to_type)) : (!input-type) -> !output-type)
    :rewrite %output :with
    (%ctx = (mlir-get-hipsr-context-arg op))
    (%placeholder = (mlir-create-placeholder-op %ctx %input !output-type 0))
    (%cast = (mlir-create-cast-op %ctx %input %placeholder !output-type)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter))

) ;; end library (patterns cast)
