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
  ;; Cast Pattern - MLIR-like Syntax with :where Clause
  ;;
  ;; Operations come first, helper bindings defined in :where (like Haskell)
  ;;===--------------------------------------------------------------------===;;

  (define-conversion-pattern onnx-cast->hipsr
    :match 
    (%output = "onnx.Cast" (%input) ((to = !to_type)) : (!input-type) -> !output-type)
    :rewrite %output :with
    (%placeholder = "hipsr.placeholder" (%ctx %input !output-device)
                    ((placeholder_type 0))
                    : (!input-device) -> !output-device)
    (%cast = "hipsr.cast" (%ctx %input %placeholder !output-device)
             ((cast_attrs))
             : (!input-device !output-device) -> !output-device)
    :where
    [%ctx (mlir-get-hipsr-context-arg op)]
    [!input-device (mlir-tensor-type-in-device-space !input-type)]
    [!output-device (mlir-tensor-type-in-device-space !output-type)])

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter))

) ;; end library (patterns cast)
