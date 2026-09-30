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

(library (passes onnx-to-hipsr cast)
  (export populate-cast-patterns
          onnx-cast->hipsr)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects shape)
          (mlir ddr))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern - MLIR-like Syntax with :where Clause
  ;; Operations come first, helper bindings defined in :where (like Haskell)
  ;;===--------------------------------------------------------------------===;;

  ;; Use the pattern DSL macro with explicit parameters
  (define-conversion-pattern (onnx-cast->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Cast (%input)
    :then-let
        ([%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (make-mlir-tensor-in-device-space !output-type)]
         [!shape-type    (mlir-shape.shape-type (mlir-operation-get-context op))])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %input !output-device)
                        (^bb0 ((%shape-in : !shape-type))
                              (hipsr.shape_yield (%shape-in)))
                        -> !output-device)
        (%cast = hipsr.cast (%ctx %input %placeholder !output-device)
                 -> !output-device))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter))

) ;; end library (onnx-to-hipsr cast)
