#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Cast → hipsr.cast
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr cast)
  (export populate-cast-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects tensor)
          (mlir dialects shape)
          (mlir ddr))

  (define-conversion-pattern (onnx-cast->hipsr op operands-ref rewriter type-converter)
    :if-match
        %output = onnx.Cast (%input)
    :then-let
        ([ctx            (mlir-operation-get-context op)]
         [%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (mlir-tensor-type-with-encoding !output-type
                            (make-hipsr-device-space-attr ctx))]
         [!shape-type    (mlir-shape.shape-type ctx)])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %input)
                        (^bb0 ((%shape-in : !shape-type))
                              (hipsr.shape_yield (%shape-in)))
                        -> !output-device)
        (%cast = hipsr.cast (%ctx %input %placeholder)
                 -> !output-device))

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter 1))

) ;; end library (onnx-to-hipsr cast)
