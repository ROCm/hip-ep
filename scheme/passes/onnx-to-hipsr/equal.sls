#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Equal → hipsr.equal
;;
;; Output shape = broadcast(lhs_shape, rhs_shape).
;;
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr equal)
  (export populate-equal-patterns)
  (import (except (rnrs (6)) =)
          (mlir core ir)
          (mlir core conversion)
          (mlir dialects hipsr)
          (mlir dialects tensor)
          (mlir dialects shape)
          (mlir ddr))

  (define-conversion-pattern (onnx-equal->hipsr op operands-ref rewriter type-converter)
    :if-match
        %output = onnx.Equal (%lhs %rhs)
    :then-let
        ([ctx            (mlir-operation-get-context op)]
         [%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (mlir-tensor-type-with-encoding !output-type
                            (make-hipsr-device-space-attr ctx))]
         [!shape-type    (mlir-shape.shape-type ctx)])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %lhs %rhs)
                        (^bb0 ((%lhs-shape : !shape-type) (%rhs-shape : !shape-type))
                              (%bcast = shape.broadcast (%lhs-shape %rhs-shape) -> !shape-type)
                              (hipsr.shape_yield (%bcast)))
                        -> !output-device)
        (%result = hipsr.equal (%ctx %lhs %rhs %placeholder)
                   -> !output-device))

  (define (populate-equal-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Equal"
                                      onnx-equal->hipsr type-converter 1))

) ;; end library (onnx-to-hipsr equal)
