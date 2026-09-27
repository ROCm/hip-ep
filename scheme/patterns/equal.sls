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

(library (patterns equal)
  (export populate-equal-patterns
          onnx-equal->hipsr)
  (import (except (rnrs (6)) =)
          (mlir ffi)
          (mlir hipsr)
          (mlir pattern-macro))

  (define-conversion-pattern (onnx-equal->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Equal (%lhs %rhs)
    :then-let
        ([%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (mlir-tensor-type-in-device-space! !output-type)]
         [!shape-type    (mlir-get-shape-shape-type (mlir-operation-get-context op))])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %lhs %rhs !output-device)
                        :regions ((^bb0 ((%lhs-shape : !shape-type) (%rhs-shape : !shape-type))
                                    (%bcast = shape.broadcast (%lhs-shape %rhs-shape) -> !shape-type)
                                    (%yield = hipsr.shape_yield (%bcast) -> ())))
                        -> !output-device)
        (%result = hipsr.equal (%ctx %lhs %rhs %placeholder !output-device)
                   -> !output-device))

  (define (populate-equal-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Equal"
                                      onnx-equal->hipsr type-converter))

) ;; end library (patterns equal)
