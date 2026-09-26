#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.ScatterND → hipsr.scatter_nd
;;
;; Output shape = data shape (identity: scatter writes into a copy of data).
;;
;;===----------------------------------------------------------------------===;;

(library (patterns scatter-nd)
  (export populate-scatter-nd-patterns
          onnx-scatter-nd->hipsr)
  (import (except (rnrs (6)) =)
          (mlir ffi)
          (mlir hipsr)
          (mlir pattern-macro))

  (define-conversion-pattern (onnx-scatter-nd->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.ScatterND (%data %indices %updates)
    :then-let
        ([%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (mlir-tensor-type-in-device-space! !output-type)]
         [!shape-type    (mlir-get-shape-shape-type (mlir-operation-get-context op))])
    :rewrite %output :with
        ;; placeholder ins = (%data) only: scatter output has data's shape
        (%placeholder = hipsr.placeholder (%ctx %data !output-device)
                        :regions ((^bb0 ((%data-shape : !shape-type))
                                    (%yield = hipsr.shape_yield (%data-shape) -> ())))
                        -> !output-device)
        (%result = hipsr.scatter_nd (%ctx %data %indices %updates %placeholder !output-device)
                   -> !output-device))

  (define (populate-scatter-nd-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.ScatterND"
                                      onnx-scatter-nd->hipsr type-converter))

) ;; end library (patterns scatter-nd)
