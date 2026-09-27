#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.MatMul → hipsr.matmul
;;
;; Creates a Normal placeholder with ins=(a, b) and hipsr.matmul consuming it.
;; The placeholder shape region is left empty and filled by hipsr-populate-shape-region.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns matmul)
  (export populate-matmul-patterns
          onnx-matmul->hipsr)
  (import (except (rnrs (6)) =)
          (mlir ffi)
          (mlir hipsr)
          (mlir pattern-macro))

  (define-conversion-pattern (onnx-matmul->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.MatMul (%a %b)
    :then-let
        ([%ctx           (mlir-get-hipsr-context-arg op)]
         [!output-type   (mlir-value-get-type %output)]
         [!output-device (mlir-tensor-type-in-device-space! !output-type)])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %a %b !output-device)
                        -> !output-device)
        (%result = hipsr.matmul (%ctx %a %b %placeholder !output-device)
                   -> !output-device))

  (define (populate-matmul-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.MatMul"
                                      onnx-matmul->hipsr type-converter))

) ;; end library (patterns matmul)
