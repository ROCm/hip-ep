#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Expand → hipsr.expand
;;
;; Creates a Barrier placeholder with ins=(input, shape) then hipsr.expand.
;; The shape region (barrier layout: ctx, input-tensor, shape-tensor) is
;; filled by hipsr-populate-shape-region.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns expand)
  (export populate-expand-patterns)
  (import (rnrs (6))
          (mlir ffi)
          (mlir hipsr))

  (define (onnx-expand->hipsr op operands-ref rewriter type-converter)
    (let* ((input   (value-array-ref-at operands-ref 0))
           (shape   (value-array-ref-at operands-ref 1))
           (ctx     (mlir-get-hipsr-context-arg op))
           (out-type (mlir-value-get-type (mlir-operation-get-result op 0)))
           (out-dev  (mlir-tensor-type-in-device-space! out-type)))
      ;; Barrier placeholder with ins=(input, shape)
      (mlir-set-insertion-point-before rewriter op)
      (let* ((ph-op  (mlir-build-op rewriter op "hipsr.placeholder"
                        (list ctx input shape out-dev) (list out-dev)))
             (ph-val (mlir-operation-get-result ph-op 0)))
        ;; Upgrade placeholder_type to Barrier
        (mlir-placeholder-set-barrier-type ph-op)
        ;; hipsr.expand
        (mlir-set-insertion-point-before rewriter op)
        (let* ((e-op (mlir-build-op rewriter op "hipsr.expand"
                        (list ctx input shape ph-val out-dev) (list out-dev))))
          (mlir-replace-op rewriter op (mlir-operation-get-result e-op 0))
          #t))))

  (define (populate-expand-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Expand"
                                      onnx-expand->hipsr type-converter))

) ;; end library (patterns expand)
