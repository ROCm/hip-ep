#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Gather → hipsr.gather  (device data path)
;;
;; Only the device-data path is implemented here. Host-data (compile-time
;; constant indices, hipsr.compute expansion) is not yet ported and falls
;; through to failure.
;;
;; Creates a Normal placeholder with ins=(data, indices) then hipsr.gather{axis}.
;; The placeholder shape region is filled by hipsr-populate-shape-region.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns gather)
  (export populate-gather-patterns)
  (import (rnrs (6))
          (mlir ffi)
          (mlir hipsr))

  (define (onnx-gather->hipsr op operands-ref rewriter type-converter)
    (let* ((data    (value-array-ref-at operands-ref 0))
           (indices (value-array-ref-at operands-ref 1))
           (data-type (mlir-value-get-type data)))
      ;; Only handle device data
      (if (not (= 1 (mlir-type-is-device-tensor data-type)))
          #f
          (let* ((ctx      (mlir-get-hipsr-context-arg op))
                 (out-type (mlir-value-get-type (mlir-operation-get-result op 0)))
                 (out-dev  (mlir-tensor-type-in-device-space! out-type))
                 ;; axis attribute (ONNX stores as si64; default 0)
                 (axis     (let ((a (mlir-operation-get-integer-attr op "axis" 0)))
                              (if (< a 0) (+ a (mlir-type-get-rank data-type)) a))))
            ;; Normal placeholder with ins=(data, indices)
            (mlir-set-insertion-point-before rewriter op)
            (let* ((ph-op  (mlir-build-op rewriter op "hipsr.placeholder"
                              (list ctx data indices out-dev) (list out-dev)))
                   (ph-val (mlir-operation-get-result ph-op 0)))
              ;; hipsr.gather with axis attribute
              (mlir-set-insertion-point-before rewriter op)
              (let* ((g-op (mlir-build-op rewriter op "hipsr.gather"
                               (list ctx data indices ph-val out-dev) (list out-dev))))
                (mlir-operation-set-attr g-op "axis" axis)
                (mlir-replace-op rewriter op (mlir-operation-get-result g-op 0))
                #t))))))

  (define (populate-gather-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Gather"
                                      onnx-gather->hipsr type-converter))

) ;; end library (patterns gather)
