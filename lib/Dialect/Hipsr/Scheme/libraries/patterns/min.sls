#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Min → chain of hipsr.min
;;
;; N=1: identity (replace with the single input).
;; N≥2: chain binary hipsr.min ops. Each step creates a Normal placeholder
;;      with ins=(lhs, rhs) — shape region filled by hipsr-populate-shape-region.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns min)
  (export populate-min-patterns)
  (import (rnrs (6))
          (mlir ffi)
          (mlir hipsr))

  (define (make-binary-min! rewriter loc-op ctx lhs rhs out-type)
    ;; Create placeholder then hipsr.min; return the min result value.
    (mlir-set-insertion-point-before rewriter loc-op)
    (let* ((ph-op (mlir-build-op rewriter loc-op "hipsr.placeholder"
                    (list ctx lhs rhs out-type) (list out-type)))
           (ph-val (mlir-operation-get-result ph-op 0)))
      (mlir-set-insertion-point-before rewriter loc-op)
      (let* ((min-op (mlir-build-op rewriter loc-op "hipsr.min"
                        (list ctx lhs rhs ph-val out-type) (list out-type))))
        (mlir-operation-get-result min-op 0))))

  (define (onnx-min->hipsr op operands-ref rewriter type-converter)
    (let ((n (value-array-ref-size operands-ref)))
      (cond
        ((= n 0) #f)
        ((= n 1)
         (mlir-replace-op rewriter op (value-array-ref-at operands-ref 0))
         #t)
        (else
         (let* ((ctx       (mlir-get-hipsr-context-arg op))
                (out-type  (mlir-tensor-type-in-device-space!
                              (mlir-value-get-type (mlir-operation-get-result op 0)))))
           (let loop ((i 1)
                      (acc (value-array-ref-at operands-ref 0)))
             (if (= i n)
                 (begin
                   (mlir-replace-op rewriter op acc)
                   #t)
                 (loop (+ i 1)
                       (make-binary-min! rewriter op ctx acc
                                         (value-array-ref-at operands-ref i)
                                         out-type)))))))))

  (define (populate-min-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Min"
                                      onnx-min->hipsr type-converter))

) ;; end library (patterns min)
