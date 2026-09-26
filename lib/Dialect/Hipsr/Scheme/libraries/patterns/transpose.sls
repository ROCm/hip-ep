#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Transpose → hipsr.transpose
;;
;; Reads the perm attribute from the ONNX op (absent = reverse permutation).
;; Creates a Normal placeholder with ins=(input) then hipsr.transpose{perm}.
;; The placeholder shape region is filled by hipsr-populate-shape-region.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns transpose)
  (export populate-transpose-patterns)
  (import (rnrs (6))
          (mlir ffi)
          (mlir hipsr))

  ;; Returns (rank-1 rank-2 ... 1 0) — the reverse permutation.
  (define (reverse-perm rank)
    (let loop ((i 0) (acc '()))
      (if (= i rank) acc (loop (+ i 1) (cons i acc)))))

  (define (onnx-transpose->hipsr op operands-ref rewriter type-converter)
    (let* ((input     (value-array-ref-at operands-ref 0))
           (ctx       (mlir-get-hipsr-context-arg op))
           (in-type   (mlir-value-get-type input))
           (rank      (mlir-type-get-rank in-type))
           (out-type  (mlir-value-get-type (mlir-operation-get-result op 0)))
           (out-dev   (mlir-tensor-type-in-device-space! out-type))
           ;; perm attribute: ArrayAttr on ONNX op; absent = reverse permutation
           (perm      (let ((p (mlir-operation-get-integer-array-attr op "perm")))
                        (if (null? p) (reverse-perm rank) p))))
      ;; Normal placeholder with ins=(input)
      (mlir-set-insertion-point-before rewriter op)
      (let* ((ph-op  (mlir-build-op rewriter op "hipsr.placeholder"
                        (list ctx input out-dev) (list out-dev)))
             (ph-val (mlir-operation-get-result ph-op 0)))
        ;; hipsr.transpose with perm attribute
        (mlir-set-insertion-point-before rewriter op)
        (let* ((t-op (mlir-build-op rewriter op "hipsr.transpose"
                        (list ctx input ph-val out-dev) (list out-dev))))
          (mlir-operation-set-dense-i64-array t-op "perm" perm)
          (mlir-replace-op rewriter op (mlir-operation-get-result t-op 0))
          #t))))

  (define (populate-transpose-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Transpose"
                                      onnx-transpose->hipsr type-converter))

) ;; end library (patterns transpose)
