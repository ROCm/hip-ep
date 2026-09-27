#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; onnx.Constant → hipsr.constant (or arith.constant for rank-0 scalars)
;;
;; Handled cases:
;;   - Rank > 0 dense inline value → hipsr.constant with copied `value` attr
;;   - Rank 0 scalar → arith.constant with copied `value` attr
;;
;; NOT handled (returns #f, pattern fails):
;;   - External data (absent `value`, present `location`) — these models require
;;     file-mapped DenseResourceElementsAttr construction not yet available in Scheme.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns constant)
  (export populate-constant-patterns)
  (import (rnrs (6))
          (mlir ffi)
          (mlir hipsr))

  (define (onnx-constant->hipsr op operands-ref rewriter type-converter)
    ;; Fail fast if there is no inline value (external data path)
    (if (= 0 (mlir-operation-has-attr op "value"))
        #f
        (let* ((out-type  (mlir-value-get-type (mlir-operation-get-result op 0)))
               (out-dev   (mlir-tensor-type-in-device-space! out-type))
               (rank      (mlir-type-get-rank out-type)))
          (if (= rank 0)
              ;; Rank-0 scalar: arith.constant keeps the raw (unencoded) result type
              (begin
                (mlir-set-insertion-point-before rewriter op)
                (let* ((c-op (mlir-build-op rewriter op "arith.constant" '() (list out-type))))
                  (mlir-operation-copy-attr c-op "value" op "value")
                  (mlir-replace-op rewriter op (mlir-operation-get-result c-op 0))
                  #t))
              ;; Rank > 0: hipsr.constant with device result type
              (begin
                (mlir-set-insertion-point-before rewriter op)
                (let* ((c-op (mlir-build-op rewriter op "hipsr.constant" '() (list out-dev))))
                  (mlir-operation-copy-attr c-op "value" op "value")
                  (mlir-replace-op rewriter op (mlir-operation-get-result c-op 0))
                  #t))))))

  (define (populate-constant-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Constant"
                                      onnx-constant->hipsr type-converter))

) ;; end library (patterns constant)
