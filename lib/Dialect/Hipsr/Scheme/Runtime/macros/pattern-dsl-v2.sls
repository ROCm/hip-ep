#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Pattern DSL - Simplified Implementation
;;
;; This is a simplified version that generates straightforward code
;; for easier debugging and iteration.
;;
;;===----------------------------------------------------------------------===;;

(library (pattern-dsl-v2)
  (export define-conversion-pattern-simple)
  (import (rnrs (6))
          (mlir ffi))

  ;; Simplified pattern macro - manual expansion template
  ;;
  ;; User writes:
  ;;   (define-conversion-pattern-simple onnx-cast->hipsr "onnx.Cast"
  ;;     (lambda (op rewriter %input)
  ;;       (mlir-build ...)))
  ;;
  ;; Expands to:
  ;;   (define (onnx-cast->hipsr op rewriter)
  ;;     (and (string=? (mlir-operation-name op) "onnx.Cast")
  ;;          (= (mlir-operation-num-operands op) 1)
  ;;          (let ([%input (mlir-operation-get-operand-value op 0)])
  ;;            ((lambda (op rewriter %input) ...) op rewriter %input))))
  ;;
  (define-syntax define-conversion-pattern-simple
    (syntax-rules ()
      [(_ pattern-name op-name rewrite-fn)
       (define (pattern-name op rewriter)
         (and (string=? (mlir-operation-name op) op-name)
              (let ([%input (mlir-operation-get-operand-value op 0)])
                (rewrite-fn op rewriter %input))))]))

) ;; end library (pattern-dsl-v2)
