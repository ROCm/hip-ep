#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; Min Pattern - Using Builders Library
;;
;; Cleaner version using hipsr-builders helpers
;;
;;===----------------------------------------------------------------------===;;

(library (patterns min-with-builders)
  (export populate-min-patterns-with-builders
          onnx-min->hipsr-with-builders)
  (import (rnrs (6))
          (only (chezscheme) format)
          (mlir ffi)
          (pattern-dsl-simple)
          (hipsr-builders))

  ;;===--------------------------------------------------------------------===;;
  ;; Min Pattern
  ;;===--------------------------------------------------------------------===;;

  (define-simple-pattern onnx-min->hipsr-with-builders "onnx.Min"
    (lambda (op rewriter)
      ;; Match: Check at least one input
      (and (> (mlir-operation-num-operands op) 0)
           (= (mlir-operation-num-results op) 1)

           ;; Rewrite
           (let* ([num-operands (mlir-operation-num-operands op)]
                  [ctx (mlir-get-hipsr-context-arg op)]
                  [result-value (mlir-operation-get-result-value op 0)]
                  [result-type (mlir-value-get-type result-value)])

             ;; Single input - identity
             (if (= num-operands 1)
                 (let ([input0 (mlir-operation-get-operand-value op 0)])
                   (mlir-replace-op op input0)
                   #t)

                 ;; Multiple inputs - fold with min
                 (let loop ([i 1]
                            [accumulate (mlir-operation-get-operand-value op 0)])
                   (if (>= i num-operands)
                       (begin
                         (mlir-replace-op op accumulate)
                         #t)

                       (let* ([rhs (mlir-operation-get-operand-value op i)]
                              [placeholder (create-hipsr-placeholder ctx accumulate result-type)]
                              [min-result (create-hipsr-min ctx accumulate rhs placeholder result-type)])
                         (loop (+ i 1) min-result)))))))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-min-patterns-with-builders converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Min" onnx-min->hipsr-with-builders))

) ;; end library (patterns min-with-builders)
