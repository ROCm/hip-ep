#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Min Pattern - Manual Implementation (No Macros)
;;
;; Equivalent to lib/Conversion/OnnxToHipsr/MinConversion.cpp
;;
;;===----------------------------------------------------------------------===;;

(library (patterns min-manual)
  (export populate-min-patterns
          onnx-min->hipsr-manual)
  (import (rnrs (6))
          (only (chezscheme) format)
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; Min Pattern - Hand-written
  ;;===--------------------------------------------------------------------===;;

  ;; Pattern function: takes (op rewriter) and returns #t on successful match+rewrite
  (define (onnx-min->hipsr-manual op rewriter)
    ;; Match: Check operation name
    (and (string=? (mlir-operation-name op) "onnx.Min")

         ;; Match: Check at least one input
         (> (mlir-operation-num-operands op) 0)

         ;; Match: Check result count
         (= (mlir-operation-num-results op) 1)

         ;; Rewrite
         (let* ([num-operands (mlir-operation-num-operands op)]
                [ctx (mlir-get-hipsr-context-arg op)]
                [loc (mlir-operation-get-loc op)]
                [result-value (mlir-operation-get-result-value op 0)]
                [result-type (mlir-value-get-type result-value)])

           ;; Single input - just replace with identity
           (if (= num-operands 1)
               (let ([input0 (mlir-operation-get-operand-value op 0)])
                 (mlir-replace-op op input0)
                 #t)

               ;; Multiple inputs - fold with min operations
               (let loop ([i 1]
                          [accumulate (mlir-operation-get-operand-value op 0)])
                 (if (>= i num-operands)
                     ;; Done - replace with final accumulate
                     (begin
                       (mlir-replace-op op accumulate)
                       #t)

                     ;; Create placeholder and min op
                     (let* ([rhs (mlir-operation-get-operand-value op i)]
                            [operands-for-placeholder (list accumulate rhs)]
                            [types-for-placeholder (list result-type)]

                            ;; Create placeholder
                            [placeholder-op (mlir-create-placeholder-op
                                              ctx accumulate result-type 0)]

                            ;; Create min operation
                            ;; hipsr.min(%ctx, %lhs, %rhs, %init) : result-type
                            [operands-for-min (list ctx accumulate rhs placeholder-op)]
                            [operands-list (cons ctx (cons accumulate (cons rhs (cons placeholder-op '()))))]
                            [result-types-list (list result-type)]

                            [min-op (mlir-create-generic-op "hipsr.min"
                                                            operands-list
                                                            result-types-list)]
                            [min-result (mlir-operation-get-result-value-from-op min-op 0)])

                       ;; Continue loop with new accumulate
                       (loop (+ i 1) min-result))))))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-min-patterns converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Min" onnx-min->hipsr-manual))

) ;; end library (patterns min-manual)
