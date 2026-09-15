#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; Equal Pattern - Simplified (device operands only)
;;
;; Note: This simplified version assumes both operands are device-resident.
;; The full C++ version handles host constants - that requires additional FFI.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns equal-simple)
  (export populate-equal-patterns-simple
          onnx-equal->hipsr-simple)
  (import (rnrs (6))
          (only (chezscheme) format)
          (mlir ffi)
          (pattern-dsl-simple)
          (hipsr-builders))

  ;;===--------------------------------------------------------------------===;;
  ;; Equal Pattern (Simplified)
  ;;===--------------------------------------------------------------------===;;

  (define-simple-pattern onnx-equal->hipsr-simple "onnx.Equal"
    (lambda (op rewriter)
      ;; Match: Check 2 operands, 1 result
      (and (= (mlir-operation-num-operands op) 2)
           (= (mlir-operation-num-results op) 1)

           ;; Rewrite
           (let* ([ctx (mlir-get-hipsr-context-arg op)]
                  [lhs (mlir-operation-get-operand-value op 0)]
                  [rhs (mlir-operation-get-operand-value op 1)]
                  [result-value (mlir-operation-get-result-value op 0)]
                  [result-type (mlir-value-get-type result-value)])

             ;; TODO: Handle host constants (needs constantOnDevice FFI)
             ;; For now, assume both operands are device-resident

             (let* ([placeholder (create-hipsr-placeholder ctx lhs result-type)]
                    [equal-result (create-hipsr-equal ctx lhs rhs placeholder result-type)])

               (mlir-replace-op op equal-result)
               #t)))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-equal-patterns-simple converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Equal" onnx-equal->hipsr-simple))

) ;; end library (patterns equal-simple)
