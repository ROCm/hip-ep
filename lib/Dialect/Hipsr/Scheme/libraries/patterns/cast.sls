#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Cast Pattern - Manual Implementation (No Macros)
;;
;; This shows what the macro should generate. Once this works,
;; we can build the macro to automate generation.
;;
;;===----------------------------------------------------------------------===;;

(library (patterns cast)
  (export populate-cast-patterns
          onnx-cast->hipsr-manual)
  (import (rnrs (6))
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern - Hand-written
  ;;===--------------------------------------------------------------------===;;

  ;; Pattern function: takes (op rewriter) and returns #t on successful match+rewrite
  (define (onnx-cast->hipsr-manual op rewriter)
    ;; Match: Check operation name
    (and (string=? (mlir-operation-name op) "onnx.Cast")

         ;; Match: Check operand count
         (= (mlir-operation-num-operands op) 1)

         ;; Match: Check result count
         (= (mlir-operation-num-results op) 1)

         ;; Extract operands and types
         (let* ([%input (mlir-operation-get-operand-value op 0)]
                [%output (mlir-operation-get-result-value op 0)]

                ;; Get context
                [ctx (mlir-get-hipsr-context-arg op)]

                ;; Get types
                [input-type (mlir-value-get-type %input)]
                [output-type (mlir-value-get-type %output)])

           ;; Rewrite: Create replacement operations
           ;; mlir-create-placeholder-op: (ctx input result-type placeholder-type-int)
           (let* ([%placeholder (mlir-create-placeholder-op ctx %input output-type 0)]
                  [%cast (mlir-create-cast-op ctx %input %placeholder output-type)])

             ;; Replace original op with new op
             (mlir-replace-op op %cast)
             #t))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr-manual))

) ;; end library (patterns cast)
