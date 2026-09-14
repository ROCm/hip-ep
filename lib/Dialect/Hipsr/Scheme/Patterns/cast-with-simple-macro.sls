#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; Cast Pattern - Using Simple Macro
;;
;; Demonstrates the simple pattern macro
;;
;;===----------------------------------------------------------------------===;;

(library (patterns cast-with-simple-macro)
  (export populate-cast-patterns-macro
          onnx-cast->hipsr-with-macro)
  (import (rnrs (6))
          (only (chezscheme) format)
          (mlir ffi)
          (pattern-dsl-simple))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern - Using Macro
  ;;===--------------------------------------------------------------------===;;

  (define-simple-pattern onnx-cast->hipsr-with-macro "onnx.Cast"
    (lambda (op rewriter)
      ;; Match: Check operand and result counts
      (and (= (mlir-operation-num-operands op) 1)
           (= (mlir-operation-num-results op) 1)

           ;; Rewrite: Create replacement operations
           (let* ([%input (mlir-operation-get-operand-value op 0)]
                  [%output (mlir-operation-get-result-value op 0)]
                  [ctx (mlir-get-hipsr-context-arg op)]
                  [loc (mlir-operation-get-loc op)]
                  [input-type (mlir-value-get-type %input)]
                  [output-type (mlir-value-get-type %output)])

             (let* ([%placeholder (mlir-create-placeholder-op ctx %input input-type 0)]
                    [%cast (mlir-create-cast-op ctx %input %placeholder output-type)])

               ;; Replace original op with new op
               (mlir-replace-op op %cast)
               #t)))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns-macro converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr-with-macro))

) ;; end library (patterns cast-with-simple-macro)
