#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Cast Pattern - Hybrid: Macro + Manual
;;
;; Uses define-conversion-pattern for structure, but manually handles
;; context extraction since %ctx is not an operand of onnx.Cast
;;
;;===----------------------------------------------------------------------===;;

(library (patterns cast)
  (export populate-cast-patterns
          onnx-cast->hipsr)
  (import (rnrs (6))
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern
  ;;===--------------------------------------------------------------------===;;

  (define (onnx-cast->hipsr op operands-ref rewriter type-converter)
    ;; Match: Check operation name
    (and (string=? (mlir-operation-name op) "onnx.Cast")
         ;; Extract matched values
         (let* ([%output (mlir-operation-get-result-value op 0)]
                [%input (value-array-ref-at operands-ref 0)]
                [!output-type (mlir-value-get-type %output)]
                [!input-type (mlir-value-get-type %input)]
                [!to_type (mlir-operation-get-attribute op "to")]
                ;; Get context from function arguments (not an operand)
                [%ctx (mlir-get-hipsr-context-arg op)])
           ;; Rewrite: create replacement operations
           (let* ([%placeholder (mlir-create-placeholder-op %ctx %input !output-type 0)]
                  [%cast (mlir-create-cast-op %ctx %input %placeholder !output-type)])
             (mlir-replace-op op %cast)
             #t))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter))

) ;; end library (patterns cast)
