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

  ;; Pattern function: takes (op operands-ref rewriter type-converter) and returns #t on successful match+rewrite
  ;; New signature mirrors C++ OpConversionPattern::matchAndRewrite
  (define (onnx-cast->hipsr-manual op operands-ref rewriter type-converter)
    (mlir-log-debug "Cast pattern: checking match")
    ;; Match: Check operation name
    (and (string=? (mlir-operation-name op) "onnx.Cast")

         ;; Match: Check operand count
         (= (mlir-operation-num-operands op) 1)

         ;; Match: Check result count
         (= (mlir-operation-num-results op) 1)

         ;; Extract operands and types
         ;; operands-ref is a ValueArrayRef* (pointer to struct with data + size)
         (let* ([%input (value-array-ref-at operands-ref 0)]  ; Get first operand from array
                [%output (mlir-operation-get-result-value op 0)]

                ;; Get context VALUE from function argument (not MLIRContext*)
                [%ctx (mlir-get-hipsr-context-arg op)]

                ;; Get types
                [input-type (mlir-value-get-type %input)]
                [output-type (mlir-value-get-type %output)])

           (mlir-log-debug "Cast pattern: matched, rewriting")
           ;; Rewrite: Create replacement operations
           ;; mlir-create-placeholder-op: (ctx-value input result-type placeholder-type-int)
           (let* ([%placeholder (mlir-create-placeholder-op %ctx %input output-type 0)]
                  [%cast (mlir-create-cast-op %ctx %input %placeholder output-type)])

             (mlir-log-debug "Cast pattern: replacing op")
             ;; Replace original op with new op
             (mlir-replace-op op %cast)
             (mlir-log-debug "Cast pattern: success")
             #t))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  ;; Register Cast pattern with the conversion framework
  ;;
  ;; Parameters:
  ;;   type-converter - MLIR TypeConverter (passed to pattern callback)
  ;;   patterns       - RewritePatternSet to add patterns to
  ;;   ctx            - MLIR Context (unused here, kept for API consistency)
  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr-manual type-converter))

) ;; end library (patterns cast)
