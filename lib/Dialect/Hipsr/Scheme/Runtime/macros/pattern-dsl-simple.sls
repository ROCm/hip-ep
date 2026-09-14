#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; Pattern DSL - Simplified macro for single-operation patterns
;;
;; Usage:
;;   (define-simple-pattern pattern-name "onnx.OpName"
;;     (lambda (op rewriter)
;;       ;; Rewrite body - can access op, rewriter
;;       ...))
;;
;;===----------------------------------------------------------------------===;;

(library (pattern-dsl-simple)
  (export define-simple-pattern)
  (import (rnrs (6))
          (mlir ffi))

  ;; Simplified pattern macro
  ;; Generates a function that:
  ;; 1. Checks operation name
  ;; 2. Calls user-provided rewrite function
  ;;
  ;; Syntax:
  ;;   (define-simple-pattern pattern-name "op-name" rewrite-fn)
  ;;
  ;; Expands to:
  ;;   (define (pattern-name op rewriter)
  ;;     (and (string=? (mlir-operation-name op) "op-name")
  ;;          (rewrite-fn op rewriter)))
  ;;
  (define-syntax define-simple-pattern
    (syntax-rules ()
      [(_ pattern-name op-name-string rewrite-fn)
       (define (pattern-name op rewriter)
         (and (string=? (mlir-operation-name op) op-name-string)
              (rewrite-fn op rewriter)))]))

) ;; end library (pattern-dsl-simple)
