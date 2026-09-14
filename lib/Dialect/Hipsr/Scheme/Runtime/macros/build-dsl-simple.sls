#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; Build DSL - Simplified version for common patterns
;;
;; Usage:
;;   (define-op-builder create-hipsr-min "hipsr.min"
;;     (ctx lhs rhs init result-type)
;;     ...)
;;
;;===----------------------------------------------------------------------===;;

(library (build-dsl-simple)
  (export define-op-builder)
  (import (rnrs (6))
          (mlir ffi))

  ;; Simple operation builder macro
  ;; Creates a helper function for building a specific operation
  ;;
  ;; Syntax:
  ;;   (define-op-builder builder-name "op-name"
  ;;     (param ...)
  ;;     body ...)
  ;;
  ;; Example:
  ;;   (define-op-builder create-min "hipsr.min"
  ;;     (ctx lhs rhs init result-type)
  ;;     (let ([operands (list ctx lhs rhs init)]
  ;;           [types (list result-type)])
  ;;       (mlir-create-generic-op "hipsr.min" operands types)))
  ;;
  (define-syntax define-op-builder
    (syntax-rules ()
      [(_ builder-name op-name-string (param ...) body ...)
       (define (builder-name param ...)
         body ...)]))

) ;; end library (build-dsl-simple)
