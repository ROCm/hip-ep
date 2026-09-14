#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; HipSR Operation Builders
;;
;; Helper functions for creating common HipSR operations
;;
;;===----------------------------------------------------------------------===;;

(library (hipsr-builders)
  (export create-hipsr-placeholder
          create-hipsr-cast
          create-hipsr-min
          create-hipsr-equal
          create-hipsr-add
          create-hipsr-mul)
  (import (rnrs (6))
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; Placeholder
  ;;===--------------------------------------------------------------------===;;

  (define (create-hipsr-placeholder ctx input result-type)
    (mlir-create-placeholder-op ctx input result-type 0))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast
  ;;===--------------------------------------------------------------------===;;

  (define (create-hipsr-cast ctx input placeholder result-type)
    (mlir-create-cast-op ctx input placeholder result-type))

  ;;===--------------------------------------------------------------------===;;
  ;; Binary Operations (min, equal, add, mul, etc.)
  ;;===--------------------------------------------------------------------===;;

  (define (create-hipsr-min ctx lhs rhs init result-type)
    (let* ([operands (list ctx lhs rhs init)]
           [types (list result-type)]
           [op (mlir-create-generic-op "hipsr.min" operands types)])
      (mlir-operation-get-result-value-from-op op 0)))

  (define (create-hipsr-equal ctx lhs rhs init result-type)
    (let* ([operands (list ctx lhs rhs init)]
           [types (list result-type)]
           [op (mlir-create-generic-op "hipsr.equal" operands types)])
      (mlir-operation-get-result-value-from-op op 0)))

  (define (create-hipsr-add ctx lhs rhs init result-type)
    (let* ([operands (list ctx lhs rhs init)]
           [types (list result-type)]
           [op (mlir-create-generic-op "hipsr.add" operands types)])
      (mlir-operation-get-result-value-from-op op 0)))

  (define (create-hipsr-mul ctx lhs rhs init result-type)
    (let* ([operands (list ctx lhs rhs init)]
           [types (list result-type)]
           [op (mlir-create-generic-op "hipsr.mul" operands types)])
      (mlir-operation-get-result-value-from-op op 0)))

  ;;===--------------------------------------------------------------------===;;
  ;; Unary Operations (transpose, shape, etc.)
  ;;===--------------------------------------------------------------------===;;

  ;; Note: Transpose needs perm attribute - requires attribute FFI
  ;; For now, create without attributes (will add later)
  (define (create-hipsr-transpose ctx input init result-type)
    (let* ([operands (list ctx input init)]
           [types (list result-type)]
           [op (mlir-create-generic-op "hipsr.transpose" operands types)])
      (mlir-operation-get-result-value-from-op op 0)))

  (define (create-hipsr-shape ctx input init result-type)
    (let* ([operands (list ctx input init)]
           [types (list result-type)]
           [op (mlir-create-generic-op "hipsr.shape" operands types)])
      (mlir-operation-get-result-value-from-op op 0)))

) ;; end library (hipsr-builders)
