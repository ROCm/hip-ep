#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Pattern DSL - High-level Scheme DSL for MLIR pattern matching
;;
;; Provides ergonomic macros and combinators for writing MLIR conversion
;; patterns in Scheme, inspired by MLIR's TableGen pattern syntax.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir pattern-dsl)
  (export
    ;; Pattern matching
    op-matches?
    get-hipsr-context
    has-ranked-tensor-type?
    make-pattern-matcher

    ;; Constraints
    has-n-operands
    has-n-results
    result-0-is-ranked-tensor

    ;; Rewrite actions
    rewrite-with-placeholder-and-cast

    ;; Pattern definition
    define-conversion-pattern
    apply-pattern
    apply-patterns
    )

  (import (rnrs (6))
          (mlir ffi)
          (rime loop))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Matching Helpers
  ;;===--------------------------------------------------------------------===;;

  ;; Check if operation matches a name pattern
  (define (op-matches? op op-name)
    (string=? (mlir-operation-name op) op-name))

  ;; Get context argument (first function argument)
  (define (get-hipsr-context op)
    (mlir-operation-get-block-argument op 0))

  ;; Check if value has ranked tensor type
  (define (has-ranked-tensor-type? value)
    (let ((type (mlir-value-get-type value)))
      (= 1 (mlir-type-is-ranked-tensor type))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Builders
  ;;===--------------------------------------------------------------------===;;

  ;; Build a pattern matcher that checks operation name and constraints
  ;; Returns: #t if pattern matches, #f otherwise
  (define (make-pattern-matcher op-name constraints)
    (lambda (op)
      (and (op-matches? op op-name)
           (loop :for constraint :in constraints
                 :break #f :unless (constraint op)
                 :finally #t))))

  ;; Constraint: operation must have N operands
  (define (has-n-operands n)
    (lambda (op)
      (= (mlir-operation-num-operands op) n)))

  ;; Constraint: operation must have N results
  (define (has-n-results n)
    (lambda (op)
      (= (mlir-operation-num-results op) n)))

  ;; Constraint: first result must be ranked tensor
  (define (result-0-is-ranked-tensor)
    (lambda (op)
      (and (> (mlir-operation-num-results op) 0)
           (has-ranked-tensor-type?
             (mlir-operation-get-result-value op 0)))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Rewrite Helpers
  ;;===--------------------------------------------------------------------===;;

  ;; Pattern rewrite action: creates placeholder and cast operations
  ;; This implements the ONNX Cast -> HipSR conversion with device memory space
  (define (rewrite-with-placeholder-and-cast op)
    (let* ((ctx (get-hipsr-context op))
           (input (mlir-operation-get-operand-value op 0))
           (input-type (mlir-value-get-type input))
           (result-value (mlir-operation-get-result-value op 0))
           (result-type (mlir-value-get-type result-value))
           ;; Set memory space to Device (1) for both input and result types
           (device-input-type (mlir-type-set-memory-space input-type 1))
           (device-result-type (mlir-type-set-memory-space result-type 1)))

      (mlir-log-debug "Pattern matched successfully")
      (mlir-log-info (string-append "Converting input to device memory space"))
      (mlir-log-info (string-append "Creating: hipsr.placeholder(ctx, device-input) with device type"))
      (mlir-log-info (string-append "Creating: hipsr.cast(ctx, device-input, placeholder) with device type"))
      (mlir-log-info (string-append "Replacing: " (mlir-operation-name op) " with cast result"))

      ;; Convert input to device memory space using unrealized_conversion_cast
      (let* ((device-input (mlir-create-unrealized-conversion-cast input device-input-type))
             (placeholder (mlir-create-placeholder-op ctx device-input device-result-type 0))
             (cast (mlir-create-cast-op ctx device-input placeholder device-result-type)))
        (mlir-replace-op op cast))

      #t))

  ;;===--------------------------------------------------------------------===;;
  ;; High-Level Pattern Definition
  ;;===--------------------------------------------------------------------===;;

  ;; Define a conversion pattern
  ;; Usage: (define-conversion-pattern "onnx.Cast"
  ;;          (list (has-n-operands 1) (has-n-results 1))
  ;;          rewrite-with-placeholder-and-cast)
  (define (define-conversion-pattern op-name constraints rewrite-action)
    (let ((matcher (make-pattern-matcher op-name constraints)))
      (lambda (op)
        (if (matcher op)
            (rewrite-action op)
            (begin
              (mlir-notify-match-failure op
                (string-append "Pattern " op-name " did not match"))
              #f)))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Application
  ;;===--------------------------------------------------------------------===;;

  ;; Apply a pattern to an operation
  ;; Returns: #t if pattern matched and rewrote, #f otherwise
  (define (apply-pattern pattern op)
    (pattern op))

  ;; Apply multiple patterns to an operation (try each in order)
  ;; Returns: #t if any pattern matched, #f if none matched
  (define (apply-patterns patterns op)
    (loop :for pattern :in patterns
          :break #t :if (apply-pattern pattern op)
          :finally #f))

) ;; end library (mlir pattern-dsl)
