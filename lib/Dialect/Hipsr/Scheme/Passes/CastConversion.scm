;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; CastConversion - Convert ONNX Cast operations to HipSR dialect (Scheme)
;;
;; This demonstrates pattern matching and rewriting MLIR operations in pure
;; Scheme. It's a port of lib/Conversion/OnnxToHipsr/CastConversion.cpp
;;
;; Pattern: onnx.Cast -> hipsr.placeholder + hipsr.cast
;;
;;===----------------------------------------------------------------------===;;

;;===----------------------------------------------------------------------===;;
;; Imports
;;===----------------------------------------------------------------------===;;

(import (rime loop))

;;===----------------------------------------------------------------------===;;
;; Pattern Matching Helpers
;;===----------------------------------------------------------------------===;;

;; Check if operation is onnx.Cast
(define (is-onnx-cast? op)
  (string=? (mlir-operation-name op) "onnx.Cast"))

;; Check if type is ranked tensor
(define (is-ranked-tensor? type)
  ;; TODO: Add MLIR type checking FFI
  #t)

;;===----------------------------------------------------------------------===;;
;; Conversion Pattern Implementation
;;===----------------------------------------------------------------------===;;

;; Get HipSR context argument (first function argument)
;; Returns context value or #f if not found
(define (get-hipsr-context op)
  ;; TODO: Walk up to find parent function, get arg 0
  ;; For now, return placeholder
  (mlir-log-debug "Getting HipSR context for operation")
  #f)

;; Convert ONNX Cast to HipSR Cast pattern
;; Pattern: onnx.Cast(input, to=type) ->
;;          init = hipsr.placeholder(ctx, input, Normal)
;;          hipsr.cast(ctx, input, init)
(define (convert-cast-op op config)
  (mlir-log-trace (format "Attempting to convert: ~a" (mlir-operation-name op)))

  (if (not (is-onnx-cast? op))
      (mlir-log-trace "Not an onnx.Cast operation")
      (let* ((num-operands (mlir-operation-num-operands op))
             (num-results (mlir-operation-num-results op)))

        (mlir-log-debug (format "Found onnx.Cast: operands=~a results=~a"
                               num-operands num-results))

        ;; Pattern matching logic
        (when (and (= num-operands 1) (= num-results 1))
          (mlir-log-info "Pattern match: onnx.Cast with 1 input, 1 output")

          ;; TODO: Actual rewrite would:
          ;; 1. Get context: ctx = getHipsrContextArg(op)
          ;; 2. Get input: input = adaptor.getInput()
          ;; 3. Get result type: resultType = op.getOutput().getType()
          ;; 4. Create placeholder: init = PlaceholderOp(ctx, input, Normal)
          ;; 5. Create cast: castOp = CastOp(ctx, input, init)
          ;; 6. Replace op: replaceOp(op, castOp.getResult(0))

          (mlir-log-info "  -> Would create: hipsr.placeholder + hipsr.cast")
          (mlir-log-warning "  -> Rewrite not implemented yet (need FFI for IR modification)")))))

;;===----------------------------------------------------------------------===;;
;; Pass Configuration
;;===----------------------------------------------------------------------===;;

(define (default-config)
  (lambda (key)
    (case key
      [(operation-filter) '("onnx.Cast")]  ; Only process onnx.Cast
      [else #f])))

;;===----------------------------------------------------------------------===;;
;; Public API
;;===----------------------------------------------------------------------===;;

;; Entry point: Find and convert all onnx.Cast operations
(define (run-pass module-op . args)
  (let ((config (if (null? args) (default-config) (car args))))
    (mlir-log-info "Starting CastConversion Pass (Scheme)")
    (mlir-log-debug (format "Module: ~a" (mlir-operation-name module-op)))

    ;; Walk all operations looking for onnx.Cast
    (mlir-operation-walk module-op
      (lambda (op)
        (convert-cast-op op config)))

    (mlir-log-info "Completed CastConversion Pass (Scheme)")))
