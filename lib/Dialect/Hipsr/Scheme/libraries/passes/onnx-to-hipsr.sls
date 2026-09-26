#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; ONNX to HipSR Conversion - Scheme Implementation
;;
;; Equivalent to lib/Conversion/OnnxToHipsr/OnnxToHipsr.cpp
;;
;; Orchestrates the dialect conversion using MLIR framework primitives.
;; The C++ FFI layer provides primitives and reusable helpers, while this
;; Scheme code implements the high-level conversion logic.
;;===----------------------------------------------------------------------===;;

(library (passes onnx-to-hipsr)
  (export run-pass)
  (import (rnrs (6))
          (mlir ffi)
          (mlir hipsr)
          (patterns cast)
          (patterns scatter-nd)
          (patterns equal)
          (for (rime loop) expand))

  ;; Populate return-conversion patterns in Scheme.
  ;; onnx.Return → func.return, forwarding the (already type-converted) operands.
  ;; onnx.Return has 0 results so we erase it after inserting func.return.
  (define (onnx-return->func-return op operands-ref rewriter type-converter)
    (let ((operands (loop :for i :from 0 :below (value-array-ref-size operands-ref)
                         :collect (value-array-ref-at operands-ref i))))
      (mlir-set-insertion-point-before rewriter op)
      (mlir-build-op rewriter op "func.return" operands '())
      (mlir-erase-op rewriter op)
      #t))

  (define (populate-return-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Return"
                                      onnx-return->func-return type-converter))

  ;;===--------------------------------------------------------------------===;;
  ;; Post-processing: erase dead onnx.NoValue ops
  ;;
  ;; onnx.NoValue stands in for an omitted optional operand. Its consumer is
  ;; converted first (dropping the operand), leaving the NoValue with no uses.
  ;; Collect all dead ones during the walk, then erase after.
  ;;===--------------------------------------------------------------------===;;
  (define (erase-dead-novalue! module-op)
    (let ((dead '()))
      (mlir-operation-walk module-op
        (lambda (op)
          (when (and (string=? (mlir-operation-name op) "onnx.NoValue")
                     (= 1 (mlir-operation-use-empty op)))
            (set! dead (cons op dead)))))
      (for-each mlir-op-erase dead)))

  ;;===--------------------------------------------------------------------===;;
  ;; Post-processing: rewire placeholder inputs to follow the shape graph
  ;;
  ;; Placeholder inputs start out pointing at data-graph values (e.g. results
  ;; of hipsr.cast). After conversion we redirect each input to its shape-graph
  ;; counterpart so the placeholder's shape region sees only block arguments,
  ;; hipsr.placeholder results, or constants.
  ;;
  ;; getShapeGraphCounterpart logic (mirrors HipsrOps.cpp):
  ;;   - block argument or already-allowed result → keep as-is
  ;;   - otherwise → take the DPS init (outs slot) at the same result index
  ;;===--------------------------------------------------------------------===;;
  (define (shape-graph-counterpart value)
    (if (= 1 (mlir-value-is-block-argument value))
        value
        (let* ((def-op (mlir-value-get-defining-op value))
               (op-name (if (= 0 def-op) "" (mlir-operation-name def-op))))
          (if (or (string=? op-name "hipsr.placeholder")
                  (string=? op-name "hipsr.constant")
                  (string=? op-name "arith.constant"))
              value
              (let* ((result-idx (mlir-value-get-result-number value))
                     (num-inits  (mlir-operation-num-dps-inits def-op)))
                (if (>= result-idx num-inits)
                    value
                    (mlir-operation-get-dps-init-value def-op result-idx)))))))

  (define (rewire-placeholder-inputs! module-op)
    (mlir-operation-walk module-op
      (lambda (op)
        (when (string=? (mlir-operation-name op) "hipsr.placeholder")
          ;; operand 0 is context; inputs start at operand 1
          (let loop ((i 1))
            (when (< i (mlir-operation-num-operands op))
              (let* ((old-val (mlir-operation-get-operand-value op i))
                     (new-val (shape-graph-counterpart old-val)))
                (unless (= old-val new-val)
                  (mlir-operation-set-operand op i new-val)))
              (loop (+ i 1))))))))

  ;;===--------------------------------------------------------------------===;;
  ;; Helper: apply conversion then run post-processing
  ;; Resources are owned by the with-raii in run-pass; do not destroy here.
  ;;===--------------------------------------------------------------------===;;
  (define (do-conversion module-op target patterns)
    (mlir-log-debug "Applying full conversion...")
    (let ((success (mlir-apply-full-conversion module-op target patterns)))
      (if (= success 1)
          (begin
            (mlir-log-debug "Erasing dead NoValue ops...")
            (erase-dead-novalue! module-op)
            (mlir-log-debug "Rewiring placeholder inputs...")
            (rewire-placeholder-inputs! module-op)
            (mlir-log-info "ONNX to HipSR Conversion (Scheme): Success"))
          (begin
            (mlir-log-error "ONNX to HipSR Conversion (Scheme): FAILED")
            (error (quote run-pass) "Dialect conversion failed")))))

  (define (run-pass module-op . args)
    (mlir-log-info "=== run-pass ENTERED ===")
    (mlir-log-info "Starting ONNX to HipSR Conversion (Scheme)")
    (let ((ctx (mlir-operation-get-context module-op)))
      (with-type-converter (type-converter)
        (hipsr-type-converter-add-device-memory-conversions! type-converter)
        (with-conversion-target (target ctx)
          (hipsr-configure-conversion-target! target ctx type-converter)
          (with-rewrite-pattern-set (patterns ctx)
            ;; Scheme DSL patterns (inline shape regions)
            (populate-cast-patterns       type-converter patterns ctx)
            (populate-scatter-nd-patterns type-converter patterns ctx)
            (populate-equal-patterns      type-converter patterns ctx)
            ;; C++ patterns (shape regions filled by PopulateShapeRegionPass)
            (mlir-populate-matmul-conversion-patterns    type-converter patterns ctx)
            (mlir-populate-expand-conversion-patterns    type-converter patterns ctx)
            (mlir-populate-min-conversion-patterns       type-converter patterns ctx)
            (mlir-populate-shape-conversion-patterns     type-converter patterns ctx)
            (mlir-populate-reshape-conversion-patterns   type-converter patterns ctx)
            (mlir-populate-unsqueeze-conversion-patterns type-converter patterns ctx)
            (mlir-populate-transpose-conversion-patterns type-converter patterns ctx)
            (mlir-populate-gather-conversion-patterns    type-converter patterns ctx)
            (mlir-populate-slice-conversion-patterns     type-converter patterns ctx)
            (mlir-populate-nonzero-conversion-patterns   type-converter patterns ctx)
            (mlir-populate-constant-conversion-patterns  type-converter patterns ctx)
            ;; Infrastructure patterns
            (populate-return-patterns type-converter patterns ctx)
            (mlir-populate-func-type-conversion-pattern patterns type-converter)
            (do-conversion module-op target patterns))))))

) ;; end library (passes onnx-to-hipsr)
