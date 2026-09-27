#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; HipSR-specific MLIR helpers — pure Scheme, built on (mlir ffi) primitives.
;;
;; This library encapsulates all knowledge of the HipSR and ONNX dialects:
;; memory spaces, context conventions, conversion target configuration, and
;; type conversion rules. Nothing here is dialect-agnostic.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir hipsr)
  (export
    ;; Memory space
    hipsr-device-memory-space
    mlir-tensor-type-in-device-space!   ; creates new MLIR type in context

    ;; Context convention
    mlir-get-hipsr-context-arg           ; pure read

    ;; Type converter configuration
    hipsr-type-converter-add-device-memory-conversions!  ; mutates type-converter

    ;; Conversion target configuration
    hipsr-configure-conversion-target!   ; mutates target

    ;; Op ancestry predicates (pure)
    hipsr-has-compute-ancestor?
    hipsr-has-placeholder-ancestor?)

  (import (rnrs (6))
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; Memory Space
  ;;===--------------------------------------------------------------------===;;

  ;; MemorySpace::Device = 1  (from HipsrEnums.td: Hipsr_Device I32EnumAttrCase 1)
  (define hipsr-device-memory-space 1)

  (define (mlir-tensor-type-in-device-space! type)
    (mlir-type-set-memory-space type hipsr-device-memory-space))

  ;;===--------------------------------------------------------------------===;;
  ;; Context Convention
  ;;
  ;; By HipSR convention, the first block argument of every inference function
  ;; is the !hipsr.context value.
  ;;===--------------------------------------------------------------------===;;

  (define (mlir-get-hipsr-context-arg op)
    (mlir-operation-get-block-argument op 0))

  ;;===--------------------------------------------------------------------===;;
  ;; Type Converter Configuration
  ;;
  ;; Mirrors TypeConverter setup in OnnxToHipsr.cpp:
  ;;   - Identity conversion for all types.
  ;;   - Ranked tensors without an encoding (unplaced tensors) get device space.
  ;;   - Rank-0 tensors and tensors that already carry an encoding are left alone.
  ;;===--------------------------------------------------------------------===;;

  (define (hipsr-type-converter-add-device-memory-conversions! type-converter)
    ;; Identity: every type is legal as-is (lowest priority, tried last)
    (mlir-type-converter-add-conversion type-converter (lambda (t) t))
    ;; Device placement: unencoded ranked tensors of rank > 0 go to device space
    (mlir-type-converter-add-conversion type-converter
      (lambda (type)
        (if (and (= 1 (mlir-type-is-ranked-tensor type))
                 (> (mlir-type-get-rank type) 0)
                 (= 0 (mlir-type-get-encoding type)))
            (mlir-tensor-type-in-device-space! type)
            #f))))

  ;;===--------------------------------------------------------------------===;;
  ;; Op Ancestry Predicates
  ;;===--------------------------------------------------------------------===;;

  (define (has-ancestor-named? op name)
    (let loop ((parent (mlir-operation-get-parent op)))
      (cond
        ((= 0 parent) #f)
        ((string=? (mlir-operation-name parent) name) #t)
        (else (loop (mlir-operation-get-parent parent))))))

  (define (hipsr-has-compute-ancestor? op)
    (has-ancestor-named? op "hipsr.compute"))

  (define (hipsr-has-placeholder-ancestor? op)
    (has-ancestor-named? op "hipsr.placeholder"))

  ;;===--------------------------------------------------------------------===;;
  ;; Conversion Target Configuration
  ;;
  ;; Mirrors ConversionTarget setup in OnnxToHipsr.cpp:
  ;;   - onnx dialect: illegal (except onnx.NoValue — consumer drops it first)
  ;;   - hipsr dialect: legal
  ;;   - builtin.module, arith.constant: legal
  ;;   - func.func: legal when signature is converted
  ;;   - func.return: legal when operand types are converted
  ;;   - unknown ops: legal if nested inside hipsr.compute or hipsr.placeholder
  ;;===--------------------------------------------------------------------===;;

  (define (hipsr-configure-conversion-target! target ctx type-converter)
    (mlir-conversion-target-add-illegal-dialect target "onnx")
    (mlir-conversion-target-add-legal-op target ctx "onnx.NoValue")
    (mlir-conversion-target-add-legal-dialect target "hipsr")
    (mlir-conversion-target-add-legal-op target ctx "builtin.module")
    (mlir-conversion-target-add-legal-op target ctx "arith.constant")
    (mlir-conversion-target-add-dynamically-legal-op target ctx "func.func"
      (lambda (op)
        (= 1 (mlir-type-converter-is-signature-legal type-converter op))))
    (mlir-conversion-target-add-dynamically-legal-op target ctx "func.return"
      (lambda (op)
        (= 1 (mlir-type-converter-is-legal type-converter op))))
    (mlir-conversion-target-mark-unknown-ops-dynamically-legal target
      (lambda (op)
        (or (hipsr-has-compute-ancestor? op)
            (hipsr-has-placeholder-ancestor? op)))))

) ;; end library (mlir hipsr)
