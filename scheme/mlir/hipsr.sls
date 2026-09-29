#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; HipSR-specific MLIR helpers — pure Scheme, built on (mlir ir) + (mlir dialects conversion) primitives.
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
    hipsr-has-placeholder-ancestor?

    ;; Onnx/HipSR conversion-target convenience wrappers
    mlir-conversion-target-add-illegal-onnx
    mlir-conversion-target-add-legal-hipsr
    mlir-conversion-target-mark-unknown-ops-nested-legal

    ;; HipSR-specific type queries and op mutation (FFI bindings)
    mlir-type-is-device-tensor          ; 1 if RankedTensorType with device space
    mlir-tensor-type-in-host-space      ; clone type with host memory-space encoding
    mlir-get-hipsr-context-type         ; hipsr::ContextType from MLIRContext
    mlir-placeholder-set-barrier-type)  ; change placeholder_type attr to Barrier

  (import (rnrs (6))
          (only (chezscheme) foreign-procedure define-ftype ftype-ref make-ftype-pointer foreign-ref)
          (mlir ir)
          (mlir dialects conversion))

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
            #f)))
    ;; Source materialization: resolve unrealized casts between ranked tensor
    ;; types that differ only in shape specificity (e.g. tensor<?x32> vs tensor<?x?>)
    ;; by inserting tensor.cast. Needed when a conversion pattern infers a more
    ;; specific result type than what the type converter derives from the declared
    ;; ONNX result type. Without this, applyFullConversion fails with an unresolved
    ;; materialization error.
    (mlir-type-converter-add-tensor-widening-materialization type-converter))

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
    ;; tensor.cast is emitted by the tensor-widening source materialization to
    ;; bridge a more-specific inferred result type back to the declared-converted
    ;; type when a C++ conversion pattern produces a sharper type than expected.
    (mlir-conversion-target-add-legal-op target ctx "tensor.cast")
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


  ;;===--------------------------------------------------------------------===;;
  ;; Onnx/HipSR conversion-target convenience wrappers
  ;;===--------------------------------------------------------------------===;;

  (define mlir-conversion-target-add-illegal-onnx
    (foreign-procedure "mlir_conversion_target_add_illegal_onnx" (uptr) void))

  (define mlir-conversion-target-add-legal-hipsr
    (foreign-procedure "mlir_conversion_target_add_legal_hipsr" (uptr) void))

  (define mlir-conversion-target-mark-unknown-ops-nested-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_nested_legal" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; HipSR-specific FFI bindings
  ;;===--------------------------------------------------------------------===;;

  (define mlir-type-is-device-tensor
    (foreign-procedure "mlir_type_is_device_tensor" (uptr) int))

  (define mlir-tensor-type-in-host-space
    (foreign-procedure "mlir_tensor_type_in_host_space" (uptr) uptr))

  (define mlir-get-hipsr-context-type
    (foreign-procedure "mlir_get_hipsr_context_type" (uptr) uptr))

  (define mlir-placeholder-set-barrier-type
    (foreign-procedure "mlir_placeholder_set_barrier_type" (uptr) void))

) ;; end library (mlir hipsr)
