#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; HipSR-dialect-specific FFI bindings.
;;
;; Mirrors: hip/Dialect/Hipsr/IR/
;;
;; Contains only bindings that reference HipSR types or attributes directly
;; (MemorySpaceAttr, PlaceholderType, HipsrDialect). Generic tensor type
;; queries that happen to be used for HipSR live in (mlir dialects tensor).
;;
;;===----------------------------------------------------------------------===;;

(library (mlir hipsr ir)
  (export
    ;; Type predicates and constructors
    mlir-type-is-device-tensor
    mlir-type-set-memory-space
    mlir-tensor-type-in-device-space

    ;; Placeholder op helpers
    mlir-placeholder-set-barrier-type

    ;; Conversion target configuration
    mlir-conversion-target-add-legal-hipsr
    mlir-conversion-target-mark-unknown-ops-nested-legal

    ;; Pattern population
    mlir-populate-cast-conversion-patterns)

  (import (chezscheme))

  ;;===--------------------------------------------------------------------===;;
  ;; Type predicates and constructors
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Returns 1 if type is a RankedTensorType with HipSR device memory space.
  ;;; @param type-ptr Type* as uptr
  ;;; @return 1 if device tensor, 0 otherwise
  (define mlir-type-is-device-tensor
    (foreign-procedure "mlir_type_is_device_tensor" (uptr) int))

  ;;; @brief Clone a RankedTensorType with the given HipSR MemorySpace attribute.
  ;;; @param type-ptr Type* as uptr
  ;;; @param space    HipSR MemorySpace enum value as int (1 = Device)
  ;;; @return New Type* with updated memory-space encoding as uptr
  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (uptr int) uptr))

  ;;; @brief Clone a tensor type replacing its encoding with HipSR device memory space.
  ;;; @param type-ptr Type* as uptr
  ;;; @return New Type* with device memory-space encoding as uptr
  (define mlir-tensor-type-in-device-space
    (foreign-procedure "mlir_tensor_type_in_device_space" (uptr) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Placeholder op helpers
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Set placeholder_type attribute on a hipsr.placeholder op to Barrier.
  ;;; @param op-ptr Operation* as uptr
  (define mlir-placeholder-set-barrier-type
    (foreign-procedure "mlir_placeholder_set_barrier_type" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Conversion target configuration
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Mark the entire HipsrDialect as legal in a ConversionTarget.
  ;;; @param target-ptr ConversionTarget* as uptr
  (define mlir-conversion-target-add-legal-hipsr
    (foreign-procedure "mlir_conversion_target_add_legal_hipsr" (uptr) void))

  ;;; @brief Mark unknown ops dynamically legal when nested inside a
  ;;;        hipsr.compute or hipsr.placeholder region.
  ;;; @param target-ptr ConversionTarget* as uptr
  (define mlir-conversion-target-mark-unknown-ops-nested-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_nested_legal" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern population
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Populate C++-defined conversion patterns for hipsr.cast.
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param patterns-ptr  RewritePatternSet* as uptr
  ;;; @param context-ptr   MLIRContext* as uptr
  (define mlir-populate-cast-conversion-patterns
    (foreign-procedure "mlir_populate_cast_conversion_patterns"
                       (uptr uptr uptr) void))

) ;; end library (mlir hipsr ir)
