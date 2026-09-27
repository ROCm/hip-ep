#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Func dialect FFI — populate functions for func.func and func.return patterns.
;;
;; Mirrors: mlir/Dialect/Func/IR/FuncOps.h
;;
;;===----------------------------------------------------------------------===;;

(library (mlir dialects func)
  (export
    mlir-populate-func-type-conversion-pattern
    mlir-populate-return-conversion-patterns)

  (import (chezscheme))

  ;;; @brief Add Return operation conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param context-ptr MLIRContext* as uptr
  ;;; @note Populates patterns for converting func.return operations
  (define mlir-populate-return-conversion-patterns
    (foreign-procedure "mlir_populate_return_conversion_patterns"
                       (uptr uptr uptr) void))

  ;;; @brief Add function type conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @note Populates patterns for converting function signatures
  (define mlir-populate-func-type-conversion-pattern
    (foreign-procedure "mlir_populate_func_type_conversion_pattern"
                       (uptr uptr) void))

) ;; end library (mlir dialects func)
