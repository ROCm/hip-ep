#!r6rs
;;===----------------------------------------------------------------------===;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; (mlir dialects func) — func dialect populate helpers
;;
;;===----------------------------------------------------------------------===;;
(library (mlir dialects func)
  (export
    mlir-populate-func-type-conversion-pattern
    mlir-populate-return-conversion-patterns)
  (import (chezscheme))

  (define mlir-populate-func-type-conversion-pattern
    (foreign-procedure "mlir_populate_func_type_conversion_pattern" (uptr uptr) void))

  (define mlir-populate-return-conversion-patterns
    (foreign-procedure "mlir_populate_return_conversion_patterns" (uptr uptr uptr) void))

) ;; end library (mlir dialects func)
