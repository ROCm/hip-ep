#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; PrintPass - Simple MLIR operation printer
;;
;; Walks all operations and prints their names. Demonstrates basic Scheme
;; MLIR pass structure.
;;
;;===----------------------------------------------------------------------===;;

(library (printpass)
  (export run-pass)
  (import (rnrs (6))
          (only (chezscheme) format)  ; format is Chez-specific
          (mlir ffi))

  ;; Entry point called from C++
  (define (run-pass module-op . args)
    (mlir-log-info "Starting PrintPass (Scheme)")

    ;; Walk all operations and print their names
    (mlir-operation-walk module-op
      (lambda (op)
        (mlir-log-trace (format "Operation: ~s" (mlir-operation-name op)))))

    (mlir-log-info "Completed PrintPass (Scheme)"))

) ;; end library (printpass)
