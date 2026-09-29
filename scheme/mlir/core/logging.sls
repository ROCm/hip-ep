#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core logging) — Scheme-level logging bound to the MLIR log system.
;;
;; Mirrors lib/Scheme/Bindings/Logging.cpp.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core logging)
  (export
    mlir-log-trace
    mlir-log-debug
    mlir-log-info
    mlir-log-warning
    mlir-log-error
    mlir-log-fatal)
  (import (chezscheme))

  (define mlir-log-trace   (foreign-procedure "mlir_log_trace"   (string) void))
  (define mlir-log-debug   (foreign-procedure "mlir_log_debug"   (string) void))
  (define mlir-log-info    (foreign-procedure "mlir_log_info"    (string) void))
  (define mlir-log-warning (foreign-procedure "mlir_log_warning" (string) void))
  (define mlir-log-error   (foreign-procedure "mlir_log_error"   (string) void))
  (define mlir-log-fatal   (foreign-procedure "mlir_log_fatal"   (string) void))

) ;; end library (mlir core logging)
