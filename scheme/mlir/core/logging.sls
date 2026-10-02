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
;; All functions take a single string message and return void.
;; The HIPDNN_EP_LOG_LEVEL environment variable controls which levels
;; are emitted at runtime; higher-severity levels are always emitted.
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

  ;; Emit a TRACE-level log message (most verbose; off by default).
  ;; msg: message string
  (define mlir-log-trace   (foreign-procedure "mlir_log_trace"   (string) void))

  ;; Emit a DEBUG-level log message (verbose developer information).
  ;; msg: message string
  (define mlir-log-debug   (foreign-procedure "mlir_log_debug"   (string) void))

  ;; Emit an INFO-level log message (normal operational events).
  ;; msg: message string
  (define mlir-log-info    (foreign-procedure "mlir_log_info"    (string) void))

  ;; Emit a WARNING-level log message (unexpected but recoverable condition).
  ;; msg: message string
  (define mlir-log-warning (foreign-procedure "mlir_log_warning" (string) void))

  ;; Emit an ERROR-level log message (non-fatal error).
  ;; msg: message string
  (define mlir-log-error   (foreign-procedure "mlir_log_error"   (string) void))

  ;; Emit a FATAL-level log message (unrecoverable; may abort the process).
  ;; msg: message string
  (define mlir-log-fatal   (foreign-procedure "mlir_log_fatal"   (string) void))

) ;; end library (mlir core logging)
