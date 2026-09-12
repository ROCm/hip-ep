#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; CastConversion - R6RS Module Re-export
;;
;; Re-exports run-pass from (mlir conversion cast) for C++ to import.
;;
;;===----------------------------------------------------------------------===;;

(library (cast-conversion)
  (export run-pass)
  (import (mlir conversion cast))

  ;; run-pass is directly re-exported from (mlir conversion cast)
  ;; No additional wrapper needed
)
