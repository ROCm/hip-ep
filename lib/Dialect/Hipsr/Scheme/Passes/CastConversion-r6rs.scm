;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; CastConversion Entry Point - R6RS Version
;;
;; This is the entry point that loads the R6RS library structure and
;; exposes run-pass for the C++ SchemeScriptPass to call.
;;
;;===----------------------------------------------------------------------===;;

;; Import the R6RS library
(import (mlir conversion cast))

;; Re-export run-pass for C++ to call
;; (run-pass is already defined in (mlir conversion cast))
