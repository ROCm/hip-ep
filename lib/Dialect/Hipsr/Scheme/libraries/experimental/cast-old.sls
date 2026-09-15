#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Cast Conversion Patterns - ONNX to HipSR
;;
;; Equivalent to lib/Conversion/OnnxToHipsr/CastConversion.cpp
;;
;;===----------------------------------------------------------------------===;;

(library (patterns cast)
  (export populate-cast-patterns
          onnx-cast->hipsr)
  (import (rnrs (6))
          (mlir ffi)
          (pattern-dsl)
          (build-dsl))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern
  ;;===--------------------------------------------------------------------===;;

  ;; onnx.Cast -> hipsr.Cast
  ;;
  ;; The ONNX `saturate` attribute (clamp-vs-wrap when casting to float8) is
  ;; not modeled on hipsr.cast yet, so it is currently ignored.
  (define-conversion-pattern onnx-cast->hipsr
    :if-match
      %cast = "onnx.Cast" (%input) (:to $dtype) :type (!input-type) -> !output-type
    :rewrite %cast
      (mlir-build
        %placeholder = "hipsr.Placeholder" (%input) :type (!input-type) -> !input-type
        %result = "hipsr.Cast" (%input %placeholder) :type (!input-type) -> !output-type))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr))

) ;; end library (patterns cast)
