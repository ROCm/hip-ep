#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Cast Pattern - Using define-conversion-pattern Macro
;;
;;===----------------------------------------------------------------------===;;

(library (patterns cast)
  (export populate-cast-patterns
          onnx-cast->hipsr)
  (import (rnrs (6))
          (mlir ffi)
          (mlir pattern-macro))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern - MLIR Operation Syntax
  ;;
  ;; Uses operation syntax: (%v = "op.name" (operands) ((attrs)) ...)
  ;; The macro recognizes "hipsr.placeholder" and "hipsr.cast" and translates
  ;; to specialized FFI calls.
  ;;
  ;; mlir-tensor-type-in-device-space: adds #hipsr.mem<device> to tensor types
  ;;===--------------------------------------------------------------------===;;

  (define-conversion-pattern onnx-cast->hipsr
    :match 
    (%output = "onnx.Cast" (%input) ((to = !to_type)) : (!input-type) -> !output-type)
    :rewrite %output :with
    (%ctx = (mlir-get-hipsr-context-arg op))
    (%placeholder = "hipsr.placeholder" (%ctx %input (mlir-tensor-type-in-device-space !output-type))
                    ((placeholder_type 0))
                    : ((mlir-tensor-type-in-device-space !input-type)) 
                    -> (mlir-tensor-type-in-device-space !output-type))
    (%cast = "hipsr.cast" (%ctx %input %placeholder (mlir-tensor-type-in-device-space !output-type))
             ((cast_attrs))
             : ((mlir-tensor-type-in-device-space !input-type) 
                (mlir-tensor-type-in-device-space !output-type)) 
             -> (mlir-tensor-type-in-device-space !output-type)))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter))

) ;; end library (patterns cast)
