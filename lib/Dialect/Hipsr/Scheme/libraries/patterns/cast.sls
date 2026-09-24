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
  (import (except (rnrs (6)) =)
          (only (chezscheme) format)
          (mlir ffi)
          (mlir pattern-macro))

  ;;===--------------------------------------------------------------------===;;
  ;; Cast Pattern - MLIR-like Syntax with :where Clause
  ;; Operations come first, helper bindings defined in :where (like Haskell)
  ;;===--------------------------------------------------------------------===;;

  ;; Use the pattern DSL macro with explicit parameters
  (define-conversion-pattern (onnx-cast->hipsr op operands-ref rewriter type-converter)
    :match
        %output = onnx.Cast (%input)
    :then-let
        ([%ctx (mlir-get-hipsr-context-arg op)]
         [!output-type (mlir-value-get-type %output)]
         [!output-device (mlir-tensor-type-in-device-space !output-type)])
    :rewrite %output :with
        (%placeholder = hipsr.placeholder (%ctx %input !output-device)
                        -> !output-device)
        (%cast = hipsr.cast (%ctx %input %placeholder !output-device)
                 -> !output-device))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-log-info "Registering onnx.Cast pattern")
    (mlir-log-info (string-append "onnx-cast->hipsr is a procedure? " (if (procedure? onnx-cast->hipsr) "yes" "no")))
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter)
    (mlir-log-info "onnx.Cast pattern registered successfully"))

) ;; end library (patterns cast)
