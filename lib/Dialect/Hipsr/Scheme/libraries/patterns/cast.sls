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
  ;;
  ;; Operations come first, helper bindings defined in :where (like Haskell)
  ;;===--------------------------------------------------------------------===;;

  ;; Implement full conversion manually
  (define (onnx-cast->hipsr op operands-ref rewriter type-converter)
    (let* ([%input (mlir-operation-get-operand op 0)]
           [%output (mlir-operation-get-result op 0)]
           [%ctx (mlir-get-hipsr-context-arg op)]
           [!input-type (mlir-value-get-type %input)]
           [!output-type (mlir-value-get-type %output)]
           [!input-device (mlir-tensor-type-in-device-space !input-type)]
           [!output-device (mlir-tensor-type-in-device-space !output-type)])
      (display "Creating hipsr.placeholder...\n")
      (let ([%placeholder (mlir-create-generic-op "hipsr.placeholder"
                            (list %ctx %input !output-device)
                            (list !output-device))])
        (display "Created placeholder\n")
        (mlir-operation-set-attr %placeholder "placeholder_type" 0)
        (let ([%placeholder-result (mlir-operation-get-result-value-from-op %placeholder 0)])
          (display "Creating hipsr.cast...\n")
          (let ([%cast (mlir-create-generic-op "hipsr.cast"
                         (list %ctx %input %placeholder-result !output-device)
                         (list !output-device))])
            (display "Created cast\n")
            (let ([%cast-result (mlir-operation-get-result-value-from-op %cast 0)])
              (display "Replacing operation...\n")
              (mlir-replace-op op %cast-result)
              (display "SUCCESS! Replaced onnx.Cast with hipsr.cast\n")
              #t))))))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Population
  ;;===--------------------------------------------------------------------===;;

  (define (populate-cast-patterns type-converter patterns ctx)
    (mlir-log-info "Registering onnx.Cast pattern")
    (mlir-register-conversion-pattern patterns "onnx.Cast" onnx-cast->hipsr type-converter)
    (mlir-log-info "onnx.Cast pattern registered successfully"))

) ;; end library (patterns cast)
