#!r6rs
(import (rnrs (6))
        (patterns cast)
        (mlir pattern-macro))

;; Print the generated code to see what is wrong
(define-conversion-pattern test-cast
  :match
      %output = onnx.Cast (%input)
  :then-let
      ([%ctx (mlir-get-hipsr-context-arg op)])
  :rewrite %output :with
      (%placeholder = hipsr.placeholder (%ctx %input))
  :debug-codegen)
