#!r6rs
(import (rnrs (6))
        (mlir ddr))

(define-conversion-pattern test-pattern
  :match
      %output = onnx.Cast (%input)
  :then-let
      ([%ctx (mlir-get-hipsr-context-arg op)])
  :rewrite %output :with
      (%result = hipsr.placeholder (%ctx %input) -> !type)
  :debug-codegen)

(display "Pattern defined successfully\n")
