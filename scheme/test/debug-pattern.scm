#!r6rs
(import (rnrs (6))
        (mlir ddr))

;; Simple pattern to see generated code
(define-conversion-pattern test-pattern
  :match
      %output = onnx.Cast (%input)
  :rewrite %output :with
      (%result = hipsr.placeholder (%input))
  :debug-codegen)
