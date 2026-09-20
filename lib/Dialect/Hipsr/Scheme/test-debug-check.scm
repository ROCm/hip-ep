#!r6rs
(import (except (chezscheme) =) (mlir pattern-macro))
(define-conversion-pattern :debug-ast test-check
  :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
  :rewrite %out :with ((%new = "new.op" (%in) -> !out-type)))
(display test-check)
(newline)
