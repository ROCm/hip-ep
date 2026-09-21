#!/usr/bin/env scheme-script
;; Test Case 4: unbound result variable used as operand

(import (rnrs)
        (only (chezscheme) pretty-print)
        (mlir pattern-macro))

;; Pattern where %b uses %a (result from op1) as operand
(define-conversion-pattern :debug-ast test-case4
  :match ((%a = "op1" (%input) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%a) () -> !t3)))

(display "=== Test Case 4: Result Variable as Operand ===\n\n")
(display "Pattern structure:\n")
(display "  %a = op1(%input) - produces result %a\n")
(display "  %b = op2(%a)     - uses %a as operand (Case 4)\n\n")

(display "Generated match-actions:\n")
(let ([actions (memq 'match-actions test-case4)])
  (if actions
      (pretty-print (cadr actions))
      (display "No match-actions found\n")))
