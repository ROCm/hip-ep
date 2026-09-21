#!/usr/bin/env scheme-script
;; Test Case 4: unbound result variable used as operand

(import (rnrs)
        (mlir pattern-macro))

;;Simple pattern: %a produces result, %b uses %a as operand
(define-conversion-pattern test-case4
  :match
  ((%a = "op1" (%input) () : (!t1) -> !t2)
   (%b = "op2" (%a) () : (!t2) -> !t3))    ;; %a is result (is-result? = #t), initially unbound
  :rewrite %b :with
  ((%output = "op3" (%a) () -> !t3))
  :debug-ast)

(display "Test Case 4 result:\n")
(display test-case4)
(newline)
