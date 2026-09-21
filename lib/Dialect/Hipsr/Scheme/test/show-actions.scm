#!/usr/bin/env scheme-script
(import (rnrs)
        (only (chezscheme) pretty-print format)
        (mlir pattern-macro))

;; Simple single-operation pattern
(define-conversion-pattern :debug-ast test-simple
  :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
  :rewrite %out :with
  ((%new = "new.op" (%in) -> !out-type)))

(display "=== Simple Pattern (single operation) ===\n")
(display "match-actions:\n")
(let ([actions (memq 'match-actions test-simple)])
  (when actions
    (for-each (lambda (action)
                (format #t "  ~a\n" action))
              (cadr actions))))

(newline)
(newline)

;; Two-operation pattern with dependency
(define-conversion-pattern :debug-ast test-two-ops
  :match ((%a = "op1" (%x) () : (!t1) -> !t2)
          (%b = "op2" (%a) () : (!t2) -> !t3))
  :rewrite %b :with
  ((%out = "op3" (%a) -> !t3)))

(display "=== Two Operations (%b uses %a) ===\n")
(display "match-actions:\n")
(let ([actions (memq 'match-actions test-two-ops)])
  (when actions
    (for-each (lambda (action)
                (format #t "  ~a\n" action))
              (cadr actions))))
