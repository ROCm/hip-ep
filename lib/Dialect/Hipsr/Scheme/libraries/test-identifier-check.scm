(import (chezscheme) (mlir pattern-macro))

(define-syntax test-id-match
  (lambda (x)
    (syntax-case x ()
      [(_ eq-id)
       (begin
         (display "Checking if eq-id matches our = keyword:\n")
         (display "  free-identifier=? result: ")
         (display (free-identifier=? #'eq-id #'=))
         (newline)
         #'(void))])))

;; Test: does the = we write match the exported =?
(test-id-match =)
