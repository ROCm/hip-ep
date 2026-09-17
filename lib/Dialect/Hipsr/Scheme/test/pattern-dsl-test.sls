#!r6rs
;;===----------------------------------------------------------------------===;;
;; Unit Tests for define-conversion-pattern
;;===----------------------------------------------------------------------===;;

(library (test pattern-dsl-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework)
          (mlir pattern-dsl))

  (define (run-tests)
    (test-begin "define-conversion-pattern")

    ;; Test: Pattern returns a procedure
    (test-assert "returns a procedure"
      (procedure? (define-conversion-pattern "test.op"
                    '()
                    (lambda (op operands-ref rewriter type-converter) #t))))

    ;; Test: Pattern accepts 4 parameters (op, operands-ref, rewriter, type-converter)
    (test-equal "accepts 4 parameters and calls rewrite action"
      (let ([pattern (define-conversion-pattern "test.op"
                       '()
                       (lambda (op operands-ref rewriter type-converter)
                         (+ op operands-ref rewriter type-converter)))])
        (pattern 1 2 3 4))
      10)  ; 1+2+3+4 = 10

    ;; Test: Pattern returns #f when constraints fail
    (test-equal "returns #f when constraint fails"
      (let ([pattern (define-conversion-pattern "test.op"
                       (list (lambda (op operands-ref rewriter type-converter) #f))
                       (lambda (op operands-ref rewriter type-converter) #t))])
        (pattern 0 0 0 0))
      #f)

    ;; Test: Pattern calls rewrite action when all constraints pass
    (test-equal "calls rewrite action when all constraints pass"
      (let ([pattern (define-conversion-pattern "test.op"
                       (list (lambda (op operands-ref rewriter type-converter) #t)
                             (lambda (op operands-ref rewriter type-converter) #t))
                       (lambda (op operands-ref rewriter type-converter) 'success))])
        (pattern 0 0 0 0))
      'success)

    ;; Test: Pattern returns result of rewrite action
    (test-equal "returns result of rewrite action"
      (let ([pattern (define-conversion-pattern "test.op"
                       '()
                       (lambda (op operands-ref rewriter type-converter)
                         (list op operands-ref rewriter type-converter)))])
        (pattern 'a 'b 'c 'd))
      '(a b c d))

    (test-end))

) ;; end library
