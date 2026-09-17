#!r6rs
;;===----------------------------------------------------------------------===;;
;; Unit Tests for Pattern DSL
;;===----------------------------------------------------------------------===;;

(library (test pattern-dsl-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework)
          (mlir pattern-dsl))

  (define (run-tests)
    (test-begin "pattern-dsl")

    ;;===------------------------------------------------------------------===;;
    ;; Test Constraint Builders
    ;;===------------------------------------------------------------------===;;

    (test-assert "has-n-operands returns a procedure"
      (procedure? (has-n-operands 1)))

    (test-assert "has-n-results returns a procedure"
      (procedure? (has-n-results 2)))

    (test-assert "result-0-is-ranked-tensor returns a procedure"
      (procedure? (result-0-is-ranked-tensor)))

    ;;===------------------------------------------------------------------===;;
    ;; Test Pattern Definition
    ;;===------------------------------------------------------------------===;;

    (test-assert "define-conversion-pattern returns a procedure"
      (let ([pattern (define-conversion-pattern "test.op"
                       (list (has-n-operands 1))
                       (lambda (op operands-ref rewriter type-converter)
                         #t))])
        (procedure? pattern)))

    (test-assert "pattern accepts 4 parameters"
      (let ([pattern (define-conversion-pattern "test.op"
                       '()
                       (lambda (op operands-ref rewriter type-converter)
                         #t))])
        (procedure? pattern)))

    ;;===------------------------------------------------------------------===;;
    ;; Test Pattern Application
    ;;===------------------------------------------------------------------===;;

    (test-equal "apply-patterns with empty list returns #f"
      (apply-patterns '() 0 0 0 0)
      #f)

    (test-equal "apply-patterns with false pattern returns #f"
      (let ([pattern (lambda (op operands-ref rewriter type-converter) #f)])
        (apply-patterns (list pattern) 0 0 0 0))
      #f)

    (test-equal "apply-patterns with true pattern returns #t"
      (let ([pattern (lambda (op operands-ref rewriter type-converter) #t)])
        (apply-patterns (list pattern) 0 0 0 0))
      #t)

    (test-equal "apply-pattern with matching pattern returns #t"
      (let ([pattern (lambda (op operands-ref rewriter type-converter) #t)])
        (apply-pattern pattern 0 0 0 0))
      #t)

    (test-equal "apply-pattern with non-matching pattern returns #f"
      (let ([pattern (lambda (op operands-ref rewriter type-converter) #f)])
        (apply-pattern pattern 0 0 0 0))
      #f)

    (test-end))

) ;; end library
