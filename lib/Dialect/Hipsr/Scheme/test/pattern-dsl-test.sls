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

    ;; Test constraint functions
    (test-assert "has-n-operands returns a function"
      (procedure? (has-n-operands 1)))

    (test-assert "has-n-results returns a function"
      (procedure? (has-n-results 1)))

    (test-assert "result-0-is-ranked-tensor returns a function"
      (procedure? (result-0-is-ranked-tensor)))

    ;; Test apply-patterns with empty list
    (test-equal "apply-patterns with empty list returns #f"
      (apply-patterns '() 0 0 0 0)  ; Dummy args
      #f)

    (test-end))

) ;; end library
