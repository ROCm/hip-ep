#!r6rs

(library (test pattern-macro-incremental-test)
  (export run-tests)
  (import (except (chezscheme) =)  ;; Exclude chezscheme's = to use mlir pattern-macro's =
          (test test-framework)
          (mlir pattern-macro))
  
  ;; Test 1: Basic pattern with :debug-ast - returns AST record
  (define-conversion-pattern :debug-ast test-basic-ast
    :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
    :rewrite %out :with
    ((%new = "new.op" (%in) -> !out-type)))

  ;; Test 2: Pattern without :debug-ast - returns lambda
  (define-conversion-pattern test-lambda-mode
    :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
    :rewrite %out :with
    ((%new = "new.op" (%in) -> !out-type)))

  ;; Test 3: Pattern with :where clause
  (define-conversion-pattern :debug-ast test-with-where
    :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
    :rewrite %out :with
    ((%new = "new.op" (%in) -> !out-type))
    :where ((%ctx (get-context))
            (%val (compute-value))))
  
  (define (run-tests)
    (test-begin "pattern-macro-incremental")
    (display "\n=== Testing clause splitting ===\n\n")
    
    ;; Basic AST structure tests
    (test-equal "debug mode returns ast-pattern" #t (ast-pattern? test-basic-ast))
    (test-equal "function name" 'test-basic-ast (ast-pattern-function-name test-basic-ast))
    (test-equal "root-op-name extracted" "test.op" (ast-pattern-root-op-name test-basic-ast))
    (test-equal "debug-ast flag" #t (ast-pattern-debug-ast? test-basic-ast))
    
    ;; Match operations
    (test-equal "match ops captured" #t (list? (ast-pattern-match test-basic-ast)))
    (test-equal "match ops not empty" #t (pair? (ast-pattern-match test-basic-ast)))
    
    ;; Rewrite operations
    (test-equal "rewrite ops captured" #t (list? (ast-pattern-rewrite test-basic-ast)))
    (test-equal "rewrite ops not empty" #t (pair? (ast-pattern-rewrite test-basic-ast)))
    
    ;; Where clause  
    (test-equal "where clause empty" (list) (ast-pattern-where test-basic-ast))
    
    ;; Lambda mode test
    (test-equal "lambda mode returns procedure" #t (procedure? test-lambda-mode))
    
    ;; :where pattern tests
    (test-equal ":where pattern is ast-pattern" #t (ast-pattern? test-with-where))
    (test-equal ":where bindings captured" #t (list? (ast-pattern-where test-with-where)))
    (test-equal ":where bindings not empty" #t (pair? (ast-pattern-where test-with-where)))
    (test-equal ":where has 2 bindings" 2 (length (ast-pattern-where test-with-where)))
    
    (test-end)))
