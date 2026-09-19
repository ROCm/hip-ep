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

  ;; Test 4: Pattern with region containing blocks
  (define-conversion-pattern :debug-ast test-with-region
    :match ((%out = "scf.if" (%cond) () : (!cond-type) -> !result-type))
    :rewrite %out :with
    ((%r = "scf.if" (%cond)
       :regions
         ((^then ()
            (%t = "arith.constant" () :attrs [value 1] -> i32))
          (^else ()
            (%f = "arith.constant" () :attrs [value 0] -> i32)))
       -> i32)))
  
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

    ;; Region and block tests
    ;; Since expansion-time records aren't exported, we just test that it parses
    (test-equal "region pattern is ast-pattern" #t (ast-pattern? test-with-region))
    ;; The rewrite list contains raw s-expressions (not expansion-time records)
    (test-equal "has rewrite operations" #t (pair? (ast-pattern-rewrite test-with-region)))

    (test-end)))
