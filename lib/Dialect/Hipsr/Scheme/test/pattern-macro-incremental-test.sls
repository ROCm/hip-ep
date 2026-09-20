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

  ;; Test 5: func.func with block arguments - function that adds two i32s
  (define-conversion-pattern :debug-ast test-func-func
    :match ((%func = "func.func" () () : () -> !func-type))
    :rewrite %func :with
    ((%new-func = "func.func" ()
       :regions
         ((^entry ((%a : i32) (%b : i32))
            (%sum = "arith.addi" (%a %b) -> i32)
            ("func.return" (%sum) -> ())))
       :attrs [sym_name "add"] [function_type "(i32, i32) -> i32"]
       -> ())))
  
  ;; Helper to get field from AST list
  (define (ast-get ast key)
    (let loop ([rest ast])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  (define (run-tests)
    (test-begin "pattern-macro-incremental")
    (display "\n=== Testing clause splitting ===\n\n")

    ;; Basic AST structure tests
    (test-equal "debug mode returns list" #t (list? test-basic-ast))
    (test-equal "function name" 'test-basic-ast (ast-get test-basic-ast 'function-name))
    (test-equal "root-op-name extracted" "test.op" (ast-get test-basic-ast 'root-op-name))
    (test-equal "debug-ast flag" #t (ast-get test-basic-ast 'debug-ast?))

    ;; Match operations
    (test-equal "match ops captured" #t (list? (ast-get test-basic-ast 'match)))
    (test-equal "match ops not empty" #t (pair? (ast-get test-basic-ast 'match)))

    ;; Rewrite operations
    (test-equal "rewrite ops captured" #t (list? (ast-get test-basic-ast 'rewrite)))
    (test-equal "rewrite ops not empty" #t (pair? (ast-get test-basic-ast 'rewrite)))

    ;; Where clause
    (test-equal "where clause empty" (list) (ast-get test-basic-ast 'where))

    ;; Lambda mode test
    (test-equal "lambda mode returns procedure" #t (procedure? test-lambda-mode))

    ;; :where pattern tests
    (test-equal ":where pattern is list" #t (list? test-with-where))
    (test-equal ":where bindings captured" #t (list? (ast-get test-with-where 'where)))
    (test-equal ":where bindings not empty" #t (pair? (ast-get test-with-where 'where)))
    (test-equal ":where has 2 bindings" 2 (length (ast-get test-with-where 'where)))

    ;; Region and block tests
    (test-equal "region pattern is list" #t (list? test-with-region))
    (test-equal "has rewrite operations" #t (pair? (ast-get test-with-region 'rewrite)))

    ;; func.func test - block arguments and attrs
    (test-equal "func.func is list" #t (list? test-func-func))
    (test-equal "func.func has rewrite ops" #t (pair? (ast-get test-func-func 'rewrite)))

    (test-end)))
