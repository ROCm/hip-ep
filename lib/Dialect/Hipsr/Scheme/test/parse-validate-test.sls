#!r6rs
;;===----------------------------------------------------------------------===;;
;; Parse & Validate Test - Test all 4 phases of pattern macro
;;===----------------------------------------------------------------------===;;

(library (test parse-validate-test)
  (export run-tests)
  (import (except (chezscheme) =)  ;; Exclude chezscheme's = to use mlir pattern-macro's =
          (test test-framework)
          (mlir pattern-macro))
  
  ;; Test 1: Basic pattern with :debug-parse - returns AST record
  (define-conversion-pattern :debug-parse test-basic-ast
    :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
    :rewrite %out :with
    ((%new = "new.op" (%in) -> !out-type)))

  ;; Test 2: Pattern without :debug-parse - returns lambda
  (define-conversion-pattern test-lambda-mode
    :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
    :rewrite %out :with
    ((%new = "new.op" (%in) -> !out-type)))

  ;; Test 3: Pattern with :then-let clause
  (define-conversion-pattern :debug-parse test-with-then-let
    :match ((%out = "test.op" (%in) () : (!in-type) -> !out-type))
    :then-let ((%ctx (get-context))
               (%val (compute-value)))
    :rewrite %out :with
    ((%new = "new.op" (%in) -> !out-type)))

  ;; Test 4: Pattern with region containing blocks
  (define-conversion-pattern :debug-parse test-with-region
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
  (define-conversion-pattern :debug-parse test-func-func
    :match ((%func = "func.func" () () : () -> !func-type))
    :rewrite %func :with
    ((%new-func = "func.func" ()
       :regions
         ((^entry ((%a : i32) (%b : i32))
            (%sum = "arith.addi" (%a %b) -> i32)
            ("func.return" (%sum) -> ())))
       :attrs [sym_name "add"] [function_type "(i32, i32) -> i32"]
       -> ())))

  ;; Test 6: Pattern with per-operation :where guard
  (define-conversion-pattern :debug-parse test-with-where-guard
    :match ((%a = "onnx.Conv" (%x %w)
               :where (let ([$ks (mlir-operation-get-attribute %a "kernel_shape")])
                        (and $ks (is-1x1-kernel? $ks)))))
    :rewrite %a :with
    ((%out = "hipsr.matmul" (%x %w))))

  ;; Test 7: Pattern with &optional operands
  (define-conversion-pattern :debug-parse test-with-optional
    :match ((%a = "test.op" (%x (&optional %y %z))))
    :rewrite %a :with
    ((%out = "new.op" (%x))))

  ;; Test 8: Pattern with &variadic operands
  (define-conversion-pattern :debug-parse test-with-variadic
    :match ((%a = "test.concat" (%x (&variadic %rest))))
    :rewrite %a :with
    ((%out = "new.concat" (%x))))

  ;; Test 9: Pattern with mixed optional and variadic
  (define-conversion-pattern :debug-parse test-mixed-operands
    :match ((%a = "test.op" (%x (&optional %y) %z (&variadic %rest))))
    :rewrite %a :with
    ((%out = "new.op" (%x %z))))

  ;; Test 10: Pattern with :where guard and :then-let
  (define-conversion-pattern :debug-parse test-where-and-then-let
    :match ((%a = "onnx.Conv" (%x %w)
               :where (mlir-operation-get-attribute %a "kernel_shape")))
    :then-let ((%ctx (mlir-get-context %a))
               (%device (get-device-type %x)))
    :rewrite %a :with
    ((%out = "hipsr.conv" (%ctx %x %w))))

  ;; Helper to get field from AST list
  (define (ast-get ast key)
    (let loop ([rest ast])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  (define (run-tests)
    (test-begin "parse-validate")
    (display "\n=== Testing Parse & Validate Phases ===\n\n")

    ;; Basic AST structure tests
    (test-equal "debug mode returns list" #t (list? test-basic-ast))
    (test-equal "function name" 'test-basic-ast (ast-get test-basic-ast 'function-name))
    (test-equal "root-op-name extracted" "test.op" (ast-get test-basic-ast 'root-op-name))
    (test-equal "debug-parse flag" #t (ast-get test-basic-ast 'debug-parse?))

    ;; Match operations
    (test-equal "match ops captured" #t (list? (ast-get test-basic-ast 'match)))
    (test-equal "match ops not empty" #t (pair? (ast-get test-basic-ast 'match)))

    ;; Rewrite operations
    (test-equal "rewrite ops captured" #t (list? (ast-get test-basic-ast 'rewrite)))
    (test-equal "rewrite ops not empty" #t (pair? (ast-get test-basic-ast 'rewrite)))

    ;; :then-let clause
    (test-equal "then-let clause empty when absent" (list) (ast-get test-basic-ast 'where))

    ;; Lambda mode test
    (test-equal "lambda mode returns procedure" #t (procedure? test-lambda-mode))

    ;; :then-let pattern tests
    (test-equal ":then-let pattern is list" #t (list? test-with-then-let))
    (test-equal ":then-let bindings captured" #t (list? (ast-get test-with-then-let 'where)))
    (test-equal ":then-let bindings not empty" #t (pair? (ast-get test-with-then-let 'where)))
    (test-equal ":then-let has 2 bindings" 2 (length (ast-get test-with-then-let 'where)))

    ;; Region and block tests
    (test-equal "region pattern is list" #t (list? test-with-region))
    (test-equal "has rewrite operations" #t (pair? (ast-get test-with-region 'rewrite)))

    ;; func.func test - block arguments and attrs
    (test-equal "func.func is list" #t (list? test-func-func))
    (test-equal "func.func has rewrite ops" #t (pair? (ast-get test-func-func 'rewrite)))

    ;; :where guard tests (Test 6)
    (test-equal ":where guard pattern parses" #t (list? test-with-where-guard))
    (test-equal ":where guard has match ops" #t (pair? (ast-get test-with-where-guard 'match)))
    ;; TODO: Add test to check where-expr field in match operation AST

    ;; &optional operands tests (Test 7)
    (test-equal "&optional pattern parses" #t (list? test-with-optional))
    (test-equal "&optional has match ops" #t (pair? (ast-get test-with-optional 'match)))
    ;; TODO: Add test to verify operands field contains (&optional %y %z) group

    ;; &variadic operands tests (Test 8)
    (test-equal "&variadic pattern parses" #t (list? test-with-variadic))
    (test-equal "&variadic has match ops" #t (pair? (ast-get test-with-variadic 'match)))
    ;; TODO: Add test to verify operands field contains (&variadic %rest) group

    ;; Mixed operands tests (Test 9)
    (test-equal "mixed operands pattern parses" #t (list? test-mixed-operands))
    (test-equal "mixed has match ops" #t (pair? (ast-get test-mixed-operands 'match)))

    ;; Combined :where and :then-let tests (Test 10)
    (test-equal ":where + :then-let pattern parses" #t (list? test-where-and-then-let))
    (test-equal ":where + :then-let has match" #t (pair? (ast-get test-where-and-then-let 'match)))
    (test-equal ":where + :then-let has bindings" #t (pair? (ast-get test-where-and-then-let 'where)))
    (test-equal ":where + :then-let has 2 bindings" 2 (length (ast-get test-where-and-then-let 'where)))

    (test-end)))

