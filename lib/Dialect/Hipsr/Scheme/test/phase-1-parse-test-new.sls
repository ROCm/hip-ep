#!r6rs
;;===----------------------------------------------------------------------===;;
;; Phase 1: Parse Tests
;;===----------------------------------------------------------------------===;;
;;
;; Tests the parsing phase using :debug-parse flag.
;; Patterns are generated from test-pattern-bodies.scm automatically.
;;
;;===----------------------------------------------------------------------===;;

(library (test phase-1-parse-test-new)
  (export run-tests)
  (import (rnrs)
          (test test-framework)
          (test test-pattern-generator)
          (mlir pattern-ast))

  ;; Generate ALL test patterns with :debug-parse flag and "-parse" suffix
  ;; This creates: basic-parse, required-parse, optional-parse, variadic-parse,
  ;;               mixed-parse, where-parse, then-let-parse, two-ops-parse,
  ;;               reuse-parse, combined-parse, region-parse
  (generate-test-patterns :debug-parse "-parse")

  ;; Helper: Extract field from AST (pattern is a list when :debug-parse is used)
  (define (ast-get ast key)
    (let loop ([rest ast])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  (define (run-tests)
    (test-begin "phase-1-parse")

    ;;=======================================================================
    ;; Pattern 1: Basic single operation
    ;;=======================================================================
    (test-equal "basic: returns AST list" #t (list? basic-parse))
    (test-equal "basic: has function-name" 'basic-parse
      (ast-get basic-parse 'function-name))
    (test-equal "basic: has root-var" #t
      (identifier? (ast-get basic-parse 'root-var)))

    ;;=======================================================================
    ;; Pattern 2: Required operands
    ;;=======================================================================
    (test-equal "required: returns AST list" #t (list? required-parse))
    (let* ([match-list (ast-get required-parse 'match)]
           [first-op (car match-list)]
           [operands (ast-match-expand-operands first-op)])
      (test-equal "required: 2 operands" 2 (length operands))
      (test-equal "required: first is required" 'required
        (ast-operand-kind (car operands)))
      (test-equal "required: second is required" 'required
        (ast-operand-kind (cadr operands))))

    ;;=======================================================================
    ;; Pattern 3: Optional operands (tests flattening)
    ;;=======================================================================
    (test-equal "optional: returns AST list" #t (list? optional-parse))
    (let* ([match-list (ast-get optional-parse 'match)]
           [first-op (car match-list)]
           [operands (ast-match-expand-operands first-op)])
      (test-equal "optional: flattened to 3 operands" 3 (length operands))
      (test-equal "optional: first is required" 'required
        (ast-operand-kind (car operands)))
      (test-equal "optional: second is optional" 'optional
        (ast-operand-kind (cadr operands)))
      (test-equal "optional: third is optional" 'optional
        (ast-operand-kind (caddr operands))))

    ;;=======================================================================
    ;; Pattern 4: Variadic operands
    ;;=======================================================================
    (test-equal "variadic: returns AST list" #t (list? variadic-parse))
    (let* ([match-list (ast-get variadic-parse 'match)]
           [first-op (car match-list)]
           [operands (ast-match-expand-operands first-op)])
      (test-equal "variadic: 2 operands" 2 (length operands))
      (test-equal "variadic: first is required" 'required
        (ast-operand-kind (car operands)))
      (test-equal "variadic: second is variadic" 'variadic
        (ast-operand-kind (cadr operands))))

    ;;=======================================================================
    ;; Pattern 5: Mixed operands (required + optional + variadic)
    ;;=======================================================================
    (test-equal "mixed: returns AST list" #t (list? mixed-parse))
    (let* ([match-list (ast-get mixed-parse 'match)]
           [first-op (car match-list)]
           [operands (ast-match-expand-operands first-op)])
      (test-equal "mixed: 4 operands total" 4 (length operands))
      (test-equal "mixed: [0]=required" 'required
        (ast-operand-kind (list-ref operands 0)))
      (test-equal "mixed: [1]=optional" 'optional
        (ast-operand-kind (list-ref operands 1)))
      (test-equal "mixed: [2]=required" 'required
        (ast-operand-kind (list-ref operands 2)))
      (test-equal "mixed: [3]=variadic" 'variadic
        (ast-operand-kind (list-ref operands 3))))

    ;;=======================================================================
    ;; Pattern 6: :where guard
    ;;=======================================================================
    (test-equal "where: returns AST list" #t (list? where-parse))
    (let* ([match-list (ast-get where-parse 'match)]
           [first-op (car match-list)]
           [guard-expr (ast-match-expand-where-expr first-op)])
      (test-equal "where: guard is syntax object" #t
        (identifier? guard-expr)))

    ;;=======================================================================
    ;; Pattern 7: :then-let bindings
    ;;=======================================================================
    (test-equal "then-let: returns AST list" #t (list? then-let-parse))
    (let ([bindings (ast-get then-let-parse 'where)])
      (test-equal "then-let: has 2 bindings" 2 (length bindings)))

    ;;=======================================================================
    ;; Pattern 8: Two operations (DAG)
    ;;=======================================================================
    (test-equal "two-ops: returns AST list" #t (list? two-ops-parse))
    (let ([match-list (ast-get two-ops-parse 'match)])
      (test-equal "two-ops: has 2 match operations" 2 (length match-list)))

    ;;=======================================================================
    ;; Pattern 9: Variable reuse
    ;;=======================================================================
    (test-equal "reuse: returns AST list" #t (list? reuse-parse))
    (let* ([match-list (ast-get reuse-parse 'match)]
           [first-op (car match-list)]
           [operands (ast-match-expand-operands first-op)])
      (test-equal "reuse: 2 operands (same var twice)" 2 (length operands)))

    ;;=======================================================================
    ;; Pattern 10: Combined features
    ;;=======================================================================
    (test-equal "combined: returns AST list" #t (list? combined-parse))
    (let ([match-list (ast-get combined-parse 'match)]
          [bindings (ast-get combined-parse 'where)])
      (test-equal "combined: 2 match ops" 2 (length match-list))
      (test-equal "combined: 1 binding" 1 (length bindings)))

    ;;=======================================================================
    ;; Pattern 11: Regions with blocks
    ;;=======================================================================
    (test-equal "region: returns AST list" #t (list? region-parse))
    (let ([rewrite-list (ast-get region-parse 'rewrite)])
      (test-equal "region: has rewrite ops" #t (pair? rewrite-list)))

    (test-end)))
