#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          :match :rewrite :with :where :debug-ast :debug-matching
          = : -> :region :regions :attrs)
  (import (except (rnrs (6)) =)
          (for (only (chezscheme) syntax->list) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand))

  ;; Define keywords as syntax (for cross-library hygiene)
  (define-syntax :match (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :rewrite (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :with (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :where (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-ast (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-matching (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax = (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax : (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :region (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :regions (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :attrs (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))

  (define-syntax define-conversion-pattern
    (lambda (stx)
      ;;=======================================================================
      ;; Phase 1: Parse whole syntax to AST record - pure pattern matching
      ;;=======================================================================
      
      (define (parse-to-ast whole-stx)
        (syntax-case whole-stx ()
          [(_ . rest)
           (parse-rest #'rest (make-ast-pattern-expand #f #f #f '() #f #f '() '() #f #f))]))
      
      ;; Parse flags and clauses
      (define (parse-rest rest ast)
        (syntax-case rest (:debug-ast :debug-matching :match :rewrite :with :where)
          ;; Debug flags
          [(:debug-ast . more)
           (begin
             (ast-pattern-expand-debug-ast?-set! ast #t)
             (parse-rest #'more ast))]
          
          [(:debug-matching . more)
           (begin
             (ast-pattern-expand-debug-matching?-set! ast #t)
             (parse-rest #'more ast))]
          
          ;; Main pattern structure with guards
          [(fname 
            :match (match-op ...)
            :rewrite root :with (rewrite-op ...)
            . where-rest)
           (and (identifier? #'fname)
                (identifier? #'root)
                (not (null? (syntax->list #'(match-op ...))))
                (not (null? (syntax->list #'(rewrite-op ...)))))
           (begin
             ;; Store syntax objects, not datums
             (ast-pattern-expand-function-name-set! ast #'fname)
             (ast-pattern-expand-root-var-set! ast #'root)
             ;; Parse each match operation into ast-match-expand using map
             (ast-pattern-expand-match-set! ast 
               (map parse-match-operation (syntax->list #'(match-op ...))))
             ;; Parse each rewrite operation into ast-operation-expand using map
             (ast-pattern-expand-rewrite-set! ast
               (map parse-rewrite-operation (syntax->list #'(rewrite-op ...))))
             (parse-where #'where-rest ast)
             ast)]
          
          [_ (syntax-violation 'define-conversion-pattern "Invalid pattern syntax (expected fname :match (...) :rewrite root :with (...))" rest)]))
      
      ;;=======================================================================
      ;; Parse individual match operation
      ;; Input: (%out = "onnx.Cast" (%in) () : (!in-type) -> !out-type)
      ;; Output: ast-match-expand record with syntax objects
      ;;=======================================================================
      (define (parse-match-operation op-stx)
        (syntax-case op-stx (= : ->)
          ;; Full pattern: (result = "op.name" (operands ...) (attrs ...) : (input-types ...) -> output-type)
          [(result = op-name operands attrs : input-types -> output-type)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'attrs
             #'input-types
             #'output-type)]

          ;; No types: (result = "op.name" (operands ...) (attrs ...))
          [(result = op-name operands attrs)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'attrs
             #'()              ;; Empty input-types
             #'())]            ;; Empty output-type

          ;; No attrs, with types: (result = "op.name" (operands ...) : (input-types ...) -> output-type)
          [(result = op-name operands : input-types -> output-type)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'()              ;; Empty attrs
             #'input-types
             #'output-type)]

          ;; Minimal: (result = "op.name" (operands ...))
          [(result = op-name operands)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'()              ;; Empty attrs
             #'()              ;; Empty input-types
             #'())]            ;; Empty output-type

          [_ (syntax-violation 'parse-match-operation
               "Invalid match operation syntax (expected: result = \"op.name\" operands [attrs] [: types -> type])"
               op-stx)]))
      
      ;;=======================================================================
      ;; Parse individual rewrite operation
      ;; Keywords introduce sections that accumulate elements until next keyword
      ;;   (result-list =)? "op.name" (operands) [:regions region1 region2 ...]? [:attrs attr1 attr2 ...]? -> result-types
      ;;
      ;; Examples:
      ;;   (%r = "op" (%a %b) -> i32)                        - with operands
      ;;   (%r = "op" (%a) :attrs [a1 v1] [a2 v2] -> i32)    - with attrs (flat!)
      ;;   (%r = "op" (%a) :regions [region1] -> i32)        - with one region (flat!)
      ;;   (%r = "op" (%a) :regions [r1] [r2] -> i32)        - with two regions (flat!)
      ;;   (%r = "op" (%a) :regions [r1] :attrs [a1 v1] -> i32)  - full
      ;;   ((%r1 %r2) = "op" (%a) -> (i32 i32))              - multiple results
      ;;   ("op" (%a) -> ())                                 - no results (void)
      ;;
      ;; Keywords as section markers (not wrappers):
      ;;   :regions [r1] [r2] :attrs [a1] [a2] -> types
      ;;   ^^^^^^^^ ^^^^^^^^^ ^^^^^^ ^^^^^^^^^
      ;;   Section  Elements  Section Elements
      ;;=======================================================================
      (define (parse-rewrite-operation op-stx)
        ;; Step 1: Extract result(s), =, op-name, and rest
        (syntax-case op-stx (=)
          ;; Pattern: (result-part = op-name . rest) or ((results ...) = op-name . rest)
          [(result-part = op-name . rest)
           (parse-rewrite-rest #'result-part #'op-name #'rest)]

          ;; Pattern: (op-name . rest) - no result
          ;; Validation will check if op-name is valid string/symbol
          [(op-name . rest)
           (parse-rewrite-rest #'() #'op-name #'rest)]

          [_ (syntax-violation 'parse-rewrite-operation
               "Invalid operation syntax (expected: [result =] \"op.name\" (operands) ...)"
               op-stx)]))

      ;; Step 2: Parse the rest: (operands) [:regions (...)]? [:attrs [...]]? [-> result-types]?
      ;; Create record with operands, then mutate as we find optional parts
      (define (parse-rewrite-rest result-stx op-name-stx rest-stx)
        (syntax-case rest-stx ()
          [(operands . more)
           (let ([rec (make-ast-operation-expand
                        result-stx
                        op-name-stx
                        #'operands
                        #'()  ;; Empty regions (may be updated)
                        #'()  ;; Empty attrs (may be updated)
                        #'())]) ;; Empty result-types (may be updated)
             (parse-rewrite-optional rec #'more)
             rec)]))

      ;; Helper: parse sections :regions, :attrs, then REQUIRED -> result-types
      ;; Sections accumulate elements until next keyword
      (define (parse-rewrite-optional rec rest-stx)
        (syntax-case rest-stx (:regions :attrs ->)
          ;; :regions starts region section - accumulate until :attrs, ->, or end
          [(:regions . more)
           (parse-regions-section rec #'more '())]

          ;; :attrs starts attrs section - accumulate until -> or end
          [(:attrs . more)
           (parse-attrs-section rec #'more '())]

          ;; -> result-types (REQUIRED - must be present)
          [(-> result-types)
           (ast-operation-expand-result-types-set! rec #'result-types)]

          ;; Error - -> result-types is mandatory
          [_ (syntax-violation 'parse-rewrite-optional
               "Missing -> result-types (required for operations with results)"
               rest-stx)]))

      ;; Accumulate regions until hitting :attrs, ->, or end
      ;; Each region is a list of blocks
      (define (parse-regions-section rec rest-stx regions-acc)
        (syntax-case rest-stx (:attrs ->)
          ;; Hit :attrs - done with regions, start attrs section
          [(:attrs . more)
           (begin
             (ast-operation-expand-regions-set! rec (reverse regions-acc))
             (parse-attrs-section rec #'more '()))]

          ;; Hit -> - done with regions, parse result types
          [(-> result-types)
           (begin
             (ast-operation-expand-regions-set! rec (reverse regions-acc))
             (ast-operation-expand-result-types-set! rec #'result-types))]

          ;; Another region - parse it and accumulate
          [(region . more)
           (let ([region-rec (parse-region #'region)])
             (parse-regions-section rec #'more (cons region-rec regions-acc)))]

          ;; Error - must have -> result-types
          [_ (syntax-violation 'parse-regions-section
               "Expected region, :attrs, or -> result-types"
               rest-stx)]))

      ;; Parse a region: a list of blocks wrapped in parens
      ;; Region syntax: ([block1] [block2] ...)
      ;; Returns ast-region-expand
      (define (parse-region region-stx)
        (syntax-case region-stx ()
          [(block ...)
           (let ([blocks (map parse-block (syntax->list #'(block ...)))])
             (make-ast-region-expand blocks))]
          [_ (syntax-violation 'parse-region
               "Invalid region syntax (expected: list of blocks)"
               region-stx)]))

      ;; Parse a block: (^label (args...) operations...)
      ;; Block syntax: (^bb0 ((%arg0 : i32) (%arg1 : i32)) (%op1 = ...) (%op2 = ...))
      ;; Returns ast-block-expand
      (define (parse-block block-stx)
        (syntax-case block-stx (:)
          ;; ((%var : type) ...) matches zero or more arguments
          [(label ((%var : type) ...) operation ...)
           (let ([args (syntax->list #'((%var type) ...))]
                 [ops (map parse-rewrite-operation (syntax->list #'(operation ...)))])
             (make-ast-block-expand #'label args ops))]

          [_ (syntax-violation 'parse-block
               "Invalid block syntax (expected: (^label ((%var : type) ...) operations...))"
               block-stx)]))

      ;; Accumulate attrs until hitting -> or end
      (define (parse-attrs-section rec rest-stx attrs-acc)
        (syntax-case rest-stx (->)
          ;; Hit -> - done with attrs, parse result types
          [(-> result-types)
           (begin
             (ast-operation-expand-attributes-set! rec (reverse attrs-acc))
             (ast-operation-expand-result-types-set! rec #'result-types))]

          ;; Another attr - accumulate it
          [(attr . more)
           (parse-attrs-section rec #'more (cons #'attr attrs-acc))]

          ;; Error - must have -> result-types
          [_ (syntax-violation 'parse-attrs-section
               "Expected attr or -> result-types"
               rest-stx)]))
      
      ;; Helper: parse optional :where clause
      (define (parse-where rest-stx ast)
        (syntax-case rest-stx (:where)
          [(:where ((var expr) ...))
           (ast-pattern-expand-where-set! ast 
             (map (lambda (binding)
                    (syntax-case binding ()
                      [(v e)
                       (identifier? #'v)
                       (make-ast-where-binding-expand #'v #'e)]
                      [_ (syntax-violation 'parse-where "Invalid where binding (expected: (var expr))" binding)]))
                  (syntax->list #'((var expr) ...))))]
          [()
           (if #f #f)]
          [_ (syntax-violation 'define-conversion-pattern "Expected :where ((var expr) ...) or end" rest-stx)]))

      ;;=======================================================================
      ;; Phase 2: Validate AST record and normalize fields
      ;;=======================================================================
      (define (validate-ast ast-rec)
        ;; Validate function name
        (unless (identifier? (ast-pattern-expand-function-name ast-rec))
          (syntax-violation 'validate-ast "Function name must be an identifier"
                           (ast-pattern-expand-function-name ast-rec)))

        ;; Validate root-var
        (unless (identifier? (ast-pattern-expand-root-var ast-rec))
          (syntax-violation 'validate-ast "Root variable must be an identifier"
                           (ast-pattern-expand-root-var ast-rec)))

        ;; Validate and normalize match operations
        (validate-and-analyze-pattern-matching ast-rec)

        ;; Validate and normalize each rewrite operation
        (for-each validate-operation (ast-pattern-expand-rewrite ast-rec))

        ;; Validate where bindings
        (for-each (lambda (where-binding)
                   (unless (identifier? (ast-where-binding-expand-var where-binding))
                     (syntax-violation 'validate-ast "Where binding variable must be an identifier"
                                      (ast-where-binding-expand-var where-binding))))
                 (ast-pattern-expand-where ast-rec))

        ast-rec)

      ;; Validate and analyze match operations (coordinator)
      (define (validate-and-analyze-pattern-matching ast-rec)
        (validate-match-operations ast-rec)
        (analyze-match-operations ast-rec))

      ;; Helper: validate identifier starts with %
      (define (validate-%-identifier var context-msg)
        (unless (identifier? var)
          (syntax-violation 'validate-match-operations
            (string-append context-msg " must be identifier") var))
        (let ([var-name (symbol->string (syntax->datum var))])
          (unless (char=? (string-ref var-name 0) #\%)
            (syntax-violation 'validate-match-operations
              (string-append context-msg " must start with %") var))))

      ;; Helper: normalize op-name (string or symbol -> string)
      (define (normalize-op-name match-op)
        (let* ([op-name-stx (ast-match-expand-op-name match-op)]
               [op-name-datum (syntax->datum op-name-stx)])
          (unless (or (string? op-name-datum) (symbol? op-name-datum))
            (syntax-violation 'validate-match-operations
              "Operation name must be string or symbol" op-name-stx))
          (when (symbol? op-name-datum)
            (ast-match-expand-op-name-set! match-op
              (datum->syntax op-name-stx (symbol->string op-name-datum))))))

      ;; Validate match operations
      (define (validate-match-operations ast-rec)
        (loop :for match-op :in (ast-pattern-expand-match ast-rec)
              :do (normalize-op-name match-op)
              (:loop :for var :in (let ([rv (ast-match-expand-result-var match-op)])
                                    (if (identifier? rv) (list rv) (syntax->list rv)))
                     :do (validate-%-identifier var "Result"))
              (:loop :for var :in (syntax->list (ast-match-expand-operands match-op))
                     :do (validate-%-identifier var "Operand"))))

      ;; Analyze match operations - build match-bindings and match-actions
      (define (analyze-match-operations ast-rec)
        ;; Convert match list to vector for index-based access
        (let* ([match-vec (list->vector (ast-pattern-expand-match ast-rec))]
               [bindings (make-eq-hashtable)]
               [actions '()])

          ;; Walk through each match operation
          (loop :for op-idx :from 0 :below (vector-length match-vec)
                :rime-with match-op := (vector-ref match-vec op-idx)
                :rime-with op-name := (let ([op-name-datum (syntax->datum (ast-match-expand-op-name match-op))])
                                         (if (string? op-name-datum)
                                             op-name-datum
                                             (symbol->string op-name-datum)))
                :rime-with results := (let ([rv (ast-match-expand-result-var match-op)])
                                        (if (identifier? rv) (list rv) (syntax->list rv)))
                :rime-with operands := (syntax->list (ast-match-expand-operands match-op))

                ;; Create :match-operation action
                :do (set! actions (cons (list ':match-operation op-name op-idx) actions))

                ;; Process results with nested loop
                (:loop :for result-idx :from 0
                       :for var :in results
                       :do (let ([matcher (list ':match-result result-idx op-idx)])
                             (hashtable-set! bindings var matcher)
                             (set! actions (cons matcher actions))))

                ;; Process operands with nested loop
                (:loop :for operand-idx :from 0
                       :for var :in operands
                       :do (if (hashtable-ref bindings var #f)
                               ;; Reuse: create :match-exact-value
                               (set! actions (cons (list ':match-exact-value var operand-idx op-idx) actions))
                               ;; First occurrence: create :match-operand
                               (let ([matcher (list ':match-operand operand-idx op-idx)])
                                 (hashtable-set! bindings var matcher)
                                 (set! actions (cons matcher actions))))))

          ;; Store results (reverse actions to maintain order)
          (ast-pattern-expand-match-bindings-set! ast-rec bindings)
          (ast-pattern-expand-match-actions-set! ast-rec (reverse actions))

          ;; Find root operation and extract its name
          (let ([root-var (ast-pattern-expand-root-var ast-rec)])
            (loop :for op-idx :from 0 :below (vector-length match-vec)
                  :rime-with match-op := (vector-ref match-vec op-idx)
                  :rime-with result-var := (ast-match-expand-result-var match-op)
                  :rime-with is-root? := (if (identifier? result-var)
                                            (free-identifier=? result-var root-var)
                                            (loop :for var :in (syntax->list result-var)
                                                  :break #t :if (free-identifier=? var root-var)
                                                  :finally #f))
                  :do (when is-root?
                        (ast-pattern-expand-root-op-name-set! ast-rec
                          (ast-match-expand-op-name match-op)))))))

      ;; Validate a single operation and walk its structure
      (define (validate-operation op)
        ;; Validate result-var is identifier (or empty for void operations)
        (let ([result (ast-operation-expand-result-var op)])
          (unless (or (identifier? result)
                      (null? (syntax->datum result)))
            (syntax-violation 'validate-operation "Operation result must be identifier or ()"
                             result)))
        ;; Validate and normalize op-name
        (let* ([op-name-stx (ast-operation-expand-op-name op)]
               [op-name-datum (syntax->datum op-name-stx)])
          (unless (or (string? op-name-datum) (symbol? op-name-datum))
            (syntax-violation 'validate-operation "Operation name must be string or symbol"
                             op-name-stx))
          ;; Normalize: convert symbol to string in-place
          (unless (string? op-name-datum)
            (ast-operation-expand-op-name-set! op
              (datum->syntax op-name-stx (symbol->string op-name-datum)))))
        ;; Walk regions - regions is a list of ast-region-expand records
        (let ([regions (ast-operation-expand-regions op)])
          (when (pair? regions)  ;; Only walk if non-empty
            (for-each validate-region regions))))

      ;; Validate a region and walk its blocks
      (define (validate-region region)
        (for-each validate-block (ast-region-expand-blocks region)))

      ;; Validate a block and walk its operations
      (define (validate-block block)
        ;; Walk operations in this block
        (for-each validate-operation (ast-block-expand-operations block)))

      ;;=======================================================================
      ;; Helper: Convert expansion-time AST to runtime data (datums)
      ;;=======================================================================
      ;; Convert ast-match-expand to datum for runtime AST record
      (define (match-expand->datum match-exp)
        (list 'match
              (syntax->datum (ast-match-expand-result-var match-exp))
              (syntax->datum (ast-match-expand-op-name match-exp))
              (syntax->datum (ast-match-expand-operands match-exp))
              (syntax->datum (ast-match-expand-attributes match-exp))
              (syntax->datum (ast-match-expand-input-types match-exp))
              (syntax->datum (ast-match-expand-output-type match-exp))))

      ;; Convert ast-operation-expand to datum for runtime AST record
      (define (operation-expand->datum op-exp)
        (list 'rewrite
              (syntax->datum (ast-operation-expand-result-var op-exp))
              (syntax->datum (ast-operation-expand-op-name op-exp))
              (syntax->datum (ast-operation-expand-operands op-exp))
              (syntax->datum (ast-operation-expand-regions op-exp))
              (syntax->datum (ast-operation-expand-attributes op-exp))
              (syntax->datum (ast-operation-expand-result-types op-exp))))

      ;; Convert ast-where-binding-expand to datum for runtime AST record
      (define (where-binding-expand->datum where-exp)
        (list (syntax->datum (ast-where-binding-expand-var where-exp))
              (syntax->datum (ast-where-binding-expand-expr where-exp))))

      ;;=======================================================================
      ;; Phase 3: Generate code from validated AST record
      ;;=======================================================================
      (define (generate-code whole-stx ast-rec)
        (syntax-case whole-stx ()
          [(macro-name . _)
           (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                        [root-op-name (ast-pattern-expand-root-op-name ast-rec)])
             (let ([match-ops (ast-pattern-expand-match ast-rec)]
                   [rewrite-ops (ast-pattern-expand-rewrite ast-rec)]
                   [where-bindings (ast-pattern-expand-where ast-rec)]
                   [debug-ast? (ast-pattern-expand-debug-ast? ast-rec)]
                   [debug-matching? (ast-pattern-expand-debug-matching? ast-rec)])
               (if debug-ast?
                   ;; For :debug-ast mode, return list with AST data
                   ;; Convert syntax objects to datums for runtime inspection
                   (let ([fname-sym (syntax->datum #'fname)]
                         [root-op-str (syntax->datum #'root-op-name)]
                         [match-data (map match-expand->datum match-ops)]
                         [rewrite-data (map operation-expand->datum rewrite-ops)]
                         [where-data (map where-binding-expand->datum where-bindings)])
                     (with-syntax ([ast-list (datum->syntax #'macro-name
                                               `(list 'function-name ',fname-sym
                                                      'root-op-name ,root-op-str
                                                      'match ',match-data
                                                      'rewrite ',rewrite-data
                                                      'where ',where-data
                                                      'debug-ast? ,debug-ast?
                                                      'debug-matching? ,debug-matching?))])
                       #'(define fname ast-list)))
                   ;; For normal mode, generate lambda using syntax objects directly
                   #'(define fname
                       (lambda (op operands-ref rewriter type-converter)
                         #f)))))]))

      (let* ([ast-rec (parse-to-ast stx)]
             [validated (validate-ast ast-rec)])
        (generate-code stx validated))))

) ;; end library
