;; Test case: basic
;; Simplest test - operation with no operands

(:pattern
  (:match %out = test.op ()
   :rewrite %out :with (new.op () -> !t))

 ;; Phase 1: Parse only - validation and analysis skipped
 ;; - root-op-name, root-op-index, root-result-idx: #f (filled by validation)
 ;; - match-bindings, match-actions: #f (filled by analysis)
 :expect-parse
 ((pattern-type . conversion)
  (function-name . pattern-basic)
  (root-var . %out)
  (root-op-name . #f)
  (root-op-index . #f)
  (root-result-idx . #f)
  (match
    ((result-var . %out)
     (op-name . test.op)
     (operands)
     (where-expr . #f)))
  (match-bindings . #f)
  (match-actions . #f)
  (rewrite
    ((result-var)
     (op-name . new.op)
     (operands)
     (regions)
     (attributes)
     (result-types . !t)))
  (where)
  (debug-parse? . #t)
  (debug-validate? . #f)
  (debug-analyze? . #f)
  (debug-codegen? . #f)
  (debug-matching? . #f))

 ;; Phase 2: Parse + Validate only, analysis skipped
 ;; - root-op-name, root-op-index, root-result-idx: filled by validation
 ;; - match-bindings, match-actions: still #f (analysis not run yet)
 ;; - op-name converted to strings by validation phase
 :expect-validate
 ((pattern-type . conversion)
  (function-name . pattern-basic)
  (root-var . %out)
  (root-op-name . "test.op")
  (root-op-index . 0)
  (root-result-idx . 0)
  (match
    ((result-var %out)
     (op-name . "test.op")
     (operands)
     (where-expr . #f)))
  (match-bindings . #f)
  (match-actions . #f)
  (rewrite
    ((result-var)
     (op-name . "new.op")
     (operands)
     (regions)
     (attributes)
     (result-types . !t)))
  (where)
  (debug-parse? . #f)
  (debug-validate? . #t)
  (debug-analyze? . #f)
  (debug-codegen? . #f)
  (debug-matching? . #f))

 ;; Phase 3: Parse + Validate + Analyze - all phases complete
 ;; - root-op-name, root-op-index, root-result-idx: filled by validation
 ;; - match-bindings: filled by analysis (hashtable with binding info)
 ;; - match-actions: filled by analysis (list of matching actions)
 :expect-analyze
 ((pattern-type . conversion)
  (function-name . pattern-basic)
  (root-var . %out)
  (root-op-name . "test.op")
  (root-op-index . 0)
  (root-result-idx . 0)
  (match
    ((result-var %out)
     (op-name . "test.op")
     (operands)
     (where-expr . #f)))
  (match-bindings
    (bindings-table
      (%out
        (id . %out)
        (is-result? . #t)
        (result-op-idx . 0)
        (result-idx . 0)
        (operand-op-idx . #f)
        (operand-idx . #f)
        (bound? . #t))))
  (match-actions
    (:set-current-op (op-idx . 0) (var . %out))
    (:check-op (op-idx . 0)))
  (rewrite
    ((result-var)
     (op-name . "new.op")
     (operands)
     (regions)
     (attributes)
     (result-types . !t)))
  (where)
  (debug-parse? . #f)
  (debug-validate? . #f)
  (debug-analyze? . #t)
  (debug-codegen? . #f)
  (debug-matching? . #f)))
