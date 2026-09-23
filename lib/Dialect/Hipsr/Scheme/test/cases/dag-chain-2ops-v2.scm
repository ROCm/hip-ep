;; Test case: dag-chain-2ops-v2
;; Root operation has two operands: one from producer (%a), one free variable (%y)
;; Demonstrates :bind-argument-operand for both operand types

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a %y)
   :rewrite %b :with (op3 (%a %y) -> !t))

 :expect-analyze
   ((pattern-type . conversion)
    (function-name . pattern-dag-chain-2ops-v2)
    (root-var . %b)
    (root-op-name . "op2")
    (root-op-index . 1)
    (root-result-idx . 0)
    (match
      ((result-var %a)
       (op-name . "op1")
       (operands ((kind . required) (var . %x)))
       (where-expr . #f))
      [(result-var %b)
       (op-name . "op2")
       (operands
         ((kind . required) (var . %a))
         ((kind . required) (var . %y)))
       (where-expr . #f)])
    (match-bindings
      (bindings-table
        (%a (id . %a) (is-result? . #t) (result-op-idx . 0)
            (result-idx . 0) (operand-op-idx . #f) (operand-idx . #f)
            (bound? . #t))
        (%b (id . %b) (is-result? . #t) (result-op-idx . 1)
            (result-idx . 0) (operand-op-idx . #f) (operand-idx . #f)
            (bound? . #t))
        (%x (id . %x) (is-result? . #f) (result-op-idx . #f)
            (result-idx . #f) (operand-op-idx . 0) (operand-idx . 0)
            (bound? . #t))
        (%y (id . %y) (is-result? . #f) (result-op-idx . #f)
            (result-idx . #f) (operand-op-idx . 1) (operand-idx . 1)
            (bound? . #t))))
    (match-actions
      (:set-current-op (op-idx . 1) (var . %b))
      (:check-op (op-idx . 1))
      (:bind-argument-operand (operand-idx . 0) (var . %a))
      (:set-current-op (op-idx . 0) (var . %a))
      (:check-op (op-idx . 0))
      (:bind-operand (op-idx . 0) (operand-idx . 0) (var . %x))
      (:bind-argument-operand (operand-idx . 1) (var . %y)))
    (rewrite
      ((result-var) (op-name . "op3") (operands %a %y) (regions)
        (attributes) (result-types . !t)))
    (where)
    (debug-parse? . #f)
    (debug-validate? . #f)
    (debug-analyze? . #t)
    (debug-codegen? . #f)
    (debug-matching? . #f))

 :expect-codegen
   (define pattern-dag-chain-2ops-v2
     (lambda (op operands-ref rewriter type-converter)
       (let ([%x (make-unbound-value)]
             [%y (make-unbound-value)]
             [%a (make-unbound-value)]
             [%b (make-unbound-value)]
             [all-operations (make-vector 2 (make-unbound-value))])

         ;; Bind root result: %b = result 0 of root operation
         (set! %b (mlir-operation-get-result op 0))

         (if (and
               ;; Navigate: %b -> defining op (op2) and store in all-operations[1]
               (let ([def-op (mlir-value-get-defining-op %b)])
                 (and def-op
                      (begin (vector-set! all-operations 1 def-op) #t)))

               ;; Check: op2 name and result count
               (and (string=? (mlir-operation-name (vector-ref all-operations 1)) "op2")
                    (= (mlir-operation-num-results (vector-ref all-operations 1)) 1))

               ;; Bind: %a = argument operand 0 (from operands-ref for conversion pattern)
               ;; Check bounds before accessing
               (begin
                 (if (< 0 (value-array-ref-size operands-ref))
                     (begin
                       (set! %a (value-array-ref-at operands-ref 0))
                       #t)
                     #f))

               ;; Navigate: %a -> defining op (op1) and store in all-operations[0]
               (let ([def-op (mlir-value-get-defining-op %a)])
                 (and def-op
                      (begin (vector-set! all-operations 0 def-op) #t)))

               ;; Check: op1 name and result count
               (and (string=? (mlir-operation-name (vector-ref all-operations 0)) "op1")
                    (= (mlir-operation-num-results (vector-ref all-operations 0)) 1))

               ;; Bind: %x = operand 0 of op1 (non-root operation, use get-operand-value)
               (begin
                 (set! %x (mlir-operation-get-operand-value
                           (vector-ref all-operations 0) 0))
                 #t)

               ;; Bind: %y = argument operand 1 (free variable from operands-ref)
               ;; Check bounds before accessing
               (begin
                 (if (< 1 (value-array-ref-size operands-ref))
                     (begin
                       (set! %y (value-array-ref-at operands-ref 1))
                       #t)
                     #f)))

             ;; Rewrite: TODO - not implemented yet
             (error 'todo "rewrite not implemented yet")
             #f)))))
