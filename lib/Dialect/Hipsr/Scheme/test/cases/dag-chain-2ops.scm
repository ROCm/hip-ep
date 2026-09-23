;; Test case: dag-chain-2ops
;; Simple chain - A uses result from B

(:pattern
  (:match %a = op1 (%x)
          %b = op2 (%a)
   :rewrite %b :with (op3 (%a) -> !t))

 :expect-codegen
   (define pattern-dag-chain-2ops
     (lambda (op operands-ref rewriter type-converter)
       (let ([%x (make-unbound-value)]
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

               ;; Bind: %a = operand 0 of op2
               (begin
                 (set! %a (mlir-operation-get-operand-value
                           (vector-ref all-operations 1) 0))
                 #t)

               ;; Navigate: %a -> defining op (op1) and store in all-operations[0]
               (let ([def-op (mlir-value-get-defining-op %a)])
                 (and def-op
                      (begin (vector-set! all-operations 0 def-op) #t)))

               ;; Check: op1 name and result count
               (and (string=? (mlir-operation-name (vector-ref all-operations 0)) "op1")
                    (= (mlir-operation-num-results (vector-ref all-operations 0)) 1))

               ;; Bind: %x = operand 0 of op1
               (begin
                 (set! %x (mlir-operation-get-operand-value
                           (vector-ref all-operations 0) 0))
                 #t))

             ;; Rewrite: TODO - not implemented yet
             (error 'todo "rewrite not implemented yet")
             #f)))))
