#!r6rs
(library (mlir pattern-actions)
  (export action:set-current-op
          action:check-op
          action:bind-operand
          action:bind-argument-operand
          action:check-eq)
  (import (rnrs))

  ;; Action constructors with labeled fields using pairs
  ;; Output format: (:tag (field-name . value) ...)
  ;;
  ;; Actions represent runtime pattern matching algorithm:
  ;; - :set-current-op - Navigate to operation in DAG
  ;; - :check-op - Verify operation type matches
  ;; - :bind-operand - Bind free variable to operation's operand (can match any input)
  ;; - :bind-argument-operand - Bind root operand from operands-ref (conversion patterns only)
  ;; - :check-eq - Verify bound variable equals current operand

  (define (action:set-current-op op-idx var)
    (list ':set-current-op
          (cons 'op-idx op-idx)
          (cons 'var var)))

  (define (action:check-op op-idx)
    (list ':check-op
          (cons 'op-idx op-idx)))

  (define (action:bind-operand op-idx operand-idx var)
    (list ':bind-operand
          (cons 'op-idx op-idx)
          (cons 'operand-idx operand-idx)
          (cons 'var var)))

  (define (action:bind-argument-operand operand-idx var)
    (list ':bind-argument-operand
          (cons 'operand-idx operand-idx)
          (cons 'var var)))

  (define (action:check-eq op-idx operand-idx var)
    (list ':check-eq
          (cons 'op-idx op-idx)
          (cons 'operand-idx operand-idx)
          (cons 'var var))))
