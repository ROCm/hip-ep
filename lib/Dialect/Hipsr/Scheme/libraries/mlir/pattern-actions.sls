#!r6rs
(library (mlir pattern-actions)
  (export action:bind-root
          action:set-current-op
          action:check-op
          action:bind-operand
          action:check-eq)
  (import (rnrs))

  ;; Action constructors with labeled fields using pairs
  ;; Output format: (:tag (field-name . value) ...)

  (define (action:bind-root var op-idx result-idx)
    (list ':bind-root
          (cons 'var var)
          (cons 'op-idx op-idx)
          (cons 'result-idx result-idx)))

  (define (action:set-current-op op-idx var result-idx)
    (list ':set-current-op
          (cons 'op-idx op-idx)
          (cons 'var var)
          (cons 'result-idx result-idx)))

  (define (action:check-op op-idx)
    (list ':check-op
          (cons 'op-idx op-idx)))

  (define (action:bind-operand op-idx var operand-idx)
    (list ':bind-operand
          (cons 'op-idx op-idx)
          (cons 'var var)
          (cons 'operand-idx operand-idx)))

  (define (action:check-eq op-idx operand-idx var)
    (list ':check-eq
          (cons 'op-idx op-idx)
          (cons 'operand-idx operand-idx)
          (cons 'var var))))
