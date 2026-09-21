#!/usr/bin/env scheme-script
;; Debug Case 4 with trace

(import (rnrs)
        (only (chezscheme) trace-define trace-lambda))

;; Minimal reproduction
(define visited (make-vector 2 #f))

(define (build-match-actions op-idx depth)
  (printf "ENTER: op-idx=~a depth=~a visited[~a]=~a\n"
          op-idx depth op-idx (vector-ref visited op-idx))
  (if (vector-ref visited op-idx)
      (begin
        (printf "EXIT: op-idx=~a (already visited)\n" op-idx)
        '())
      (begin
        (printf "VISIT: marking op-idx=~a as visited\n" op-idx)
        (vector-set! visited op-idx #t)
        (printf "RECURSE: from op-idx=~a calling op-idx=~a\n"
                op-idx (- 1 op-idx))
        (let ([child (build-match-actions (- 1 op-idx) (+ depth 1))])
          (printf "EXIT: op-idx=~a depth=~a\n" op-idx depth)
          (list 'action op-idx)))))

(printf "Starting from op-idx=1\n")
(build-match-actions 1 0)
