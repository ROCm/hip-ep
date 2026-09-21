#!/usr/bin/env scheme-script
;; Test if mutating inside loop with :rime-with causes issues

(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

(define counter 0)

(define (expensive-computation x)
  (set! counter (+ counter 1))
  (format #t "expensive-computation called (count=~a) for x=~a\n" counter x)
  (* x 2))

(define items '(1 2 3))
(define actions '())

(format #t "Starting loop\n")
(loop :for item :in items
      :for idx :from 0
      :rime-with result := (expensive-computation item)
      :do (begin
            (format #t "Processing item=~a idx=~a result=~a\n" item idx result)
            ;; Mutate actions - does this cause :rime-with to re-evaluate?
            (set! actions (cons (list 'action idx result) actions))
            (format #t "After mutation, actions=~a\n" actions)))

(format #t "\nFinal actions: ~a\n" actions)
(format #t "expensive-computation was called ~a times\n" counter)
