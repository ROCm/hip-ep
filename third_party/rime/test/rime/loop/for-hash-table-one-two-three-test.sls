#!r6rs
(library (test rime loop for-hash-table-one-two-three-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define-test
    for-i-in-hash-one-two-three
    (lambda ()
      (define ht (make-eq-hashtable))
      (hashtable-set! ht 1 "one")
      (hashtable-set! ht 2 "two")
      (hashtable-set! ht 3 "three")
      (CHECK equal?
             (list-sort (lambda (x y) (< (car x) (car y)))
                        (loop :for (k v) :in-hashtable ht
                              :collect (cons k v)))
             '((1 . "one") (2 . "two") (3 . "three")))
      (CHECK equal?
             (list-sort < (loop :for k :in-hashtable ht
                                :collect k ))
             '(1 2 3))))
  (define (main)
    (run-all-tests)))
