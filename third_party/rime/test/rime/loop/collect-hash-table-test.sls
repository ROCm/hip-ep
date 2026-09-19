#!r6rs
(library (test rime loop collect-hash-table-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-collect-hash-table
    (let ()
      (let [(ht (loop :for i :in '(1 2 3)
                      :for j :in '("a" "b" "c")
                      :collect (i j) :as :hash-table
                      ))]
        (CHECK equal?
               (list-sort fx<? (vector->list (hashtable-keys ht)))
               '(1 2 3))
        (CHECK equal?
               (map (lambda (k) (hashtable-ref ht k #t)) (list-sort fx<? (vector->list (hashtable-keys ht))))
               '("a" "b" "c")))
      ))

  (define-test
    test-collect-hash-table-2
    (let ()
      (let [(ht (loop :for i :in '(1 2 1 3)
                      :for j :in '("a" "b" "A" "c")
                      :collect (i j) :as :hash-table
                      ))]
        (CHECK equal?
               (list-sort fx<? (vector->list (hashtable-keys ht)))
               '(1 2 3))
        (CHECK equal?
               (map (lambda (k) (hashtable-ref ht k #t)) (list-sort fx<? (vector->list (hashtable-keys ht))))
               '("A" "b" "c")))
      )))
