#!r6rs
(library (test rime loop group-by-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define (sorted-ht ht)
    (map
     (lambda (k)
       (cons k (hashtable-ref ht k #t)))
     (list-sort fx<? (vector->list (hashtable-keys ht)))))

  (define-test
    test-collect-hash-table-1
    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 3)
                  :for j :in '("a" "b" "c")
                  :group j :by i
                  ))
           '((1 "a") (2 "b") (3 "c")))

    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 3)
                  :for j :in '("a" "b" "A" "c")
                  :group j :by i
                  ))
           '((1 "a" "A") (2 "b") (3 "c")))

    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 3)
                  :for j :in '("a" "b" "A" "c")
                  :group j :by i :into var
                  :group j :by 100 :into var
                  :finally var
                  ))
           '((1 "a" "A") (2 "b") (3 "c") (100 "a" "b" "A" "c")))

    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 3)
                  :for j :in '("a" "b" "A" "c")
                  :group j :by i :into var :if (odd? i)
                  :group j :by 100 :into var
                  :finally var
                  ))
           '((1 "a" "A") (3 "c") (100 "a" "b" "A" "c")))
    ))
