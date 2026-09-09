#!r6rs
(library (test rime loop count-by-test)
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
    test-count
    (CHECK equal?
           (loop :for i :in '(1 2 3)
                 :count :if (odd? i)
                 )
           2)
    )
  (define-test
    test-count-by-0
    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 3)
                  :count :by i
                  ))
           '((1 . 1) (2 . 1) (3 . 1)))
    )
  (define-test
    test-count-by-1
    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 2 3 1 2 3)
                  :count :by i
                  ))
           '((1 . 3) (2 . 3) (3 . 2))))

  (define-test
    test-count-by-3
    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 2 3 1 2 3)
                  :count :by i :if (odd? i)
                  ))
           '((1 . 3) (3 . 2))))

  (define-test
    test-make-hash-table
    (let ()
      (define (sorted-ht ht)
        (map
         (lambda (k)
           (cons k (hashtable-ref ht k #t)))
         (list-sort string<? (vector->list (hashtable-keys ht)))))
      (CHECK equal?
             (sorted-ht
              (loop :for str :in (map string (string->list "Hello World"))
                    :count :by str
                    :make-hash-table (make-hashtable string-hash string=?)))
             '((" "  . 1) ("H" . 1) ("W" . 1) ("d" . 1) ("e" . 1) ("l" . 3) ("o" . 2) ("r" . 1)))))

    )
