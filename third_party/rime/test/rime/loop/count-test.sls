#!r6rs
(library (test rime loop count-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop))

  (define (main)
    (run-all-tests))

  ;; Helper for sorting hashtable results
  (define (sorted-ht ht)
    (map
     (lambda (k)
       (cons k (hashtable-ref ht k #t)))
     (list-sort fx<? (vector->list (hashtable-keys ht)))))

  (define (sorted-ht-string ht)
    (map
     (lambda (k)
       (cons k (hashtable-ref ht k #t)))
     (list-sort string<? (vector->list (hashtable-keys ht)))))

  ;; ========================================
  ;; Basic Counting Tests
  ;; ========================================

  (define-test
    test-count-basic
    ;; Basic counting - count iterations
    (CHECK equal? (loop :for i :upfrom 0 :to 10 :count) 11)
    (CHECK equal? (loop :for i :from 1 :to 5 :count) 5)
    (CHECK equal? (loop :for i :from 0 :below 5 :count) 5))

  (define-test
    test-count-with-finally
    ;; Basic count with :finally transformation
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :count
                        :finally (+ :return-value 100))
           111))

  ;; ========================================
  ;; Conditional Counting Tests
  ;; ========================================

  (define-test
    test-count-conditional
    ;; Conditional counting with :if/:when/:unless
    (CHECK equal? (loop :for ch :in-string "Hello World"
                        :count :if (char-upper-case? ch)) 2)
    (CHECK equal? (loop :for i :from 1 :to 10
                        :count :when (even? i)) 5)
    (CHECK equal? (loop :for i :from 1 :to 10
                        :count :unless (< i 5)) 6)
    (CHECK equal? (loop :for i :in '(1 2 3)
                        :count :if (odd? i)) 2))

  ;; ========================================
  ;; Custom Increment Tests with :step
  ;; ========================================

  (define-test
    test-count-step
    ;; Custom increment with :step
    (CHECK equal? (loop :for i :from 1 :to 5 :count :step 2) 10)
    (CHECK equal? (loop :for i :from 1 :to 5 :count :step i) 15)
    (CHECK equal? (loop :for i :from 1 :to 3 :count :step 10) 30))

  ;; ========================================
  ;; :into Variable Tests
  ;; ========================================

  (define-test
    test-count-into
    ;; Count into variable
    (CHECK equal? (loop :for i :from 1 :to 5
                        :count :into total
                        :finally total) 5)
    (CHECK equal? (loop :for i :from 1 :to 5
                        :count :into sum :step 2
                        :finally sum) 10))

  (define-test
    test-count-combined
    ;; Combined :into :step :if
    (CHECK equal? (loop :for i :from 1 :to 10
                        :count :into sum :step (* 2 i) :if (odd? i)
                        :finally sum) 50)
    (CHECK equal? (loop :for i :from 1 :to 10
                        :count :into sum :step i :if (even? i)
                        :finally sum) 30))

  (define-test
    test-count-multiple-counters
    ;; Multiple counters in same loop
    (CHECK equal? (loop :for i :from 1 :to 10
                        :count :into evens :step i :if (even? i)
                        :count :into odds :step i :if (odd? i)
                        :finally (cons evens odds))
           '(30 . 25)))

  ;; ========================================
  ;; Hashtable Grouping Tests with :by
  ;; ========================================

  (define-test
    test-count-by-grouping
    ;; :by creates hashtable for grouping
    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 3)
                  :count :by i))
           '((1 . 1) (2 . 1) (3 . 1)))

    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 2 3 1 2 3)
                  :count :by i))
           '((1 . 3) (2 . 3) (3 . 2))))

  (define-test
    test-count-by-with-condition
    ;; :by with :if condition
    (CHECK equal?
           (sorted-ht
            (loop :for i :in '(1 2 1 2 3 1 2 3)
                  :count :by i :if (odd? i)))
           '((1 . 3) (3 . 2))))

  (define-test
    test-count-by-custom-hashtable
    ;; :by with custom hash function using :make-hash-table
    (CHECK equal?
           (sorted-ht-string
            (loop :for str :in (map string (string->list "Hello World"))
                  :count :by str
                  :make-hash-table (make-hashtable string-hash string=?)))
           '((" " . 1) ("H" . 1) ("W" . 1) ("d" . 1) ("e" . 1) ("l" . 3) ("o" . 2) ("r" . 1))))

  ;; ========================================
  ;; Practical Examples
  ;; ========================================

  (define-test
    test-count-practical
    ;; String length counting
    (CHECK equal? (loop :for ch :in-string "Hello World" :count) 11)

    ;; Weighted scoring
    (CHECK equal? (loop :for score :in '(80 90 100 70 60)
                        :for weight :in '(1 1 2 1 1)
                        :count :into total :step (* score weight)
                        :finally total) 500)

    ;; Original requested example that was broken
    (CHECK equal? (loop :initially sum := 0
                        :for i :from 0 :below 10
                        :count :into sum :step (* 2 i) :if (odd? i)
                        :finally sum) 50))

  ;; ========================================
  ;; Edge Cases
  ;; ========================================

  (define-test
    test-count-edge-cases
    ;; Empty iteration
    (CHECK equal? (loop :for i :from 1 :to 0 :count) 0)

    ;; Count with no matching conditions
    (CHECK equal? (loop :for i :from 1 :to 10
                        :count :if (> i 20)) 0)

    ;; Grouping empty list
    (CHECK equal? (hashtable-size
                   (loop :for i :in '()
                         :count :by i)) 0)))
