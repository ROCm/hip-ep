#!r6rs
(library (test rime loop break-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))

  (define-test
    test-break
    (CHECK equal? (loop :trace-parser :trace-codegen
                        :for i :upfrom 0 :to 10
                        :with i+ := (fx+ 1 i)
                        :with i++ := (fx+ 1 i+)
                        :break
                        :break
                        :break)
           (if #f 0))

    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++
                        :break)
           '(2))
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++
                        :break i++)
           2)

    (CHECK equal? (loop :trace-parser :trace-codegen :for i :upfrom 0 :to 10
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++
                        :break :if (> i 2)
                        :finally (map (lambda (test-for-break) (+ 100 test-for-break)) :return-value))
           '(102 103 104 105))

    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++
                        :break :if (>= i 0)
                        :finally (map (lambda (test-for-break) (+ 100 test-for-break)) :return-value))
           '(102))
    ))
