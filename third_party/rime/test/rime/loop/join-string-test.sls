#!r6rs
(library (test rime loop join-string-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))

  (define-test
    test-basic-join-string
    (let ()
      (CHECK equal? (loop :for i :in '(1 2 3)
                          :join-string i
                          )
             "123")
      (CHECK equal? (loop :for i :in '(1 2 3)
                          :join-string i :seperator ","
                          )
             "1,2,3")
      (CHECK equal? (loop :for i :in '(1 2 3)
                          :for j :in '(a b c)
                          :join-string i :seperator ","
                          :join-string j :seperator ":"
                          )
             "1,a:2,b:3,c")

      (CHECK equal? (loop :for i :in '(1 2 3)
                          :for j :in '(a b c)
                          :join-string i :seperator "," :into ic
                          :join-string j :seperator ":" :into jc
                          :finally (cons ic jc)
                          )
             (cons "1,2,3" "a:b:c"))
      )))
