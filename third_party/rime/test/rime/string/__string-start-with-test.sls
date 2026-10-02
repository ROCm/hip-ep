(library (test rime string __string-start-with-test)
  (export main)
  (import (rnrs (6))
          (rime string __string-start-with)
          (rime unit-test))
  (define-test test-string-start-with
    (CHECK equal?
           (string-starts-with "abc" 0 "abcDEF")
           #t)
    (CHECK equal?
           (string-starts-with "abc" 4 "01234abc")
           #f)
    (CHECK equal?
           (string-starts-with "abc" 5 "01234abc")
           #t)
    (CHECK equal?
           (string-starts-with "abc" 4 "ab")
           #f)
    (CHECK equal?
           (string-starts-with "abc" 4 "abc")
           #f)
    (CHECK equal?
           (string-starts-with "" 4 "abc")
           #t)
    )
  (define (main)
    (run-all-tests)))
