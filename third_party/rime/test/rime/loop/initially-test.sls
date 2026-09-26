#!r6rs
(library (test rime loop initially-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-initially
    (let ( ;; [init-var1 '(A B C)]
          [init-var2 (list 'A 'B 'C)])
      (CHECK equal? (loop :initially :=  init-var2
                          :for ch :in '(d e f)
                          :collect ch
                          :do (map display (list "RET=" :return-value "\n")))
             '(A B C d e f))
      (CHECK equal? init-var2 '(A B C d e f))
      ;; warning, this is undefined behavior I guess, because collect deconstructly append values.
      ;; (CHECK equal? (loop :initially := init-var1
      ;;                     :for ch :in '(d e f)
      ;;                     :collect ch
      ;;                     :do (printf "RET=~s~n" :return-value))
      ;;        '(A B C d e f))
      ;; (printf "init-var1=~s ~s~n" init-var1 (equal? init-var1 (list 'A 'B 'C 'd 'e 'f)))
      ;; (for-each (lambda (x y)
      ;;             (printf "x=~s y=~s (eq? x y)=~s~n" x y (eq? x y)))
      ;;           init-var1 '(A B C d e f))
      ;; (CHECK eq? (equal? init-var1 '(A B C d e f)) #f)
      ))
  (define-test
    test-initially-more-vars-exprs
    (let ()
      (CHECK equal? (loop :initially := (list "something")
                          characters := '(d e f)
                          numbers := '(1 2 3)
                          :for ch :in characters
                          :for number :in numbers
                          :collect (cons number ch))
             '("something" (1 . d) (2 . e) (3 . f)))
      )))
