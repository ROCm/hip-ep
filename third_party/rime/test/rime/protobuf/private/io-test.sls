#!r6rs
(library (test rime protobuf private io-test)
  (export main)
  (import (rnrs (6))
          (rnrs mutable-pairs (6))
          (rime unit-test)
          (rime logging)
          (rime loop)
          (rime protobuf private io))
  (define test-io '())
  (define (test-reader who reader bytevector expected-values)
    (call-with-port (open-bytevector-input-port bytevector)
      (lambda (port)
        (loop :for expected-value :in expected-values
              :with pos := (if (port-has-port-position? port)
                               (port-position port)
                               'N/A)
              :with actual-value := (reader port)
              :with succeeded? :=  (equal? actual-value expected-value)
              :do (if (not succeeded?)
                      (logger :error :who who
                              expected-value " is expected "
                              "but " actual-value " is given. pos=" pos "\n")
                      (logger :info "TEST READER: [" who "]" " OK! "
                              actual-value " == " expected-value "\n"))
              :break :unless succeeded?
              ;; it is GOOD if eof is reached.
              :break succeeded? :if (port-eof? port)
              ))))

  (define (generate-cases cases)
    (let* ([ret-values (list '())]
           [bytes (call-with-bytevector-output-port
                   (lambda (port)
                     (loop :for data :in cases
                           :with value := (car data)
                           :with bytes := (cdr data)
                           :if (fixnum? value)
                           :collect value
                           :do (put-bytevector port bytes)
                           :finally (set-car! ret-values :return-value))))])
      (list bytes (car ret-values))))

  (define-test
    test-read-varint
    (apply test-reader
           'read-varint
           read-varint
           (generate-cases
            (list
             (cons 8 #vu8(#x8))
             (cons 150 #vu8(#x96 #x01))
             (cons 137 #vu8(#x89 #x01))
             (cons -1 #vu8(#xff #xff #xff #xff #xff #xff #xff #xff #xff #x01))
             (cons -2 #vu8(#xfe #xff #xff #xff #xff #xff #xff #xff #xff #x01))
             (cons -72057594037927936 #vu8(#x80 #x80 #x80 #x80 #x80 #x80 #x80 #x80 #xff #x01))
             ;; max 0xE0 ....
             (cons -2305843009213693952 #vu8(#x80 #x80 #x80 #x80 #x80 #x80 #x80 #x80 #xe0 #x01))
             ;; max 0x10 ....
             (cons -1152921504606846976 #vu8(#x80 #x80 #x80 #x80 #x80 #x80 #x80 #x80 #xf0 #x01))))))

  (define-test
    test-read-int32
    (apply test-reader
           'read-int32
           read-int32
           (generate-cases
            (list
             (cons -2147483648 #vu8(#x80 #x80 #x80 #x80 #xf8 #xff #xff #xff #xff #x01))
             (cons 2147483647  #vu8(#xff #xff #xff #xff #x07))
             (cons 2147483646 #vu8(#xfe #xff #xff #xff #x07))))))

  (define-test
    test-read-string
    (apply test-reader
           'read-string
           read-string
           (generate-cases
            (list
             (cons "John"  #vu8(#x04 #x4a #x6f #x68 #x6e))
             (cons "Kpio" #vu8(#x04 #x4b #x70 #x69 #x6f))
             (cons "诸葛亮" #vu8(#x09 #xe8 #xaf #xb8 #xe8 #x91 #x9b #xe4 #xba #xae))))))

  (define-test
    test-read-sint32
    (apply test-reader
           'read-sint32
           read-sint32
           (generate-cases
            (list
             (cons 0 #vu8(#x00))
             (cons -1 #vu8(#x01))
             (cons 1 #vu8(#x02))
             (cons -2 #vu8(#x03))
             (cons 2 #vu8( #x04))
             (cons 150 #vu8(#xac #x02))))))
  (define (main)
    (run-all-tests)))
