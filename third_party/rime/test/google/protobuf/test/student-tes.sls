#!r6rs
(library (test google protobuf test student-test)
  (export main)
  (import (rnrs (6))
          (rime loop)
          (rime protobuf private vec)
          (rime unit-test)
          (rime protobuf private message)
          (rime logging)
          (test google protobuf test student))
  (define test-student (list google-protobuf-test-student))
  (define-test
    test-deserialize-student
    (call-with-port (open-bytevector-input-port
                     #vu8(
                          ;; id = 150 (0, len=3)
                          #x08 #x96 #x01
                               ;; name= John (3, len=6)
                          #x12 #x04 #x4a #x6f #x68 #x6e
                          ;; location = (9, len=6)
                          #x1a #x04 #x08 #x14 #x10 #x27
                          ;; school (15,len=13)
                          #x22 #x0b #x08 #xae #x02 #x12 #x06 #x4f #x78 #x66 #x6f #x72 #x64
                          ;; friend Tom (28,len=5)
                          #x32 #x03 #x54 #x6f #x6d
                          ;; friend Jerry (33, len=7)
                          #x32 #x05 #x4a #x65 #x72 #x72 #x79
                          ;; sex: female (40, len=2)
                          #x40 #x01
                          ))
      (lambda (ip)
        (let ([student (make-message '(google protobuf test Student))])
          (logger :info " student = " (student ':debug-string))
          (student ':<< ip)
          (logger :info " student = " (student ':debug-string))
          ))))
  (define (main)
    (log-level 'trace)
    (run-all-tests)))
