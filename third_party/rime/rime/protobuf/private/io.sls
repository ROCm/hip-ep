(library (rime protobuf private io)
  (export read-varint
          read-packed-int32
          read-int32
          read-sint32
          read-string
          make-sub-input-port)
  (import (rename (rnrs (6))
                  (get-u8 $get-u8)
                  (port-position $port-position)
                  (get-bytevector-n! $get-bytevector-n!)
                  (get-string-n $get-string-n))
          (rime loop)
          (rime logging))

  (define (sub-bytevector b start n)
    (let ((ret (make-bytevector n)))
      (bytevector-copy! b start ret 0 n)
      ret))

  (define (port-position port)
    (if (port-has-port-position? port)
        ($port-position port)
        'N/A))

  (define (get-u8 port)
    ($get-u8 port))

  (define (get-bytevector-n! binary-input-port bytevector start n)
    (let ([ret ($get-bytevector-n! binary-input-port bytevector start n)])
      ret))

  (define NUM_OF_LOSS_BITS (fx- 64 (fixnum-width)))
  (define ACCURACE_CHECK (if (and (fx>? (fixnum-width) 32)
                                  (fx>=? NUM_OF_LOSS_BITS 1)
                                  (fx<=? NUM_OF_LOSS_BITS 7))
                             #t
                             (assertion-violation
                              'protobuf-io-limitation " unsupported fixnum-width"
                              (fixnum-width))))
  (define MSB_VALID_BITS (fx- 8 NUM_OF_LOSS_BITS))
  (define MAX_MSB (fxarithmetic-shift 1 MSB_VALID_BITS))
  (define MSB_MASK (fx- MAX_MSB 1))
  (define MSB_LOSS_MASK (fx- 256 MAX_MSB))

  (define (read-varint port)
    (define (return acc)
      (let* ((lsb (car acc))
             (msb (cdr acc))
             (valid-msb (fxand msb MSB_MASK))
             (msb-loss (fxand msb MSB_LOSS_MASK)) ;; we have 2 or 3 bits loss in accuracy
             )
        (and #f (logger :trace " acc = " acc
                                 " valid-msb = " valid-msb
                                 " msb-loss = " msb-loss
                                 " MSB_LOSS_MASK = " MSB_LOSS_MASK
                              ))
        (let ([overflow? (cond
                          [(fx=? msb-loss 0) #f]
                          [(and (eq? msb-loss MSB_LOSS_MASK)
                                ;; it is overflow for negative value.
                                (not (fx=? valid-msb 0))) #f]
                          [else #t])])
          (when overflow?
            ;; it might be a ok.
            (assertion-violation
             'read-varint "overflow !"
             (object-to-string " acc = ("  (car acc) " " (cdr acc) ")"
                               " msb-loss = " msb-loss))))
        ;; (fxarithmetic-shift msb-61 56)
        (if (fx=? msb-loss MSB_LOSS_MASK)
            (fx+ lsb (fxarithmetic-shift (fx- valid-msb MAX_MSB) 56))
            (fx+ lsb (fxarithmetic-shift valid-msb 56)))))
    (loop :for n :upfrom 0 :by 7
          :with pos := (if (port-has-port-position? port)
                           (port-position port)
                           'N/A)
          :with value := (get-u8 port)
          :do (when (eof-object? value)
                (assertion-violation
                 'read-varint "end of file"
                 ( " acc = " acc
                                   " n= " n)))
          :do (and #f (logger :trace " READ value = " value))
          :do (when (fx>? n 63)
                (assertion-violation
                 'read-varint "overflow!"
                 (object-to-string " pos = " (port-position port)
                                   " acc =" acc
                                   " value= " value
                                   " n= " n)))
          :with stop? := (fxzero? (fxand value #x80))
          :with new-value := (if (fx<? n 56)
                                 (fxarithmetic-shift (fxand value #x7F) n)
                                 (fxarithmetic-shift (fxand value #x7F) (fx- n 56)))
          :with new-acc :=  (if (fx<? n 56)
                                (cons (fx+ new-value (car acc)) (cdr acc))
                                (cons (car acc) (fx+ new-value (cdr acc))))
          :do (and #f (logger :trace
                       " pos = " pos
                       " acc = " acc
                       " n = " n
                       " acc = " acc
                       " value = " value
                       " new-value = " new-value
                       " new-acc = " new-acc
                       " stop = " stop?
                       "\n"))
          :with acc := new-acc :initially (cons 0 0)
          :break :if stop?
          :finally (let ([ret (return acc)])
                     (and #f (logger :trace " acc = " acc " ret = " ret))
                     ret)
          ))

  (define (read-int32 port)
    ;; it supposed to read an unsigned varint from a port, scheme
    ;; represent 61-bit signed fix number.
    (let ((max-int32 (- (expt 2 (- 32 1)) 1))
          (min-int32 (- (expt 2 (- 32 1)))))
      (fxmax (fxmin (read-varint port) max-int32) min-int32)))

  (define (read-sint32 port)
    (let* [(value (read-varint port))
           (zigzag-decoded (fxxor (fxarithmetic-shift-right value 1) (fx* -1 (fxand value 1))))]
      (let [(max-int32 (- (expt 2 (- 32 1)) 1))
            (min-int32 (- (expt 2 (- 32 1))))]
        (fxmax (fxmin zigzag-decoded max-int32) min-int32))))

  (define (read-packed maker reader setter!)
    (lambda (port)
      (let* ([len (read-varint port)]
             [ret (maker len)])
        (let loop ((i 0))
          (when (< i len)
            (setter! ret i (reader port))
            (loop (+ 1 i)))))))

  (define read-packed-int32
    (read-packed make-vector read-int32 vector-set!))

  (define (make-sub-input-port port len)
    (let ([current-pos (if (port-has-port-position? port)
                           (port-position port)
                           0)]
          [read-pos 0])
      (make-custom-binary-input-port
       (call-with-string-output-port
        (lambda (op)
          (display "(" op)
          (display port op)
          (display ")" op)
          (display len op)))
       ;; r!
       (lambda (bytevector start n)
         (let ((bytes-left (fxmax (fx- len (fx- (port-position port) current-pos)) 0)))
           (let ([ret ($get-bytevector-n! port bytevector start (fxmin n bytes-left))])
             (unless (eof-object? ret)
               (set! read-pos (fx+ read-pos ret)))
             ret)))
       ;; gp
       (lambda ()
         (fx+ current-pos read-pos))
       ;; sp!
       #f
       ;; close
       (lambda ()
         (close-port port)))))
  (define (read-string port)
    ;; it supposed to read an unsigned varint from a port, scheme
    ;; represent 61-bit signed fix number.
    (let* [(len (read-varint port))
           (new-port (make-sub-input-port port len))]
      (let* ([new-textual-port (transcoded-port new-port (make-transcoder (utf-8-codec) 'none))]
             ;; len is larger than the number of characters so that it is
             ;; safe to call it.
             [ret ($get-string-n new-textual-port len)])
        ret)))

  )
