(library (rime protobuf private vec)
  (export make-vec vec vec?
          vec-length
          vec-ref vec-set!
          vec->list vec-fill!
          vec-map
          vec-for-each
          vec-append
          vec-equal?)
  (import (rename (rnrs (6)))
          (rime  loop))
  (define-record-type (r-vector new-vector vec?)
    (fields (mutable data vec-data vec-data-set!)
            (mutable length vec-length vec-length-set!)
            (mutable capacity vec-capacity vec-capacity-set!)))

  (define make-vec
    (case-lambda
      [(k) (new-vector (make-vector k) k k)]
      [(k fill) (new-vector (make-vector k fill) k k)]))

  (define (vec . args)
    (let ([data (apply vector args)])
      (new-vector data (vector-length data) (vector-length data))))

  (define (vec-equal? v1 v2)
    (let ([len1 (vec-length v1)]
          [len2 (vec-length v2)])
      (cond
       [(fx=? len1 len2)
        (loop :initially := #t
              :for k :upfrom 0 :below len1
              :with :return-value := (and :return-value (equal? (vec-ref-unsafe v1 k)
                                                                (vec-ref-unsafe v2 k)))
              :break :unless :return-value)
        ]
       [else #f])))
  (define (check-out-of-range who vector k)
    (unless (vec? vector)
      (assertion-violation
       who
       "not a vec"
       (vec-length vector) k))
    (unless (fx<? k (vec-length vector))
      (assertion-violation
       who
       "out of range"
       (vec-length vector) k)))

  (define (vec-ref vector k)
    (check-out-of-range 'vec-ref vector k)
    (vec-ref-unsafe vector k))

  (define (vec-ref-unsafe vector k)
    (vector-ref (vec-data vector) k))

  (define (vec-set! vector k obj)
    (check-out-of-range 'vec-ref vector k)
    (vec-set!-unsafe vector k obj))

  (define (vec-set!-unsafe vector k obj)
    (vector-set! (vec-data vector) k obj))

  (define (vec-fill! vector fill)
    (loop :for k :upfrom 0 :below (vec-length vector)
          :collect (vec-set!-unsafe vector k fill)))

  (define (vec->list vector)
    (loop :for k :upfrom 0 :below (vec-length vector)
            :collect (vec-ref-unsafe vector k)))

  (define (check-same-size who length vectors)
    (loop :for vector :in vectors
          :unless (fx=? length (vec-length vector))
          :do (assertion-violation
               who
               "vectors must have same length"
               (cons length
                     (loop :for vector :in vectors
                           :collect (vec-length vector))))))

  (define (list->vec list)
    (let ([data (list->vector list)])
      (new-vector data (vector-length data) (vector-length data))))

  (define (vec-for-each proc vector . vectors)
    (let ([length (vec-length vector)])
      (check-same-size 'vec-for-each length vectors)
      (loop :for i :from 0 :below length
            :do (apply proc (vector-ref (vec-data vector) i)
                       (loop :for v :in vectors
                             :collect (vec-ref-unsafe v i))))))

  (define (vec-map proc vector . vectors)
    (let* ([length (vec-length vector)]
           [ret (make-vec length)])
      (check-same-size 'vec-map length vectors)
      (loop :for i :from 0 :below length
            :do
            (vec-set!-unsafe ret i
                             (apply proc (vector-ref (vec-data vector) i)
                                    (loop :for v :in vectors
                                          :collect (vec-ref-unsafe v i)))))
      ret))


  (define (calculate-new-capacity capacity new-length)
    (cond
     [(fx<=? new-length capacity)
      capacity]
     [else (fx* 2 new-length)]))

  (define (maybe-grow-size vec delta)
    (let* ([data (vec-data vec)]
           [length (vec-length vec)]
           [capacity (vec-capacity vec)]
           [new-length (fx+ length delta)]
           [new-capacity (calculate-new-capacity capacity new-length)])
      (when (fx<? capacity new-capacity)
        (let [(new-data (make-vector new-capacity))]
          (loop :for k :upfrom 0 :below length
                :do (vector-set! new-data k (vector-ref data k)))
          (vec-data-set! vec new-data)
          (vec-capacity-set! vec new-capacity)))))

  (define (vec-append vec . args)
    (let ([delta (length args)])
      (maybe-grow-size vec delta)
      (loop :for k :upfrom (vec-length vec)
            :for arg :in args
            :do (vec-set!-unsafe vec k arg))
      (vec-length-set! vec (fx+ (vec-length vec) delta)))))
