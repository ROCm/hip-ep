#!r6rs
(library (rime protobuf private platform-info)
  (export platform-info check-feature check-library)
  (import (rnrs (6))
          (rnrs eval (6)))
  (define platform-info-data
    (guard
        (error
         [else '()])
      (call-with-port
          (open-file-input-port "platform-info.data"
                                (file-options) (buffer-mode block) (native-transcoder))
        (lambda (port)
          (get-datum port)))))

  (define (platform-info key)
    (let ([result (assq key platform-info-data)])
      (if result
          (cdr result)
          #f)))

  (define (check-feature syn)
    (let [(result (and (identifier? syn)
                       (platform-info (syntax->datum syn))))]
      result))

  (define (check-library import-spec)
    (let ([result (guard
                      (error
                       [else #f])
                    (eval #t (environment import-spec)))])
      result)))
