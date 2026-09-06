#!/usr/bin/env scheme-script
;; Build custom Chez boot file with rime embedded
;; Usage: scheme --script build-chez-boot.scm petite.boot scheme.boot output.boot

(import (chezscheme))

(define (main args)
  (when (< (length args) 3)
    (fprintf (current-error-port)
             "Usage: scheme --script build-chez-boot.scm petite.boot scheme.boot output.boot\n")
    (exit 1))

  (let ([petite-boot (list-ref args 0)]
        [scheme-boot (list-ref args 1)]
        [output-boot (list-ref args 2)])

    (fprintf (current-output-port) "Building custom boot file...\n")
    (fprintf (current-output-port) "  Base boots: ~a, ~a\n" petite-boot scheme-boot)
    (fprintf (current-output-port) "  Output: ~a\n" output-boot)

    ;; For now, just create a combined boot from petite+scheme
    ;; TODO: compile rime into the boot file
    (make-boot-file output-boot
                    (list petite-boot scheme-boot)
                    ""    ;; base library name (empty = no additional libs)
                    "")

    (fprintf (current-output-port) "Boot file created successfully.\n")))

(main (cdr (command-line)))
