# Vendored web fonts

These files are served from the site itself rather than from
`fonts.googleapis.com`. A documentation site that fetches its fonts from a
third-party CDN renders differently — or not at all — on networks where that CDN
is unreachable, which includes many corporate networks. Self-hosting also
removes a third-party request from every page load.

| File | Family | Weights | Source |
|---|---|---|---|
| `space-grotesk-var.woff2` | Space Grotesk | Variable, `300 700` | [google/fonts `ofl/spacegrotesk`](https://github.com/google/fonts/tree/main/ofl/spacegrotesk) |
| `ibm-plex-mono-400.woff2` | IBM Plex Mono | 400 | [google/fonts `ofl/ibmplexmono`](https://github.com/google/fonts/tree/main/ofl/ibmplexmono) |
| `ibm-plex-mono-500.woff2` | IBM Plex Mono | 500 | same |

Only the `latin` subset is vendored (`U+0000-00FF` plus the punctuation,
currency and arrow ranges Google groups with it). The site is written in
English; any character outside that range falls through to the next family in
the stack, which is the platform's own UI font.

Both families are licensed under the SIL Open Font License 1.1. The license text
is included verbatim as `OFL-space-grotesk.txt` and `OFL-ibm-plex-mono.txt`, as
that license requires.

## Refreshing

The `@font-face` declarations live in `../styles.css`. To pull newer builds,
request the same families from the Google Fonts CSS API with a browser
`User-Agent` (an older agent is served `woff`/`ttf` instead of `woff2`) and take
the URL from the block whose `unicode-range` begins with `U+0000-00FF`:

```bash
curl -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
  (KHTML, like Gecko) Chrome/131.0 Safari/537.36' \
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=IBM+Plex+Mono:wght@400;500&display=swap'
```

Space Grotesk is a variable font, so one file covers every weight the site uses
and the `@font-face` rule declares the range `300 700`. IBM Plex Mono is static
on Google Fonts, so each weight is a separate file — do not point two weights at
one file, or the browser will render both at whichever weight the file actually
contains instead of synthesizing the difference.
