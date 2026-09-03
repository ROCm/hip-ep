#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Emit the hipRTC name-expression subset the offline LUT can actually pick.

    python gen_rtc_names.py --arch gfx1151 --output gqa_rtc_tuned_names.h

gqa_kernel.hip instantiates every (head_dim, HpG, block, ...) combination its
dispatch ladders can reach, but the autotuner only ever selects the configs
recorded in lut/<arch>.json. Registering the rest with hipRTC costs compile
time for code no shape resolves to, so the build narrows the registration set
to the LUT's winners; a launch site whose name expression is left out
transparently falls back to its AOT kernel.

The instantiation set is read back out of gqa_kernel.hip's GQA_*_LIST X-macros,
so a LUT row naming a config the dispatch ladders cannot reach is reported as
drift and dropped rather than emitted as an unresolvable name expression.

Without --lut (an arch that has never been tuned) the output only defines
HIPDNN_GQA_RTC_TUNED_SUBSET to 0 and gqa_kernel.hip registers everything.
"""
import argparse
import re
import sys
import json
from pathlib import Path

HERE = Path(__file__).parent
LUT_DIR = HERE.parent / 'lut'
KERNEL_SOURCE = HERE.parent.parent.parent / 'gqa_kernel.hip'

# Both are always instantiated: the LUT records kv_dtype "Any" because the
# quantised cache reuses the fp16 tuning.
KV_FMTS = ['KvDtype::kF16', 'KvDtype::kI8']


def parse_x_macro(src, name):
    """Return the argument tuples of a `#define <name>(X) X(..) X(..)` list."""
    m = re.search(r'#define\s+{}\(X\)((?:[^\n]*\\\n)*[^\n]*)'.format(name), src)
    if not m:
        raise SystemExit('gen_rtc_names: {} not found in the kernel source'
                         .format(name))
    body = m.group(1).replace('\\\n', ' ')
    return [tuple(a.strip() for a in args.split(','))
            for args in re.findall(r'X\(([^)]*)\)', body)]


def instantiated_names(src):
    """Every tuned name expression gqa_kernel.hip instantiates.

    Mirrors the appendGqa*Names spellings; stringizing a template argument list
    collapses each run of whitespace to one space, hence ", " between args.
    """
    hpgs = [a for (a,) in parse_x_macro(src, 'GQA_DECODE_HPG_LIST')]
    head_dims = [a for (a,) in parse_x_macro(src, 'GQA_DECODE_D_LIST')]

    names = set()
    for d, hpg, bkv in parse_x_macro(src, 'GQA_DECODE_WMMA_LIST'):
        for fmt in KV_FMTS:
            names.add('gqa_flash_decode_wmma_kernel<{}, {}, {}, {}>'
                      .format(d, hpg, bkv, fmt))
    for hpg in hpgs:
        for d in head_dims:
            for fmt in KV_FMTS:
                names.add('gqa_flash_decode_kernel<{}, {}, {}>'
                          .format(d, hpg, fmt))
    for mt, bkv, d, hw in parse_x_macro(src, 'GQA_PREFILL_V5_LIST'):
        names.add('gqa_flash_prefill_v5_kernel<{}, {}, {}, {}>'
                  .format(mt, bkv, d, hw))
    for nw, bkv, d, mt in parse_x_macro(src, 'GQA_PREFILL_V7_LIST'):
        names.add('gqa_flash_prefill_v7_kernel<{}, {}, {}, {}>'
                  .format(nw, bkv, d, mt))
    for nd, mt, bkv in parse_x_macro(src, 'GQA_PREFILL_V8_LIST'):
        names.add('gqa_flash_prefill_v8_kernel<{}, {}, {}, 256>'
                  .format(nd, mt, bkv))
    return names, hpgs


def winner_names(rows, hpgs, unmapped):
    """Map every LUT row to the name expression its config would launch."""
    names = set()
    for r in rows:
        phase, cfg = r['phase'], r['config']
        d = r['head_dim'][1:]
        # hpg 0 is the wildcard row: it can be selected for any supported
        # grouping, so it stands for the whole ladder.
        row_hpgs = hpgs if r['hpg'] == 0 else [str(r['hpg'])]
        if r['kv_dtype'] != 'Any':
            unmapped.add('kv_dtype={}'.format(r['kv_dtype']))
            continue
        if phase == 'Decode':
            m = re.fullmatch(r'WmmaBkv(\d+)', cfg)
            for hpg in row_hpgs:
                for fmt in KV_FMTS:
                    if cfg == 'Scalar':
                        names.add('gqa_flash_decode_kernel<{}, {}, {}>'
                                  .format(d, hpg, fmt))
                    elif m:
                        names.add('gqa_flash_decode_wmma_kernel<{}, {}, {}, {}>'
                                  .format(d, hpg, m.group(1), fmt))
                    else:
                        unmapped.add('{} {}'.format(phase, cfg))
        elif phase == 'PrefillV5':
            m = re.fullmatch(r'MT(\d+)_BKV(\d+)', cfg)
            if not m:
                unmapped.add('{} {}'.format(phase, cfg))
                continue
            # A NoWindow row cannot select the windowed instantiation; an Any
            # row can end up on either.
            windows = ['false'] if r['window'] == 'NoWindow' else ['true', 'false']
            for hw in windows:
                names.add('gqa_flash_prefill_v5_kernel<{}, {}, {}, {}>'
                          .format(m.group(1), m.group(2), d, hw))
        elif phase == 'PrefillV7':
            m = re.fullmatch(r'NW(\d+)_BKV(\d+)_MT(\d+)', cfg)
            if not m:
                unmapped.add('{} {}'.format(phase, cfg))
                continue
            names.add('gqa_flash_prefill_v7_kernel<{}, {}, {}, {}>'
                      .format(m.group(1), m.group(2), d, m.group(3)))
        elif phase == 'PrefillV8':
            m = re.fullmatch(r'ND(\d+)_MT(\d+)_BKV(\d+)', cfg)
            if not m:
                unmapped.add('{} {}'.format(phase, cfg))
                continue
            names.add('gqa_flash_prefill_v8_kernel<{}, {}, {}, {}>'
                      .format(m.group(1), m.group(2), m.group(3), d))
        else:
            unmapped.add('phase={}'.format(phase))
    return names


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def render(arch, lut_path, names):
    head = [
        '//',
        '// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.',
        '// Licensed under the MIT License.',
        '//',
        '',
        '// Generated by hip/autotune/gqa/scripts/gen_rtc_names.py -- edits are',
        '// overwritten by the build. Re-run the script (or just rebuild) after',
        '// changing the LUT.',
        '',
        '#pragma once',
        '',
    ]
    if not names:
        head += [
            '// No LUT-derived winner set for {}, so gqa_kernel.hip registers'.format(arch),
            '// every instantiation.',
            '#define HIPDNN_GQA_RTC_TUNED_SUBSET 0',
            '',
        ]
        return '\n'.join(head)
    head += [
        '// Winners of {}, one name expression per hipRTC registration.'.format(
            Path(lut_path).name),
        '#define HIPDNN_GQA_RTC_TUNED_SUBSET 1',
        '',
        '#define HIPDNN_GQA_RTC_TUNED_NAME_LIST(X) \\',
    ]
    body = ['  X("{}")'.format(n) for n in sorted(names, key=natural_key)]
    return '\n'.join(head + [' \\\n'.join(body), ''])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', required=True, help='GPU arch, e.g. gfx1151')
    ap.add_argument('--lut', default=None,
                    help='LUT JSON (default: lut/<arch>.json if it exists)')
    ap.add_argument('--kernel-source', default=str(KERNEL_SOURCE),
                    help='gqa_kernel.hip, read for the GQA_*_LIST X-macros')
    ap.add_argument('--output', default=None,
                    help='Header to write (default: stdout)')
    args = ap.parse_args()

    lut_path = args.lut or LUT_DIR / '{}.json'.format(args.arch)
    src = Path(args.kernel_source).read_text(encoding='utf-8')
    instantiated, hpgs = instantiated_names(src)

    names = set()
    if Path(lut_path).exists():
        rows = json.loads(Path(lut_path).read_text(encoding='utf-8'))['rows']
        unmapped = set()
        names = winner_names(rows, hpgs, unmapped)
        drift = sorted(names - instantiated, key=natural_key)
        names -= set(drift)
        for cfg in sorted(unmapped):
            print('gen_rtc_names: {}: LUT config not understood: {}'
                  .format(args.arch, cfg), file=sys.stderr)
        for name in drift:
            print('gen_rtc_names: {}: LUT wins a config gqa_kernel.hip does not '
                  'instantiate, dropped: {}'.format(args.arch, name),
                  file=sys.stderr)
        print('gen_rtc_names: {}: {} of {} tuned instantiations kept'
              .format(args.arch, len(names), len(instantiated)))
    else:
        print('gen_rtc_names: {}: no LUT at {}, registering every '
              'instantiation'.format(args.arch, lut_path))

    text = render(args.arch, lut_path, names)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
