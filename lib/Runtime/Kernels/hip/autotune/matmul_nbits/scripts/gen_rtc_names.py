#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Emit the hipRTC name-expression subset the offline LUT can actually pick.

    python gen_rtc_names.py --arch gfx1151 --output matmul_nbits_rtc_tuned_names.h

matmul_nbits_kernel.hip instantiates every config its dispatch ladders can
reach, but the autotuner only ever selects the ones recorded in
lut/<arch>.json. Registering the rest with hipRTC costs compile time for code
no shape resolves to, so the build narrows the registration set to the LUT's
winners; a launch site whose name expression is left out transparently falls
back to its AOT kernel.

Only the bits=4 families are narrowed. The LUT's model_key is `bits=4`, so its
three phases cover exactly the u4 GEMV (Decode), the u4 DP4A GEMV
(DecodeDp4a) and the u4 WMMA / dequant+GEMM prefill (Prefill). The i8 / u3 / u2
families and the shape-independent helpers have no LUT rows and stay fully
registered.

The instantiation set is read back out of matmul_nbits_kernel.hip's MNB_*_LIST
X-macros, so a LUT row naming a config the dispatch ladders cannot reach is
reported as drift and dropped rather than emitted as an unresolvable name
expression.

Without --lut (an arch that has never been tuned) the output only defines
HIPDNN_MATMUL_NBITS_RTC_TUNED_SUBSET to 0 and matmul_nbits_kernel.hip
registers everything.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
LUT_DIR = HERE.parent / 'lut'
KERNEL_SOURCE = HERE.parent.parent.parent / 'matmul_nbits_kernel.hip'

# BC (the bounds-checked variant) and col_major are picked from the shape at
# launch, not from the config, so a tuned config needs both spellings.
BOOLS = ['false', 'true']


def parse_x_macro(src, name):
    """Return the argument tuples of a `#define <name>(X) X(..) X(..)` list."""
    m = re.search(r'#define\s+{}\(X\)((?:[^\n]*\\\n)*[^\n]*)'.format(name), src)
    if not m:
        raise SystemExit('gen_rtc_names: {} not found in the kernel source'
                         .format(name))
    body = m.group(1).replace('\\\n', ' ')
    return [tuple(a.strip() for a in args.split(','))
            for args in re.findall(r'X\(([^)]*)\)', body)]


def gemv_names(bs, tn):
    return ['matmul_nbits_gemv_kernel<{}, {}, {}>'.format(bs, tn, b)
            for b in BOOLS]


def dp4a_names(bs, tn):
    return ['matmul_nbits_gemv_dp4a_kernel<{}, {}>'.format(bs, tn)]


def fused_names(bm, bn, wtm, wtn, weu, bk):
    return ['MatMulNBits{}<{}, {}, {}, {}, {}, {}, {}>'
            .format(kern, bm, bn, wtm, wtn, weu, b, bk)
            for kern in ('WMMA_ZP', 'WMMA_NoZP') for b in BOOLS]


def dequant_names(bm, bn, wtm, wtn, weu, bk):
    return ['MatMulNBitsFp16GEMM<{}, {}, {}, {}, {}, {}, {}>'
            .format(bm, bn, wtm, wtn, weu, b, bk) for b in BOOLS]


def instantiated(src):
    """Every name expression the three LUT-covered families instantiate.

    Mirrors appendGemvNames / appendDp4aNames / appendWmmaNames; stringizing a
    template argument list collapses each run of whitespace to one space, hence
    ", " between args.
    """
    names = set()
    for _id, bs, tn in parse_x_macro(src, 'MNB_GEMV_CONFIG_LIST'):
        names.update(gemv_names(bs, tn))
        names.update(dp4a_names(bs, tn))
    for tile in parse_x_macro(src, 'MNB_WMMA_TILE_LIST'):
        names.update(dequant_names(*tile))
        names.update(fused_names(*tile))
    for tile in parse_x_macro(src, 'MNB_WMMA_FUSED_ONLY_TILE_LIST'):
        names.update(fused_names(*tile))
    return names


def wmma_tile_index(src):
    """(BM, BN, wt_m, wt_n, bk) -> the tile's full X-macro argument tuple.

    The dispatch ladder keys on exactly these five values, so they identify the
    instantiation a WMMA config resolves to; WEU comes from the table.
    """
    index = {}
    for lst in ('MNB_WMMA_TILE_LIST', 'MNB_WMMA_FUSED_ONLY_TILE_LIST'):
        for tile in parse_x_macro(src, lst):
            bm, bn, wtm, wtn, _weu, bk = tile
            index[(int(bm), int(bn), int(wtm), int(wtn), int(bk))] = tile
    return index


def winner_names(lut, tiles, unmapped):
    """Map every LUT config to the name expressions its winner would launch."""
    cfgs = lut['configs']
    used = {p['config'] for p in lut['points']}
    used.update(f['config'] for f in lut['fallbacks'])
    phase_of = {}
    for p in lut['points']:
        phase_of.setdefault(p['config'], set()).add(p['phase'])
    for f in lut['fallbacks']:
        phase_of.setdefault(f['config'], set()).add(f['phase'])

    names = set()
    for cid in sorted(used):
        c = cfgs[cid]
        phases = phase_of[cid]
        if c['kind'] == 'Gemv':
            for phase in phases:
                if phase == 'Decode':
                    names.update(gemv_names(c['threads'], c['tile_n']))
                elif phase == 'DecodeDp4a':
                    names.update(dp4a_names(c['threads'], c['tile_n']))
                else:
                    unmapped.add('{} on a Gemv config'.format(phase))
        elif c['kind'] == 'Wmma':
            key = (c['bm16'] * 16, c['bn16'] * 16, c['wt_m'], c['wt_n'], c['bk'])
            tile = tiles.get(key)
            if tile is None:
                unmapped.add('Wmma tile BM={} BN={} wt=({},{}) bk={}'.format(*key))
                continue
            names.update(fused_names(*tile) if c['fused']
                         else dequant_names(*tile))
        else:
            unmapped.add('kind={}'.format(c['kind']))
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
        '// Generated by hip/autotune/matmul_nbits/scripts/gen_rtc_names.py --',
        '// edits are overwritten by the build. Re-run the script (or just',
        '// rebuild) after changing the LUT.',
        '',
        '#pragma once',
        '',
    ]
    if not names:
        head += [
            '// No LUT-derived winner set for {}, so matmul_nbits_kernel.hip'.format(arch),
            '// registers every instantiation.',
            '#define HIPDNN_MATMUL_NBITS_RTC_TUNED_SUBSET 0',
            '',
        ]
        return '\n'.join(head)
    head += [
        '// Winners of {}, one name expression per hipRTC registration.'.format(
            Path(lut_path).name),
        '#define HIPDNN_MATMUL_NBITS_RTC_TUNED_SUBSET 1',
        '',
        '#define HIPDNN_MATMUL_NBITS_RTC_TUNED_NAME_LIST(X) \\',
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
                    help='matmul_nbits_kernel.hip, read for the MNB_*_LIST X-macros')
    ap.add_argument('--output', default=None,
                    help='Header to write (default: stdout)')
    args = ap.parse_args()

    lut_path = args.lut or LUT_DIR / '{}.json'.format(args.arch)
    src = Path(args.kernel_source).read_text(encoding='utf-8')
    reachable = instantiated(src)

    names = set()
    if Path(lut_path).exists():
        lut = json.loads(Path(lut_path).read_text(encoding='utf-8'))
        unmapped = set()
        names = winner_names(lut, wmma_tile_index(src), unmapped)
        drift = sorted(names - reachable, key=natural_key)
        names -= set(drift)
        for cfg in sorted(unmapped):
            print('gen_rtc_names: {}: LUT config not understood: {}'
                  .format(args.arch, cfg), file=sys.stderr)
        for name in drift:
            print('gen_rtc_names: {}: LUT wins a config matmul_nbits_kernel.hip '
                  'does not instantiate, dropped: {}'.format(args.arch, name),
                  file=sys.stderr)
        print('gen_rtc_names: {}: {} of {} tuned instantiations kept'
              .format(args.arch, len(names), len(reachable)))
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
