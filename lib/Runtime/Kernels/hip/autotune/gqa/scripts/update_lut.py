#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Update the GQA autotune LUT from fresh measurements.

Steps, run individually or all at once:

    python update_lut.py plan             # list shapes to measure -> shapes_gfx1151.json (config=TBD)
    python update_lut.py measure          # sweep GPU, write CSVs to --data
    python update_lut.py build            # *_best.csv -> gfx1151.json (dedup + saturation-prune)
    python update_lut.py compile          # gfx1151.json -> gfx1151.fb (needs flatc)
    python update_lut.py all              # measure + build + compile

To re-measure only some head dims (or only prefill) without re-measuring the rest
of an existing table:

    python update_lut.py measure --phase prefill --head-dim 64,128,256
    python update_lut.py build --merge-existing --head-dim 64,128,256
    python update_lut.py compile

or only some geometries (H:G:d), keeping every other geometry's points:

    python update_lut.py measure --phase decode --geometry 24:4:256
    python update_lut.py build --merge-existing --geometry 24:4:256

The sweep executable must be built first:
    cmake --build <build-dir> --target gqa_autotune_sweep

Measurement data is written to --data (default: ./scripts/data/).
The LUT files live in ./lut/ relative to this script's parent directory.

The supported build is the standalone one (build reads the *_best.csv the sweep
wrote and emits the nearest-neighbour schema this loader expects).

Do NOT use --rdpcapture yet: that path delegates to RdpCapture's build_lut.py,
which still emits the old tier schema (buckets + per-field wildcards) that
gqa_autotune.cpp rejects. The flag is kept for when build_lut.py is ported to the
new configs[]/points[] format.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).parent
LUT_DIR = HERE.parent / 'lut'
FBS_FILE = HERE.parent / 'gqa_autotune.fbs'
SWEEP_EXE = (HERE.parent.parent.parent.parent / 'test' / 'example' / 'gqa' /
             'autotune' / 'build' / 'gqa_autotune_sweep.exe')

# Defined again (with KERNEL_ABI) in the build section below; kept here only so
# the module reads top-down. Both are the same value.
SCHEMA_VERSION = 9

# All measured geometries: (H, G, d, sink, group, prefill_kernel)
# Every entry becomes a row in the decode LUT; entries with a prefill_kernel
# also drive prefill measurement.
GEOMETRIES = [
    # ---- primary models -------------------------------------------------------
    (64,  8,  64,  1, 'gpt-oss',           'prefill_v6x'),
    (32,  8, 128,  0, 'llama-mistral',      'prefill_v6x'),   # Llama-3.1-8B, Mistral-7B
    (40, 10, 128,  0, 'phi4',               'prefill_v6x'),   # phi-4
    (40,  8, 128,  0, 'qwen2.5-14b',        'prefill_v6x'),   # Qwen2.5-14B
    (64,  8, 128,  0, 'deepseek-70b',       'prefill_v6x'),   # DeepSeek-R1-Distill-70B
    ( 8,  4, 256,  0, 'gemma3-4b',          'prefill_v8x'),   # gemma3-4b
    (16,  4, 256,  0, 'qwen3.5-9b',         'prefill_v8x'),   # Qwen3.5-9B
    (16,  2, 256,  0, 'qwen3.5-35b',        'prefill_v8x'),   # Qwen3.5/3.6-35B-A3B
    (24,  4, 256,  0, 'qwen3.6-27b',        'prefill_v8x'),   # Qwen3.6-27B, Qwen3.8-27B
    (24,  6, 256,  0, 'qwen3.6-27b-hpg4',   'prefill_v8x'),
    (24,  3, 256,  0, 'qwen3.6-27b-alt',    'prefill_v8x'),
    # ---- fill: close the (head_dim, HpG) matrix ------------------------------
    (32, 32,  64,  0, 'mha-d64',            'prefill_v6x'),
    (16,  8,  64,  0, 'hpg2-d64',           'prefill_v6x'),
    (24,  8,  64,  0, 'hpg3-d64',           'prefill_v6x'),
    (32,  8,  64,  0, 'llama3.2-1b',        'prefill_v6x'),
    (40,  8,  64,  0, 'hpg5-d64',           'prefill_v6x'),
    (32,  2,  64,  0, 'hpg16-d64',          'prefill_v6x'),
    (32, 32, 128,  0, 'llama2-7b',          'prefill_v6x'),
    (16,  8, 128,  0, 'qwen3-1.7b',         'prefill_v6x'),
    (24,  8, 128,  0, 'llama3.2-3b',        'prefill_v6x'),
    (32,  2, 128,  0, 'glm4-9b',            'prefill_v6x'),
    (16, 16, 256,  0, 'mha-d256',           'prefill_v8x'),
    (24,  8, 256,  0, 'hpg3-d256',          'prefill_v8x'),
    (40,  8, 256,  0, 'hpg5-d256',          'prefill_v8x'),
    (32,  2, 256,  0, 'hpg16-d256',         'prefill_v8x'),
    # ---- low-parallelism anchor (each HpG x head_dim pair) -------------------
    ( 8,  8,  64,  0, 'hold-1-d64',         'prefill_v6x'),
    ( 8,  4,  64,  0, 'hold-2-d64',         'prefill_v6x'),
    (12,  4,  64,  0, 'hold-3-d64',         'prefill_v6x'),
    (16,  4,  64,  0, 'hold-4-d64',         'prefill_v6x'),
    (20,  4,  64,  0, 'hold-5-d64',         'prefill_v6x'),
    ( 8,  1,  64,  0, 'hold-8-d64-mqa',     'prefill_v6x'),
    (16,  1,  64,  0, 'hold-16-d64-mqa',    'prefill_v6x'),
    (16, 16, 128,  0, 'hold-1-d128',        'prefill_v6x'),
    (64, 32, 128,  0, 'hold-2-d128',        'prefill_v6x'),
    (12,  4, 128,  0, 'hold-3-d128',        'prefill_v6x'),
    ( 8,  2, 128,  0, 'hold-4-d128',        'prefill_v6x'),
    (20,  4, 128,  0, 'hold-5-d128',        'prefill_v6x'),
    (128,16, 128,  0, 'hold-8-d128',        'prefill_v6x'),
    (16,  1, 128,  0, 'hold-16-d128-mqa',   'prefill_v6x'),
    ( 8,  8, 256,  0, 'hold-1-d256',        'prefill_v8x'),
    (32, 16, 256,  0, 'hold-2-d256',        'prefill_v8x'),
    (12,  4, 256,  0, 'falcon3-7b',         'prefill_v8x'),
    ( 8,  2, 256,  0, 'hold-4-d256',        'prefill_v8x'),
    (20,  4, 256,  0, 'hold-5-d256',        'prefill_v8x'),
    (48,  6, 256,  0, 'hold-8-d256',        'prefill_v8x'),
    (64,  4, 256,  0, 'hold-16-d256',       'prefill_v8x'),
    # ---- floor and holdout (second head count per HpG) -----------------------
    (16, 16,  64,  0, 'hold2-1-d64',        'prefill_v6x'),
    (32, 16,  64,  0, 'hold2-2-d64',        'prefill_v6x'),
    (32,  4,  64,  0, 'hold2-8-d64',        'prefill_v6x'),
    ( 8,  4, 128,  0, 'hold2-2-d128',       'prefill_v6x'),
    (32,  4, 128,  0, 'hold2-8-d128',       'prefill_v6x'),
    (10,  2, 128,  0, 'hold2-5-d128',       'prefill_v6x'),
    (64, 16, 128,  0, 'hold2-4-d128',       'prefill_v6x'),
    (48, 16, 256,  0, 'hold2-3-d256',       'prefill_v8x'),
    (48, 12, 256,  0, 'hold2-4-d256',       'prefill_v8x'),
    (24, 24, 256,  0, 'hold2-1-d256',       'prefill_v8x'),
    # ---- second head count at already-covered (head_dim, HpG) ---------------
    (16,  4, 128,  0, 'hpg4-lowpar-d128',   'prefill_v6x'),
    (40, 40, 128,  0, 'llama2-13b',         'prefill_v6x'),
    (32, 16, 128,  0, 'gemma2-27b',         'prefill_v6x'),
    (16,  8, 256,  0, 'gemma2-9b',          'prefill_v8x'),
    (64,  4, 128,  0, 'hpg16-highpar',      'prefill_v6x'),
    # ---- head_dim 96: Phi-3-mini plus every templated HpG --------------------
    (32, 32,  96,  0, 'phi3-mini',          'prefill_v6x'),  # Phi-3/3.5-mini
    (16, 16,  96,  0, 'hold-1-d96',         'prefill_v6x'),
    (16,  8,  96,  0, 'hpg2-d96',           'prefill_v6x'),
    (24,  8,  96,  0, 'hpg3-d96',           'prefill_v6x'),
    (32,  8,  96,  0, 'hpg4-d96',           'prefill_v6x'),
    (40,  8,  96,  0, 'hpg5-d96',           'prefill_v6x'),
    (64,  8,  96,  0, 'hpg8-d96',           'prefill_v6x'),
    (32,  2,  96,  0, 'hpg16-d96',          'prefill_v6x'),
    (16,  1,  96,  0, 'hold-16-d96-mqa',    'prefill_v6x'),
]

PHASE_OF_KERNEL = {
    'flash_decode': 'Decode',
    'prefill_v6x': 'PrefillV6x',
    'prefill_v8x': 'PrefillV8x',
}

# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def _gen_shapes(data_dir, phase):
    """Write shape grid CSVs to data_dir; return list of CSV paths."""
    try:
        import gen_lut_grid as glg
    except ImportError:
        sys.exit('gen_lut_grid not found -- add RdpCapture/ops_analyze/gqa/tools to PYTHONPATH')

    paths = []
    for ph in (['decode', 'prefill'] if phase == 'both' else [phase]):
        out = data_dir / ('shapes_' + ph + '.csv')
        argv_bak = sys.argv
        sys.argv = ['gen_lut_grid.py', '--phase', ph, '--status', 'all', '--out', str(out)]
        try:
            glg.main()
        except SystemExit:
            pass
        finally:
            sys.argv = argv_bak
        if out.exists():
            paths.append(out)
    return paths


def run_measure(args):
    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)

    sweep = Path(args.sweep)
    if not sweep.exists():
        sys.exit('Sweep executable not found: ' + str(sweep) +
                 '\nBuild it first: cmake --build <build-dir> --target gqa_autotune_sweep')

    # Use RdpCapture gen_lut_grid if available, else fall back to built-in geometries
    if args.rdpcapture:
        tools = Path(args.rdpcapture) / 'ops_analyze' / 'gqa' / 'tools'
        sys.path.insert(0, str(tools))
        shape_files = _gen_shapes(data_dir, args.phase)
    else:
        shape_files = _write_builtin_shapes(data_dir, args.phase,
                                            args.head_dim, args.geometry)

    for shapes_csv in shape_files:
        tag = shapes_csv.stem
        out_csv = data_dir / (tag + '_results.csv')
        best_csv = data_dir / (tag + '_best.csv')
        cmd = [str(sweep),
               '--shapes', str(shapes_csv),
               '--csv', str(out_csv),
               '--best-csv', str(best_csv),
               '--target-ms', str(args.target_ms),
               '--rounds', str(args.rounds)]
        print('[measure] ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)

    print('[measure] done -- results in ' + str(data_dir))


# ---------------------------------------------------------------------------
# Coverage grid: the shapes to measure. One place defines what the sweep runs
# and what the plan manifest lists.
#
#   decode  : every model geometry x a length ladder (octave boundaries plus
#             interior points, since a generation loop's seq_kv is essentially
#             never a power of two -- the nearest-neighbour metric then answers
#             any length between two measured ones), batch 1, window folded into
#             the scanned length so no separate windowed rows are needed.
#   prefill : every model geometry x chunk size (seq_q) x a short and a long KV.
#             The config is largely seq_kv-independent above the short end, so
#             two KV points per (geometry, seq_q) bracket it.
# ---------------------------------------------------------------------------
SHAPE_FIELDS = ['id', 'group', 'phase', 'B', 'H', 'G', 'd', 'sq', 'skv',
                'max_seq', 'window', 'sink', 'grid_role', 'note']

DECODE_BOUNDARIES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
INTERIOR_FRACTIONS = [1.03, 1.25, 1.5, 1.75]
PREFILL_SQ = [128, 256, 512, 1024, 2048, 4096]
PREFILL_SKV = [128, 65536]

VARIANT_OF_HEAD_DIM = {64: 'PrefillV6x', 96: 'PrefillV6x', 128: 'PrefillV6x',
                       256: 'PrefillV8x'}


def _interior(lo, hi):
    out = []
    for fr in INTERIOR_FRACTIONS:
        v = int(round(lo * fr))
        if lo < v < hi:
            out.append(v)
    return sorted(set(out))


def _decode_lengths():
    lens = set(DECODE_BOUNDARIES)
    for lo, hi in zip(DECODE_BOUNDARIES, DECODE_BOUNDARIES[1:]):
        lens.update(_interior(lo, hi))
    return sorted(lens)


def _geometries(head_dims=None, only=None):
    """GEOMETRIES, restricted to `head_dims` and to the (H, G, d) triples in
    `only` when given (an incremental sweep)."""
    out = GEOMETRIES
    if head_dims:
        out = [g for g in out if g[2] in head_dims]
    if only:
        out = [g for g in out if tuple(g[:3]) in only]
    return out


def _grid_shapes(head_dims=None, only=None):
    """Canonical list of shapes to measure, as sweep-CSV row dicts."""
    shapes = []
    lengths = _decode_lengths()
    geometries = _geometries(head_dims, only)
    di = 0
    for H, G, d, sink, group, _kernel in geometries:
        for skv in lengths:
            di += 1
            shapes.append(dict(
                id='D{:04d}'.format(di), group=group, phase='decode', B=1,
                H=H, G=G, d=d, sq=1, skv=skv, max_seq=skv, window=-1,
                sink=sink,
                grid_role='boundary' if skv in DECODE_BOUNDARIES else 'interior',
                note=''))
    pi = 0
    for H, G, d, sink, group, _kernel in geometries:
        for sq in PREFILL_SQ:
            for skv in PREFILL_SKV:
                if skv < sq:
                    continue
                pi += 1
                shapes.append(dict(
                    id='P{:04d}'.format(pi), group=group, phase='prefill', B=1,
                    H=H, G=G, d=d, sq=sq, skv=skv, max_seq=skv, window=-1,
                    sink=sink, grid_role='boundary', note=''))
    return shapes


def _shape_to_tbd_point(s):
    """A grid shape as a table point with the config still to be measured."""
    if s['phase'] == 'decode':
        phase, seq_q, seq_kv = 'Decode', 1, s['skv']
    else:
        phase = VARIANT_OF_HEAD_DIM[s['d']]
        seq_q, seq_kv = s['sq'], s['skv']
    return OrderedDict([
        ('phase', phase),
        ('head_dim', 'D{}'.format(s['d'])),
        ('kv_dtype', 'Fp16'),
        ('num_heads', s['H']),
        ('kv_num_heads', s['G']),
        ('batch', s['B']),
        ('seq_q', seq_q),
        ('seq_kv', seq_kv),
        ('config', 'TBD'),
        ('splits', 'TBD'),
    ])


def run_plan(args):
    """Write the shapes-to-measure manifest: every planned point with its config
    marked TBD. This is the coverage plan a sweep fills in; `build` produces the
    real table with configs resolved from the measurements."""
    shapes = _grid_shapes(args.head_dim, args.geometry)
    points = [_shape_to_tbd_point(s) for s in shapes]
    # De-dup on the point key (a geometry can repeat a length across groups).
    seen, uniq = set(), []
    for p in points:
        key = (p['phase'], p['head_dim'], p['num_heads'], p['kv_num_heads'],
               p['batch'], p['seq_q'], p['seq_kv'])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    uniq.sort(key=lambda p: (p['phase'], p['head_dim'], p['num_heads'],
                             p['kv_num_heads'], p['batch'], p['seq_q'],
                             p['seq_kv']))
    out = LUT_DIR / ('shapes_gfx' + args.arch + '.json')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write(' "note": "GQA shapes to measure. config/splits = TBD until a '
                'sweep fills them in; run update_lut.py measure then build.",\n')
        f.write(' "schema_version": {},\n'.format(SCHEMA_VERSION))
        f.write(' "gpu_arch": "gfx{}",\n'.format(args.arch))
        f.write(' "kernel_abi": "{}",\n'.format(KERNEL_ABI))
        f.write(' "points": [\n')
        f.write(',\n'.join('  ' + json.dumps(p, sort_keys=True) for p in uniq))
        f.write('\n ]\n}\n')
    n_dec = sum(1 for p in uniq if p['phase'] == 'Decode')
    print('[plan] {} shapes ({} decode, {} prefill) -> {}'.format(
        len(uniq), n_dec, len(uniq) - n_dec, out))


def _write_builtin_shapes(data_dir, phase, head_dims=None, only=None):
    """Write the coverage grid (decode + prefill) to sweep CSVs. A head_dim or
    geometry subset gets its own file names, so it lands beside a full sweep's
    results instead of overwriting them."""
    paths = []
    phases = ['decode', 'prefill'] if phase == 'both' else [phase]
    shapes = _grid_shapes(head_dims, only)
    tag = ''.join('_d{}'.format(d) for d in sorted(head_dims or []))
    tag += ''.join('_h{}g{}d{}'.format(*g) for g in sorted(only or []))
    for ph in phases:
        rows = [s for s in shapes if s['phase'] == ph]
        out = data_dir / ('shapes_' + ph + tag + '.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(SHAPE_FIELDS)
            for s in rows:
                w.writerow([s[k] for k in SHAPE_FIELDS])
        print('[measure] {}: {} shapes -> {}'.format(ph, len(rows), out))
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build: measured *_best.csv -> new-format lut/<arch>.json
#
# The table follows the matmul_nbits layout: a de-duplicated `configs` list plus
# `points` that reference a config by 1-byte index, so thousands of measured
# shapes cost one config byte each instead of inlining the knobs. A lookup
# matches the categorical key (phase, head_dim, kv_dtype) exactly, then takes the
# nearest measured point in log space over the numeric dims (num_heads,
# kv_num_heads, batch, seq_q, seq_kv); see gqa_autotune.cpp.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 9
KERNEL_ABI = 'gqa-v4'
ROCM_VERSION = 70151803

HEAD_DIM_CLASS = {64: 'D64', 96: 'D96', 128: 'D128', 256: 'D256'}


def _int(row, key, default=0):
    try:
        return int(row.get(key, ''))
    except (TypeError, ValueError):
        return default


def _config_and_splits(row):
    """(config-dict, splits) for a *_best.csv row, or (None, 0) if unusable.

    A config carries every knob field (the .fbs struct has no defaults), with the
    ones a variant does not use left at 0. Decode's split count is per-point, not
    part of the config, because the runtime clamps it per request.
    """
    kernel = row.get('kernel', '')
    if kernel == 'flash_decode':
        cfg = {'kind': 'Decode',
               'use_wmma': 1 if 'wmma' in row.get('cfg_impl', '').lower() else 0,
               'bkv': _int(row, 'cfg_BKV', 16) or 16,
               'hb': 0, 'qw': 0, 'mt': 0, 'nd': 0}
        return cfg, (_int(row, 'cfg_splits', 1) or 1)
    if kernel == 'prefill_v6x':
        return {'kind': 'Prefill', 'use_wmma': 0,
                'bkv': _int(row, 'cfg_BKV', 32) or 32,
                'hb': _int(row, 'cfg_HB', 1) or 1,
                'qw': _int(row, 'cfg_QW', 4) or 4,
                'mt': _int(row, 'cfg_MT', 1) or 1, 'nd': 0}, 0
    if kernel == 'prefill_v8x':
        return {'kind': 'Prefill', 'use_wmma': 0,
                'bkv': _int(row, 'cfg_BKV', 16) or 16,
                'hb': _int(row, 'cfg_HB', 1) or 1, 'qw': 0, 'mt': 0,
                'nd': _int(row, 'cfg_ND', 2) or 2}, 0
    return None, 0


# Deterministic, always-runnable last resort per (phase, head_dim). Decode uses
# scalar (valid for every geometry) with a split count the runtime clamps down.
# A prefill fallback serves any heads-per-group, so it runs one head per block
# (hb=1 divides every group): v6x 1x4 at M_TILES=1/BKV=32 (also the windowed
# instance), v8x 2 waves at BKV=16.
# D96 decode saturates on far fewer splits (its own heuristic in
# gqa_autotune.cpp), so its fallback starts at 8.
def _fallback_specs():
    return [
        ('Decode', 'D64', {'kind': 'Decode', 'use_wmma': 0, 'bkv': 16,
                           'hb': 0, 'qw': 0, 'mt': 0, 'nd': 0}, 32),
        ('Decode', 'D128', {'kind': 'Decode', 'use_wmma': 0, 'bkv': 16,
                            'hb': 0, 'qw': 0, 'mt': 0, 'nd': 0}, 32),
        ('Decode', 'D256', {'kind': 'Decode', 'use_wmma': 0, 'bkv': 16,
                            'hb': 0, 'qw': 0, 'mt': 0, 'nd': 0}, 32),
        ('Decode', 'D96', {'kind': 'Decode', 'use_wmma': 0, 'bkv': 16,
                           'hb': 0, 'qw': 0, 'mt': 0, 'nd': 0}, 8),
        ('PrefillV6x', 'D64', {'kind': 'Prefill', 'use_wmma': 0, 'bkv': 32,
                               'hb': 1, 'qw': 4, 'mt': 1, 'nd': 0}, 0),
        ('PrefillV6x', 'D128', {'kind': 'Prefill', 'use_wmma': 0, 'bkv': 32,
                                'hb': 1, 'qw': 4, 'mt': 1, 'nd': 0}, 0),
        ('PrefillV6x', 'D96', {'kind': 'Prefill', 'use_wmma': 0, 'bkv': 32,
                               'hb': 1, 'qw': 4, 'mt': 1, 'nd': 0}, 0),
        ('PrefillV8x', 'D256', {'kind': 'Prefill', 'use_wmma': 0, 'bkv': 16,
                                'hb': 1, 'qw': 0, 'mt': 0, 'nd': 2}, 0),
    ]


def _cfg_key(cfg):
    return (cfg['kind'], cfg['use_wmma'], cfg['bkv'], cfg['hb'], cfg['qw'],
            cfg['mt'], cfg['nd'])


_DIST_DIMS = ('num_heads', 'kv_num_heads', 'batch', 'seq_q', 'seq_kv')


def _dist2(a, b):
    """Squared weighted-log2 distance between two points over the numeric dims,
    mirroring the runtime metric with the shipped weights (all 1.0)."""
    s = 0.0
    for dim in _DIST_DIMS:
        s += math.log2(a[dim] / b[dim]) ** 2
    return s


def _nearest_answer(coord, pool):
    """(config, splits) of the nearest point to `coord` in `pool`, and whether the
    result is ambiguous (a differently-answered point sits at the same distance).
    Ambiguity is treated as unsafe so pruning never depends on the runtime's
    tie-break order."""
    eps = 1e-9
    best_d = None
    best_ans = None
    tied = False
    for q in pool:
        d = _dist2(coord, q)
        ans = (q['config'], q['splits'])
        if best_d is None or d < best_d - eps:
            best_d, best_ans, tied = d, ans, False
        elif abs(d - best_d) <= eps and ans != best_ans:
            tied = True
    return best_ans, tied


def _prune_saturated(points):
    """Drop points a nearest-neighbour lookup can reconstruct exactly.

    Within each exact-geometry group (phase, head_dim, kv_dtype, num_heads,
    kv_num_heads, batch) the swept axes are seq_q and seq_kv. A point is removed
    only when every original coordinate in the group still resolves to the same
    (config, splits) against the reduced set -- so the answer at every measured
    shape is unchanged, and a dropped coordinate is served by a neighbour that
    was measured to want the same config. This is the nearest-neighbour analogue
    of the old tier table's bucketing: a run of lengths that share a config
    collapses to its endpoints (e.g. once the decode split count saturates, or
    since a prefill config barely depends on seq_kv), while every boundary where
    the answer changes is kept.

    Returns (kept_points, n_removed).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for p in points:
        groups[(p['phase'], p['head_dim'], p['kv_dtype'],
                p['num_heads'], p['kv_num_heads'], p['batch'])].append(p)

    kept = []
    removed = 0
    for _, lst in groups.items():
        # Fixed reference of every original coordinate's required answer.
        required = [(p, (p['config'], p['splits'])) for p in lst]
        # Try interior points first (sorted by the swept axes) so runs collapse
        # to their endpoints.
        survivors = sorted(lst, key=lambda p: (p['seq_q'], p['seq_kv']))
        changed = True
        while changed:
            changed = False
            for p in list(survivors):
                if len(survivors) == 1:
                    break
                trial = [q for q in survivors if q is not p]
                if all(_nearest_answer(c, trial) == (ans, False)
                       for c, ans in required):
                    survivors = trial
                    removed += 1
                    changed = True
        kept.extend(survivors)
    return kept, removed


def _load_merge_base(out_json, arch):
    """The existing lut/<arch>.json a --merge-existing build extends."""
    if not Path(out_json).exists():
        sys.exit('[build] --merge-existing: {} does not exist'.format(out_json))
    with open(out_json, encoding='utf-8') as f:
        base = json.load(f)
    if (base.get('schema_version') != SCHEMA_VERSION or
            base.get('kernel_abi') != KERNEL_ABI or
            base.get('gpu_arch') != 'gfx' + arch):
        sys.exit('[build] --merge-existing: {} is schema {} / {} / {}, expected '
                 '{} / {} / gfx{}; rebuild it in full instead'.format(
                     out_json, base.get('schema_version'),
                     base.get('kernel_abi'), base.get('gpu_arch'),
                     SCHEMA_VERSION, KERNEL_ABI, arch))
    return base


def _build_from_csvs(data_dir, out_json, arch, merge_existing=False,
                     head_dims=None, only=None):
    """Build gfx<arch>.json (new nearest-neighbour layout) from *_best.csv.

    One point per measured shape, literal dims (no bucketing), configs
    de-duplicated and referenced by index. Then `_prune_saturated` drops points a
    nearest-neighbour lookup can reconstruct exactly (saturated length runs),
    which is answer-preserving on every measured shape and roughly halves the
    table. Outlier repair is not done here (the sweep's *_best.csv is already the
    per-shape winner).

    merge_existing adds the measured (phase, head_dim) groups to the current
    lut/<arch>.json instead of replacing the table, for an incremental sweep
    (`measure --head-dim` / `--phase`). Every other group, every existing config
    index and the table scalars are kept verbatim; a group present in both is
    replaced by the new measurements. Pruning is per geometry group, so the
    result equals a full rebuild over the union of the measurements. head_dims
    and `only` limit which *_best.csv rows are read, so older sweeps left in
    data_dir cannot replace a group by accident. With `only` (the --geometry
    (H, G, d) triples) the replaced group narrows to (phase, head_dim, H, G), so
    re-measuring one geometry keeps every other geometry at that head_dim.
    """
    base = _load_merge_base(out_json, arch) if merge_existing else None
    configs = []          # list of config dicts
    cfg_index = {}        # _cfg_key -> index

    def intern(cfg):
        k = _cfg_key(cfg)
        if k not in cfg_index:
            cfg_index[k] = len(configs)
            configs.append(cfg)
        return cfg_index[k]

    if base is not None:
        for cfg in base['configs']:
            intern(cfg)
    n_base_configs = len(configs)

    points = {}           # point-key -> point dict (dedup, last wins)
    n_rows = 0
    for csv_path in sorted(Path(data_dir).glob('*_best.csv')):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                cfg, splits = _config_and_splits(row)
                if cfg is None:
                    continue
                d = _int(row, 'd')
                if d not in HEAD_DIM_CLASS or (head_dims and d not in head_dims):
                    continue
                H, G, B = _int(row, 'H'), _int(row, 'G'), _int(row, 'B', 1) or 1
                if only and (H, G, d) not in only:
                    continue
                sq, skv = _int(row, 'sq', 1) or 1, _int(row, 'skv', 1) or 1
                window = _int(row, 'window', -1)
                phase = PHASE_OF_KERNEL.get(row.get('kernel', ''))
                if phase is None or H <= 0 or G <= 0:
                    continue
                if phase == 'Decode':
                    # The window folds into the scanned length: a windowed decode
                    # does the work of its effective length, which is what the
                    # runtime queries with.
                    seq_q = 1
                    seq_kv = min(skv, window) if window > 0 else skv
                else:
                    seq_q, seq_kv = sq, skv
                cfg_id = intern(cfg)
                key = (phase, d, H, G, B, seq_q, seq_kv)
                points[key] = {
                    'phase': phase, 'head_dim': HEAD_DIM_CLASS[d],
                    'kv_dtype': 'Fp16', 'config': cfg_id, 'splits': splits,
                    'num_heads': H, 'kv_num_heads': G, 'batch': B,
                    'seq_q': seq_q, 'seq_kv': seq_kv}
                n_rows += 1

    if not points:
        sys.exit('[build] no usable *_best.csv rows in ' + str(data_dir))

    fallbacks = []
    if base is not None:
        fallbacks = [dict(r) for r in base.get('fallbacks', [])]
    have_fallback = {(r['phase'], r['head_dim']) for r in fallbacks}
    for phase, hd, cfg, splits in _fallback_specs():
        if (phase, hd) in have_fallback:
            continue
        fallbacks.append({'phase': phase, 'head_dim': hd,
                          'config': intern(cfg), 'splits': splits})

    if base is not None:
        scalars = OrderedDict((k, v) for k, v in base.items()
                              if k not in ('configs', 'points', 'fallbacks'))
    else:
        scalars = OrderedDict([
            ('schema_version', SCHEMA_VERSION),
            ('gpu_arch', 'gfx' + arch),
            ('rocm_version', ROCM_VERSION),
            ('kernel_abi', KERNEL_ABI),
            ('model_key', 'update_lut/best-csv'),
            ('weight_num_heads', 1.0),
            ('weight_kv_num_heads', 1.0),
            ('weight_batch', 1.0),
            ('weight_seq_q', 1.0),
            ('weight_seq_kv', 1.0),
        ])
    # Head counts and batch are ubyte in the schema (packs the point to 16
    # bytes); fail loudly if a geometry ever exceeds that rather than let flatc
    # truncate silently.
    for p in points.values():
        for dim in ('num_heads', 'kv_num_heads', 'batch'):
            if p[dim] > 255:
                sys.exit('[build] {}={} exceeds the ubyte schema field; widen '
                         'GqaTunePoint.{} and bump schema_version'.format(
                             dim, p[dim], dim))

    # Drop points a nearest-neighbour lookup reconstructs exactly (saturated
    # runs), then garbage-collect configs that no surviving point/fallback uses
    # and renumber the indices so `configs` stays tight.
    n_before = len(points)
    kept, n_removed = _prune_saturated(list(points.values()))
    if base is not None:
        # The base table is already pruned against its own measurements;
        # pruning it again against fewer coordinates would drop more of it.
        def group(p):
            if only:
                return (p['phase'], p['head_dim'], p['num_heads'],
                        p['kv_num_heads'])
            return (p['phase'], p['head_dim'])
        rebuilt = {group(p) for p in kept}
        kept = [dict(p) for p in base['points']
                if group(p) not in rebuilt] + kept
    used = ({p['config'] for p in kept} | {r['config'] for r in fallbacks})
    # A merge keeps every base config where it was, so the base points read
    # the same in the merged table; only unused new configs are dropped.
    used = sorted(used | set(range(n_base_configs)))
    remap = {old: new for new, old in enumerate(used)}
    configs = [configs[old] for old in used]
    for p in kept:
        p['config'] = remap[p['config']]
    for r in fallbacks:
        r['config'] = remap[r['config']]
    print('[build] pruned {} saturated points ({} -> {}), {} configs'.format(
        n_removed, n_before, n_before - n_removed, len(configs)))

    point_rows = sorted(kept, key=lambda p: (
        p['phase'], p['head_dim'], p['num_heads'], p['kv_num_heads'],
        p['batch'], p['seq_q'], p['seq_kv']))

    # One entry per line (matmul_nbits style): compact enough to keep the file
    # reviewable and to make a regeneration diff readable, without the blowup of
    # indent-per-field.
    with open(out_json, 'w', encoding='utf-8') as f:
        f.write('{\n')
        for k in scalars:
            f.write(' "{}": {},\n'.format(k, json.dumps(scalars[k])))
        f.write(' "configs": [\n')
        f.write(',\n'.join('  ' + json.dumps(c, sort_keys=True) for c in configs))
        f.write('\n ],\n "points": [\n')
        f.write(',\n'.join('  ' + json.dumps(p, sort_keys=True)
                           for p in point_rows))
        f.write('\n ],\n "fallbacks": [\n')
        f.write(',\n'.join('  ' + json.dumps(r, sort_keys=True)
                           for r in fallbacks))
        f.write('\n ]\n}\n')
    print('[build] {} points ({} rows), {} configs, {} fallbacks -> {}'.format(
        len(point_rows), n_rows, len(configs), len(fallbacks), out_json))


def run_build(args):
    lut_json = LUT_DIR / ('gfx' + args.arch + '.json')
    if args.rdpcapture:
        tools = Path(args.rdpcapture) / 'ops_analyze' / 'gqa' / 'tools'
        cmd = [sys.executable, str(tools / 'build_lut.py'),
               '--store', '--prune-tolerance', str(args.prune_tolerance),
               '--fbs', str(FBS_FILE), '--arch', args.arch,
               '--json', str(lut_json)]
        print('[build] ' + ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=str(tools.parent))
    else:
        _build_from_csvs(Path(args.data), lut_json, args.arch,
                         merge_existing=args.merge_existing,
                         head_dims=args.head_dim, only=args.geometry)
    print('[build] wrote ' + str(lut_json))


def run_compile(args):
    lut_json = LUT_DIR / ('gfx' + args.arch + '.json')
    lut_fb = LUT_DIR / ('gfx' + args.arch + '.fb')
    with open(lut_json) as f:
        doc = json.load(f)
    print('[compile] schema_version={} configs={} points={}'.format(
        doc.get('schema_version'), len(doc.get('configs', [])),
        len(doc.get('points', []))))
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [args.flatc, '--binary', '--strict-json', '-o', tmp,
               str(FBS_FILE), str(lut_json)]
        print('[compile] ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)
        generated = list(Path(tmp).glob('*.bin'))
        if not generated:
            sys.exit('[compile] flatc produced no .bin file')
        import shutil
        shutil.copy(generated[0], lut_fb)
    print('[compile] wrote {} ({} KB)'.format(lut_fb, lut_fb.stat().st_size // 1024))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command',
                    choices=['plan', 'measure', 'build', 'compile', 'all'])
    ap.add_argument('--arch', default='1151', help='GPU arch suffix (default: 1151)')
    ap.add_argument('--data', default=str(HERE / 'data'),
                    help='Directory of *_best.csv (default: scripts/data/)')
    ap.add_argument('--sweep', default=str(SWEEP_EXE),
                    help='Path to gqa_autotune_sweep executable')
    ap.add_argument('--rdpcapture', default=None,
                    help='RdpCapture root. UNSUPPORTED: build_lut.py still emits '
                         'the old tier schema; use the standalone build instead.')
    ap.add_argument('--phase', choices=['decode', 'prefill', 'both'], default='both',
                    help='Which phase to measure (default: both)')
    ap.add_argument('--target-ms', type=float, default=40.0,
                    help='Per-candidate target time for sweep (ms)')
    ap.add_argument('--rounds', type=int, default=7,
                    help='Measurement rounds per shape')
    ap.add_argument('--prune-tolerance', type=float, default=1.02,
                    help='build_lut prune tolerance (--rdpcapture mode only)')
    ap.add_argument('--flatc', default='flatc',
                    help='flatc executable (for compile step)')
    ap.add_argument('--head-dim', type=lambda s: [int(x) for x in s.split(',')],
                    default=None,
                    help='plan/measure/build only these head dims, e.g. '
                         '64,128,256 (comma list); pair with build '
                         '--merge-existing')
    ap.add_argument('--geometry',
                    type=lambda s: {tuple(int(x) for x in g.split(':'))
                                    for g in s.split(',')},
                    default=None,
                    help='plan/measure/build only these H:G:d geometries, e.g. '
                         '24:4:256 (comma list); with build --merge-existing '
                         'only those geometries are replaced')
    ap.add_argument('--merge-existing', action='store_true',
                    help='build: add the measured (phase, head_dim) groups to '
                         'the current lut/gfx<arch>.json instead of replacing '
                         'the table')
    args = ap.parse_args()

    cmds = ['measure', 'build', 'compile'] if args.command == 'all' else [args.command]
    for cmd in cmds:
        print('\n=== {} ==='.format(cmd))
        {'plan': run_plan, 'measure': run_measure, 'build': run_build,
         'compile': run_compile}[cmd](args)


if __name__ == '__main__':
    main()
