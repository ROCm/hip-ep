#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Update the GQA autotune LUT from fresh measurements.

Three steps, run individually or all at once:

    python update_lut.py measure          # sweep GPU, write CSVs to --data
    python update_lut.py build            # CSVs -> gfx1151.json
    python update_lut.py compile          # gfx1151.json -> gfx1151.fb (needs flatc)
    python update_lut.py all              # measure + build + compile

The sweep executable must be built first:
    cmake --build <build-dir> --target gqa_autotune_sweep

Measurement data is written to --data (default: ./scripts/data/).
The LUT files live in ./lut/ relative to this script's parent directory.

Optional: point --rdpcapture at RdpCapture/ops_analyze/gqa to use the full
measurement store (deduplication, outlier repair, prune-tolerance tuning).
Without it, build uses only the CSVs written by the latest measure run.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict, OrderedDict
from pathlib import Path

HERE = Path(__file__).parent
LUT_DIR = HERE.parent / 'lut'
FBS_FILE = HERE.parent / 'gqa_autotune.fbs'
SWEEP_EXE = HERE.parent.parent.parent.parent.parent / 'test' / 'example' / 'gqa' / 'autotune' / 'build' / 'gqa_autotune_sweep.exe'

SCHEMA_VERSION = 7  # bumped: prefill Length/HeadGroup/ExactHeadGroup rows now use seq_kv=Any

# All measured geometries: (H, G, d, sink, group, prefill_kernel)
# Every entry becomes a row in the decode LUT; entries with a prefill_kernel
# also drive prefill measurement.
GEOMETRIES = [
    # ---- primary models -------------------------------------------------------
    (64,  8,  64,  1, 'gpt-oss',           'prefill_v5'),
    (32,  8, 128,  0, 'llama-mistral',      'prefill_v7'),   # Llama-3.1-8B, Mistral-7B
    (40, 10, 128,  0, 'phi4',               'prefill_v7'),   # phi-4
    (40,  8, 128,  0, 'qwen2.5-14b',        'prefill_v7'),   # Qwen2.5-14B
    (64,  8, 128,  0, 'deepseek-70b',       'prefill_v7'),   # DeepSeek-R1-Distill-70B
    ( 8,  4, 256,  0, 'gemma3-4b',          'prefill_v8'),   # gemma3-4b
    (16,  4, 256,  0, 'qwen3.5-9b',         'prefill_v8'),   # Qwen3.5-9B
    (16,  2, 256,  0, 'qwen3.5-35b',        'prefill_v8'),   # Qwen3.5/3.6-35B-A3B
    (24,  4, 256,  0, 'qwen3.6-27b',        'prefill_v8'),   # Qwen3.6-27B, Qwen3.8-27B
    (24,  6, 256,  0, 'qwen3.6-27b-hpg4',   'prefill_v8'),
    (24,  3, 256,  0, 'qwen3.6-27b-alt',    'prefill_v8'),
    # ---- fill: close the (head_dim, HpG) matrix ------------------------------
    (32, 32,  64,  0, 'mha-d64',            'prefill_v5'),
    (16,  8,  64,  0, 'hpg2-d64',           'prefill_v5'),
    (24,  8,  64,  0, 'hpg3-d64',           'prefill_v5'),
    (32,  8,  64,  0, 'llama3.2-1b',        'prefill_v5'),
    (40,  8,  64,  0, 'hpg5-d64',           'prefill_v5'),
    (32,  2,  64,  0, 'hpg16-d64',          'prefill_v5'),
    (32, 32, 128,  0, 'llama2-7b',          'prefill_v7'),
    (16,  8, 128,  0, 'qwen3-1.7b',         'prefill_v7'),
    (24,  8, 128,  0, 'llama3.2-3b',        'prefill_v7'),
    (32,  2, 128,  0, 'glm4-9b',            'prefill_v7'),
    (16, 16, 256,  0, 'mha-d256',           'prefill_v8'),
    (24,  8, 256,  0, 'hpg3-d256',          'prefill_v8'),
    (40,  8, 256,  0, 'hpg5-d256',          'prefill_v8'),
    (32,  2, 256,  0, 'hpg16-d256',         'prefill_v8'),
    # ---- low-parallelism anchor (each HpG x head_dim pair) -------------------
    ( 8,  8,  64,  0, 'hold-1-d64',         'prefill_v5'),
    ( 8,  4,  64,  0, 'hold-2-d64',         'prefill_v5'),
    (12,  4,  64,  0, 'hold-3-d64',         'prefill_v5'),
    (16,  4,  64,  0, 'hold-4-d64',         'prefill_v5'),
    (20,  4,  64,  0, 'hold-5-d64',         'prefill_v5'),
    ( 8,  1,  64,  0, 'hold-8-d64-mqa',     'prefill_v5'),
    (16,  1,  64,  0, 'hold-16-d64-mqa',    'prefill_v5'),
    (16, 16, 128,  0, 'hold-1-d128',        'prefill_v7'),
    (64, 32, 128,  0, 'hold-2-d128',        'prefill_v7'),
    (12,  4, 128,  0, 'hold-3-d128',        'prefill_v7'),
    ( 8,  2, 128,  0, 'hold-4-d128',        'prefill_v7'),
    (20,  4, 128,  0, 'hold-5-d128',        'prefill_v7'),
    (128,16, 128,  0, 'hold-8-d128',        'prefill_v7'),
    (16,  1, 128,  0, 'hold-16-d128-mqa',   'prefill_v7'),
    ( 8,  8, 256,  0, 'hold-1-d256',        'prefill_v8'),
    (32, 16, 256,  0, 'hold-2-d256',        'prefill_v8'),
    (12,  4, 256,  0, 'falcon3-7b',         'prefill_v8'),
    ( 8,  2, 256,  0, 'hold-4-d256',        'prefill_v8'),
    (20,  4, 256,  0, 'hold-5-d256',        'prefill_v8'),
    (48,  6, 256,  0, 'hold-8-d256',        'prefill_v8'),
    (64,  4, 256,  0, 'hold-16-d256',       'prefill_v8'),
    # ---- floor and holdout (second head count per HpG) -----------------------
    (16, 16,  64,  0, 'hold2-1-d64',        'prefill_v5'),
    (32, 16,  64,  0, 'hold2-2-d64',        'prefill_v5'),
    (32,  4,  64,  0, 'hold2-8-d64',        'prefill_v5'),
    ( 8,  4, 128,  0, 'hold2-2-d128',       'prefill_v7'),
    (32,  4, 128,  0, 'hold2-8-d128',       'prefill_v7'),
    (10,  2, 128,  0, 'hold2-5-d128',       'prefill_v7'),
    (64, 16, 128,  0, 'hold2-4-d128',       'prefill_v7'),
    (48, 16, 256,  0, 'hold2-3-d256',       'prefill_v8'),
    (48, 12, 256,  0, 'hold2-4-d256',       'prefill_v8'),
    (24, 24, 256,  0, 'hold2-1-d256',       'prefill_v8'),
    # ---- second head count at already-covered (head_dim, HpG) ---------------
    (16,  4, 128,  0, 'hpg4-lowpar-d128',   'prefill_v7'),
    (40, 40, 128,  0, 'llama2-13b',         'prefill_v7'),
    (32, 16, 128,  0, 'gemma2-27b',         'prefill_v7'),
    (16,  8, 256,  0, 'gemma2-9b',          'prefill_v8'),
    (64,  4, 128,  0, 'hpg16-highpar',      'prefill_v7'),
]

# (H, G) pairs with an ExactHeadGroup enum entry -- mirrors GqaHeadCountClass in .fbs
HEAD_COUNT_CLASS = {
    ( 8,  4): 'H8_G4',
    (16,  2): 'H16_G2',
    (16,  4): 'H16_G4',
    (24,  4): 'H24_G4',
    (32,  8): 'H32_G8',
    (32, 32): 'H32_G32',
    (40,  8): 'H40_G8',
    (40, 10): 'H40_G10',
    (64,  8): 'H64_G8',
}

PHASE_OF_KERNEL = {
    'flash_decode': 'Decode',
    'prefill_v5': 'PrefillV5',
    'prefill_v7': 'PrefillV7',
    'prefill_v8': 'PrefillV8',
}

SEQ_BUCKETS = [1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,
               768,1024,1280,1536,1792,2048,2560,3072,3584,4096,
               5120,6144,7168,8192,10240,12288,14336,16384,
               20480,24576,28672,32768,40960,49152,57344,65536]


def seq_bucket(length):
    for label in SEQ_BUCKETS:
        if length <= label:
            return 'S' + str(label)
    return 'S' + str(SEQ_BUCKETS[-1])


def head_dim_class(d):
    return 'D' + str(d) if d in (64, 128, 256) else 'Any'


def par_class(batch, heads):
    work = max(1, batch) * max(1, heads)
    for p in [1,2,4,8,16,32,64,128,256,512,1024]:
        if work <= p:
            return 'P' + str(p)
    return 'P1024'


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
        shape_files = _write_builtin_shapes(data_dir, args.phase)

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


def _write_builtin_shapes(data_dir, phase):
    """Fallback: write simple decode shape grid from GEOMETRIES."""
    BOUNDARIES = [128,256,512,1024,2048,4096,8192,16384,32768,65536]
    paths = []
    for ph in (['decode', 'prefill'] if phase == 'both' else [phase]):
        if ph != 'decode':
            print('[measure] prefill shape generation requires --rdpcapture; skipping')
            continue
        out = data_dir / 'shapes_decode.csv'
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id','group','phase','B','H','G','d','sq','skv',
                        'max_seq','window','sink','grid_role','note'])
            idx = 1
            for H, G, d, sink, group, _ in GEOMETRIES:
                for skv in BOUNDARIES:
                    w.writerow(['D{:04d}'.format(idx), group, 'decode', 1,
                                H, G, d, 1, skv, skv, -1, sink, 'boundary', ''])
                    idx += 1
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def run_build(args):
    data_dir = Path(args.data)

    if args.rdpcapture:
        # Delegate to the full build_lut.py pipeline (measurement store, pruning, etc.)
        tools = Path(args.rdpcapture) / 'ops_analyze' / 'gqa' / 'tools'
        lut_json = LUT_DIR / ('gfx' + args.arch + '.json')
        cmd = [sys.executable, str(tools / 'build_lut.py'),
               '--store',
               '--prune-tolerance', str(args.prune_tolerance),
               '--fbs', str(FBS_FILE),
               '--arch', args.arch,
               '--json', str(lut_json)]
        print('[build] ' + ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=str(tools.parent))
    else:
        # Standalone: read *_best.csv files from data_dir and build JSON directly
        lut_json = LUT_DIR / ('gfx' + args.arch + '.json')
        _build_from_csvs(data_dir, lut_json, args.arch)

    print('[build] wrote ' + str(lut_json))


def _build_from_csvs(data_dir, out_json, arch):
    """Build gfx<arch>.json directly from *_best.csv measurement files.

    Groups readings by their LUT key, picks the best config per group,
    and emits Geometry + ExactHeadGroup + HeadGroup + Length + Fallback rows.
    No pruning or outlier repair -- use --rdpcapture mode for that.
    """
    # Load all *_best.csv
    readings = []
    for csv_path in sorted(data_dir.glob('*_best.csv')):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    readings.append({
                        'phase': row['phase'],
                        'kernel': row['kernel'],
                        'B': int(row['B']),
                        'H': int(row['H']),
                        'G': int(row['G']),
                        'd': int(row['d']),
                        'sq': int(row['sq']),
                        'skv': int(row['skv']),
                        'window': int(row['window']),
                        'sink': int(row['sink']),
                        'config': row['best_config'],
                    })
                except (KeyError, ValueError):
                    pass

    if not readings:
        sys.exit('No *_best.csv files found in ' + str(data_dir))

    # Emit rows for each tier
    rows = []
    _emit_fallback(rows)
    _emit_length(readings, rows)
    _emit_head_group(readings, rows)
    _emit_exact_head_group(readings, rows)
    _emit_geometry(readings, rows)

    doc = OrderedDict([
        ('schema_version', SCHEMA_VERSION),
        ('gpu_arch', 'gfx' + arch),
        ('rocm_version', 70151803),
        ('kernel_abi', 'gqa-v2'),
        ('model_key', 'update_lut/direct'),
        ('rows', rows),
    ])
    with open(out_json, 'w') as f:
        json.dump(doc, f, indent=1)
        f.write('\n')


def _config_parts(config_str):
    """Split 'scalar_SPLITS32' -> ('Scalar', 32) etc."""
    s = config_str.lower()
    if '_splits' in s:
        impl, _, splits = s.partition('_splits')
        name = {'scalar': 'Scalar', 'wmma': 'Wmma',
                'wmma_bkv16': 'WmmaBkv16', 'wmma_bkv32': 'WmmaBkv32'}.get(impl, impl)
        return name, int(splits)
    # prefill configs
    name_map = {
        'mt1_bkv32': 'MT1_BKV32', 'mt2_bkv32': 'MT2_BKV32',
        'nw2_bkv32_mt1': 'NW2_BKV32_MT1', 'nw4_bkv32_mt1': 'NW4_BKV32_MT1',
        'nw2_bkv64_mt1': 'NW2_BKV64_MT1', 'nw4_bkv64_mt1': 'NW4_BKV64_MT1',
        'nw1_bkv32_mt1': 'NW1_BKV32_MT1', 'nw1_bkv64_mt1': 'NW1_BKV64_MT1',
        'nw1_bkv32_mt2': 'NW1_BKV32_MT2', 'nw2_bkv32_mt2': 'NW2_BKV32_MT2',
        'nw4_bkv32_mt2': 'NW4_BKV32_MT2', 'nw1_bkv64_mt2': 'NW1_BKV64_MT2',
        'nw2_bkv64_mt2': 'NW2_BKV64_MT2', 'nw4_bkv64_mt2': 'NW4_BKV64_MT2',
        'nd2_mt1_bkv16': 'ND2_MT1_BKV16', 'nd2_mt1_bkv32': 'ND2_MT1_BKV32',
        'nd2_mt1_bkv64': 'ND2_MT1_BKV64', 'nd2_mt2_bkv32': 'ND2_MT2_BKV32',
        'nd2_mt2_bkv64': 'ND2_MT2_BKV64',
        'nd4_mt1_bkv16': 'ND4_MT1_BKV16', 'nd4_mt1_bkv32': 'ND4_MT1_BKV32',
        'nd4_mt2_bkv32': 'ND4_MT2_BKV32',
        'nd8_mt1_bkv16': 'ND8_MT1_BKV16',
    }
    return name_map.get(s, config_str), 0


def _best_config(group):
    """Pick config seen most often (majority vote among best readings)."""
    counts = defaultdict(int)
    for r in group:
        counts[r['config']] += 1
    return max(counts, key=counts.__getitem__)


def _make_row(phase, tier, head_dim, hpg, head_count, par, batch,
              seq_q, seq_kv, window, config_str):
    name, splits = _config_parts(config_str)
    return OrderedDict([
        ('phase', phase), ('tier', tier), ('kv_dtype', 'Any'),
        ('head_dim', head_dim), ('hpg', hpg), ('head_count', head_count),
        ('par', par), ('batch', 'Any'), ('seq_q', seq_q), ('seq_kv', seq_kv),
        ('window', window), ('config', name), ('splits', splits),
    ])


def _emit_fallback(rows):
    for phase, d, cfg, splits in [
        ('Decode', 'D128', 'Scalar', 32), ('Decode', 'D256', 'Scalar', 32),
        ('Decode', 'D64', 'Scalar', 48),
        ('PrefillV5', 'D64', 'MT1_BKV32', 0),
        ('PrefillV7', 'D128', 'NW2_BKV32_MT1', 0),
        ('PrefillV8', 'D256', 'ND4_MT1_BKV32', 0),
    ]:
        rows.append(OrderedDict([
            ('phase', phase), ('tier', 'Fallback'), ('kv_dtype', 'Any'),
            ('head_dim', d), ('hpg', 0), ('head_count', 'Any'),
            ('par', 'Any'), ('batch', 'Any'), ('seq_q', 'Any'), ('seq_kv', 'Any'),
            ('window', 'Any'), ('config', cfg), ('splits', splits),
        ]))


def _emit_length(readings, rows):
    groups = defaultdict(list)
    for r in readings:
        is_decode = r['kernel'] == 'flash_decode'
        # decode: key on seq_kv; prefill: key on seq_q (config independent of seq_kv)
        sq_key  = 'Any'                      if is_decode else seq_bucket(r['sq'])
        skv_key = seq_bucket(r['skv'])       if is_decode else 'Any'
        key = (PHASE_OF_KERNEL.get(r['kernel'], r['kernel']),
               head_dim_class(r['d']),
               sq_key, skv_key,
               'NoWindow' if r['window'] <= 0 else 'Any')
        groups[key].append(r)
    seen = set()
    for key, group in groups.items():
        if key in seen:
            continue
        seen.add(key)
        phase, hd, sq, skv, win = key
        rows.append(_make_row(phase, 'Length', hd, 0, 'Any', 'Any', 'Any',
                              sq, skv, win, _best_config(group)))


def _emit_head_group(readings, rows):
    groups = defaultdict(list)
    for r in readings:
        if not r['G'] or r['H'] % r['G'] != 0:
            continue
        hpg = r['H'] // r['G']
        is_decode = r['kernel'] == 'flash_decode'
        sq_key  = 'Any'                if is_decode else seq_bucket(r['sq'])
        skv_key = seq_bucket(r['skv']) if is_decode else 'Any'
        key = (PHASE_OF_KERNEL.get(r['kernel'], r['kernel']),
               head_dim_class(r['d']), hpg,
               sq_key, skv_key,
               'NoWindow' if r['window'] <= 0 else 'Any')
        groups[key].append(r)
    seen = set()
    for key, group in groups.items():
        if key in seen:
            continue
        seen.add(key)
        phase, hd, hpg, sq, skv, win = key
        rows.append(_make_row(phase, 'HeadGroup', hd, hpg, 'Any', 'Any', 'Any',
                              sq, skv, win, _best_config(group)))


def _emit_exact_head_group(readings, rows):
    groups = defaultdict(list)
    for r in readings:
        hcc = HEAD_COUNT_CLASS.get((r['H'], r['G']))
        if not hcc:
            continue
        if r['H'] % r['G'] != 0:
            continue
        hpg = r['H'] // r['G']
        is_decode = r['kernel'] == 'flash_decode'
        sq_key  = 'Any'                if is_decode else seq_bucket(r['sq'])
        skv_key = seq_bucket(r['skv']) if is_decode else 'Any'
        key = (PHASE_OF_KERNEL.get(r['kernel'], r['kernel']),
               head_dim_class(r['d']), hpg, hcc,
               sq_key, skv_key,
               'NoWindow' if r['window'] <= 0 else 'Any')
        groups[key].append(r)
    seen = set()
    for key, group in groups.items():
        if key in seen:
            continue
        seen.add(key)
        phase, hd, hpg, hcc, sq, skv, win = key
        rows.append(_make_row(phase, 'ExactHeadGroup', hd, hpg, hcc, 'Any', 'Any',
                              sq, skv, win, _best_config(group)))


def _emit_geometry(readings, rows):
    groups = defaultdict(list)
    for r in readings:
        if not r['G'] or r['H'] % r['G'] != 0:
            continue
        hpg = r['H'] // r['G']
        par = par_class(r['B'], r['H'])
        is_decode = r['kernel'] == 'flash_decode'
        sq_key  = 'Any'                if is_decode else seq_bucket(r['sq'])
        skv_key = seq_bucket(r['skv']) if is_decode else 'Any'
        key = (PHASE_OF_KERNEL.get(r['kernel'], r['kernel']),
               head_dim_class(r['d']), hpg,
               par, sq_key, skv_key,
               'NoWindow' if r['window'] <= 0 else 'Any')
        groups[key].append(r)
    seen = set()
    for key, group in groups.items():
        if key in seen:
            continue
        seen.add(key)
        phase, hd, hpg, par, sq, skv, win = key
        rows.append(_make_row(phase, 'Geometry', hd, hpg, 'Any', par, 'Any',
                              sq, skv, win, _best_config(group)))


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def run_compile(args):
    lut_json = LUT_DIR / ('gfx' + args.arch + '.json')
    lut_fb = LUT_DIR / ('gfx' + args.arch + '.fb')

    with open(lut_json) as f:
        doc = json.load(f)
    print('[compile] schema_version={} rows={}'.format(
        doc.get('schema_version'), len(doc.get('rows', []))))

    flatc = args.flatc
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [flatc, '--binary', '--strict-json',
               '-o', tmp, str(FBS_FILE), str(lut_json)]
        print('[compile] ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)

        # flatc names the output after the root_type; find it
        generated = list(Path(tmp).glob('*.bin'))
        if not generated:
            sys.exit('[compile] flatc produced no .bin file')
        import shutil
        shutil.copy(generated[0], lut_fb)

    print('[compile] wrote ' + str(lut_fb) +
          ' ({} KB)'.format(lut_fb.stat().st_size // 1024))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['measure', 'build', 'compile', 'all'])
    ap.add_argument('--arch', default='1151', help='GPU arch suffix (default: 1151)')
    ap.add_argument('--data', default=str(HERE / 'data'),
                    help='Directory for measurement CSVs (default: scripts/data/)')
    ap.add_argument('--sweep', default=str(SWEEP_EXE),
                    help='Path to gqa_autotune_sweep executable')
    ap.add_argument('--rdpcapture', default=None,
                    help='Path to RdpCapture root. Enables full pipeline: '
                         'measurement store, outlier repair, prune-tolerance tuning.')
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
    args = ap.parse_args()

    cmds = ['measure', 'build', 'compile'] if args.command == 'all' else [args.command]
    for cmd in cmds:
        print('\n=== {} ==='.format(cmd))
        {'measure': run_measure, 'build': run_build, 'compile': run_compile}[cmd](args)


if __name__ == '__main__':
    main()
