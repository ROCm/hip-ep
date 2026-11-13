# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""CSV/catalog/evidence plumbing for native UT ranking; contains no timer."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess

HEADER = ['leaf', 'shape', 'priority', 'config']


def private_path(path, installed):
    if not path:
        raise ValueError('rank commands require an explicit --out path')
    path = Path(path).resolve()
    if path == installed.resolve() or installed.resolve() in path.parents:
        raise ValueError('rank outputs must be outside the installed LUT directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError('expected leaf,shape,priority,config CSV')
        rows = list(reader)
    if not rows or any(not r['leaf'] or not r['shape'] or not r['config'] or
                       r['priority'] not in ('0', '1') for r in rows):
        raise ValueError('empty or malformed rank CSV')
    return rows


def records(paths, prefix):
    result = []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.startswith(prefix + ' '):
                result.append(json.loads(line[len(prefix) + 1:]))
    if not result:
        raise ValueError('no ' + prefix + ' records')
    keys = [(r['leaf'], r['shape']) for r in result]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate leaf/shape records')
    return result


def candidates(args, installed):
    seed = read_csv(args.candidates)
    catalog = {(r['leaf'], r['shape']): r for r in records(args.catalog, 'UT_CANDIDATES')}
    output = private_path(args.out, installed)
    seen = set()
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, HEADER)
        writer.writeheader()
        for row in seed:
            key = row['leaf'], row['shape']
            if key in seen:
                continue
            seen.add(key)
            writer.writerow({**row, 'config': 'default'})
            identity = set()
            for config in catalog[key]['configs']:
                if config['final'] not in identity:
                    identity.add(config['final'])
                    writer.writerow({**row, 'config': config['request']})
    print('candidate CSV:', output, 'shapes:', len(seen))


def reference_candidates(args, installed):
    catalog = {(r['leaf'], r['shape']): r for r in records(args.catalog, 'UT_CANDIDATES')}
    ranked = evidence(args.rank_log)
    output = private_path(args.out, installed)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, HEADER)
        writer.writeheader()
        for record in ranked:
            final_to_request = {c['final']: c['request'] for c in catalog[record['leaf'], record['shape']]['configs']}
            baseline = record['candidates'][0]['config']
            default = 'default'
            if record['leaf'].startswith('matmul_nbits/') and baseline in final_to_request:
                default += ';' + final_to_request[baseline]
            row = dict(leaf=record['leaf'], shape=record['shape'], priority='0')
            writer.writerow({**row, 'config': default})
            measurements = record['phases'][-1]['measurements']
            top = sorted(measurements, key=lambda m: m['versus_default']['median'])[:3]
            for candidate in top:
                index = candidate['candidate']
                if index:
                    identity = record['candidates'][index]['config']
                    writer.writerow({**row, 'config': final_to_request[identity]})
    print('independent reference CSV:', output)


def measure(args, installed):
    rows = read_csv(args.candidates)
    output = private_path(args.out, installed)
    # Existing Makefiles own compilation. A launcher may acquire/release the
    # owning host lease around each executable, independently of this tool.
    prefix = json.loads(args.launcher_json) if args.launcher_json else []
    with output.open('w', encoding='utf-8') as f:
        for leaf in dict.fromkeys(r['leaf'] for r in rows):
            if not leaf.startswith(args.op + '/') or '..' in Path(leaf).parts:
                raise ValueError('unexpected leaf ' + leaf)
            directory = Path(args.leaf_root) / leaf / args.build_dir
            executables = sorted(directory.glob('test*.exe')) + sorted(directory.glob('test_*'))
            executables = list(dict.fromkeys(p for p in executables if p.is_file() and p.suffix != '.obj'))
            if len(executables) != 1:
                raise ValueError('expected one native executable in ' + str(directory))
            env = os.environ.copy()
            env['HIPDNN_UT_RANK'] = str(Path(args.candidates).resolve())
            for name in ('HIPDNN_UT_RANK_CATALOG', 'HIPDNN_UT_RANK_REFERENCE'):
                env.pop(name, None)
            if args.command == 'rank-catalog':
                env['HIPDNN_UT_RANK_CATALOG'] = '1'
            if args.reference:
                env['HIPDNN_UT_RANK_REFERENCE'] = '1'
            proc = subprocess.run(prefix + [str(executables[0].resolve())],
                                  cwd=directory.parent, env=env, stdout=f, stderr=subprocess.STDOUT)
            f.flush()
            if proc.returncode:
                raise RuntimeError('native leaf failed: ' + leaf + ' rc=' + str(proc.returncode))
    print('native log:', output)


def evidence(paths):
    ranked = records(paths, 'UT_RANK')
    for r in ranked:
        if r.get('version') not in ('ut-rank-v1', 'ut-rank-v2', 'ut-rank-v3') or r.get('policy') != 'ut-adaptive-v3' or \
                r.get('sequential_reference') or r.get('diagnostic') or r.get('default') != 0:
            raise ValueError('unsupported ranking evidence')
        configs = r['candidates']
        if not configs or any(c.get('cpu_correct') is not True or
                              not math.isfinite(c['rel_l2']) for c in configs):
            raise ValueError('candidate CPU check absent or failed')
        if not r['phases'] or r['phases'][-1]['name'] not in ('standard', 'confirm'):
            raise ValueError('ranking final phase absent')
        for phase in r['phases']:
            excluded = phase.get('excluded', [])
            if 0 in excluded or set(excluded) & {m['candidate'] for m in phase['measurements']}:
                raise ValueError('default excluded or excluded candidate measured')
            for m in phase['measurements']:
                means, totals, calls = m['group_means_ms'], m['group_gpu_ms'], m['group_calls']
                if len(means) != 5 or len(totals) != 5 or len(calls) != 5 or \
                        any(c <= 0 or t <= 0 or not math.isfinite(t) or
                            not math.isclose(t / c, v, rel_tol=1e-10)
                            for v, t, c in zip(means, totals, calls)):
                    raise ValueError('invalid complete-call groups')
        final = r['phases'][-1]
        final_ids = [m['candidate'] for m in final['measurements']]
        if len(set(final_ids)) != len(final_ids) or 0 not in final_ids or r['winner'] not in final_ids:
            raise ValueError('missing default/winner or duplicate final candidate')
        for phase in r['phases']:
            by_id = {m['candidate']: m for m in phase['measurements']}
            if 0 not in by_id:
                raise ValueError('phase omits default')
            for m in phase['measurements']:
                if m['candidate'] < 0 or m['candidate'] >= len(configs):
                    raise ValueError('unknown phase candidate')
                ratios = [c / d for c, d in zip(m['group_means_ms'], by_id[0]['group_means_ms'])]
                observed = m['versus_default']
                if any(not math.isclose(a, b, rel_tol=1e-10) for a, b in zip(ratios, observed['groups'])) or \
                        not math.isclose(sorted(ratios)[2], observed['median'], rel_tol=1e-10):
                    raise ValueError('inconsistent paired ratios')
        final_ratio = next(m['versus_default']['median'] for m in final['measurements'] if m['candidate'] == r['winner'])
        if final_ratio > min(m['versus_default']['median'] for m in final['measurements']) + 1e-10:
            raise ValueError('winner is not the final ranked best')
        if not math.isclose(final_ratio, r['versus_default']['median'], rel_tol=1e-10):
            raise ValueError('final decision ratio differs from measurements')
        selected = r.get('selected', r['winner'])
        if r.get('version') == 'ut-rank-v2' and (
                not isinstance(r.get('uncertain'), bool) or selected != (0 if r['uncertain'] else r['winner']) or
                r.get('decision_status') != ('uncertain' if r['uncertain'] else 'resolved')):
            raise ValueError('inconsistent uncertainty fallback')
        if r.get('version') == 'ut-rank-v3':
            # v3 inserts the ranked winner whenever its median beats the
            # default by >2%, even when its order against the runner-up is
            # unresolved; uncertainty is diagnostic.
            insert = r['winner'] != 0 and r['versus_default']['median'] < .98
            if not isinstance(r.get('uncertain'), bool) or r['insert_nondefault'] != insert or \
                    selected != (r['winner'] if insert else 0) or \
                    r.get('decision_status') != ('winner_unresolved' if r['uncertain'] else 'resolved'):
                raise ValueError('inconsistent v3 entry decision')
        if not 0 <= r['winner'] < len(configs) or selected not in final_ids or \
                r['insert_nondefault'] != (selected != 0 and r['versus_default']['median'] < .98):
            raise ValueError('inconsistent entry decision')
    return ranked


def fields(text):
    out = {}
    for token in text.split('_'):
        n = next((i for i, c in enumerate(token) if c.isdigit()), len(token))
        out[token[:n]] = int(token[n:]) if n < len(token) else 1
    return out


def knobs(text):
    return dict(field.split('=', 1) for field in text.split(';'))


def add_arguments(ap):
    ap.add_argument('--candidates', help='native leaf/shape/priority/config CSV')
    ap.add_argument('--catalog', nargs='+', help='UT_CANDIDATES logs for rank-candidates')
    ap.add_argument('--rank-log', nargs='+', help='UT_RANK logs for rank-build')
    ap.add_argument('--base', help='base JSON; preserve weights, fallbacks and unrelated points')
    ap.add_argument('--leaf-root', help='existing unit_test directory containing built leaves')
    ap.add_argument('--build-dir', default='out_rank')
    ap.add_argument('--launcher-json', help='JSON argv prefix; host lease wrapper around each native process')
    ap.add_argument('--reference', action='store_true', help='independent sequential confirm')


def dispatch(args, installed, build):
    if args.command in ('rank-seed', 'rank-report'):
        import ut_lut_from_table
        action = ut_lut_from_table.seed if args.command == 'rank-seed' else ut_lut_from_table.report
        action(args, installed)
    elif args.command == 'rank-candidates':
        candidates(args, installed)
    elif args.command == 'rank-reference-candidates':
        reference_candidates(args, installed)
    elif args.command in ('rank-catalog', 'rank-measure'):
        measure(args, installed)
    elif args.command == 'rank-build':
        build(args)
    else:
        raise ValueError('unknown rank command')
