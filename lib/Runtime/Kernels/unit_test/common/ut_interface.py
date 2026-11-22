# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Read ut_interface.json, build kernel-UT commands from parameter values and run them.

  python ut_interface.py list
  python ut_interface.py show <leaf>
  python ut_interface.py command <leaf> [name=value ...]
  python ut_interface.py run <leaf> --bin <bin> [name=value ...]
  python ut_interface.py verify --bin <bin> [--leaf <leaf> ...]

<leaf> is the `leaf` field, e.g. gqa/prefill/Xfp16_Wfp16_Yfp16. Values are the
`name` fields of the leaf's custom.params. With no values, run/command use the
suite at --coverage. run and verify start hipdnn-kernel-ut.exe from <bin> (the
package's bin directory) with --leaf, so the leaf gets the package runtime,
environment and stderr capture, and read back bin/out/results.csv.
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = (HERE / 'ut_interface.json', HERE.parent / 'ut_interface.json')


def load(path=None):
    for candidate in ([Path(path)] if path else CANDIDATES):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding='utf-8'))
    raise SystemExit('ut_interface.json not found; pass --interface <path>')


def leaf_entry(doc, name):
    for leaf in doc['leaves']:
        if leaf['leaf'] == name:
            return leaf
    raise SystemExit(f'unknown leaf {name}; see "list"')


def convert(param, value):
    """Return the validated value; raise ValueError with a caller-facing message."""
    kind = param.get('type', 'string')
    name = param['name']
    if param.get('positional'):
        text = str(value)
        parts = text.split('x')
        if param.get('format') == 'MxKxN' and (len(parts) != 3 or not all(p.isdigit() and int(p) > 0 for p in parts)):
            raise ValueError(f'{name} must be MxKxN with positive integers, got {text!r}')
        return text
    if kind == 'bool':
        if isinstance(value, bool):
            return value
        if str(value).lower() in ('1', 'true', 'yes', 'on'):
            return True
        if str(value).lower() in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(f'{name} must be true or false, got {value!r}')
    if kind == 'int':
        try:
            number = int(str(value), 10)
        except ValueError:
            raise ValueError(f'{name} must be an integer, got {value!r}') from None
        value = number
    elif kind == 'float':
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f'{name} must be a number, got {value!r}') from None
        if math.isnan(number):
            raise ValueError(f'{name} must not be NaN')
        value = value if isinstance(value, str) else number
    else:
        value = str(value)
    if 'choices' in param and value not in param['choices'] and str(value) not in map(str, param['choices']):
        raise ValueError(f'{name} must be one of {param["choices"]}, got {value!r}')
    return value


def check_names(custom):
    """Names a check may use: every param, M/K/N of an MxKxN shape, and min."""
    names = {p['name'] for p in custom['params']} | {'min'}
    if any(p.get('format') == 'MxKxN' for p in custom['params']):
        names |= {'M', 'K', 'N'}
    return names


def check_constraints(custom, values):
    """Evaluate custom.checks on the given values, falling back to exe defaults.

    A check is skipped when a name it uses has neither a value nor a default."""
    env = {}
    for p in custom['params']:
        value = values.get(p['name'], p.get('exe_default'))
        if value is None:
            continue
        value = convert(p, value)
        if p.get('type') == 'float':
            value = float(value)
        env[p['name']] = value
        if p.get('format') == 'MxKxN':
            env['M'], env['K'], env['N'] = (int(x) for x in str(value).split('x'))
    for check in custom.get('checks', []):
        used = set(re.findall(r'[A-Za-z_]\w*', re.sub(r"'[^']*'", '', check))) - {'and', 'or', 'not', 'min'}
        if not used <= set(env):
            continue
        if not eval(check, {'__builtins__': {}, 'min': min}, dict(env)):  # noqa: S307 - expressions come from the checked-in JSON
            raise ValueError(f'constraint not met: {check}')


def leaf_arguments(leaf, values):
    """Leaf argv for a custom case, in params order; raise ValueError on bad input."""
    custom = leaf.get('custom')
    if not custom:
        raise ValueError(f'{leaf["leaf"]} has no custom-case interface (gaps: {", ".join(leaf.get("gaps", []))})')
    params = {p['name']: p for p in custom['params']}
    unknown = sorted(set(values) - set(params))
    if unknown:
        raise ValueError(f'unknown parameter(s) {unknown}; known: {list(params)}')
    missing = [n for n, p in params.items() if p.get('required') and n not in values]
    if missing:
        raise ValueError(f'missing required parameter(s) {missing}')
    check_constraints(custom, values)
    argv = []
    for name, param in params.items():
        if name not in values:
            continue
        value = convert(param, values[name])
        if param.get('positional'):
            argv.insert(0, value)
        elif param.get('type') == 'bool':
            if value:
                argv.append(param['flag'])
        else:
            argv += [param['flag'], str(value)]
    return argv


def make_command(leaf, values):
    custom = leaf.get('custom') or {}
    params = {p['name']: p for p in custom.get('params', [])}
    if custom.get('make_args'):
        return f'make test_custom HIP_SDK=<sdk> {custom["make_args"]}="{" ".join(leaf_arguments(leaf, values))}"'
    parts = []
    for name, value in values.items():
        param = params[name]
        if not param.get('make'):
            return None
        value = convert(param, value)
        if param.get('type') == 'bool':
            if value:
                parts.append(f'{param["make"]}=1')
        else:
            parts.append(f'{param["make"]}={value}')
    return 'make test_custom HIP_SDK=<sdk> ' + ' '.join(parts)


def controller_argv(leaf, values, coverage, timing, mode):
    argv = ['hipdnn-kernel-ut.exe', '--leaf', leaf['leaf'], '--timing', timing, '--mode', mode]
    if values:
        return argv + ['--'] + leaf_arguments(leaf, values)
    return argv + ['--coverage', str(coverage)]


def parse_values(items):
    values = {}
    for item in items:
        if '=' not in item:
            raise SystemExit(f'expected name=value, got {item!r}')
        name, value = item.split('=', 1)
        values[name] = value
    return values


def quote(arg):
    return f'"{arg}"' if (not arg or any(c in arg for c in ' \t"')) else arg


def run_case(bin_dir, leaf, values, coverage, timing, mode):
    bin_dir = Path(bin_dir).resolve()
    exe = bin_dir / 'hipdnn-kernel-ut.exe'
    if not exe.is_file():
        raise SystemExit(f'{exe} not found; --bin must be the package bin directory')
    argv = controller_argv(leaf, values, coverage, timing, mode)
    argv[0] = str(exe)
    proc = subprocess.run(argv, cwd=bin_dir, capture_output=True, text=True, errors='replace')
    op, name = leaf['leaf'].rsplit('/', 1)
    rows = []
    report = bin_dir / 'out' / 'results.csv'
    if report.is_file():
        with report.open(newline='', encoding='utf-8', errors='replace') as f:
            rows = [r for r in csv.DictReader(f) if r['op'] == op and r['leaf'] == name]
    return proc, rows


def cmd_list(doc, _args):
    for leaf in doc['leaves']:
        params = leaf['custom']['params'] if leaf.get('custom') else []
        inputs = ' '.join(('<' + p['name'] + '>' if p.get('positional') else p['name'] + ('*' if p.get('required') else ''))
                          for p in params) or '(no custom case; suite only)'
        print(f'{leaf["leaf"]:42s} {inputs}')
    print('\n* = required. "show <leaf>" for flags, defaults and constraints.')


def cmd_show(doc, args):
    leaf = leaf_entry(doc, args.leaf)
    caps = leaf['capabilities']
    print(f'{leaf["leaf"]}  ({leaf["package_executable"]})')
    print(f'  coverage={caps["coverage"]} timing={caps["timing"]} mode={caps["mode"]} rank={caps["rank"]}'
          + (f'  gaps={",".join(leaf["gaps"])}' if leaf.get('gaps') else ''))
    custom = leaf.get('custom')
    if custom:
        print(f'  custom case selected by: {custom["selected_by"]}')
        rows = [('name', 'flag', 'make', 'type', 'required', 'exe_default', 'make_default', 'description')]
        for p in custom['params']:
            kind = p.get('type', 'string') if not p.get('positional') else f'positional {p.get("format", "")}'
            if 'choices' in p:
                kind += ' ' + '|'.join(map(str, p['choices']))
            rows.append((p['name'], p.get('flag', '(positional)'), p.get('make') or custom.get('make_args') and '(in ARGS)' or '-',
                         kind, 'yes' if p.get('required') else '', str(p.get('exe_default', '')),
                         str(p.get('make_default', '')), p.get('description', '')))
        widths = [max(len(str(r[i])) for r in rows) for i in range(7)]
        for r in rows:
            print('  ' + '  '.join(str(r[i]).ljust(widths[i]) for i in range(7)) + '  ' + r[7])
        for constraint in custom.get('constraints', []):
            print(f'  constraint: {constraint}')
        for check in custom.get('checks', []):
            print(f'  check: {check}')
    for flag in leaf.get('diagnostic_flags', []):
        print(f'  diagnostic {flag["flag"]}: {flag["description"]}')
    for env in leaf.get('env', []):
        print(f'  env {env["name"]}: {env["description"]}')
    for example in leaf.get('examples', []):
        values = example.get('values', {})
        argv = controller_argv(leaf, values, example.get('coverage', 1), 'standard', 'lookup')
        print(f'  example {example["name"]}: ' + ' '.join(quote(a) for a in argv))
        if values and (make := make_command(leaf, values)):
            print(f'    make: {make}')


def cmd_command(doc, args):
    leaf = leaf_entry(doc, args.leaf)
    values = parse_values(args.values)
    try:
        argv = controller_argv(leaf, values, args.coverage, args.timing, args.mode)
        make = make_command(leaf, values) if values else f'make test HIP_SDK=<sdk> COVERAGE={args.coverage} TIMING={args.timing} MODE={args.mode}'
    except ValueError as error:
        raise SystemExit(f'error: {error}')
    print('package (cd <bin>): ' + ' '.join(quote(a) for a in argv))
    print('standalone (cd ' + leaf['leaf'] + '): ' + (make or 'not expressible through make (a parameter has no Make variable); use the package command'))


def cmd_run(doc, args):
    leaf = leaf_entry(doc, args.leaf)
    try:
        proc, rows = run_case(args.bin, leaf, parse_values(args.values), args.coverage, args.timing, args.mode)
    except ValueError as error:
        raise SystemExit(f'error: {error}')
    sys.stdout.write(proc.stdout[-4000:])
    for row in rows:
        print(f'{row["shape"]:50s} {row["config"][:48]:48s} {row["time_ms"]:>12s} {row["verdict"]}')
    return proc.returncode


def cmd_verify(doc, args):
    selected = [leaf for leaf in doc['leaves'] if not args.leaf or leaf['leaf'] in args.leaf]
    failures = 0
    for leaf in selected:
        for example in leaf.get('examples', []):
            values = example.get('values', {})
            proc, rows = run_case(args.bin, leaf, values, example.get('coverage', 1), args.timing, args.mode)
            cases = [r for r in rows if r['shape'] != 'suite']
            suite = [r for r in rows if r['shape'] == 'suite']
            expect_rows = 'results_csv' not in leaf.get('gaps', [])
            ok = (proc.returncode == 0 and suite and suite[-1]['verdict'] == 'PASS' and
                  all(r['verdict'] == 'PASS' for r in cases) and (cases or not expect_rows))
            failures += not ok
            detail = f'{len(cases)} case row(s)'
            if cases and values:
                detail += f', {cases[0]["shape"]}, {cases[0]["config"][:60]}'
            print(f'{"PASS" if ok else "FAIL"}  {leaf["leaf"]} {example["name"]}: rc={proc.returncode}, {detail}', flush=True)
            if not ok:
                print('  ' + ' '.join(quote(a) for a in proc.args))
                print('  ' + proc.stdout[-1500:].replace('\n', '\n  '))
                log = Path(args.bin) / 'out' / (Path(leaf['package_executable']).stem + '.stderr.log')
                if log.is_file():
                    print('  stderr: ' + log.read_text(errors='replace')[-1500:].replace('\n', '\n  '))
    print(f'{"PASS" if not failures else "FAIL"}: {failures} failing example(s)')
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--interface', help='path to ut_interface.json (default: next to this script or its parent)')
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    p = sub.add_parser('show'); p.add_argument('leaf')
    for name in ('command', 'run'):
        p = sub.add_parser(name)
        p.add_argument('leaf')
        p.add_argument('values', nargs='*', help='name=value')
        if name == 'run':
            p.add_argument('--bin', required=True)
        p.add_argument('--coverage', type=int, choices=[1, 2, 3], default=1)
        p.add_argument('--timing', choices=['screen', 'standard', 'confirm'], default='standard')
        p.add_argument('--mode', choices=['lookup', 'autotune'], default='lookup')
    p = sub.add_parser('verify')
    p.add_argument('--bin', required=True)
    p.add_argument('--leaf', nargs='+')
    p.add_argument('--timing', choices=['screen', 'standard', 'confirm'], default='screen')
    p.add_argument('--mode', choices=['lookup', 'autotune'], default='lookup')
    args = ap.parse_args(argv)
    doc = load(args.interface)
    return {'list': cmd_list, 'show': cmd_show, 'command': cmd_command, 'run': cmd_run,
            'verify': cmd_verify}[args.command](doc, args) or 0


if __name__ == '__main__':
    sys.exit(main())
