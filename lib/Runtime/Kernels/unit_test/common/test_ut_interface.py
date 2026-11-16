# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Checks unit_test/ut_interface.json against the leaves it describes."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ut_interface as tool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAPS = {'custom_shape', 'coverage', 'shared_timer', 'results_csv', 'make_api', 'arch_default'}
# Leaves that predate the required-capability standard. Their gaps may shrink;
# no leaf may add a gap and a new leaf has none.
LEGACY_GAPS = {
    'gqa/ring/Xfp16_Wfp16_Yfp16': {'custom_shape', 'shared_timer', 'make_api'},
    'qmoe_ragged_mm/Xfp16_Wu4_Yfp16': {'custom_shape'},
    'qmoe_ragged_mm/Xfp16_Wu2_Yfp16': {'custom_shape', 'coverage', 'shared_timer', 'results_csv', 'make_api'},
    'gather_block_quantized/Xi64_Wu4_Yfp16': {'custom_shape', 'shared_timer', 'arch_default'},
    'gather_block_quantized/Xi64_Wu8_Yfp16': {'custom_shape', 'shared_timer', 'arch_default'},
    'gather_block_quantized/Xi64_Wfp8_Yfp16': {'custom_shape', 'shared_timer', 'arch_default'},
    'cast/Xany_Yany_bf16': {'custom_shape', 'shared_timer', 'arch_default'},
}


def check(doc):
    errors = []
    fail = errors.append
    if doc.get('schema') != 'hipdnn-kernel-ut-interface/v1':
        fail('unknown schema')
    controller = (ROOT / 'kernel_ut_main.cpp').read_text(encoding='utf-8')
    registered = {f'{op}/{name}': exe for op, name, exe in
                  re.findall(r'\{"([^"]+)",\s*"([^"]+)",\s*"([^"]+\.exe)"\}', controller)}
    for arg in doc['controller']['args']:
        if f'"{arg["flag"]}"' not in controller:
            fail(f'controller flag {arg["flag"]} not parsed by kernel_ut_main.cpp')
    cmake = (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
    packaged = {f'{exe}.exe': src for exe, src in
                re.findall(r'(hipdnn-kernel-ut-[a-z0-9-]+)\s+([\w/]+\.cpp)', cmake)}
    leaves = {leaf['leaf']: leaf for leaf in doc['leaves']}
    if len(leaves) != len(doc['leaves']):
        fail('duplicate leaf entry')
    if set(leaves) != set(registered):
        fail(f'leaves differ from kLeaves: missing {sorted(set(registered) - set(leaves))}, '
             f'unregistered {sorted(set(leaves) - set(registered))}')
    for name, leaf in leaves.items():
        where = f'{name}:'
        exe = leaf['package_executable']
        if registered.get(name) != exe:
            fail(f'{where} executable differs from kLeaves')
        if packaged.get(exe) != f'{name}/{leaf["source"]}':
            fail(f'{where} source differs from CMakeLists.txt')
        source_path = ROOT / name / leaf['source']
        if not source_path.is_file():
            fail(f'{where} missing source')
            continue
        source = source_path.read_text(encoding='utf-8')
        custom = leaf.get('custom')
        params = custom['params'] if custom else []
        flags = [p['flag'] for p in params if not p.get('positional')]
        flags += [f['flag'] for f in leaf.get('diagnostic_flags', [])]
        flags += leaf['suite']['args']
        for flag in flags:
            if f'"{flag}"' not in source:
                fail(f'{where} {flag} is not parsed by {leaf["source"]}')
        for env in leaf.get('env', []):
            if env['name'] not in source:
                fail(f'{where} {env["name"]} is not read by {leaf["source"]}')
        examples = leaf.get('examples', [])
        if not examples:
            fail(f'{where} no examples')
        for example in examples:
            if 'values' in example:
                try:
                    tool.leaf_arguments(leaf, example['values'])
                except ValueError as error:
                    fail(f'{where} example {example.get("name")}: {error}')
            elif example.get('coverage') not in (1, 2, 3):
                fail(f'{where} example {example.get("name")} has neither values nor coverage 1|2|3')
        if params and not any('values' in example for example in examples):
            fail(f'{where} no custom-case example')
        if custom:
            known = tool.check_names(custom)
            for check in custom.get('checks', []):
                names = set(re.findall(r'[A-Za-z_]\w*', re.sub(r"'[^']*'", '', check))) - {'and', 'or', 'not'}
                if not names <= known:
                    fail(f'{where} check uses unknown names {sorted(names - known)}: {check}')
                try:
                    compile(check, 'check', 'eval')
                except SyntaxError:
                    fail(f'{where} check is not an expression: {check}')
        caps = leaf['capabilities']
        if (caps['timing'] == 'shared') != ('adaptiveRotatedMeanMs' in source):
            fail(f'{where} timing capability does not match the source')
        if caps['rank'] != ('HIPDNN_UT_RANK' in source):
            fail(f'{where} rank capability does not match the source')
        gaps = set(leaf.get('gaps', []))
        if gaps - GAPS:
            fail(f'{where} unknown gaps {sorted(gaps - GAPS)}')
        if not gaps <= LEGACY_GAPS.get(name, set()):
            fail(f'{where} gaps {sorted(gaps - LEGACY_GAPS.get(name, set()))} are not allowed; '
                 'see unit_test/GUIDELINES.md "Required capabilities in every leaf"')
        if ('custom_shape' in gaps) != (not params):
            fail(f'{where} custom params and custom_shape gap disagree')
        if ('shared_timer' in gaps) != (caps['timing'] != 'shared'):
            fail(f'{where} timing capability and shared_timer gap disagree')
        if ('coverage' in gaps) != (not caps['coverage']):
            fail(f'{where} coverage capability and coverage gap disagree')
        makefile = ROOT / name / 'Makefile'
        if ('make_api' in gaps) != (not makefile.is_file()):
            fail(f'{where} Makefile presence and make_api gap disagree')
        if not makefile.is_file():
            continue
        make = makefile.read_text(encoding='utf-8')
        defined = set(re.findall(r'^([A-Z_]+)\s*\?=', make, re.M))
        for param in params:
            if param.get('make') and param['make'] not in defined:
                fail(f'{where} make variable {param["make"]} is not defined')
            if 'make_default' in param and param.get('make'):
                value = re.search(rf'^{param["make"]}[ \t]*\?=[ \t]*(.*?)[ \t]*$', make, re.M).group(1)
                if value != str(param['make_default']):
                    fail(f'{where} {param["make"]} default is {value!r}, not {param["make_default"]!r}')
        if custom and custom.get('make_args') and custom['make_args'] not in defined:
            fail(f'{where} make variable {custom["make_args"]} is not defined')
        if caps['timing'] == 'shared' and not re.search(r'^TIMING\s*\?=\s*standard\s*$', make, re.M):
            fail(f'{where} Makefile lacks TIMING ?= standard')
        if caps['rank'] and 'RANK' not in defined:
            fail(f'{where} Makefile lacks RANK')
        generic = re.search(r'^OFFLOAD\s*\?=\s*--offload-arch=gfx11-generic\s*$', make, re.M)
        if ('arch_default' in gaps) == bool(generic):
            fail(f'{where} OFFLOAD default and arch_default gap disagree')
    return errors


def main():
    doc = json.loads((ROOT / 'ut_interface.json').read_text(encoding='utf-8'))
    errors = check(doc)
    for error in errors:
        print(f'ut_interface.json: {error}', file=sys.stderr)
    if errors:
        return 1
    print(f'UT interface: {len(doc["leaves"])} leaves match kLeaves, CMake, Makefiles and sources')
    return 0


if __name__ == '__main__':
    sys.exit(main())
