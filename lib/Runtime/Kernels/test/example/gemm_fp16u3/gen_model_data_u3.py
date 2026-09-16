#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
Generate bits=3 vs bits=4 comparison test data for every shape defined in a
model config JSON. Mirrors ../gemm_fp16u4/gen_model_data.py but calls
gen_matmul_nbits_u3_data.py (which emits both uint3 and uint4 data per shape).

Usage:
    python gen_model_data_u3.py test_model/gpt_oss_20b.json --group-size 128
    python gen_model_data_u3.py test_model/gpt_oss_20b.json --group-size 128 --no-zeros
"""

import json
import subprocess
import sys
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Generate MatMulNBits bits=3 vs bits=4 test data for all model shapes')
    parser.add_argument('model_json', help='Path to model config JSON')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--no-zeros', action='store_true')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C (faster)')
    parser.add_argument('--out-dir', default='data_model',
                        help='Root output directory (default: data_model/)')
    args = parser.parse_args()

    with open(args.model_json) as f:
        cfg = json.load(f)

    M_array = cfg['M_array']
    K_array = cfg['KN_pairs']['K']
    N_array = cfg['KN_pairs']['N']

    if len(K_array) != len(N_array):
        print(f'ERROR: K ({len(K_array)}) and N ({len(N_array)}) arrays '
              f'must have the same length', file=sys.stderr)
        sys.exit(1)

    total = len(K_array) * len(M_array)
    print(f'Model: {args.model_json}')
    print(f'  {len(M_array)} M values x {len(K_array)} KN pairs = {total} shapes')
    print(f'  group_size={args.group_size}  '
          f'{"no-zeros" if args.no_zeros else "with-zeros"}')
    print()

    gen_script = os.path.join(os.path.dirname(__file__),
                              'gen_matmul_nbits_u3_data.py')
    idx = 0
    skipped = []

    for k, n in zip(K_array, N_array):
        for m in M_array:
            idx += 1
            shape = f'{m}x{k}x{n}'
            if k % 32 != 0:
                print(f'[{idx}/{total}] Skipping {shape}: K={k} not a '
                      f'multiple of 32')
                skipped.append(shape)
                continue
            out = os.path.join(args.out_dir, shape)
            print(f'[{idx}/{total}] Generating {shape} -> {out}/')

            cmd = [sys.executable, gen_script, shape,
                   '--group-size', str(args.group_size),
                   '--dir', out]
            if args.no_zeros:
                cmd.append('--no-zeros')
            if args.no_ref:
                cmd.append('--no-ref')

            subprocess.check_call(cmd)

    print(f'\nDone: {total - len(skipped)} shapes generated in {args.out_dir}/'
          f'{f" ({len(skipped)} skipped: K not %32)" if skipped else ""}')


if __name__ == '__main__':
    main()
