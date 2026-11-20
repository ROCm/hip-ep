#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
from hipep_perf import bench, stats

LOG = """\
Batch size: 1, prompt tokens: 1973, tokens to generate: 128
Prompt processing (time to first token):
\tavg (us):       1717406.62
\tavg (tokens/s): 1148.83
\tp50 (us):       1708013.50
\tstddev (us):    26765.44
\tn:              4 * 1973 token(s)
Token generation:
\tavg (us):       23544.76
\tavg (tokens/s): 42.47
\tp50 (us):       23394.00
\tstddev (us):    577.03
\tn:              508 * 1 token(s)
Token sampling:
\tavg (us):       419.83
\tavg (tokens/s): 2381.94
\tp50 (us):       424.90
\tstddev (us):    18.56
\tn:              4 * 1 token(s)
"""


def test_blocks_do_not_bleed_into_each_other():
    lines = LOG.splitlines()
    pre = bench.parse_block(lines, bench.PREFILL_HEADER)
    dec = bench.parse_block(lines, bench.DECODE_HEADER)
    assert pre["avg_us"] == 1717406.62 and pre["n"] == 4
    # The sampler block follows with the same field names; it must not win.
    assert dec["avg_us"] == 23544.76 and dec["tps"] == 42.47 and dec["n"] == 508
    assert bench.parse_prompt_tokens(lines) == 1973


def test_scientific_notation():
    lines = [
        "Prompt processing (time to first token):",
        "\tavg (us):       1.00924e+07",
    ]
    assert bench.parse_block(lines, bench.PREFILL_HEADER)["avg_us"] == 1.00924e07


def test_t_crit_matches_published_table():
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        5: 2.571,
        10: 2.228,
        20: 2.086,
        30: 2.042,
        60: 2.000,
    }
    for df, want in table.items():
        assert abs(stats.t_crit(df) - want) < 1e-3, df


def test_paired_interval():
    base = [100.0, 101.0, 99.0, 100.5]
    arm = [98.0, 99.2, 97.1, 98.4]
    p = stats.paired(base, arm)
    assert p.n == 4 and p.mean_diff < 0 and not p.spans_zero
    same = stats.paired(base, [b + d for b, d in zip(base, (0.3, -0.2, 0.1, -0.25))])
    assert same.spans_zero
