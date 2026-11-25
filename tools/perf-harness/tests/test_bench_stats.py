#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import csv

import pytest

from hipep_perf import ab, bench, stats

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


def test_token_stats_skips_post_prefill_step_and_counts_stalls():
    # Two iterations of four steps; each starts with the slow post-prefill step.
    clean = [220.0, 70.0, 71.0, 72.0] * 2
    s = bench.token_stats(clean, reps=2)
    assert s["p10_ms"] == 70.0 and s["slow_pct"] == 0.0
    stalled = [220.0, 70.0, 200.0, 72.0, 220.0, 71.0, 70.0, 205.0]
    s = bench.token_stats(stalled, reps=2)
    assert s["slow_pct"] == pytest.approx(100.0 * 2 / 6)
    assert bench.token_stats([], reps=3) is None


def test_append_csv_widens_an_older_header(tmp_path):
    path = tmp_path / "tps_summary.csv"
    path.write_text("tag,tps\nab_main_r1,10.0\n")
    bench._append_csv(path, {"tag": "ab_main_r2", "tps": 11.0, "slow_pct": 3.5})
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == {"tag": "ab_main_r1", "tps": "10.0", "slow_pct": ""}
    assert rows[1] == {"tag": "ab_main_r2", "tps": "11.0", "slow_pct": "3.5"}


def _summary_csv(tmp_path, rows):
    path = tmp_path / "tps_summary.csv"
    for tag, p50, p10, slow in rows:
        bench._append_csv(
            path,
            {
                "tag": tag,
                "ms_per_token": p50 * (1 + slow / 50),
                "p50_ms": p50,
                "p10_ms": p10,
                "slow_pct": slow,
            },
        )
    return path


def test_summary_flags_stalled_rounds_and_compares_p10(tmp_path, capsys):
    path = _summary_csv(
        tmp_path,
        [
            ("ab_main_r1", 77.0, 76.0, 0.8),
            ("ab_pr_r1", 76.0, 75.0, 0.8),
            ("ab_main_r2", 77.2, 76.1, 40.0),
            ("ab_pr_r2", 76.1, 75.2, 0.8),
            ("ab_main_r3", 77.1, 76.0, 0.8),
            ("ab_pr_r3", 76.2, 75.1, 0.8),
        ],
    )
    res = ab.summarize(path, "main", "tps_p10")
    out = capsys.readouterr().out
    assert "<- stalls: 40% slow tokens" in out and "--drop 2" in out
    assert res[0]["arm"] == "pr" and res[0]["pct"] < 0 and not res[0]["spans_zero"]


def test_ab_run_stops_at_the_first_failed_run(tmp_path, monkeypatch):
    class Env:
        out_root = tmp_path
        bin = tmp_path

        def require(self, *names):
            pass

    calls = []

    def fake_run(env, spec):
        calls.append(spec.tag)
        return None

    monkeypatch.setattr(bench, "run", fake_run)
    monkeypatch.delenv("HIPEP_LOCK_OWNER", raising=False)
    arms = [ab.Arm("main", [], {"X": "0"}), ab.Arm("pr", [], {"X": "1"})]
    with pytest.raises(SystemExit, match="prime_main produced no result"):
        ab.run(Env(), arms, bench.BenchSpec(tag="", metric="tps"), rounds=2)
    assert calls == ["prime_main"]


def test_summary_p10_needs_per_token_times(tmp_path):
    path = tmp_path / "tps_summary.csv"
    for tag in ("ab_main_r1", "ab_pr_r1"):
        bench._append_csv(path, {"tag": tag, "ms_per_token": 30.0, "p10_ms": ""})
    with pytest.raises(SystemExit, match="tps_p50"):
        ab.summarize(path, "main", "tps_p10")
