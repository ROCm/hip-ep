# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Host-only JSON/CSV/native-input/import/report contract checks."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import ut_lut_from_table as table
import ut_rank_tools as rank

HERE = Path(__file__).resolve().parent
AUTOTUNE = HERE.parents[1] / 'hip/autotune'


def synthetic(point, singleton=False):
    old = point['old_config']
    values = table.config_values(old)
    field = 'swizzle' if 'route' in values else 'bkv'
    values[field] += 1
    default = old if singleton else table.config_text(values)
    configs = [default] if singleton else [default, old]
    winner = 0 if singleton else 1
    means = [[1.] * 5] if singleton else [[1.] * 5, [.9] * 5]
    measurements = [dict(candidate=i, group_means_ms=v, group_gpu_ms=v, group_calls=[1] * 5,
                         versus_default=dict(groups=v, median=v[0])) for i, v in enumerate(means)]
    return dict(version='ut-rank-v3', policy='ut-adaptive-v3', leaf=point['leaf'], shape=point['shape'], default=0,
                candidates=[dict(config=c, cpu_correct=True, rel_l2=0.) for c in configs],
                phases=[dict(name='standard', measurements=measurements, excluded=[])],
                winner=winner, selected=winner, insert_nondefault=bool(winner), uncertain=False,
                decision_status='resolved', versus_default=measurements[winner]['versus_default'], wall_ms=10.)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, op, *args, success=True):
        result = subprocess.run([os.environ.get('PYTHON', 'python3'), str(AUTOTUNE / op / 'scripts/update_lut.py'), *map(str, args)],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_full_existing_tables_roundtrip(self):
        for op in ('matmul_nbits', 'gqa', 'gemm'):
            source = AUTOTUNE / op / 'lut/gfx1151.json'
            doc = table.read_table(source)
            csv_path = self.root / (op + '.csv')
            self.cli(op, 'rank-seed', '--from-lut', source, '--out', csv_path)
            rows = rank.read_csv(csv_path)
            seed = json.loads(csv_path.with_suffix('.seed.json').read_text())
            points = seed['selected']
            self.assertEqual(len(points) + len(seed['skipped']), len(doc['points']))
            self.assertEqual({(r['leaf'], r['shape']) for r in rows}, {(p['leaf'], p['shape']) for p in points})
            by_id = {}
            for row in rows:
                by_id.setdefault((row['leaf'], row['shape']), []).append(row['config'])
            evidence_points = []
            for point in points:
                requests = by_id[point['leaf'], point['shape']]
                self.assertEqual(requests, ['default', point['old_config']])
                evidence_points.append(dict(point, old_config=requests[1]))
            log = self.root / (op + '.log')
            log.write_text(''.join('UT_RANK ' + json.dumps(synthetic(p, i % 7 == 0)) + '\n' for i, p in enumerate(evidence_points)))
            self.assertEqual(len(rank.evidence([log])), len(points))
            output = self.root / (op + '-rebuilt.json')
            self.cli(op, 'rank-build', '--base', source, '--rank-log', log, '--out', output)
            self.assertEqual(json.loads(output.read_text()), doc)
            print(f'ROUNDTRIP {op}: {len(points)} representable, {len(seed["skipped"])} skipped, full JSON equality PASS')

    def test_sampling_filters_and_keys(self):
        for op in ('matmul_nbits', 'gqa', 'gemm'):
            source = AUTOTUNE / op / 'lut/gfx1151.json'
            outputs = [self.root / (op + str(i) + '.csv') for i in range(2)]
            for out in outputs:
                self.cli(op, 'rank-seed', '--from-lut', source, '--sample', 20, '--seed', 111313, '--priority', '--out', out)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            rows = rank.read_csv(outputs[0])
            self.assertEqual(len({(r['leaf'], r['shape']) for r in rows}), 20)
            self.assertTrue(all(r['priority'] == '1' for r in rows))
            points = json.loads(outputs[0].with_suffix('.seed.json').read_text())['selected']
            keys = self.root / 'keys.txt'
            keys.write_text(points[0]['key'] + '\n')
            out = self.root / 'filtered.csv'
            self.cli(op, 'rank-seed', '--from-lut', source, '--keys', keys, '--leaf', points[0]['leaf'], '--out', out)
            self.assertEqual({r['shape'] for r in rank.read_csv(out)}, {points[0]['shape']})
            self.cli(op, 'rank-seed', '--from-lut', source, '--rank-phase', 'decode', '--dtype', 'fp16', '--sample', 4, '--out', out)
            selected = json.loads(out.with_suffix('.seed.json').read_text())['selected']
            self.assertTrue(all(p['point']['phase'] in ('GemvNt', 'GemvNn') if op == 'gemm' else p['point']['phase'] == 'Decode' for p in selected))

    def test_catalog_default_dedup_and_append(self):
        source = AUTOTUNE / 'gqa/lut/gfx1151.json'
        p = table.inventory('gqa', table.read_table(source))[0][0]
        keys = self.root / 'keys.txt'
        keys.write_text(p['key'] + '\n')
        log = self.root / 'catalog.log'
        log.write_text('UT_CANDIDATES ' + json.dumps(dict(leaf=p['leaf'], shape=p['shape'], default=p['old_config'],
            configs=[dict(request='id=1', final=p['old_config']), dict(request='id=2', final='other'), dict(request='id=3', final='other')])) + '\n')
        out = self.root / 'candidates.csv'
        self.cli('gqa', 'rank-seed', '--from-lut', source, '--keys', keys, '--with-catalog', '--catalog', log, '--out', out)
        self.assertEqual([r['config'] for r in rank.read_csv(out)], ['default', 'id=2'])

    def test_gemm_catalog_and_reference_aliases(self):
        source = AUTOTUNE / 'gemm/lut/gfx1151.json'
        point = table.inventory('gemm', table.read_table(source))[0][0]
        keys = self.root / 'keys.txt'
        keys.write_text(point['key'] + '\n')
        log = self.root / 'catalog.log'
        catalog = dict(leaf=point['leaf'], shape=point['shape'], default='canonical',
                       configs=[dict(request='id=0', final='canonical', geometry=point['old_config']),
                                dict(request='id=1', final='other', geometry='other')])
        log.write_text('UT_CANDIDATES ' + json.dumps(catalog) + '\n')
        output = self.root / 'seed.csv'
        self.cli('gemm', 'rank-seed', '--from-lut', source, '--keys', keys, '--with-catalog', '--catalog', log, '--out', output)
        self.assertEqual([r['config'] for r in rank.read_csv(output)], ['default', 'id=1'])
        evidence = self.root / 'rank.log'
        evidence.write_text('UT_RANK ' + json.dumps(synthetic(point)) + '\n')
        reference = self.root / 'reference.csv'
        self.cli('gemm', 'rank-reference-candidates', '--catalog', log, '--rank-log', evidence, '--out', reference)
        self.assertEqual([r['config'] for r in rank.read_csv(reference)], ['default', 'id=0'])

    def test_invalid_and_unsupported_inputs(self):
        source = AUTOTUNE / 'matmul_nbits/lut/gfx1151.json'
        self.cli('matmul_nbits', 'rank-seed', '--from-lut', source.with_suffix('.fb'), '--out', self.root / 'bad.csv', success=False)
        self.cli('matmul_nbits', 'rank-seed', '--from-lut', source, '--sample', 99999, '--out', self.root / 'bad.csv', success=False)
        doc = table.read_table(source)
        points, skipped = table.inventory('matmul_nbits', doc)
        self.assertEqual(len(skipped), 924)
        self.assertEqual({s['reason'] for s in skipped}, {'padded row stride is not expressible by the leaf',
                                                         'DecodeDp4a activation/entry is not covered by these leaves'})
        bad = copy.deepcopy(doc)
        bad['points'].append(copy.deepcopy(bad['points'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            table.inventory('matmul_nbits', bad)
        record = synthetic(points[0])
        record['phases'][0]['measurements'][1]['versus_default']['median'] = .8
        log = self.root / 'bad.log'
        log.write_text('UT_RANK ' + json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, 'paired ratios'):
            rank.evidence([log])

    def test_preserves_duplicate_config_index_and_fields(self):
        for op in ('matmul_nbits', 'gqa', 'gemm'):
            doc = table.read_table(AUTOTUNE / op / 'lut/gfx1151.json')
            p = copy.deepcopy(table.inventory(op, doc)[0][0]['point'])
            doc['configs'].append(copy.deepcopy(doc['configs'][p['config']]))
            p.update(config=len(doc['configs']) - 1, extra_annotation='retain')
            doc['points'] = [p]
            before = copy.deepcopy(doc)
            point = table.inventory(op, doc)[0][0]
            table.update_points(op, doc, [synthetic(point)])
            self.assertEqual(doc, before)

    def test_reports_native_decision_and_same_phase_ratios(self):
        source = AUTOTUNE / 'gqa/lut/gfx1151.json'
        point = table.inventory('gqa', table.read_table(source))[0][0]
        r = synthetic(point)
        r.update(uncertain=True, decision_status='winner_unresolved')
        log = self.root / 'rank.log'
        log.write_text('UT_RANK ' + json.dumps(r) + '\n')
        out = self.root / 'report.csv'
        self.cli('gqa', 'rank-report', '--from-lut', source, '--rank-log', log, '--out', out, '--md', self.root / 'report.md')
        with out.open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(float(row['winner_over_old']), 1.)
        self.assertEqual(float(row['old_over_default']), .9)
        self.assertEqual(row['decision_status'], 'winner_unresolved')
        self.assertEqual(row['changed'], 'False')
        # The old config is excluded before standard. Report screen comparisons
        # with an explicit phase instead of dividing unrelated stage medians.
        r['phases'].insert(0, dict(name='screen', measurements=copy.deepcopy(r['phases'][0]['measurements']), excluded=[]))
        r['phases'][-1].update(measurements=[r['phases'][-1]['measurements'][0]], excluded=[1])
        r.update(winner=0, selected=0, insert_nondefault=False, uncertain=False, decision_status='resolved', versus_default=dict(groups=[1.] * 5, median=1.))
        log.write_text('UT_RANK ' + json.dumps(r) + '\n')
        self.cli('gqa', 'rank-report', '--from-lut', source, '--rank-log', log, '--out', out)
        with out.open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['old_ratio_phase'], 'screen')
        self.assertEqual(row['old_status'], 'excluded')
        self.assertAlmostEqual(float(row['winner_over_old']), 1. / .9)

    def test_reports_new_exact_keys_without_old_ratios(self):
        for op in ('matmul_nbits', 'gqa', 'gemm'):
            original = table.read_table(AUTOTUNE / op / 'lut/gfx1151.json')
            point = table.inventory(op, original)[0][0]
            record = synthetic(point)
            source = self.root / (op + '-empty.json')
            empty = copy.deepcopy(original)
            empty['points'] = []
            source.write_text(json.dumps(empty))
            log = self.root / (op + '-addition.log')
            log.write_text('UT_RANK ' + json.dumps(record) + '\n')
            out = self.root / (op + '-addition.csv')
            self.cli(op, 'rank-report', '--from-lut', source, '--rank-log', log, '--out', out)
            with out.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['key'], point['key'])
            self.assertEqual(row['added'], 'True')
            self.assertEqual(row['old_status'], 'absent_input_table')
            for field in ('old_config', 'old_index', 'winner_over_old', 'old_over_default', 'old_ratio_phase'):
                self.assertEqual(row[field], '')
            self.assertEqual(float(row['winner_over_default']), .9)
            summary = json.loads(out.with_suffix('.summary.json').read_text())
            self.assertEqual(summary['added'], 1)
            self.assertEqual(summary['unmeasured'], 0)
            self.assertEqual(summary['distribution']['unavailable'], 1)
            self.assertEqual(table.read_table(source), empty)

    def test_native_geometry_resolution_and_default_dedup(self):
        source = self.root / 'inputs.cpp'
        source.write_text(r'''
#define HIPDNN_UT_RANK_GEMM_ADAPTER
#define HIPDNN_UT_RANK_NBITS_ADAPTER
#define HIPDNN_UT_RANK_GQA_ADAPTER
#include "ut_gemm_rank_input.h"
#include "ut_matmul_nbits_rank_input.h"
#include "ut_gqa_rank_input.h"
#include <cassert>
using namespace hipdnn_ep_test::timing;
extern "C" int hipdnn_ut_nbits_config_count(const UtNbitsShape*) { return 2; }
extern "C" int hipdnn_ut_nbits_config(const UtNbitsShape*, int id, UtNbitsConfig* c) {
  if (id > 1) return -1;
  *c = UtNbitsConfig{}; c->route=1; c->bm=16; c->bn=32; c->wt_m=2; c->wt_n=2; c->bk=8;
  c->id=id < 0 ? 0 : id; c->swizzle=c->id; return 0;
}
extern "C" int hipdnn_ut_nbits_select(const UtNbitsShape*, const UtNbitsConfig*) { return 0; }
extern "C" int hipdnn_ut_gqa_config_count(const UtGqaShape*) { return 2; }
extern "C" int hipdnn_ut_gqa_config(const UtGqaShape*, int id, UtGqaConfig* c) {
  if (id < 0 || id > 1) return -1;
  *c=UtGqaConfig{}; c->bkv=16; c->splits=id ? 8 : 10; return 0;
}
extern "C" int hipdnn_ut_gemm_config_count(const UtGemmShape*) { return 3; }
extern "C" int hipdnn_ut_gemm_config(const UtGemmShape*, int id, UtGemmConfig* c) {
  if (id > 2) return -1;
  const int selected = id < 0 ? 0 : id;
  *c = {selected, "geometry" + std::to_string(selected), selected == 2 ? "different" : "same"};
  return 0;
}
extern "C" int hipdnn_ut_gemm_select(const UtGemmShape*, const UtGemmConfig*) { return 0; }
int main() {
  const auto shape = gemmRankShape("M16_N32_K64_ta0_tb1_bias_crows16_ccols1", 3);
  assert(shape.m == 16 && shape.dtype == 3 && shape.beta == 1 && shape.crows == 16 && shape.ccols == 1);
  const auto grouped = gemmRankShape("M19_N128_K256_ta0_tb1_B8_ag1_bg4_sa8192_sb33792_sy2560", 0, 1);
  assert(grouped.output == 1 && grouped.batches == 8 && grouped.b_group == 4 &&
         grouped.stride_a == 8192 && grouped.stride_b == 33792 && grouped.stride_y == 2560);
  const auto zero_stride = gemmRankShape("M2_N256_K4096_ta0_tb0_B4_ag2_bg1_sa0", 3, 1);
  assert(zero_stride.a_group == 2 && zero_stride.stride_a == 0);
  for (const auto& bad_shape : {"M0_N32_K64_ta0_tb1", "M16_N32_K64_ta2_tb1", "M16_N32_K64_ta0_tb1_extra1",
      "M16_N32_K64_ta0_tb1_bias2", "M16_N32_K64_ta0_tb1_crows16", "M16_N32_K64_ta0_tb1_bias_crows2_ccols1"}) {
    bool bad = false; try { gemmRankShape(bad_shape, 0); } catch(const std::invalid_argument&) { bad = true; }
    assert(bad);
  }
  for (const auto& bad_shape : {"M19_N128_K256_ta0_tb1_B0", "M19_N128_K256_ta0_tb1_B8_bg3",
      "M19_N128_K256_ta0_tb1_B8_sy0", "M19_N128_K256_ta0_tb1_B65536"}) {
    bool bad=false; try {gemmRankShape(bad_shape,0,1);}catch(const std::invalid_argument&){bad=true;}
    assert(bad);
  }
  auto gemm = gemmRankOptions(shape, RankRow{"", {"default", "id=1", "id=2", "geometry2"}, false}, [] {return 0.;});
  assert(gemm.configs.size() == 2 && gemm.configs[0] == "geometry0" && gemm.configs[1] == "geometry2");
  assert(gemm.requests[0] == (std::vector<std::string>{"geometry0", "geometry1", "id=1"}));
  assert(gemm.requests[1] == (std::vector<std::string>{"geometry2", "id=2"}));
  assert(gemm.final_launches == (std::vector<std::string>{"same", "different"}));
  gemm.select(1); assert(gemm.validate(0) == 0.);
  bool gemm_bad = false; try { gemmRankOptions(shape, RankRow{"", {"id=1"}, false}, [] {return 0.;}); }
  catch(const std::invalid_argument&) {gemm_bad = true;} assert(gemm_bad);
  UtNbitsShape ns{16,32,256,4,128,0,2}; UtNbitsConfig nc;
  hipdnn_ut_nbits_config(&ns,-1,&nc);
  auto no=nbitsRankOptions(ns,RankRow{"",{"default",nbitsConfigString(nc)},false}, [] {return 0.;});
  assert(no.configs.size()==1); // old == runtime default
  nc.swizzle=1;
  assert(nbitsRequestedConfig(ns,nbitsConfigString(nc)).id==1);
  bool bad=false; try { nbitsRequestedConfig(ns,"route=wmma;bogus=1"); } catch(const std::invalid_argument&) {bad=true;} assert(bad);
  UtGqaShape gs{1,8,2,128,1,160,0,0,64}; UtGqaConfig gc;
  hipdnn_ut_gqa_config(&gs,0,&gc);
  auto go=gqaRankOptions(gs,RankRow{"",{"default",gqaConfigString(gc)},false}, [](const UtGqaConfig&) {}, [] {return 0.;});
  assert(go.configs.size()==1);
  gc.splits=17; assert(gqaRequestedConfig(gs,gqaConfigString(gc)).splits==17);
  gc.splits=65; bad=false; try {gqaRequestedConfig(gs,gqaConfigString(gc));} catch(const std::invalid_argument&) {bad=true;} assert(bad);
}
''')
        binary = self.root / 'inputs'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(HERE), str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    def test_gemm_alias_report_and_preservation(self):
        source = AUTOTUNE / 'gemm/lut/gfx1151.json'
        doc = table.read_table(source)
        point = table.inventory('gemm', doc)[0][0]
        record = synthetic(point)
        record['candidates'][1].update(config=record['candidates'][0]['config'] + ';alias=1',
                                       requests=[point['old_config']], final_launch='actual old')
        record['candidates'][0]['final_launch'] = 'actual default'
        log = self.root / 'aliases.log'
        log.write_text('UT_RANK ' + json.dumps(record) + '\n')
        self.assertEqual(len(rank.evidence([log])), 1)
        output = self.root / 'alias-built.json'
        self.cli('gemm', 'rank-build', '--base', source, '--rank-log', log, '--out', output)
        self.assertEqual(table.read_table(output), doc)
        report = self.root / 'alias-report.csv'
        self.cli('gemm', 'rank-report', '--from-lut', source, '--rank-log', log, '--out', report)
        with report.open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['changed'], 'False')
        self.assertEqual(row['old_status'], 'measured_final')
        self.assertEqual(float(row['winner_over_old']), 1.)
        self.assertEqual(float(row['old_over_default']), .9)
        record['candidates'][0]['requests'] = [point['old_config']]
        log.write_text('UT_RANK ' + json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            rank.evidence([log])

    def test_gemm_skipped_and_bf16_filter(self):
        source = AUTOTUNE / 'gemm/lut/gfx1151.json'
        doc = table.read_table(source)
        rows, skipped = table.inventory('gemm', doc)
        self.assertEqual((len(rows), len(skipped)), (1254, 0))
        for change, reason in ((dict(wts_dtype='F32'), 'mixed'), (dict(act_dtype='F64'), 'dtype'),
                               (dict(phase='Any'), 'phase'), (dict(m=2), 'phase')):
            modified = copy.deepcopy(doc)
            modified['points'] = [{**doc['points'][0], **change}]
            points, skipped = table.inventory('gemm', modified)
            self.assertFalse(points)
            self.assertIn(reason, skipped[0]['reason'])
        output = self.root / 'bf16.csv'
        self.cli('gemm', 'rank-seed', '--from-lut', source, '--dtype', 'bf16', '--sample', 20, '--seed', 111616, '--out', output)
        self.assertEqual(len({r['shape'] for r in rank.read_csv(output)}), 20)
        self.assertTrue(all('_Wbf16_' in r['leaf'] for r in rank.read_csv(output)))
        record = synthetic(rows[0])
        record['candidates'][record['selected']]['config'] = 'route=default'
        table.update_points('gemm', doc, [record])
        self.assertEqual(doc, table.read_table(source))
        self.cli('gemm', 'rank-seed', '--from-lut', source.with_suffix('.fb'), '--out', output, success=False)
        self.cli('gemm', 'rank-seed', '--from-lut', source, '--out', source, success=False)

    def test_gemm_output_batch_group_partitions_roundtrip(self):
        doc = table.read_table(AUTOTUNE / 'gemm/lut/gfx1151.json')
        template = next(p for p in doc['points'] if p['phase'] == 'Wmma' and p['act_dtype'] == 'F16')
        template = dict(template, trans_b='NT')
        points = [dict(template, m=19, n=128, k=256, out_dtype=out,
                       batch_count=batch, a_group=1, b_group=group)
                  for out, batch, group in [('F16', 1, 1), ('F32', 1, 1), ('F32', 8, 4), ('F32', 8, 2)]]
        # uint16 indices survive the same importer with no renumbering.
        doc['configs'] += [copy.deepcopy(doc['configs'][template['config']]) for _ in range(258)]
        points[-1]['config'] = len(doc['configs']) - 1
        self.assertGreater(points[-1]['config'], 255)
        doc['points'] = points
        before = copy.deepcopy(doc)
        mapped, skipped = table.inventory('gemm', doc)
        self.assertFalse(skipped)
        self.assertEqual(len({p['key'] for p in mapped}), 4)
        self.assertEqual(len({(p['leaf'], p['shape']) for p in mapped}), 4)
        self.assertIn('_B8_ag1_bg4', mapped[2]['shape'])
        table.update_points('gemm', doc, [synthetic(p) for p in mapped])
        self.assertEqual(doc, before)

    def test_gpu_admission_boundary(self):
        source = self.root / 'phase.cpp'
        source.write_text(r'''
#include "ut_gpu_execution_scope.h"
int main() {
  { hipdnn_ep_test::GpuExecutionScope scope([] {}); }
}
''')
        binary = self.root / 'phase'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(HERE), str(source), '-o', str(binary)], check=True)
        prefix = self.root / 'phase'
        worker = subprocess.Popen([str(binary)], env=dict(os.environ, HIPDNN_UT_GPU_ADMISSION=str(prefix)))
        def wait(suffix):
            deadline = time.monotonic() + 10
            while not Path(str(prefix) + '.0' + suffix).exists():
                self.assertIsNone(worker.poll())
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.02)
        try:
            wait('.request')
            self.assertFalse(Path(str(prefix) + '.0.done').exists())
            Path(str(prefix) + '.0.grant').write_text('go')
            wait('.done')
            self.assertIsNone(worker.poll())
            Path(str(prefix) + '.0.released').write_text('released')
            self.assertEqual(worker.wait(timeout=10), 0)
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait()


if __name__ == '__main__':
    unittest.main()
