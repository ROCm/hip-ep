# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Existing JSON table <-> native UT inputs and reports. Contains no timer."""
import csv
import json
import random
import re
import statistics
from pathlib import Path

import ut_rank_tools as rank

NBITS_KEY = ('phase', 'bits', 'group_size', 'zero_point', 'row_stride', 'm', 'n', 'k')
GQA_KEY = ('phase', 'head_dim', 'kv_dtype', 'num_heads', 'kv_num_heads', 'batch', 'seq_q', 'seq_kv')


def key_fields(op):
    return ('shape',) if op == 'paged_attention' else NBITS_KEY if op == 'matmul_nbits' else GQA_KEY


def table_key(op, point):
    """Stable, JSON-valued key; --keys accepts one such array per line."""
    return json.dumps([point[k] for k in key_fields(op)], separators=(',', ':'))


def config_text(values):
    return ';'.join(f'{k}={v}' for k, v in values.items())


def config_values(text):
    values = rank.knobs(text)
    return {k: v if k == 'route' else int(v) for k, v in values.items()}


def paged_shape(shape):
    pattern = (r'(decode|prefill)_B(\d+)_H(\d+)_G(\d+)_D(\d+)_P(\d+)_S(\d+)_M(\d+)'
               r'_q([0-9+]+)_past([0-9+]+)_rope([01])_rD(\d+)_inter([01])_pos(\d+)'
               r'_scale([^_]+)_win(-?\d+)_cap([^_]+)_map(ordered|permuted|shared)_shared(\d+)_seed(\d+)_errnone')
    match = re.fullmatch(pattern, shape)
    if not match:
        raise ValueError('malformed/unsupported Page shape (normal cases only)')
    phase, b, h, g, d, p, size, columns, q, past, rope, rd, inter, pos, scale, window, cap, page_map, shared, seed = match.groups()
    b,h,g,d,p,size,columns,rope,rd,inter,pos,window,shared,seed=map(int,(b,h,g,d,p,size,columns,rope,rd,inter,pos,window,shared,seed))
    q,past=[int(x) for x in q.split('+')],[int(x) for x in past.split('+')]
    if min(b,h,g,p,size,columns,pos)<=0 or d not in (64,96,128,256) or h%g or len(q)!=b or len(past)!=b:
        raise ValueError('invalid Page base dimensions/arrays')
    if not sum(q) or sum(q)>2**31-1 or any(x+y>2**31-1 or (x+y+size-1)//size>columns for x,y in zip(q,past)):
        raise ValueError('invalid Page T/post/capacity')
    if phase=='decode' and (any(x>1 for x in q) or h//g not in (1,2,3,4,5,6,8,16) and not(h//g==9 and d==128)):
        raise ValueError('unsupported Page decode geometry')
    if phase=='prefill' and d==256 and window>0 or float(cap)!=0 or not(float(scale)>=0 and float(scale)<float('inf')):
        raise ValueError('unsupported Page options')
    if rope and (inter or rd not in (0,d) or any(x and x+y>pos for x,y in zip(q,past))):
        raise ValueError('invalid Page RoPE options/position')
    if shared and (page_map!='shared' or shared%size or any(x and y<shared for x,y in zip(q,past))):
        raise ValueError('invalid Page shared prefix')
    return dict(phase=phase, B=b,H=h,G=g,D=d,P=p,S=size,M=columns,q=q,past=past,rope=rope,rD=rd,inter=inter,pos=pos,scale=scale,window=window,map=page_map,shared=shared,seed=seed)


def paged_config(shape, text):
    d=paged_shape(shape)
    c=config_values(text)
    fields=set(c)
    if fields not in ({'splits','bkv'},{'splits','bkv','hb','qw','mt','nd'}):
        raise ValueError('Page config must contain legacy or versioned final geometry')
    if d['phase']=='decode':
        legal=c['bkv']==0 and (0 if len(c)==6 else 1)<=c['splits']<=64 and all(c.get(k,0)==0 for k in ('hb','qw','mt','nd'))
    else:
        legal=c['splits']==0 and c['bkv'] in ((16,32) if d['D']==256 else (16,32,64))
        if len(c)==6:
            hb,qw,mt,nd=(c[k] for k in ('hb','qw','mt','nd'))
            legacy=(hb==1 and qw==2 and mt==1 and nd==0 and d['D']<=128)
            if d['D']==256 or d['D']==128 and nd>0:
                legal=legal and d['window']<=0 and c['bkv'] in (16,32) and mt==1 and qw==0 and (nd,hb) in ((2,1),(2,2),(2,3),(2,4),(4,1),(4,2)) and (d['D']*(c['bkv']+8)+c['bkv']*(d['D']+8))*2+hb*nd*c['bkv']*16*4<=65536
            else:
                nw=hb*qw
                layouts=((1,1),(2,1),(1,4),(2,2),(4,1),(1,8),(2,4),(4,2),(8,1))
                normal=(hb,qw) in layouts and nd==0 and c['bkv'] in (32,64) and mt in ((1,2) if d['D']==64 else (1,))
                normal=normal and not(d['window']>0 and nw>4) and not(d['D']==96 and nw<4) and not(d['D']==128 and (nw==2 or c['bkv']==64 and nw!=8)) and not(mt==2 and c['bkv']==64)
                legal=legal and (legacy or normal)
            legal=legal and hb>0 and (d['H']//d['G'])%hb==0
    if not legal:raise ValueError('incompatible Page config geometry')
    return c


def paged_header(doc):
    version=doc.get('schema_version')
    expected=dict(kernel_abi='paged_fp16_pshd_v'+str(version),op='paged_attention',layout='PSHD',dtype='Fp16')
    if version not in (1,2) or any(doc.get(k)!=v for k,v in expected.items()) or not isinstance(doc.get('gpu_arch'),str):
        raise ValueError('incompatible Page op/layout/dtype/schema/ABI/arch header')


def paged_pool(doc):
    return doc['configs_v2'] if doc['schema_version']==2 else doc['configs']


def migrate_paged_v1(doc):
    """Preserve v1 arrays and indices; append only conflicting D geometries."""
    paged_header(doc)
    if doc['schema_version']!=1:raise ValueError('migration requires a v1 Page table')
    import copy
    out=copy.deepcopy(doc)
    def convert(c,d):
        if d['phase']=='decode':return dict(splits=c['splits'],bkv=c['bkv'],hb=0,qw=0,mt=0,nd=0)
        return dict(splits=c['splits'],bkv=c['bkv'],hb=1,qw=0 if d['D']==256 else 2,mt=1,nd=2 if d['D']==256 else 0)
    uses={}
    for point in out['points']:
        index=point['config'];d=paged_shape(point['shape']);paged_config(point['shape'],config_text(doc['configs'][index]))
        uses.setdefault(index,[]).append((point,d))
    pool=[]
    for index,c in enumerate(doc['configs']):
        d=uses[index][0][1] if index in uses else dict(phase='decode' if c['splits'] else 'prefill',D=64)
        pool.append(convert(c,d))
    remapped=[]
    for index,points in uses.items():
        for point,d in points:
            c=convert(doc['configs'][index],d)
            if c!=pool[index]:
                target=next((i for i,x in enumerate(pool) if x==c),len(pool))
                if target==len(pool):pool.append(c)
                point['config']=target;remapped.append(dict(shape=point['shape'],old=index,new=target))
    out.update(schema_version=2,kernel_abi='paged_fp16_pshd_v2',configs_v2=pool,fallbacks_v2=[])
    return out,dict(old_configs=len(doc['configs']),new_configs=len(pool),remapped=remapped,old_fields_preserved=True)


def native_config(op, doc, point):
    c = (paged_pool(doc) if op=='paged_attention' else doc['configs'])[point['config']]
    if op == 'paged_attention':
        return config_text({k:c[k] for k in (('splits','bkv','hb','qw','mt','nd') if doc['schema_version']==2 else ('splits','bkv'))})
    if op == 'gqa':
        return config_text(dict(wmma=c['use_wmma'], splits=point['splits'],
                                **{k: c[k] for k in ('bkv', 'hb', 'qw', 'mt', 'nd')}))
    ext = {e['config']: e['variant'] for e in doc.get('extended_configs', [])}
    return config_text(dict(route='wmma' if c['kind'] == 'Wmma' else 'gemv',
                            bm=c['bm16'] * 16, bn=c['bn16'] * 16, swizzle=c['swizzle'],
                            wt_m=c['wt_m'], wt_n=c['wt_n'], bk=c['bk'],
                            fused=c['fused'] if c['kind'] == 'Wmma' else 1,
                            ext=int(point['config'] in ext), variant=ext.get(point['config'], 0),
                            threads=c['threads'], tile_n=c['tile_n'], tile_m=0))


def point_input(op, doc, point, activation='fp16', prefill_kv='fp16'):
    """Lossless schema key mapping. Unsupported semantic classes are explicit."""
    if op == 'paged_attention':
        paged_header(doc)
        d=paged_shape(point['shape'])
        if point.get('phase')!=d['phase'] or not isinstance(point.get('config'),int) or not 0<=point['config']<len(paged_pool(doc)):
            raise ValueError('invalid Page point phase/config index')
        text=native_config(op,doc,point)
        paged_config(point['shape'],text)
        return dict(key=table_key(op,point),leaf=f"paged_attention/{d['phase']}/Xfp16_Wfp16_Yfp16",shape=point['shape'],old_config=text,point=point)
    if op == 'matmul_nbits':
        bits = {'B2': 'u2', 'B3': 'u3', 'B4': 'u4', 'B8': 'i8'}.get(point['bits'])
        if bits is None or point['group_size'] not in ('G16', 'G32', 'G64', 'G128', 'G256', 'G512'):
            raise ValueError('unsupported bits/group_size category')
        if point['zero_point'] not in ('Asymmetric', 'Symmetric'):
            raise ValueError('unsupported zero_point category')
        if point['phase'] == 'DecodeDp4a':
            raise ValueError('DecodeDp4a activation/entry is not covered by these leaves')
        if point['row_stride'] == 'Padded':
            raise ValueError('padded row stride is not expressible by the leaf')
        m, n, k = (point[x] for x in ('m', 'n', 'k'))
        gs = int(point['group_size'][1:])
        # The lookup-mode family cutoffs in the unchanged production adapter.
        wmma = m >= (8 if bits == 'u4' else 16)
        if bits != 'u4':
            wmma = wmma and k % 32 == 0 and gs >= 32 and gs % 32 == 0
        if (point['phase'] == 'Prefill') != wmma:
            raise ValueError('table phase differs from the leaf lookup-mode route (including M-blocked tile_m)')
        if point['phase'] not in ('Decode', 'Prefill') or point['row_stride'] != ('Arrival' if wmma else 'Any'):
            raise ValueError('unsupported phase/row_stride category')
        if not wmma and m > 1 and bits in ('u4', 'i8'):
            raise ValueError('M-blocked tile_m has no table config representation')
        if not wmma and bits in ('u2', 'u3') and (gs < 32 or k % 32):
            raise ValueError('naive route has no table config representation')
        leaf = f'matmul_nbits/X{activation}_W{bits}_Y{activation}'
        shape = f'M{m}_N{n}_K{k}_gs{gs}' + ('_noz' if point['zero_point'] == 'Symmetric' else '')
    else:
        d = {'D64': 64, 'D96': 96, 'D128': 128, 'D256': 256}.get(point['head_dim'])
        b, h, g, sq, skv = (point[x] for x in ('batch', 'num_heads', 'kv_num_heads', 'seq_q', 'seq_kv'))
        if d is None or h % g or h // g not in (1, 2, 3, 4, 5, 6, 8, 16):
            raise ValueError('head geometry is not covered by the fused leaf')
        decode = point['phase'] == 'Decode'
        kv = {'Fp16': 'fp16', 'Int8': 'int8'}.get(point['kv_dtype']) if decode else prefill_kv
        if kv is None or (not decode and point['kv_dtype'] != 'Fp16'):
            raise ValueError('unsupported KV dtype (prefill lookup is after dequantization)')
        if decode and sq != 1 or not decode and (sq <= 1 or skv < sq):
            raise ValueError('sequence dimensions do not match decode/prefill leaf semantics')
        if not decode and point['phase'] != ('PrefillV8x' if d == 256 else 'PrefillV6x'):
            raise ValueError('prefill variant differs from the leaf head_dim route')
        if not decode and kv == 'int8' and skv != sq:
            raise ValueError('INT8 prefill leaf only covers past=0')
        leaf = f'gqa/{"decode" if decode else "prefill"}/Xfp16_W{"i8" if kv == "int8" else "fp16"}_Yfp16'
        shape = f'B{b}_H{h}_G{g}_D{d}_len{skv}' if decode else f'B{b}_H{h}_G{g}_D{d}_sq{sq}' + (f'_past{skv - sq}' if kv == 'fp16' else '')
        if decode and not 1 <= point['splits'] <= 64 or not decode and point['splits'] != 0:
            raise ValueError('splits outside the configured leaf range')
    if any(point[k] <= 0 for k in (('m', 'n', 'k') if op == 'matmul_nbits' else ('batch', 'num_heads', 'kv_num_heads', 'seq_q', 'seq_kv'))):
        raise ValueError('nonpositive dimensions')
    return dict(key=table_key(op, point), leaf=leaf, shape=shape,
                old_config=native_config(op, doc, point), point=point)


def inventory(op, doc, activation='fp16', prefill_kv='fp16'):
    if op=='paged_attention': paged_header(doc)
    points, skipped, seen = [], [], set()
    for p in doc['points']:
        key = table_key(op, p)
        if key in seen:
            raise ValueError('ambiguous duplicate table key ' + key)
        seen.add(key)
        try:
            points.append(point_input(op, doc, p, activation, prefill_kv))
        except ValueError as error:
            skipped.append(dict(key=key, reason=str(error)))
    identities = [(p['leaf'], p['shape']) for p in points]
    if len(set(identities)) != len(identities):
        raise ValueError('ambiguous table key -> leaf/shape mapping')
    return points, skipped


def read_table(path):
    if Path(path).suffix.lower() != '.json':
        raise ValueError('--from-lut accepts JSON only; FB is compiled from JSON')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed(args, installed):
    doc = read_table(args.from_lut)
    points, skipped = inventory(args.op, doc, getattr(args, 'activation', 'fp16'), getattr(args, 'prefill_kv', 'fp16'))
    requested = None
    if args.keys:
        requested = {json.dumps(json.loads(line), separators=(',', ':')) for line in Path(args.keys).read_text().splitlines() if line.strip()}
        unknown = requested - {table_key(args.op, p) for p in doc['points']}
        if unknown:
            raise ValueError('unknown --keys: ' + ', '.join(sorted(unknown)))
    selected = []
    for p in points:
        reason = None
        if requested is not None and p['key'] not in requested:
            reason = 'not selected by --keys'
        if args.leaf and p['leaf'] not in args.leaf:
            reason = 'filtered leaf'
        dtype = getattr(args, 'activation', 'fp16') if args.op == 'matmul_nbits' else ('int8' if '_Wi8_' in p['leaf'] else 'fp16')
        if args.dtype and dtype not in args.dtype:
            reason = 'filtered dtype'
        phase = 'decode' if ('/decode/' in p['leaf'] or p['point']['phase'] == 'Decode') else 'prefill'
        if args.rank_phase != 'both' and phase != args.rank_phase:
            reason = 'filtered phase'
        if reason:
            skipped.append(dict(key=p['key'], reason=reason))
        else:
            selected.append(p)
    if args.sample is not None:
        if args.sample <= 0 or args.sample > len(selected):
            raise ValueError('--sample must be positive and no larger than the eligible point count')
        chosen = set(random.Random(args.seed).sample(range(len(selected)), args.sample))
        skipped.extend(dict(key=p['key'], reason='not sampled') for i, p in enumerate(selected) if i not in chosen)
        selected = [p for i, p in enumerate(selected) if i in chosen]
    if not selected:
        raise ValueError('no eligible table points')
    catalog = {}
    if args.with_catalog:
        if not args.catalog:
            raise ValueError('--with-catalog requires --catalog from rank-catalog')
        catalog = {(r['leaf'], r['shape']): r for r in rank.records(args.catalog, 'UT_CANDIDATES')}
    output = rank.private_path(args.out, installed)
    rows = []
    for p in selected:
        c = catalog.get((p['leaf'], p['shape']))
        if args.with_catalog and c is None:
            raise ValueError('catalog omits ' + p['shape'])
        baseline = c.get('default') if c else None
        configs = ['default']
        if p['old_config'] != baseline:
            configs.append(p['old_config'])
        identities = {p['old_config'], baseline}
        for cfg in c['configs'] if c else []:
            if cfg['final'] not in identities:
                identities.add(cfg['final'])
                configs.append(cfg['request'])
        rows.extend(dict(leaf=p['leaf'], shape=p['shape'], priority=str(int(args.priority)), config=config) for config in configs)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, rank.HEADER)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.seed.json').write_text(json.dumps(dict(from_lut=str(Path(args.from_lut).resolve()), selected=selected, skipped=skipped,
        sample=args.sample, seed=args.seed, default_resolution='native leaf resolves no-LUT default and deduplicates final geometry'), indent=2) + '\n')
    skipped_out = rank.private_path(args.skipped or output.with_suffix('.skipped.json'), installed)
    skipped_out.write_text(json.dumps(skipped, indent=2) + '\n')
    print('candidate CSV:', output, 'points:', len(selected), 'rows:', len(rows), 'skipped:', len(skipped))


def record_point(op, record):
    if op=='paged_attention':
        d=paged_shape(record['shape'])
        if record['leaf']!=f"paged_attention/{d['phase']}/Xfp16_Wfp16_Yfp16":raise ValueError('Page phase/leaf mismatch')
        return dict(phase=d['phase'],shape=record['shape'])
    d = rank.fields(record['shape'])
    if op == 'matmul_nbits':
        # The leaf has no dp4a/padded route. A singleton gemv_m/naive stays skipped.
        c = config_values(record['candidates'][record.get('selected', record['winner'])]['config'])
        if c['route'] not in ('wmma', 'gemv'):
            raise ValueError('route has no schema representation')
        return dict(phase='Prefill' if c['route'] == 'wmma' else 'Decode', bits='B' + record['leaf'].split('_W')[1].lstrip('ui').split('_')[0],
                    group_size='G' + str(d['gs']), zero_point='Symmetric' if d.get('noz') else 'Asymmetric',
                    row_stride='Arrival' if c['route'] == 'wmma' else 'Any', m=d['M'], n=d['N'], k=d['K'])
    decode = '/decode/' in record['leaf']
    return dict(phase='Decode' if decode else 'PrefillV8x' if d['D'] == 256 else 'PrefillV6x', head_dim='D' + str(d['D']),
                kv_dtype='Int8' if decode and '_Wi8_' in record['leaf'] else 'Fp16', num_heads=d['H'], kv_num_heads=d['G'], batch=d['B'],
                seq_q=1 if decode else d['sq'], seq_kv=min(d['len'] if decode else d['sq'] + d.get('past', 0), d.get('win', 0) or 2**31 - 1))


def update_points(op, doc, records):
    """Preserve original point order, fields and duplicate config indices."""
    skipped, seen = [], set()
    by_key = {table_key(op, p): p for p in doc['points']}
    if len(by_key) != len(doc['points']):
        raise ValueError('ambiguous duplicate base table key')
    for record in records:
        try:
            p = record_point(op, record)
        except ValueError as error:
            skipped.append(dict(leaf=record['leaf'], shape=record['shape'], reason=str(error)))
            continue
        key = table_key(op, p)
        if key in seen:
            raise ValueError('ambiguous duplicate table key ' + key)
        seen.add(key)
        chosen = record['winner'] if record['insert_nondefault'] else 0
        text = record['candidates'][chosen]['config']
        old = by_key.get(key)
        # Identical selections keep the exact existing index and other fields.
        if old is not None and native_config(op, doc, old) == text:
            continue
        c = config_values(text)
        if op == 'paged_attention':
            config=paged_config(p['shape'],text)
            pool=paged_pool(doc)
        elif op == 'gqa':
            config = dict(kind='Decode' if p['phase'] == 'Decode' else 'Prefill', use_wmma=c['wmma'],
                          **{k: c[k] for k in ('bkv', 'hb', 'qw', 'mt', 'nd')})
            p['splits'] = c['splits']
            pool = doc['configs']
        else:
            pool = [dict(v) for v in doc['configs']]
            for e in doc.get('extended_configs', []):
                pool[e['config']].update(ext=1, variant=e['variant'])
            wmma = c['route'] == 'wmma'
            config = dict(kind='Wmma' if wmma else 'Gemv', bm16=c['bm'] // 16 if wmma else 0,
                          bn16=c['bn'] // 16 if wmma else 0, swizzle=c['swizzle'] if wmma else 0,
                          wt_m=c['wt_m'] if wmma else 0, wt_n=c['wt_n'] if wmma else 0,
                          bk=c['bk'] if wmma else 0, fused=c['fused'] if wmma else 0,
                          threads=0 if wmma else c['threads'], tile_n=0 if wmma else c['tile_n'])
            if wmma and c['ext']:
                config.update(ext=1, variant=c['variant'])
        if config not in pool:
            if len(pool) >= 256:
                raise ValueError('config index exceeds ubyte')
            index = len(pool)
            plain = dict(config)
            if plain.pop('ext', 0):
                doc.setdefault('extended_configs', []).append(dict(config=index, variant=plain.pop('variant')))
            if op == 'paged_attention':
                pool.append(plain)
            else:
                doc['configs'].append(plain)
        p['config'] = pool.index(config) if config in pool else len(pool)
        if old is None:
            doc['points'].append(p)
        else:
            old.update(p)
    return dict(updated=len(seen), skipped=skipped)


def report(args, installed):
    doc = read_table(args.from_lut)
    points, unsupported = inventory(args.op, doc, getattr(args, 'activation', 'fp16'), getattr(args, 'prefill_kv', 'fp16'))
    by_id = {(p['leaf'], p['shape']): p for p in points}
    records = rank.evidence(args.rank_log)
    output = rank.private_path(args.out, installed)
    rows = []
    for r in records:
        p = by_id.get((r['leaf'], r['shape']))
        if p is None:
            raise ValueError('rank record has no representable input-table key: ' + r['shape'])
        final = r['phases'][-1]
        measurements = {m['candidate']: m for m in final['measurements']}
        winner, default = r['candidates'][r['winner']]['config'], r['candidates'][0]['config']
        old_id = next((i for i, c in enumerate(r['candidates']) if c['config'] == p['old_config']), None)
        old = measurements.get(old_id)
        win = measurements[r['winner']]
        ratio = lambda a, b: statistics.median(x / y for x, y in zip(a['group_means_ms'], b['group_means_ms']))
        common = None
        for phase in reversed(r['phases']):
            same_phase = {m['candidate']: m for m in phase['measurements']}
            if old_id in same_phase and r['winner'] in same_phase:
                common = (phase['name'], same_phase)
                break
        selected = r['candidates'][r['winner'] if r['insert_nondefault'] else 0]['config']
        phases = {phase['name']: phase.get('gpu_ms', sum(sum(m['group_gpu_ms']) for m in phase['measurements'])) for phase in r['phases']}
        excluded = {i for phase in r['phases'] for i in phase.get('excluded', [])}
        rows.append(dict(key=p['key'], leaf=r['leaf'], shape=r['shape'], old_config=p['old_config'], default_config=default,
                         winner=winner, selected=selected, winner_over_default=r['versus_default']['median'],
                         winner_over_old=ratio(common[1][r['winner']], common[1][old_id]) if common else '',
                         old_over_default=ratio(common[1][old_id], common[1][0]) if common else '', old_ratio_phase=common[0] if common else '',
                         old_status='measured_final' if old else 'excluded' if old_id in excluded else 'not_in_final_phase',
                         changed=selected != p['old_config'], insert_nondefault=r['insert_nondefault'], retested=any(phase['name'] == 'confirm' for phase in r['phases']),
                         decision_status=r['decision_status'], excluded=json.dumps(sorted(excluded)),
                         screen_gpu_ms=phases.get('screen', 0), standard_gpu_ms=phases.get('standard', 0), confirm_gpu_ms=phases.get('confirm', 0),
                         gpu_ms=sum(phases.values()), pilot_gpu_ms=sum(phase.get('pilot_ms', 0) for phase in r['phases']),
                         wall_ms=r.get('wall_ms', ''), old_index=p['point']['config']))
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    distribution = dict.fromkeys(('<=0.95', '(0.95,0.98]', '(0.98,1.02]', '>1.02', 'unavailable'), 0)
    for row in rows:
        v = row['winner_over_old']
        bucket = 'unavailable' if v == '' else '<=0.95' if v <= .95 else '(0.95,0.98]' if v <= .98 else '(0.98,1.02]' if v <= 1.02 else '>1.02'
        distribution[bucket] += 1
    measured = {(r['leaf'], r['shape']) for r in records}
    summary = dict(points=len(rows), changed=sum(r['changed'] for r in rows), distribution=distribution,
                   retested_fraction=sum(r['retested'] for r in rows) / len(rows),
                   winner_unresolved_fraction=sum(r['decision_status'] == 'winner_unresolved' for r in rows) / len(rows),
                   skipped=len(unsupported), unmeasured=len(points) - len(measured),
                   gpu_ms=sum(r['gpu_ms'] for r in rows), wall_ms=sum(r['wall_ms'] for r in rows if r['wall_ms'] != ''))
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.md:
        md = rank.private_path(args.md, installed)
        md.write_text('# LUT refresh report\n\n' + '\n'.join(f'- {k}: {v}' for k, v in summary.items()) +
                      '\n\nWinner/default uses the final phase. Ratios involving the old config use the latest phase containing both the winner and old config, named in old_ratio_phase; no ratios cross phases. Excluded or absent old candidates without a common phase are unavailable. GPU totals cover recorded phases, with pilot GPU time separate; wall totals cover native rank calls and exclude build, CPU reference, launcher and resource waits. Unsupported points and eligible unmeasured points are counted separately.\n')
    print(json.dumps(summary))


def add_arguments(ap):
    ap.add_argument('--from-lut', help='existing JSON table (FB input is unsupported)')
    ap.add_argument('--leaf', nargs='+', help='rank-seed: exact leaf names')
    ap.add_argument('--dtype', nargs='+', choices=['fp16', 'fp32', 'int8'], help='rank-seed: activation (nbits) or KV input (GQA)')
    ap.add_argument('--rank-phase', choices=['decode', 'prefill', 'both'], default='both')
    ap.add_argument('--keys', help='one JSON key array per line (see *.seed.json)')
    ap.add_argument('--sample', type=int, help='reproducibly select N eligible points')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--priority', action='store_true')
    ap.add_argument('--with-catalog', action='store_true', help='append legal configs from --catalog; default + old only otherwise')
    ap.add_argument('--skipped', help='skipped JSON output (default: CSV.with_suffix(.skipped.json))')
    ap.add_argument('--md', help='rank-report: Markdown summary output')
