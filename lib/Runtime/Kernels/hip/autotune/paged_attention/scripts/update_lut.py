# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Page exact ragged-key LUT adapter for the owning native UT rank tools."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
LUT_DIR=HERE.parent/'lut'
sys.path.insert(0,str(HERE.parents[3]/'unit_test/common'))
import ut_rank_tools as rank
import ut_lut_from_table as table

def build(args):
 doc=table.read_table(args.base)
 table.inventory('paged_attention',doc)
 summary=table.update_points('paged_attention',doc,rank.evidence(args.rank_log))
 output=rank.private_path(args.out,LUT_DIR)
 output.write_text(json.dumps(doc,indent=2)+'\n')
 output.with_suffix('.rank-report.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('rank table:',output,'updated:',summary['updated'],'skipped:',len(summary['skipped']))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('command',choices=['migrate-v1','compile','rank-catalog','rank-candidates','rank-seed','rank-measure','rank-report','rank-build','rank-reference-candidates'])
 ap.add_argument('--out',required=True)
 ap.add_argument('--flatc',default='flatc')
 ap.add_argument('--fb-out')
 rank.add_arguments(ap);table.add_arguments(ap)
 args=ap.parse_args();args.op='paged_attention'
 if args.command=='migrate-v1':
  doc,receipt=table.migrate_paged_v1(table.read_table(args.base))
  output=rank.private_path(args.out,LUT_DIR);output.write_text(json.dumps(doc,indent=2)+'\n')
  output.with_suffix('.migration.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print('migrated:',output)
 elif args.command=='compile':
  table.inventory(args.op,table.read_table(args.out))
  output=rank.private_path(args.fb_out or str(Path(args.out).with_suffix('.fb')),LUT_DIR)
  subprocess.run([args.flatc,'--binary','--strict-json','-o',str(output.parent),str(HERE.parent/'paged_attention_autotune.fbs'),args.out],check=True)
  generated=output.parent/(Path(args.out).stem+'.fb')
  if generated!=output:generated.replace(output)
  print('compiled:',output)
 else:rank.dispatch(args,LUT_DIR,build)
if __name__=='__main__':main()
