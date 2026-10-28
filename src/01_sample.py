"""Stratified sampling of nebius/SWE-agent-trajectories (CC-BY-4.0).
Cap per repo and class; eval_logs and generated_patch are discarded.
exit_status and n_ai_total are stored only as diagnostic METADATA, never as features."""
import pyarrow.parquet as pq, glob, json, random, collections, sys
import pandas as pd
from common import *
random.seed(SEED)
fs = sorted(glob.glob(ROOT + '/data/nebius_raw/data/*.parquet'))
meta = []
for si, f in enumerate(fs):
    d = pq.read_table(f, columns=['instance_id', 'model_name', 'target', 'exit_status']).to_pandas()
    d['shard'] = si
    d['row'] = range(len(d))
    meta.append(d)
meta = pd.concat(meta, ignore_index=True)
meta['repo'] = meta.instance_id.map(repo_of)
print('rows', len(meta), 'repos', meta.repo.nunique(), 'prev', meta.target.mean())
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 14      # per repo and class
sel = []
for (repo, t), g in meta.groupby(['repo', 'target']):
    idx = list(g.index)
    random.shuffle(idx)
    sel += idx[:CAP]
s = meta.loc[sel].copy()
print('sample', len(s), 'pos', int(s.target.sum()), 'repos', s.repo.nunique())
out = open(ROOT + '/data/sample.jsonl', 'w')
want = collections.defaultdict(dict)
for i, r in s.iterrows():
    want[r.shard][r.row] = r
for si, f in enumerate(fs):
    if si not in want:
        continue
    pf = pq.ParquetFile(f)
    base = 0
    for b in pf.iter_batches(batch_size=500, columns=['trajectory']):
        rows = b.column('trajectory').to_pylist()
        for j, tr in enumerate(rows):
            r = want[si].get(base + j)
            if r is None:
                continue
            msgs = [{'role': m['role'], 'text': (m['text'] if m['role'] != 'system' else None)} for m in tr]
            out.write(json.dumps({'instance_id': r.instance_id, 'repo': r.repo, 'model': r.model_name,
                                  'target': bool(r.target), 'exit_status': r.exit_status,
                                  'n_ai_total': sum(m['role'] == 'ai' for m in tr), 'traj': msgs}) + '\n')
        base += len(rows)
    print('shard', si, 'ok', flush=True)
out.close()
s.drop(columns=['shard', 'row']).to_csv(ROOT + '/data/sample_meta.csv', index=False)
