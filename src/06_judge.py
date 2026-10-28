"""E2 batch runner with vLLM: resumable, records tokens and seconds.
Usage: python 06_judge.py --model PATH --tag e4b --stages "pilot:abs10,abs20;test:abs10,abs20;dev:abs10,abs20"
Output (append): OUT/<tag>__<set>__<prefix>__<judge>.jsonl ; instance_ids already present are skipped (resumable)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from judge_common import *
ROOT = os.environ.get('SJEV_ROOT', '/home/ai/proyectos/gemma4-paper')  # data/models/outputs root
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--stages', required=True)
ap.add_argument('--judges', default='holistic,jury')
ap.add_argument('--chunk', type=int, default=64)
ap.add_argument('--out', default=ROOT + '/out/e2')
ap.add_argument('--deadline_s', type=float, default=1e9, help='do not start new batches once this much time has elapsed since start')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
IDS = {'pilot': ROOT + '/out/ids_pilot50.csv', 'test': ROOT + '/out/ids_test600.csv', 'dev': ROOT + '/out/ids_dev_all.csv'}
stages = [st.split(':') for st in a.stages.split(';')]
UID = lambda iid, t: '%s|%d' % (iid, int(t))   # instance_id is NOT unique: same issue with both successful and failed attempts
ids = {s: [UID(r['instance_id'], r['target']) for r in csv.DictReader(open(IDS[s], encoding='utf8'))] for s, _ in stages}
want_ids = set()
for s in ids:
    want_ids |= set(ids[s])
trajs = {}
for l in open(ROOT + '/data/sample.jsonl'):
    r = json.loads(l)
    u = UID(r['instance_id'], r['target'])
    if u in want_ids:
        trajs[u] = r['traj']
print('trajectories loaded', len(trajs), flush=True)
from vllm import LLM, SamplingParams
t0 = time.time()
_mm = {'limit_mm_per_prompt': {'image': 0, 'audio': 0}} if 'gemma' in a.model.lower() else {}
llm = LLM(a.model, max_model_len=16384, gpu_memory_utilization=0.90, dtype='bfloat16', seed=20261027,
          enable_prefix_caching=True, **_mm)
load_s = time.time() - t0
tok = llm.get_tokenizer()
count = lambda t: len(tok.encode(t, add_special_tokens=False))
json.dump({'tag': a.tag, 'model': a.model, 'load_s': load_s, 'prompt_version': PROMPT_VERSION, 'stages': a.stages},
          open(a.out + '/%s__load.json' % a.tag, 'w'))
SP = {'holistic': SamplingParams(max_tokens=48, temperature=0, seed=20261027),
      'jury': SamplingParams(max_tokens=96, temperature=0, seed=20261027)}
for s, ps in stages:
    for spec in ps.split(','):
        df = ids[s]
        items = []
        for iid in df:
            p = get_prefix(trajs[iid], spec)
            if p is not None:
                items.append((iid, p))
        for judge in a.judges.split(','):
            fn = '%s/%s__%s__%s__%s.jsonl' % (a.out, a.tag, s, spec, judge)
            done = set()
            if os.path.exists(fn):
                done = {json.loads(l)['uid'] for l in open(fn)}
            todo = [x for x in items if x[0] not in done]
            print('[%s] %s %s %s: %d items, %d pending' % (a.tag, s, spec, judge, len(items), len(todo)), flush=True)
            for i in range(0, len(todo), a.chunk):
                if time.time() - T0 > a.deadline_s:
                    print('deadline reached', flush=True)
                    sys.exit(0)
                ch = todo[i:i + a.chunk]
                built = [build_prompt(p, judge, count) for _, p in ch]
                msgs = [[{'role': 'user', 'content': b[0]}] for b in built]
                t1 = time.time()
                outs = llm.chat(msgs, SP[judge], use_tqdm=False)
                dt = time.time() - t1
                ptoks = [len(o.prompt_token_ids) for o in outs]
                with open(fn, 'a') as f:
                    for (iid, _), b, o, pt in zip(ch, built, outs, ptoks):
                        txt = o.outputs[0].text
                        f.write(json.dumps({'uid': iid, 'instance_id': iid.split('|')[0], 'target': int(iid.split('|')[1]), 'set': s, 'prefix': spec, 'judge': judge, 'model': a.tag,
                                            'n_steps': b[1], 'prompt_tokens': pt, 'cached_tokens': o.num_cached_tokens or 0,
                                            'gen_tokens': len(o.outputs[0].token_ids), 'chunk_s': dt, 'chunk_n': len(ch),
                                            'chunk_ptoks': sum(ptoks), 'sec_est': dt * pt / sum(ptoks),
                                            'raw': txt, 'parsed': parse(txt, judge), 'pv': PROMPT_VERSION}) + '\n')
                print('  %s %s %s %s chunk %d/%d %.1fs' % (a.tag, s, spec, judge, i + len(ch), len(todo), dt), flush=True)
print('DONE', a.tag, 'total_s', time.time() - T0, flush=True)
