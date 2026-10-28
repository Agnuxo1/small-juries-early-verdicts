"""Trace features without an LLM, computed on prefixes. They use only the prefix content (including the issue text)."""
import json, re, collections
import pandas as pd, numpy as np
from common import *

CMDS = ['open', 'edit', 'search_file', 'search_dir', 'find_file', 'python', 'create', 'scroll_down', 'scroll_up',
        'goto', 'ls', 'cd', 'pytest', 'rm', 'submit', 'cat', 'grep', 'pip', 'git']


def first_cmd(text):
    m = re.search(r'```(?:\w*)\n(.*?)```', text or '', re.S)
    if not m:
        return None, ''
    body = m.group(1).strip()
    return (body.split()[0] if body.split() else ''), body


def feats(prefix):
    ai = [m for m in prefix if m['role'] == 'ai']
    users = [m for m in prefix if m['role'] == 'user']
    issue = users[0]['text'] if users else ''
    obs = users[1:]
    cmds, bodies = [], []
    for m in ai:
        c, b = first_cmd(m['text'])
        cmds.append(c)
        bodies.append(b)
    cnt = collections.Counter(cmds)
    f = {'n_ai': len(ai), 'issue_chars': len(issue or ''),
         'ai_chars_mean': float(np.mean([len(m['text'] or '') for m in ai])) if ai else 0.0,
         'obs_chars_mean': float(np.mean([len(m['text'] or '') for m in obs])) if obs else 0.0,
         'ctx_chars': sum(len(m['text'] or '') for m in prefix)}
    for c in CMDS:
        f['cmd_' + c] = cnt.get(c, 0)
    f['cmd_none'] = cnt.get(None, 0)
    f['frac_cmd_none'] = f['cmd_none'] / max(1, len(ai))
    bc = collections.Counter(b for b in bodies if b)
    f['dup_cmds'] = sum(v - 1 for v in bc.values())
    f['max_repeat'] = max(bc.values()) if bc else 0
    run = best = 1
    for a, b in zip(bodies, bodies[1:]):
        run = run + 1 if (a and a == b) else 1
        best = max(best, run)
    f['max_consec_repeat'] = best if bodies else 0
    ot = [(m['text'] or '') for m in obs]
    f['obs_traceback'] = sum('Traceback' in t for t in ot)
    f['obs_syntax_err'] = sum(('syntax error' in t.lower() or 'E999' in t) for t in ot)
    f['obs_edit_reject'] = sum('Your proposed edit has introduced' in t for t in ot)
    f['obs_error_word'] = sum(bool(re.search(r'\berror\b', t, re.I)) for t in ot)
    f['obs_not_found'] = sum(('not found' in t.lower() or 'No such file' in t or 'No matches found' in t) for t in ot)
    f['obs_tests_pass'] = sum(bool(re.search(r'\d+ passed', t)) for t in ot)
    f['obs_tests_fail'] = sum(bool(re.search(r'\d+ failed|FAILED', t)) for t in ot)
    f['obs_empty'] = sum(len(t.strip()) < 5 for t in ot)
    f['opened_files'] = len({b.split()[1] for c, b in zip(cmds, bodies) if c == 'open' and len(b.split()) > 1})
    f['created_repro'] = int(any(c == 'create' and 'reproduce' in b.lower() for c, b in zip(cmds, bodies)))
    f['first_edit_pos'] = next((i / len(ai) for i, c in enumerate(cmds) if c == 'edit'), 1.0) if ai else 1.0
    f['frac_edit'] = f['cmd_edit'] / max(1, len(ai))
    f['frac_nav'] = (f['cmd_scroll_down'] + f['cmd_scroll_up'] + f['cmd_goto']) / max(1, len(ai))
    f['hedge_words'] = sum(len(re.findall(r"\b(?:however|unfortunately|still|again|seems)\b", m['text'] or '', re.I)) for m in ai)
    return f


def build(sample_path, out_path):
    rows = []
    for line in open(sample_path):
        r = json.loads(line)
        tr = r['traj']
        specs = [('frac33', prefix_fraction(tr, 0.33)), ('frac66', prefix_fraction(tr, 0.66)),
                 ('frac100', prefix_fraction(tr, 1.0))]
        for k in (10, 20, 30):
            specs.append((f'abs{k}', prefix_steps(tr, k)))
        for name, p in specs:
            if p is None:
                continue
            d = feats(p)
            d.update(instance_id=r['instance_id'], repo=r['repo'], model=r['model'], target=int(r['target']), prefix=name)
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


if __name__ == '__main__':
    df = build(ROOT + '/data/sample.jsonl', ROOT + '/out/features.csv')
    print(df.groupby('prefix').agg(n=('target', 'size'), prev=('target', 'mean'), repos=('repo', 'nunique')))
