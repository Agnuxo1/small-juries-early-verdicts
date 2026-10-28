"""Common utilities for F0/E1. Fixed seed. No LLM."""
import os, re, math
SEED = 20261027
ROOT = os.environ.get('SJEV_ROOT', '/home/ai/proyectos/gemma4-paper')  # data/models/outputs root


def repo_of(iid):
    # 'owner__repo-123' -> 'owner__repo'
    return re.sub(r'-\d+$', '', iid)


def ai_indexes(traj):
    return [i for i, m in enumerate(traj) if m.get('role') == 'ai']


def is_submit(text):
    t = (text or '')
    return bool(re.search(r'```\s*\n?\s*submit\s*\n?```', t)) or t.strip().endswith('submit')


def prefix_fraction(traj, f):
    """Prefix covering a fraction f of the ai steps. f=1: the last submit and everything after it is removed."""
    ai = ai_indexes(traj)
    n = len(ai)
    if n == 0:
        return traj[:]
    k = max(1, math.ceil(f * n))
    if f >= 1.0:
        last = ai[-1]
        if is_submit(traj[last].get('text')):
            return traj[:last]
        return traj[:last + 1]
    end = ai[k - 1]
    return traj[:end + 2]       # action k + its observation


def prefix_steps(traj, k):
    """Absolute prefix: first k ai steps (+ observation). Requires n>k (the final step is not included)."""
    ai = ai_indexes(traj)
    if len(ai) <= k:
        return None
    return traj[:ai[k - 1] + 2]
