"""E2: prefix rendering and judge prompts (holistic and 6-dimension jury).
Prompts are tuned only on the 50-item pilot (development split), never on the test set."""
import re, json
from common import prefix_fraction, prefix_steps

PROMPT_VERSION = 'v1'
DIMS = ['localization', 'reproduction', 'coherence', 'test_evidence', 'loop', 'scope']

HEAD = ("You are an expert reviewer of autonomous software-engineering agents (SWE-agent style). Below is the BEGINNING of the "
        "working log of an agent trying to resolve a GitHub issue in a repository. The log is PARTIAL: the agent has not finished "
        "and the final outcome is NOT known to you.\n\n")

HOLISTIC_TASK = ("TASK: Estimate the probability that this agent will finish with a patch that resolves the issue and passes the hidden tests. "
                 "Most attempts fail (roughly 1 in 4 succeeds), so use the whole 0-1 range and do not default to 0.5.\n"
                 "Reply with ONLY a JSON object, no other text, exactly in this form:\n"
                 "{\"p_success\": <number between 0 and 1>}")

JURY_TASK = ("TASK: Rate the partial log on six dimensions with integers from 1 (very poor) to 5 (very good). "
             "If a dimension cannot be judged yet, use 3.\n"
             "- localization: has the agent found the files/functions that actually need to change?\n"
             "- reproduction: has the agent reproduced or understood the bug (script run, failing behaviour observed)?\n"
             "- coherence: do the edits and plan so far match what the issue actually asks for?\n"
             "- test_evidence: has the agent run tests or scripts whose output supports its current fix?\n"
             "- loop: 5 = progressing with new actions; 1 = stuck repeating the same commands or failing edits.\n"
             "- scope: 5 = changes focused and minimal on relevant code; 1 = unfocused, over-broad or touching unrelated files.\n"
             "Reply with ONLY a JSON object, no other text, exactly in this form:\n"
             "{\"localization\": <1-5>, \"reproduction\": <1-5>, \"coherence\": <1-5>, \"test_evidence\": <1-5>, \"loop\": <1-5>, \"scope\": <1-5>}")


def clip(t, n):
    t = t or ''
    if len(t) <= n:
        return t
    h = int(n * 0.7)
    return t[:h] + '\n...[truncated]...\n' + t[-(n - h):]


def issue_of(t):
    t = t or ''
    m = re.search(r'ISSUE:\s*(.*?)\s*INSTRUCTIONS:', t, re.S)
    return (m.group(1) if m else t)


def clean_obs(t):
    t = t or ''
    return re.sub(r'\n*\(Open file: ([^\n]*)\)\n\(Current directory: [^\n]*\)\nbash-\$\s*$', r'\n[open file: \1]', t)


def get_prefix(traj, spec):
    """spec: 'abs10' | 'abs20' | 'frac33' | 'frac66' | 'issue10'. None if there are not enough steps.
    'issueK' (ablation added after review): same runs as 'absK' but the log is cut before the first agent step,
    so the judge sees only the issue."""
    if spec.startswith('issue'):
        p = prefix_steps(traj, int(spec[5:]))
        if p is None:
            return None
        first_ai = next(i for i, m in enumerate(p) if m.get('role') == 'ai')
        return p[:first_ai]
    if spec.startswith('abs'):
        return prefix_steps(traj, int(spec[3:]))
    return prefix_fraction(traj, int(spec[4:]) / 100.0)


def render(prefix, caps):
    ac, oc, ic = caps
    msgs = [m for m in prefix if m['role'] != 'system']
    issue = clip(issue_of(msgs[0]['text']), ic) if msgs else ''
    parts, n = [], 0
    for m in msgs[1:]:
        if m['role'] == 'ai':
            n += 1
            parts.append('[step %d] AGENT: %s' % (n, clip(m['text'], ac)))
        else:
            parts.append('[step %d] ENVIRONMENT: %s' % (n, clip(clean_obs(m['text']), oc)))
    return issue, parts, n


CAPS = [(1500, 1200, 4000), (1000, 800, 3000), (600, 500, 2500), (400, 300, 2000), (250, 200, 1500)]


def build_prompt(prefix, judge, count_tokens, budget=13000):
    """Reduce the per-message caps until the log fits in `budget` tokens; if it still does not fit, drop intermediate steps."""
    task = HOLISTIC_TASK if judge == 'holistic' else JURY_TASK
    for caps in CAPS:
        issue, parts, n = render(prefix, caps)
        body = '\n'.join(parts)
        if count_tokens(issue + body) <= budget:
            break
    dropped = 0
    while count_tokens(issue + body) > budget and len(parts) > 8:
        mid = len(parts) // 2
        del parts[mid - 1:mid + 1]
        dropped += 2
        body = '\n'.join(parts[:len(parts) // 2] + ['...[%d middle messages omitted]...' % dropped] + parts[len(parts) // 2:])
    prompt = (HEAD + '=== ISSUE ===\n' + issue + '\n=== AGENT LOG SO FAR (%d agent steps) ===\n' % n + body +
              '\n=== END OF LOG ===\n\n' + task)
    return prompt, n


def parse(text, judge):
    m = re.search(r'\{.*?\}', text or '', re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    if judge == 'holistic':
        p = j.get('p_success')
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0 <= p <= 1):
            return None
        return {'p_success': float(p)}
    out = {}
    for d in DIMS:
        v = j.get(d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (1 <= v <= 5):
            return None
        out[d] = float(v)
    return out
