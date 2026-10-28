import json, time, sys, re, random, os
MODEL = sys.argv[1] if len(sys.argv) > 1 else '/home/ai/proyectos/gemma4-paper/models/gemma-4-E4B-it'
N = 20
random.seed(20261027)
rows = [json.loads(l) for l in open('/home/ai/proyectos/gemma4-paper/data/sample.jsonl')]
random.shuffle(rows)
rows = rows[:N]
def render(tr, f=0.66):
    ai = [i for i, m in enumerate(tr) if m['role'] == 'ai']
    end = ai[max(0, int(len(ai) * f) - 1)]
    parts = []
    for m in tr[:end + 2]:
        if m['role'] == 'system': continue
        parts.append(('AGENT: ' if m['role'] == 'ai' else 'ENV: ') + (m['text'] or '')[:1500])
    return '\n'.join(parts)[:24000]
PROMPT = ("You are a reviewer of a software-engineering agent trajectory that is still in progress (partial log below). "
          "Predict whether the agent will end up with a correct patch that resolves the issue.\n"
          "Answer ONLY with a JSON object: {\"p_success\": <number 0-1>, \"verdict\": \"success\"|\"fail\", \"reason\": \"<one sentence>\"}\n\n"
          "=== PARTIAL TRAJECTORY ===\n%s\n=== END ===")
prompts = [PROMPT % render(r['traj']) for r in rows]
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(MODEL, max_model_len=16384, gpu_memory_utilization=0.90, dtype='bfloat16', limit_mm_per_prompt={'image': 0, 'audio': 0}, seed=20261027)
load_s = time.time() - t0
tok = llm.get_tokenizer()
msgs = [[{'role': 'user', 'content': p}] for p in prompts]
# prefill only
t1 = time.time()
o1 = llm.chat(msgs, SamplingParams(max_tokens=1, temperature=0))
pre_s = time.time() - t1
ptoks = sum(len(o.prompt_token_ids) for o in o1)
t2 = time.time()
o2 = llm.chat(msgs, SamplingParams(max_tokens=200, temperature=0))
gen_s = time.time() - t2
gtoks = sum(len(o.outputs[0].token_ids) for o in o2)
valid = 0; outs = []
for r, o in zip(rows, o2):
    txt = o.outputs[0].text
    m = re.search(r'\{.*\}', txt, re.S)
    ok = False
    try:
        j = json.loads(m.group(0)); ok = isinstance(j.get('p_success'), (int, float)) and j.get('verdict') in ('success', 'fail')
    except Exception: pass
    valid += ok
    outs.append({'instance_id': r['instance_id'], 'target': r['target'], 'ok': ok, 'text': txt})
res = {'model': MODEL, 'n': N, 'json_valid': valid, 'json_valid_pct': 100 * valid / N, 'load_s': load_s,
       'prefill_prompt_tokens': ptoks, 'prefill_s': pre_s, 'prefill_tok_s': ptoks / pre_s,
       'gen_tokens': gtoks, 'gen_s': gen_s, 'gen_tok_s_incl_prefill': gtoks / gen_s, 'max_model_len': 16384}
print('E0RESULT', json.dumps(res))
json.dump({'summary': res, 'outputs': outs}, open('e0_result.json', 'w'), indent=1)
