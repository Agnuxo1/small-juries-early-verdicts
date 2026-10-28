"""Build the two published forms of the paper from paper/writeup.md:
  paper/small-juries-early-verdicts.pdf  (pandoc + XeLaTeX; local figures, captions, sized images)
  paper/kaggle_writeup.md                 (Kaggle Writeup body; figures served from the public GitHub repository)
Usage: python scripts/build_paper.py   (from the repository root; needs pandoc and a XeLaTeX install)"""
import re, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / 'paper' / 'writeup.md').read_text(encoding='utf8')
RAW = 'https://raw.githubusercontent.com/Agnuxo1/small-juries-early-verdicts/main/figures/'
assert '[[' not in SRC, 'unfilled placeholder in writeup.md'
CAPTIONS = {
    'fig2_auroc_forest.png': ('Figure 1. AUROC for predicting final success from the first k agent steps (test split, 95% '
                              'repository-cluster bootstrap CIs). Hollow markers: holistic judge; filled: six-dimension jury.', '100%'),
    'fig4_calibration.png': ('Figure 2. Reliability diagram, Gemma 4 E4B at k = 10: raw holistic probabilities vs. the jury '
                             'with a logistic layer fitted on development data. Marker area is proportional to bin size.', '55%'),
    'fig7_cost_quality.png': ('Figure 3. Discrimination vs. GPU time per judgement (k = 10, batched vLLM throughput on one '
                              'RTX 3090). Hollow: holistic; filled: jury.', '62%'),
    'fig3_budget_gain.png': ('Figure 4. Resolved tasks per agent step relative to never aborting, as a function of the abort '
                             'threshold (test split). Dots mark the threshold chosen on development data.', '100%'),
    'fig5_judge_validation.png': ('Figure 5. Left: agreement (quadratic-weighted kappa, 95% CI) between the Gemma jury and a '
                                  'blind reference annotation of 80 test prompts. Right: mean rating per dimension.', '100%'),
}


def pdf_md(s):
    def rep(m):
        f = m.group(2).split('/')[-1]
        cap, w = CAPTIONS.get(f, (m.group(1), '90%'))
        return f'![{cap}]({m.group(2)}){{width={w}}}'
    s = re.sub(r'!\[(.*?)\]\((.*?)\)', rep, s)
    s = re.sub(r'^(\[\d+\] .*)$', r'\1\\', s, flags=re.M)           # one reference per line
    s = s.replace('# Small Juries, Early Verdicts\n', '---\ntitle: "Small Juries, Early Verdicts"\n'
                  'author: "Francisco Angulo de Lafuente (Agnuxo1)"\ndate: "Gemma 4 Developer Agent Paper Track, November 2026"\n---\n', 1)
    return re.sub(r'^### .*\n', '', s, count=1)                       # subtitle is passed to pandoc as a variable


def kaggle_md(s):
    s = re.sub(r'!\[(.*?)\]\(\.\./figures/(.*?)\)', lambda m: f'![{CAPTIONS.get(m.group(2), (m.group(1),))[0]}]({RAW}{m.group(2)})', s)
    s = s.replace('(`results/metrics_preregistered.csv`.) ', '')
    return re.sub(r'^# .*\n+(### .*\n+)?', '', s, count=1)       # title/subtitle go in the Kaggle form fields


(ROOT / 'paper' / '_pdf_src.md').write_text(pdf_md(SRC), encoding='utf8')
(ROOT / 'paper' / 'kaggle_writeup.md').write_text(kaggle_md(SRC), encoding='utf8')
subprocess.run(['pandoc', '_pdf_src.md', '-o', 'small-juries-early-verdicts.pdf', '--pdf-engine=xelatex',
                '--resource-path=.:..', '-V', 'mainfont=Cambria', '-V', 'monofont=Consolas', '-V', 'geometry:margin=2.1cm',
                '-V', 'fontsize=10pt', '-V', 'colorlinks=true', '-V', 'linkcolor=blue', '-V', 'urlcolor=blue',
                '-V', 'subtitle=' + re.search(r'^### (.*)$', SRC, re.M).group(1)], cwd=ROOT / 'paper', check=True)
(ROOT / 'paper' / '_pdf_src.md').unlink()
words = len(re.findall(r'\S+', re.sub(r'!\[.*?\]\(.*?\)', '', SRC)))
print('built PDF and Kaggle markdown; words in writeup.md (figures excluded):', words)
