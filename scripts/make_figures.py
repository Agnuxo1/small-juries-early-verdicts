"""Build every figure of the paper (PNG) and of the README (animated GIF) from the CSV/JSON files in results/.
Usage: python scripts/make_figures.py            (from the repository root)
Deterministic: fixed seed, no network, matplotlib + pillow only."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score

RES, FIG = 'results', 'figures'
os.makedirs(FIG, exist_ok=True)
SEED = 20261027
# Validated categorical palette (light), fixed order; neutral gray for the no-LLM control.
C = {'e4b': '#2a78d6', 'e2b': '#1baf7a', 'q4b': '#6250d6', 'base': '#8a8984', 'cap': '#52514e', 'warn': '#eb6834'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
NAME = {'e4b': 'Gemma 4 E4B', 'e2b': 'Gemma 4 E2B', 'q4b': 'Qwen3-4B (control)', 'base': 'Trace features (no LLM)'}
DIMS = ['localization', 'reproduction', 'coherence', 'test_evidence', 'loop', 'scope']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.titleweight': 'bold', 'axes.titlesize': 11,
                     'axes.titlecolor': INK, 'legend.frameon': False, 'grid.color': GRID, 'grid.linewidth': 0.8})

ctr = pd.read_csv(f'{RES}/contrasts.csv')
pred = pd.read_csv(f'{RES}/test_predictions.csv')
curves = pd.read_csv(f'{RES}/tau_curves.csv')
cost = pd.read_csv(f'{RES}/cost.csv')
agree = pd.read_csv(f'{RES}/judge_agreement.csv')
MODELS = [m for m in ('e2b', 'e4b', 'q4b') if m in set(pred.model)]


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def auc_row(prefix, a):
    r = ctr[(ctr.kind == 'auroc') & (ctr.prefix == prefix) & (ctr.a == a)]
    return r.iloc[0] if len(r) else None


# ---------- Fig 2: AUROC forest plot ----------
def fig_auroc():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rows = [('base', 'base', 'Trace features (no LLM)')]
    for m in MODELS:
        rows += [(f'{m}_hol', m, f'{NAME[m]} · holistic'), (f'{m}_jury', m, f'{NAME[m]} · 6-dim jury')]
    rows += [('e4b_stack_jury', 'e4b', 'E4B jury + trace features')]
    for ax, pf in zip(axes, ('abs10', 'abs20')):
        for i, (key, col, lab) in enumerate(rows):
            r = auc_row(pf, key)
            if r is None:
                continue
            y = len(rows) - 1 - i
            hollow = key.endswith('_hol')
            ax.plot([r.lo, r.hi], [y, y], color=C[col], lw=2, solid_capstyle='round')
            ax.plot(r.est, y, 'o', ms=8, mfc=SURF if hollow else C[col], mec=C[col], mew=2)
            ax.text(r.hi + 0.008, y, f'{r.est:.3f}', va='center', fontsize=8.5, color=INK2)
        b = auc_row(pf, 'base')
        ax.axvline(0.5, color=GRID, lw=1)
        ax.axvline(b.est, color=C['base'], lw=1, ls=':')
        n = int(b.n)
        ax.set_title(f'First {pf[3:]} agent steps  (n={n}, {int(b.n_pos)} resolved)', loc='left')
        ax.set_xlim(0.45, 0.86)
        ax.grid(axis='x')
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[2] for r in rows][::-1])
    fig.suptitle('Small local judges read the first steps of a SWE-agent run and predict whether it will succeed',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold', color=INK)
    fig.supxlabel('AUROC for predicting final success, test split (dot = estimate, bar = 95% repo-cluster bootstrap CI)',
                  fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, 'fig2_auroc_forest')


# ---------- Fig 3: budget gain vs threshold ----------
def gain_rows(prefix, m):
    g = {}
    for a in ('jury', 'base', 'cap'):
        r = ctr[(ctr.kind == 'h4_rel_gain') & (ctr.prefix == prefix) & (ctr.a == f'{m}_{a}') & (ctr.b == 'no_abort')]
        g[a] = r.iloc[0] if len(r) else None
    return g


def fig_budget():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pf in zip(axes, ('abs10', 'abs20')):
        cv = curves[(curves.prefix == pf) & (curves.model == 'e4b')]
        g = gain_rows(pf, 'e4b')
        ax.axhline(0, color=GRID, lw=1)
        ax.axhline(g['cap'].est * 100, color=C['cap'], lw=1.2, ls='--', label='abort every run at the decision step')
        ax.plot(cv.tau, cv.base_test * 100, color=C['base'], lw=2, label=NAME['base'])
        ax.plot(cv.tau, cv.jury_test * 100, color=C['e4b'], lw=2, label='Gemma 4 E4B jury')
        for a, col in (('jury', C['e4b']), ('base', C['base'])):
            t = g[a].tau
            yv = cv.loc[(cv.tau - t).abs().idxmin(), f'{a}_test'] * 100
            ax.plot(t, yv, 'o', ms=9, color=col, mec=SURF, mew=2, zorder=4)
        box = '\n'.join(['gain at the τ chosen on dev [95% CI]',
                         f'E4B jury        {100 * g["jury"].est:+.0f}%  [{100 * g["jury"].lo:+.0f}, {100 * g["jury"].hi:+.0f}]',
                         f'trace features  {100 * g["base"].est:+.0f}%  [{100 * g["base"].lo:+.0f}, {100 * g["base"].hi:+.0f}]',
                         f'fixed cap       {100 * g["cap"].est:+.0f}%  [{100 * g["cap"].lo:+.0f}, {100 * g["cap"].hi:+.0f}]'])
        ax.text(0.98, 0.04, box, transform=ax.transAxes, ha='right', va='bottom', multialignment='left', fontsize=8,
                family='monospace', color=INK2,
                bbox=dict(boxstyle='round,pad=0.5', fc=SURF, ec=GRID))
        ax.set_title(f'Decision after {pf[3:]} steps', loc='left')
        ax.set_xlabel('abort threshold τ on P(failure)')
        ax.grid(axis='y')
    axes[0].set_ylabel('resolved tasks per agent step\nvs. never aborting (%)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].set_ylim(-22, 40)
    fig.suptitle('Jury-guided early abort turns wasted steps into more solved tasks (test split, 600 runs)', x=0.01,
                 ha='left', fontsize=11.5, fontweight='bold', color=INK)
    fig.tight_layout()
    save(fig, 'fig3_budget_gain')


# ---------- Fig 4: calibration ----------
def reliability(y, p, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    xs, ys, ns = [], [], []
    for k in range(nb):
        mk = b == k
        if mk.sum() >= 5:
            xs.append(p[mk].mean()); ys.append(y[mk].mean()); ns.append(mk.sum())
    return np.array(xs), np.array(ys), np.array(ns)


def fig_calibration():
    d = pred[(pred.model == 'e4b') & (pred.prefix == 'abs10')]
    y = d.target.values
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    met = pd.read_csv(f'{RES}/metrics_preregistered.csv')
    for col, key, lab, colr, mk in (('hol_p', 'e4b_hol', 'raw holistic probability', C['warn'], 's'),
                                   ('jury_p', 'e4b_jury', '6-dim jury + logistic (fit on dev)', C['e4b'], 'o')):
        xs, ys, ns = reliability(y, d[col].values)
        e = met[(met.prefix == 'abs10') & (met.score == key)].ece.iloc[0]
        ax.plot(xs, ys, '-', color=colr, lw=2)
        ax.scatter(xs, ys, s=12 + ns / 2, color=colr, marker=mk, edgecolor=SURF, lw=1.5, zorder=3, label=f'{lab} (ECE {e:.3f})')
    ax.set_xlabel('predicted P(success)')
    ax.set_ylabel('observed success rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Calibration, Gemma 4 E4B, first 10 steps', loc='left')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid()
    save(fig, 'fig4_calibration')


# ---------- Fig 5: judge validation ----------
def fig_agreement():
    a = agree[agree.dim != 'mean']
    ms = [m for m in ('e4b', 'e2b', 'q4b') if m in set(a.model)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={'width_ratios': [1.25, 1]})
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        r = a[a.model == m].set_index('dim').loc[DIMS]
        x = np.arange(len(DIMS)) + (i - (len(ms) - 1) / 2) * w
        ax.bar(x, r.qwk, w * 0.92, color=C[m], label=NAME[m], edgecolor=SURF, lw=2)
        ax.errorbar(x, r.qwk, yerr=[r.qwk - r.qwk_lo, r.qwk_hi - r.qwk], fmt='none', ecolor=INK2, lw=1, capsize=0)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([d.replace('_', ' ') for d in DIMS], fontsize=8.5, rotation=25, ha='right')
    ax.set_ylabel("quadratic-weighted Cohen's κ")
    ax.axhline(0, color=GRID)
    for t, lab in ((0.4, 'moderate'), (0.6, 'substantial')):
        ax.axhline(t, color=GRID, ls=':', lw=1)
        ax.text(5.45, t + 0.01, lab, fontsize=7.5, color=INK2, ha='right')
    ax.set_title('Agreement with a blind reference annotation (80 test items)', loc='left')
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    r = a[a.model == 'e4b'].set_index('dim').loc[DIMS]
    x = np.arange(len(DIMS))
    ax2.bar(x - 0.2, r.mean_ref, 0.38, color=C['base'], label='reference annotator', edgecolor=SURF, lw=2)
    ax2.bar(x + 0.2, r.mean_model, 0.38, color=C['e4b'], label='Gemma 4 E4B jury', edgecolor=SURF, lw=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace('_', ' ') for d in DIMS], fontsize=8.5, rotation=25, ha='right')
    ax2.set_ylim(1, 5)
    ax2.set_ylabel('mean rating (1-5)')
    ax2.set_title('Small judges are lenient, most on "coherence"', loc='left')
    ax2.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    save(fig, 'fig5_judge_validation')


# ---------- Fig 6: dimensions ----------
def fig_dims():
    met = pd.read_csv(f'{RES}/metrics_preregistered.csv')
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ms = [m for m in ('e2b', 'e4b') if f'{m}_dim_loop' in set(met.score)]
    w = 0.38
    for i, m in enumerate(ms):
        r = met[(met.prefix == 'abs10') & met.score.isin([f'{m}_dim_{d}' for d in DIMS])].set_index('score').loc[[f'{m}_dim_{d}' for d in DIMS]]
        x = np.arange(len(DIMS)) + (i - 0.5) * w
        ax.bar(x, r.auroc - 0.5, w * 0.92, bottom=0.5, color=C[m], label=NAME[m], edgecolor=SURF, lw=2)
        ax.errorbar(x, r.auroc, yerr=[r.auroc - r.auc_lo, r.auc_hi - r.auroc], fmt='none', ecolor=INK2, lw=1)
    ax.axhline(0.5, color=GRID)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([d.replace('_', '\n') for d in DIMS], fontsize=8.5)
    ax.set_ylabel('AUROC of the single dimension')
    ax.set_ylim(0.45, 0.72)
    ax.set_title('Which rubric dimension carries the signal? (first 10 steps)', loc='left')
    ax.legend(fontsize=8)
    ax.grid(axis='y')
    save(fig, 'fig6_dimensions')


# ---------- Fig 7: cost vs quality ----------
def fig_cost():
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for m in MODELS:
        for judge, mk, key in (('holistic', 's', 'hol'), ('jury', 'o', 'jury')):
            c = cost[(cost.model == m) & (cost.set == 'test') & (cost.prefix == 'abs10') & (cost.judge == judge)]
            r = auc_row('abs10', f'{m}_{key}')
            if c.empty or r is None:
                continue
            s = c.sec.iloc[0] / c.n.iloc[0]
            ax.plot([s, s], [r.lo, r.hi], color=C[m], lw=1.5, alpha=0.6)
            ax.plot(s, r.est, mk, ms=9, color=C[m], mfc=SURF if key == 'hol' else C[m], mew=2)
            ax.annotate(f'{NAME[m].split(" (")[0]} {judge}', (s, r.est), xytext=(7, -3), textcoords='offset points', fontsize=8, color=INK2)
    b = auc_row('abs10', 'base')
    ax.axhline(b.est, color=C['base'], ls=':', lw=1.2)
    ax.text(ax.get_xlim()[0] if False else 0.32, b.est + 0.004, 'trace features (CPU, ~0 s)', fontsize=8, color=C['base'])
    ax.set_xlabel('GPU seconds per judged run (one RTX 3090, vLLM, bf16)')
    ax.set_ylabel('AUROC (first 10 steps)')
    ax.set_title('Quality vs. cost of the judge', loc='left')
    ax.grid()
    save(fig, 'fig7_cost_quality')


# ---------- GIF A: tau sweep ----------
def gif_tau():
    pf = 'abs10'
    cv = curves[(curves.prefix == pf) & (curves.model == 'e4b')].reset_index(drop=True)
    g = gain_rows(pf, 'e4b')
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={'width_ratios': [1.6, 1]}, dpi=90)
    ax.axhline(0, color=GRID)
    ax.axhline(g['cap'].est * 100, color=C['cap'], ls='--', lw=1.2)
    ax.plot(cv.tau, cv.base_test * 100, color=C['base'], lw=2, label='trace features')
    ax.plot(cv.tau, cv.jury_test * 100, color=C['e4b'], lw=2, label='Gemma 4 E4B jury')
    ax.set_xlabel('abort threshold τ on P(failure)')
    ax.set_ylabel('solved tasks per step vs. no abort (%)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(axis='y')
    vl = ax.axvline(cv.tau[0], color=INK, lw=1)
    ax.set_title('Sweeping the abort threshold (decision at step 10)', loc='left', fontsize=10)
    labs = ['never abort', 'abort all\nat step 10', 'trace\nfeatures', 'E4B jury']
    cols = [GRID, C['cap'], C['base'], C['e4b']]
    bars = axb.bar(range(4), [0, 0, 0, 0], color=cols, edgecolor=SURF, lw=2)
    axb.set_xticks(range(4)); axb.set_xticklabels(labs, fontsize=8)
    axb.set_ylim(-30, 45); axb.axhline(0, color=GRID)
    axb.set_title('gain at this τ', loc='left', fontsize=10)
    txt = axb.text(0.02, 0.92, '', transform=axb.transAxes, fontsize=9, color=INK2)
    fig.tight_layout()
    frames = list(range(0, len(cv), 2)) + [len(cv) - 1] * 6

    def up(i):
        vl.set_xdata([cv.tau[i]] * 2)
        vals = [0, g['cap'].est * 100, cv.base_test[i] * 100, cv.jury_test[i] * 100]
        for b, v in zip(bars, vals):
            b.set_height(v)
        txt.set_text(f'τ = {cv.tau[i]:.2f}   (dev-chosen τ = {g["jury"].tau:.2f})')
        return list(bars) + [vl, txt]
    FuncAnimation(fig, up, frames=frames, blit=False).save(f'{FIG}/anim_tau_sweep.gif', writer=PillowWriter(fps=8))
    plt.close(fig)


# ---------- GIF B: budget race ----------
def gif_race():
    d = pred[(pred.model == 'e4b') & (pred.prefix == 'abs10')].set_index('uid')
    allr = pd.read_csv(f'{RES}/test_runs.csv')       # 600 test runs: uid, target, n_ai_total
    g = gain_rows('abs10', 'e4b')
    rng = np.random.default_rng(SEED)
    allr = allr.iloc[rng.permutation(len(allr))].reset_index(drop=True)
    jp = allr.uid.map(d.jury_p).fillna(1.0).values      # runs with <=10 steps are never judged/aborted
    bp = allr.uid.map(d.base_p).fillna(1.0).values
    n, y = allr.n_ai_total.values, allr.target.values
    rob = json.load(open(f'{RES}/robustness.json'))['A_B_budget_e4b']
    cap, (tj2, cp2) = rob['cap_dev'], rob['combined_dev']
    nocap = 10 ** 9
    lanes = {'never abort': (np.zeros(len(n), bool), nocap, '#b5b4ad'),
             'trace-feature abort at step 10': ((1 - bp) > g['base'].tau, nocap, C['base']),
             f'step cap at {cap} (tuned on dev)': (np.zeros(len(n), bool), cap, C['cap']),
             'Gemma 4 E4B jury abort at step 10': ((1 - jp) > g['jury'].tau, nocap, C['e4b']),
             f'jury at step 10 + cap at {cp2}': ((1 - jp) > tj2, cp2, '#123f78')}
    BUD = 6000
    series = {}
    for k, (ab, cp, col) in lanes.items():
        capped = (~ab) & (n > cp)
        steps = np.where(ab, 10, np.where(capped, cp, n))
        solved = np.where(ab | capped, 0, y)
        cs, cy = np.cumsum(steps), np.cumsum(solved)
        xs = np.linspace(0, BUD, 121)
        series[k] = (xs, np.array([cy[cs <= x][-1] if (cs <= x).any() else 0 for x in xs]), col)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=90)
    lines = {k: ax.plot([], [], lw=2.5, color=c, label=k)[0] for k, (_, _, c) in series.items()}
    lbl = {k: ax.text(0, 0, '', fontsize=9, color=INK2) for k in series}
    ax.set_xlim(0, BUD * 1.18); ax.set_ylim(0, max(v[1].max() for v in series.values()) * 1.15)
    ax.set_xlabel('agent steps spent (shared budget)')
    ax.set_ylabel('tasks resolved')
    ax.set_title('Same budget, more fixes: held-out runs replayed in one random order (illustrates H4)', loc='left', fontsize=10)
    ax.legend(loc='upper left', fontsize=8.5)
    ax.grid()
    fig.tight_layout()

    gap = ax.get_ylim()[1] * 0.045

    def up(i):
        prev = -1e9
        for k in sorted(series, key=lambda q: series[q][1][i]):
            xs, ys, _ = series[k]
            lines[k].set_data(xs[:i + 1], ys[:i + 1])
            yl = max(ys[i], prev + gap)          # keep end labels from overlapping
            prev = yl
            lbl[k].set_position((xs[i] + 60, yl))
            lbl[k].set_text(str(int(ys[i])))
        return list(lines.values()) + list(lbl.values())
    FuncAnimation(fig, up, frames=list(range(121)) + [120] * 12, blit=False).save(f'{FIG}/anim_budget_race.gif', writer=PillowWriter(fps=12))
    for k in series:
        up(120)
    fig.savefig(f'{FIG}/fig8_budget_race.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------- GIF C: how it works ----------
def gif_pipeline():
    d = pred[(pred.model == 'e4b') & (pred.prefix == 'abs10')]
    ok = d[d.target == 1].sort_values('jury_p').iloc[-3]
    bad = d[d.target == 0].sort_values('jury_p').iloc[2]
    fig = plt.figure(figsize=(9.6, 4.6), dpi=90)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 96); ax.set_ylim(0, 46); ax.axis('off')
    ax.text(2, 43, 'Small Juries, Early Verdicts', fontsize=15, fontweight='bold', color=INK)
    ax.text(2, 39.6, 'A local Gemma 4 jury reads the first 10 steps of a SWE-agent run and decides whether to keep paying for it',
            fontsize=9.5, color=INK2)
    rows = [(ok, 25.5, 'run A'), (bad, 6.5, 'run B')]
    art = []
    for r, y0, nm in rows:
        ax.text(2, y0 + 10, f'{nm}: {r.instance_id}', fontsize=8.5, color=INK2)
        steps = [ax.add_patch(FancyBboxPatch((2 + i * 3.1, y0 + 4.5), 2.5, 3.6, boxstyle='round,pad=0.1,rounding_size=0.5',
                                             fc=GRID, ec='none', alpha=0)) for i in range(10)]
        ax.text(2, y0 + 1.8, 'issue + 10 agent steps (actions & tool output)', fontsize=7.5, color=INK2)
        bars, vals = [], [getattr(r, dm) for dm in DIMS]
        for j, dm in enumerate(DIMS):
            yy = y0 + 8.6 - j * 1.55
            ax.text(44, yy, dm.replace('_', ' '), fontsize=7.5, color=INK2, ha='right', va='center')
            ax.add_patch(FancyBboxPatch((45, yy - 0.5), 13, 1.0, boxstyle='round,pad=0,rounding_size=0.4', fc=GRID, ec='none'))
            bars.append(ax.add_patch(FancyBboxPatch((45, yy - 0.5), 0.01, 1.0, boxstyle='round,pad=0,rounding_size=0.4',
                                                    fc=C['e4b'], ec='none')))
        ax.text(51.5, y0 + 10.4, 'Gemma 4 E4B jury (1-5)', fontsize=8, color=INK, ha='center', fontweight='bold')
        pt = ax.text(62, y0 + 6.2, '', fontsize=13, fontweight='bold', color=INK)
        vt = ax.text(62, y0 + 2.6, '', fontsize=9.5, color=INK2)
        art.append((steps, bars, vals, pt, vt, r))
    tau = gain_rows('abs10', 'e4b')['jury'].tau

    def up(f):
        for steps, bars, vals, pt, vt, r in art:
            for i, s in enumerate(steps):
                s.set_alpha(1.0 if f > i else 0.0)
                s.set_facecolor(C['e4b'] if f > i and f <= 10 else '#c9d9ef')
            prog = np.clip((f - 11) / 8, 0, 1)
            for b, v in zip(bars, vals):
                b.set_width(13 * v / 5 * prog)
            if f >= 20:
                pt.set_text(f'P(success) = {r.jury_p:.2f}')
                abort = (1 - r.jury_p) > tau
                vt.set_text(('ABORT, reassign budget' if abort else 'CONTINUE') + f'   ·   true outcome: {"resolved" if r.target else "failed"}')
                vt.set_color(C['warn'] if abort else C['e2b'])
            else:
                pt.set_text(''); vt.set_text('')
        return []
    FuncAnimation(fig, up, frames=list(range(0, 21)) + [20] * 18, blit=False).save(f'{FIG}/anim_how_it_works.gif', writer=PillowWriter(fps=5))
    up(20)
    fig.savefig(f'{FIG}/fig1_how_it_works.png', dpi=200)
    plt.close(fig)


# ---------- Kaggle card (560 x 280, rendered at 2x) ----------
def card():
    cv = curves[(curves.prefix == 'abs10') & (curves.model == 'e4b')]
    g = gain_rows('abs10', 'e4b')
    fig = plt.figure(figsize=(5.6, 2.8), dpi=200)
    fig.patch.set_facecolor('#0f1b2d')
    fig.text(0.05, 0.80, 'Small Juries,\nEarly Verdicts', fontsize=17, fontweight='bold', color='white', va='top', linespacing=1.05)
    fig.text(0.05, 0.40, 'Gemma 4 E4B reads 10 agent steps\nand predicts failure early', fontsize=8.5, color='#c3c2b7', va='top')
    fig.text(0.05, 0.15, f'{100 * g["jury"].est:+.0f}%', fontsize=22, fontweight='bold', color='#5aa2f0', va='bottom')
    fig.text(0.05, 0.08, 'resolved tasks per step, same budget', fontsize=7, color='#c3c2b7', va='bottom')
    ax = fig.add_axes([0.56, 0.18, 0.40, 0.66])
    ax.set_facecolor('#0f1b2d')
    ax.axhline(0, color='#3a4a60', lw=0.8)
    ax.plot(cv.tau, cv.base_test * 100, color='#8a8984', lw=1.5, label='trace features')
    ax.plot(cv.tau, cv.jury_test * 100, color='#5aa2f0', lw=2, label='Gemma 4 jury')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors='#8a8984', labelsize=6)
    ax.set_xlabel('abort threshold', fontsize=6.5, color='#8a8984')
    ax.legend(fontsize=6, labelcolor='#c3c2b7', loc='upper right')
    ax.set_ylabel('gain vs no abort (%)', fontsize=6.5, color='#8a8984')
    fig.savefig(f'{FIG}/kaggle_card.png', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig_auroc, fig_budget, fig_calibration, fig_agreement, fig_dims, fig_cost, gif_tau, gif_race, gif_pipeline, card):
        f()
        print('ok', f.__name__, flush=True)
