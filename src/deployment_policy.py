"""Portable, dev-fitted postprocessing; no test data or model API required."""
import argparse
import json
from pathlib import Path

import numpy as np


def probability(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a finite probability in [0, 1]')
    return value


def should_abort(success_probability, policy):
    """Missing judgements continue; decide at the recorded absolute prefix only."""
    if success_probability is None:
        return False
    return 1 - probability(success_probability) > policy['failure_threshold']


def calibrated_holistic(success_probability, policy):
    """Piecewise-linear sklearn isotonic export, clipped outside dev support."""
    p = probability(success_probability)
    return float(np.interp(p, policy['isotonic_x'], policy['isotonic_y']))


def fit(dev_predictions, dev_runs):
    from sklearn.isotonic import IsotonicRegression
    d = dev_predictions[(dev_predictions.model == 'e4b') &
                        (dev_predictions.prefix == 'abs10')].copy()
    if not d.uid.is_unique or not dev_runs.uid.is_unique:
        raise ValueError('Duplicate run IDs')
    z = dev_runs.merge(d[['uid', 'jury_oof']], on='uid', how='left', validate='one_to_one')
    judged = z.jury_oof.notna().to_numpy()
    p = z.jury_oof.fillna(1).to_numpy()
    taus = np.round(np.linspace(.30, .99, 70), 4)
    def efficiency(tau):
        abort = 1 - p > tau
        steps = np.where(abort, 10, z.n_ai_total.to_numpy()) + judged
        return float(np.where(abort, 0, z.target.to_numpy()).sum() / steps.sum())
    tau = max(taus, key=efficiency)  # fixed grid; ties take the first threshold
    iso = IsotonicRegression(out_of_bounds='clip').fit(d.hol_p, d.target)
    return {'schema': 1, 'model': 'e4b', 'prefix': 'abs10',
            'failure_threshold': float(tau), 'judgement_overhead_steps': 1,
            'policy_probability': 'jury_p', 'calibration_probability': 'hol_p',
            'dev_efficiency': efficiency(tau), 'fit_split': 'development_only',
            'isotonic_x': iso.X_thresholds_.tolist(),
            'isotonic_y': iso.y_thresholds_.tolist(),
            'note': 'Calibration is for holistic scores; abort threshold uses jury probabilities.'}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, default=Path('results/deployment_policy.json'))
    args = ap.parse_args()
    policy = fit(pd.read_csv(args.results / 'dev_predictions_all_models.csv'),
                 pd.read_csv(args.results / 'dev_runs.csv'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.out), 'threshold': policy['failure_threshold'],
                      'fit_split': policy['fit_split']}))


if __name__ == '__main__':
    main()
