import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.deployment_policy import fit, calibrated_holistic, should_abort

ROOT = Path(__file__).resolve().parents[1]


def test_export_matches_dev_calibrator_and_fixed_replay_threshold():
    d = pd.read_csv(ROOT/'results/dev_predictions_all_models.csv')
    r = pd.read_csv(ROOT/'results/dev_runs.csv')
    policy = json.loads(json.dumps(fit(d, r)))
    assert policy['failure_threshold'] == .75
    d = d[(d.model == 'e4b') & (d.prefix == 'abs10')]
    iso = IsotonicRegression(out_of_bounds='clip').fit(d.hol_p, d.target)
    p = np.linspace(0, 1, 101)
    np.testing.assert_allclose([calibrated_holistic(x, policy) for x in p], iso.predict(p))


def test_decision_missing_boundary_and_invalid_inputs():
    policy = {'failure_threshold': .75}
    assert not should_abort(None, policy)
    assert not should_abort(.25, policy)
    assert should_abort(.249, policy)
    assert not should_abort(1, policy)
    for p in [-.1, 1.1, float('nan')]:
        with pytest.raises(ValueError):
            should_abort(p, policy)
