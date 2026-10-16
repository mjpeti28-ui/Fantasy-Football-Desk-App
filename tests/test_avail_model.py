"""avail_model.py: design/curve algebra and the horizon curve used by valuation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import avail_model as am  # noqa: E402
import valuation as v  # noqa: E402


def _fe(**kw):
    row = {c: 0.0 for c in am.FEATURES + am.POS}
    row.update(kw)
    row["espn_id"] = 1
    return pd.DataFrame([row])


def test_curve_params_match_the_fitted_design():
    rng = np.random.default_rng(0)
    n = 4000
    Fe = pd.DataFrame({c: rng.integers(0, 2, n).astype(float) for c in am.FEATURES + am.POS})
    Fe["espn_id"] = np.arange(n)
    h = rng.integers(0, 10, n).astype(float)
    true_lin = 2.0 - 3.0 * Fe["espn0"] - 0.3 * np.log1p(h) + 1.5 * Fe["espn0"] * np.log1p(h)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_lin))).astype(float)
    m = am.fit(Fe, h, y)
    X = np.column_stack([np.ones(n), am.design(Fe, h)])
    direct = 1 / (1 + np.exp(-(X @ np.asarray(m["coef"]))))
    cp = am.curve_params(m, Fe)
    via = np.where(h == 0, cp["p_next"], am.later_prob(cp["av_a"], cp["av_b"], h))
    assert np.allclose(direct, via, atol=1e-9)


def test_avail_for_week_uses_horizon_curve_and_skips_bye():
    tbl = pd.DataFrame({"est_ppg": [10.0], "avail_now": [0.1], "avail": [0.5], "bye": [6], "from_week": [4],
                        "av_a": [0.0], "av_b": [1.0]}, index=[1])
    assert v._avail_for_week(tbl, 4)[0] == 0.1                       # next game
    assert np.isclose(v._avail_for_week(tbl, 5)[0], 1 / (1 + np.exp(-np.log1p(1))))  # h = 1
    assert v._avail_for_week(tbl, 6)[0] == 0.0                       # bye
    assert np.isclose(v._avail_for_week(tbl, 7)[0], 1 / (1 + np.exp(-np.log1p(2))))  # h = 2 (bye skipped)
