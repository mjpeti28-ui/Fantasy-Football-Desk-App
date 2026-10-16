"""The no-look-ahead rule and the small numerical helpers behind backtest/mode_strength.py."""
import numpy as np

from backtest import mode_strength as ms


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(0, 5, size=(3, ms.NW, 2)).astype(float)
    return n, rng.normal(size=n.shape) * n


def test_strength_uses_only_earlier_weeks():
    n, s = _toy()
    K = np.array([20.0, 20.0])
    d, o, _, _ = ms.strength(n, s, K, 50.0)
    n2, s2 = n.copy(), s.copy()
    n2[:, 7, :] += 3
    s2[:, 7, :] += 9           # a week-7 game must change week 8 onward, never week 7 or earlier
    d2, o2, _, _ = ms.strength(n2, s2, K, 50.0)
    assert np.allclose(d[:, :8], d2[:, :8]) and np.allclose(o[:, :8], o2[:, :8])
    assert not np.allclose(d[:, 8], d2[:, 8])
    # the deliberate look-ahead (positive control) does see week 7
    di, _, _, _ = ms.strength(n, s, K, 50.0, incl=True)
    di2, _, _, _ = ms.strength(n2, s2, K, 50.0, incl=True)
    assert not np.allclose(di[:, 7], di2[:, 7])


def test_no_plays_means_no_strength():
    n = np.zeros((1, ms.NW, 3))
    d, o, _, _ = ms.strength(n, n.copy(), np.array([10.0, 10.0, 10.0]), 30.0)
    assert np.all(d == 0) and np.all(o == 0)


def test_more_plays_means_less_shrinkage():
    K = np.array([100.0])
    few, many = np.zeros((1, ms.NW, 1)), np.zeros((1, ms.NW, 1))
    few[0, 1, 0], many[0, 1, 0] = 5, 500
    d1, _, _, _ = ms.strength(few, few * 0.3, K, 1e9)    # same +0.3 per play, 100x the sample
    d2, _, _, _ = ms.strength(many, many * 0.3, K, 1e9)
    assert 0 < d1[0, 5, 0] < d2[0, 5, 0] < 0.3


def test_clustered_ols_matches_lstsq():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 5)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = 2 + 0.5 * X[:, 1] + rng.normal(size=200)
    b, se = ms.ols_cl(y, X, np.r_[0, np.flatnonzero(np.diff(g)) + 1])
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0]) and np.all(se > 0)


def test_bh_is_monotone_and_never_below_p():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.5])
    q = ms.bh(p)
    assert np.all(q >= p - 1e-12) and np.all(q <= 1)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_db_count_parser_handles_both_formats():
    assert ms._dbs("3 CB, 2 DE, 2 DT, 1 MLB, 1 OLB, 2 SS") == 5.0
    assert ms._dbs("4 DL, 2 LB, 5 DB") == 5.0
    assert np.isnan(ms._dbs(None))
