"""The Lab's fingerprint payload: compact per-player values and per-position metadata."""
from app.engine import roles as R


def test_fp_compact_rounds_and_keeps_order():
    fp = [{"key": "adot", "raw": 9.87654, "est": 10.12345, "rel": 0.4567}, {"key": "catch_r", "raw": None, "est": 0.6, "rel": 0.0}]
    out = R.fp_compact(fp)
    assert out == {"adot": [9.8765, 10.1235, 0.46], "catch_r": [None, 0.6, 0.0]}


def test_fp_meta_covers_every_metric_with_stab():
    model = R.load_model()
    meta = R.fp_meta(model)
    for pos, ms in R.METRICS.items():
        for k in ms:
            if k in model["stab"].get(pos, {}):
                m = meta[pos][k]
                assert m["label"] == ms[k][0] and m["fmt"] == ms[k][1] and m["stab"] > 0
