"""Desk rebuilds the usage table only when nflverse has published new snap/charting data."""
import os
import time

import pandas as pd
import pytest

from app.engine import roles_data as rd

OLD = {"snap_counts": "Fri, 09 Oct 2026 11:01:37 GMT", "ftn_charting": "Sat, 10 Oct 2026 11:02:08 GMT"}
NEW = {"snap_counts": "Tue, 13 Oct 2026 11:01:32 GMT", "ftn_charting": "Tue, 13 Oct 2026 05:01:41 GMT"}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    f = tmp_path / "2026" / "roles_weekly.csv"
    f.parent.mkdir()
    pd.DataFrame({"week": [1, 2]}).to_csv(f, index=False)
    calls = []

    def fake_load(season, cache, ids, refresh=False):
        calls.append(season)
        pd.DataFrame({"week": [1, 2, 3]}).to_csv(cache / str(season) / "roles_weekly.csv", index=False)
        rd.save_stamps(cache, season, fake_load.live)
        return None

    fake_load.live = OLD
    monkeypatch.setattr(rd, "load", fake_load)
    monkeypatch.setattr(rd, "upstream_stamps", lambda season: fake_load.live)
    return tmp_path, calls, fake_load


def test_unchanged_upstream_does_not_rebuild(cache):
    path, calls, fake = cache
    rd.save_stamps(path, 2026, OLD)
    assert not rd.refresh_if_stale(2026, path, None, min_gap=0)
    assert calls == []


def test_new_upstream_rebuilds_once_and_records_stamps(cache):
    path, calls, fake = cache
    rd.save_stamps(path, 2026, OLD)
    fake.live = NEW
    assert rd.refresh_if_stale(2026, path, None, min_gap=0)
    assert calls == [2026]
    assert rd._read_stamps(path, 2026)["stamps"] == NEW
    assert not rd.refresh_if_stale(2026, path, None, min_gap=0)  # same stamps now
    assert calls == [2026]


def test_checks_are_rate_limited(cache):
    path, calls, fake = cache
    rd.save_stamps(path, 2026, OLD)
    fake.live = NEW
    assert not rd.refresh_if_stale(2026, path, None, min_gap=600)  # just checked: skip
    assert calls == []


def test_offline_keeps_the_table(cache, monkeypatch):
    path, calls, fake = cache
    monkeypatch.setattr(rd, "upstream_stamps", lambda season: {})
    assert not rd.refresh_if_stale(2026, path, None, min_gap=0)
    assert calls == []


def test_table_built_before_stamps_existed_compares_to_file_time(cache):
    path, calls, fake = cache
    f = path / "2026" / "roles_weekly.csv"
    old_t = time.mktime((2026, 10, 12, 0, 0, 0, 0, 0, 0))
    os.utime(f, (old_t, old_t))
    fake.live = NEW  # published Tuesday, after the file was written Monday
    assert rd.is_stale(2026, path)
    os.utime(f, None)  # file is newer than everything upstream
    assert not rd.is_stale(2026, path)
