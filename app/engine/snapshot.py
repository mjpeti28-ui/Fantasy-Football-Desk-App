"""Dump every view to one JSON file: python3 -m app.engine.snapshot [--out path] [--team 3]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import data as d, views

# Bump when the snapshot shape changes; the UI warns if it talks to an older engine.
ENGINE_VERSION = 3
# NFL team colors from ESPN's public teams endpoint (site.api.espn.com), saved 2026-10-07.
TEAM_COLORS = json.loads((Path(__file__).with_name("team_colors.json")).read_text())

def build(team: int = 3, season: int = 2026) -> dict:
    inp = d.load(season)
    return {"generated_at": time.time(), "season": season, "week": inp.week, "data_age": inp.ages,
            "team_week": views.team_week(inp, team), "power": views.power_table(inp),
            "ratio_q": views.ratio_quantiles(inp), "team_colors": TEAM_COLORS,
            "engine_version": ENGINE_VERSION}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(d.ROOT / "app" / "cache" / "snapshot.json"))
    ap.add_argument("--team", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(a.team), indent=1, default=str))
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
