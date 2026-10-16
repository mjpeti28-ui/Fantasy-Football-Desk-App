"""Experiment harness: run evaluate_league against real data, dump comparable metrics.
Not part of the shipped tool -- scratch file for the value-weighting experiment.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from main import evaluate_league, DEFAULT_RANKINGS, DEFAULT_PROJECTIONS, default_supplemental_path
from rosters import Fantasy_Rosters


def run(label, **overrides):
    payload = evaluate_league(
        str(DEFAULT_RANKINGS),
        projections_path=str(DEFAULT_PROJECTIONS),
        supplemental_path=default_supplemental_path(),
        custom_rosters=Fantasy_Rosters,
        save_snapshot=False,
        **overrides,
    )
    summary = {
        "label": label,
        "league_total_starter_proj": round(sum(payload["starter_projections"].values()), 2),
        "league_total_starter_vor": round(sum(payload["starters_totals"].values()), 2),
        "teams": {},
        "combined_board": payload["leaderboards"]["combined"],
        "replacement_points": payload["replacement_points"],
    }
    for team, res in payload["results"].items():
        starters = sorted(res["starters"], key=lambda p: (p["slot"]))
        summary["teams"][team] = {
            "starter_proj": payload["starter_projections"][team],
            "starter_vor": payload["starters_totals"][team],
            "bench_total": payload["bench_totals"][team],
            "starters": [
                {"slot": p["slot"], "name": p["name"], "pos": p["pos"], "proj": p.get("proj"), "rank": p.get("rank")}
                for p in starters
            ],
            "top_bench": [
                {"name": b["name"], "pos": b["pos"], "proj": b.get("proj"), "rank": b.get("rank"),
                 "vor": b.get("vor"), "oVAR": b.get("oVAR"), "BenchScore": b.get("BenchScore")}
                for b in payload["bench_tables"][team][:8]
            ],
        }
    return summary, payload


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    outpath = sys.argv[2] if len(sys.argv) > 2 else f"_experiments/{label}.json"
    summary, _ = run(label)
    Path(REPO / outpath).write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {outpath}")
    print("league_total_starter_proj:", summary["league_total_starter_proj"])
    print("league_total_starter_vor:", summary["league_total_starter_vor"])
