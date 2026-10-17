"""Probe every operation in PFF's OpenAPI spec once with small params; write a field inventory.

    python backtest/pff_probe.py        # -> .cache/pff/inventory.json (main checkout)
"""
from __future__ import annotations

import json
import urllib.request

import pff_api as api

SPEC = api.MAIN / ".cache" / "pff" / "openapi.json"
OUT = api.MAIN / ".cache" / "pff" / "inventory.json"
SEASON, WEEK, FRANCHISE, TEAM = 2025, 5, 7, "cincinnati-bengals"
# subject player per family: Burrow, Puka Nacua, Bijan Robinson, CIN OL/DL picked below, kicker/punter/returner
PLAYER = {"passing": 28022, "receiving": 100632, "rushing": 123126, "offense": 100632}


def rows_of(body):
    """Find the main list of row dicts in a response."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        if isinstance(body.get("rows"), list):
            return body["rows"]
        best = []
        for v in body.values():
            r = rows_of(v) if isinstance(v, (dict, list)) else []
            if len(r) > len(best) and (not r or isinstance(r[0], dict)):
                best = r
        return best
    return []


def flat_keys(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and len(v) < 60:
            out.update(flat_keys(v, f"{pre}{k}."))
        else:
            out[pre + k] = v
    return out


def params_for(path: str, op: dict, spec: dict, ids: dict) -> dict:
    p = {}
    for prm in op.get("parameters", []):
        if "$ref" in prm:
            o = spec
            for k in prm["$ref"].split("/")[1:]:
                o = o[k]
            prm = o
        n, req = prm["name"], prm.get("required")
        if prm.get("in") == "path":
            continue
        if n == "league":
            p[n] = "nfl"
        elif n == "season" and (req or "/facet/" in path or "/v2/" in path or "/teams" in path or "/player/" in path):
            p[n] = SEASON
        elif n == "week" and (req or "/facet/" in path or "/v2/" in path):
            p[n] = WEEK
        elif n == "franchise_id" and (req or path.startswith("/v1/games")):
            p[n] = FRANCHISE
        elif n == "player_id":
            fam = path.split("/")[3]
            p[n] = ids.get(fam, ids["default"])
        elif n == "name":
            p[n] = "Burrow"
        elif n == "category":
            p[n] = "offense-passing"
        elif n == "group":
            p[n] = "receiving"
        elif n == "weekGroup":
            p[n] = "REG"
    return p


def main():
    spec = json.loads(SPEC.read_text())
    # ids for non-skill families from the week-5 leaderboards
    ids = {"passing": 28022, "receiving": 100632, "rushing": 123126, "offense": 100632, "default": 100632}
    for fam, facet, pos in [("defense", "defense/summary", None), ("field_goal", "field_goal/summary", None),
                            ("kickoff", "kickoff/summary", None), ("punting", "punting/summary", None),
                            ("return", "return/summary", None), ("special", "special/summary", None)]:
        r = api.get(f"/v1/facet/{facet}", {"league": "nfl", "season": SEASON, "week": WEEK, "franchise_id": FRANCHISE})
        rows = rows_of(r["body"])
        if rows:
            ids[fam] = rows[0]["player_id"]
    r = api.get("/v1/facet/offense/blocking", {"league": "nfl", "season": SEASON, "week": WEEK, "franchise_id": FRANCHISE})
    ol = [x for x in rows_of(r["body"]) if x.get("position") in ("T", "G", "C")]
    if ol:
        ids["offense_blocking"] = ol[0]["player_id"]
    inv = []
    for path, ops in spec["paths"].items():
        op = ops.get("get")
        if not op or path in ("/openapi.json", "/openapi.yaml"):
            continue
        p = params_for(path, op, spec, ids)
        if "blocking" in path and path.startswith("/v1/player/"):
            p["player_id"] = ids.get("offense_blocking", p.get("player_id"))
        real = path.replace("{league}", "nfl").replace("{team}", TEAM).replace("{report}", "receiving")
        reports = ["receiving"]
        if "{report}" in path:
            for prm in op["parameters"]:
                if "$ref" in prm:
                    o = spec
                    for k in prm["$ref"].split("/")[1:]:
                        o = o[k]
                    prm = o
                if prm.get("name") == "report":
                    reports = prm["schema"]["enum"]
        for rep in reports:
            rp = path.replace("{league}", "nfl").replace("{team}", TEAM).replace("{report}", rep)
            if "{report}" in path and "/teams/" in path and rep not in ("receiving", "rushing", "passing"):
                continue  # team copy of the league report: three are enough to confirm shape
            r = api.get(rp, p)
            rows = rows_of(r["body"])
            keys = {}
            for row in rows[:400]:
                if isinstance(row, dict):
                    for k, v in flat_keys(row).items():
                        if k not in keys or keys[k] in (None, "", 0):
                            keys[k] = v
            top = list(r["body"].keys()) if isinstance(r["body"], dict) else type(r["body"]).__name__
            inv.append({"op": op.get("operationId"), "path": rp, "params": p, "status": r["status"],
                        "top_keys": top, "n_rows": len(rows), "fields": keys,
                        "error": r["body"].get("error") if isinstance(r["body"], dict) else None,
                        "remaining": r["ratelimit"].get("x-ratelimit-remaining")})
            print(r["status"], rp, len(rows), len(keys), flush=True)
    OUT.write_text(json.dumps(inv, default=str))


if __name__ == "__main__":
    main()
