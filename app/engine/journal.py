"""Decision journal: what you decided, why, how sure you were, and what the model said, scored
against what happened. Stored as JSON lines in history/journal/<season>.jsonl."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from . import data as d

KINDS = {"lineup", "waiver", "trade", "note"}
STATUSES = {"done", "offered", "skipped"}  # only "done" decisions count toward calibration


def _path() -> Path:
    p = d.ROOT / "history" / "journal" / f"{d.load().season}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read() -> List[dict]:
    p = _path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write(rows: List[dict]) -> None:
    p = _path()
    tmp = p.with_suffix(".tmpw")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, p)


def _points(pid: int, weeks) -> tuple:
    pw = d.load().players_weekly
    g = pw[(pw["espn_id"] == pid) & (pw["week"].isin(list(weeks)))]
    return float(g["points"].sum()), sorted(int(w) for w in g["week"].unique())


def _resolve(e: dict) -> dict:
    """Outcome from actual points. Lineup: final once the week is scored. Waiver/trade: running
    total of actual points since the decision week (final after week 17)."""
    inp = d.load()
    scored = set(int(w) for w in inp.team_weekly["week"].unique())
    k, p = e["kind"], e.get("payload", {})
    w = int(e.get("week") or inp.week)
    if k == "lineup" and p.get("start_id") and p.get("bench_id"):
        if w not in scored:
            return {"state": "pending"}
        a, _ = _points(p["start_id"], [w])
        b, _ = _points(p["bench_id"], [w])
        return {"state": "final", "start_pts": round(a, 1), "bench_pts": round(b, 1), "right": a >= b}
    if k in ("waiver", "trade"):
        ins = p.get("add_ids") or p.get("get_ids") or []
        outs = p.get("drop_ids") or p.get("give_ids") or []
        weeks = [x for x in sorted(scored) if x >= w]
        if not weeks:
            return {"state": "pending"}
        a = sum(_points(i, weeks)[0] for i in ins)
        b = sum(_points(i, weeks)[0] for i in outs)
        return {"state": "final" if max(weeks) >= 17 else "running", "weeks": [weeks[0], weeks[-1]],
                "in_pts": round(a, 1), "out_pts": round(b, 1), "right": a >= b}
    return {"state": "none"}


def entries() -> dict:
    rows = _read()
    for e in rows:
        e["outcome"] = _resolve(e)
    rows.sort(key=lambda e: (-(e.get("week") or 0), -e["created_at"]))
    done = [e for e in rows if e["outcome"].get("state") == "final" and "right" in e["outcome"]
            and e.get("status", "done") == "done"]
    def rate(xs, key):
        v = [x.get(key) for x in xs if x.get(key) is not None]
        return round(sum(v) / len(v), 3) if v else None
    summary = {"logged": len(rows), "resolved": len(done),
               "hit_rate": round(sum(e["outcome"]["right"] for e in done) / len(done), 3) if done else None,
               "your_conf": rate(done, "confidence"), "model_p": rate(done, "model_p"),
               "buckets": []}
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        b = [e for e in done if e.get("confidence") is not None and lo <= e["confidence"] < hi]
        summary["buckets"].append({"lo": lo, "hi": min(hi, 1.0), "n": len(b),
                                   "hit": round(sum(e["outcome"]["right"] for e in b) / len(b), 3) if b else None})
    return {"entries": rows, "summary": summary, "week": d.load().week}


def add(kind: str, title: str, why: str = "", confidence: Optional[float] = None, payload: Optional[dict] = None,
        model: Optional[dict] = None, model_p: Optional[float] = None, week: Optional[int] = None,
        status: Optional[str] = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    status = status or ("offered" if kind == "trade" else "done")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    e = {"id": uuid.uuid4().hex[:12], "created_at": time.time(), "week": week or d.load().week, "kind": kind,
         "title": title.strip(), "why": why.strip(), "confidence": confidence, "payload": payload or {},
         "model": model or {}, "model_p": model_p, "status": status}
    rows = _read()
    rows.append(e)
    _write(rows)
    return e


def update(eid: str, **fields) -> dict:
    if fields.get("status") is not None and fields["status"] not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    rows = _read()
    for e in rows:
        if e["id"] == eid:
            for k in ("why", "confidence", "status", "title"):
                if fields.get(k) is not None:
                    e[k] = fields[k]
            _write(rows)
            return e
    raise KeyError(eid)


def delete(eid: str) -> None:
    rows = _read()
    _write([e for e in rows if e["id"] != eid])
