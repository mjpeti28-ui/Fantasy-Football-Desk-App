import type { Player, Pos, TeamWeek } from "../../types";
import { f1, gameLabel, pct } from "../../format";
import { isDead } from "../../components/bits";

export const STARTERS = ["QB", "RB", "WR", "TE", "RB/WR", "WR/TE", "FLEX", "DST", "K"] as const;
export type Slot = (typeof STARTERS)[number];
export const ELIG: Record<Slot, Pos[]> = {
  QB: ["QB"], RB: ["RB"], WR: ["WR"], TE: ["TE"], K: ["K"], DST: ["DST"],
  "RB/WR": ["RB", "WR"], "WR/TE": ["WR", "TE"], FLEX: ["RB", "WR", "TE"],
};
export type Plan = Partial<Record<Slot, number>>;

export const canPlay = (p: Player, s: Slot) => ELIG[s].includes(p.pos) && p.slot !== "IR";

export function planFrom(rec: Record<string, number>): Plan {
  const out: Plan = {};
  for (const s of STARTERS) if (rec[s] != null) out[s] = rec[s];
  return out;
}

export const slotOf = (plan: Plan, id: number) => STARTERS.find((s) => plan[s] === id);

export function planPoints(plan: Plan, by: Map<number, Player>) {
  return STARTERS.reduce((a, s) => a + (plan[s] != null ? by.get(plan[s]!)?.espn_week ?? 0 : 0), 0);
}

/** Move player `id` into slot `to`. Swaps with the occupant when the occupant can play the
 *  slot being vacated; otherwise the occupant goes to the bench. Returns null if illegal. */
export function move(plan: Plan, by: Map<number, Player>, id: number, to: Slot | "BENCH"): Plan | null {
  const p = by.get(id);
  if (!p) return null;
  const from = slotOf(plan, id);
  const next: Plan = { ...plan };
  if (to === "BENCH") {
    if (!from) return null;
    delete next[from];
    return next;
  }
  if (!canPlay(p, to) || from === to) return null;
  const occ = plan[to];
  if (from) delete next[from];
  next[to] = id;
  if (occ != null && from) {
    const o = by.get(occ)!;
    if (canPlay(o, from)) next[from] = occ;
  }
  return next;
}

export function deadStarters(plan: Plan, by: Map<number, Player>) {
  return STARTERS.filter((s) => plan[s] != null && isDead(by.get(plan[s]!)!)).map((s) => ({ slot: s, p: by.get(plan[s]!)! }));
}

export function samePlan(a: Plan, b: Plan) {
  return STARTERS.every((s) => a[s] === b[s]);
}

export function planMarkdown(tw: TeamWeek, plan: Plan, by: Map<number, Player>, pWin: number | null) {
  const L = [`## ${tw.team_name}: week ${tw.week} lineup plan (from Desk)`, "",
    "| Slot | Player | Game | ESPN proj | ROS ppg | Plays | Status |", "|---|---|---|---|---|---|---|"];
  const row = (slot: string, p: Player) => {
    const g = p.game;
    L.push(`| ${slot} | ${p.name} (${p.pos}, ${p.team ?? ""}) | ${gameLabel(g, true, g?.day, g?.time)} | ${f1(p.espn_week)} | ${f1(p.est_ppg)} | ${pct(p.avail_now)} | ${p.game?.bye ? "bye" : p.injury} |`);
  };
  for (const s of STARTERS) if (plan[s] != null) row(s, by.get(plan[s]!)!);
  for (const p of tw.players) if (!slotOf(plan, p.id)) row(p.slot === "IR" ? "IR" : "BN", p);
  L.push("", `Plan projects ${f1(planPoints(plan, by))} (ESPN weekly numbers). As set in ESPN: ${f1(tw.lineup.current_pts)}.`);
  if (tw.matchup) L.push(`Opponent ${tw.matchup.opp_name}: ${f1(tw.matchup.opp_proj)}. Win chance ≈ ${pct(pWin)} (normal approximation, margin sd ${f1(tw.matchup.margin_sd)}).`);
  L.push("", `Notes: ${tw.notes.lineup_rule}`);
  return L.join("\n");
}

/** Smallest fix: replace each starter who can't score with the best healthy bench player who
 *  can play that slot (ESPN's weekly number). Keeps every other slot where it is. */
export function fixDead(plan: Plan, by: Map<number, Player>): { plan: Plan; moves: { slot: Slot; out: Player; in: Player }[] } {
  const next: Plan = { ...plan };
  const moves: { slot: Slot; out: Player; in: Player }[] = [];
  for (const s of STARTERS) {
    const id = next[s];
    if (id == null || !isDead(by.get(id)!)) continue;
    const used = new Set(Object.values(next));
    const cand = [...by.values()].filter((p) => !used.has(p.id) && !isDead(p) && canPlay(p, s)).sort((a, b) => (b.espn_week ?? 0) - (a.espn_week ?? 0))[0];
    if (cand) { moves.push({ slot: s, out: by.get(id)!, in: cand }); next[s] = cand.id; }
  }
  return { plan: next, moves };
}

/** The real alternative for a decision about `p`: for a starter, the best healthy bench player
 *  who could take his slot; for a bench player, the weakest starter he could replace. */
export function rival(p: Player, plan: Plan, by: Map<number, Player>): { other: Player; slot: Slot; role: "bench" | "starter" } | null {
  const s = slotOf(plan, p.id);
  if (s) {
    const used = new Set(Object.values(plan));
    const alt = [...by.values()].filter((o) => !used.has(o.id) && o.slot !== "IR" && !isDead(o) && canPlay(o, s))
      .sort((a, b) => (b.espn_week ?? 0) - (a.espn_week ?? 0))[0];
    return alt ? { other: alt, slot: s, role: "bench" } : null;
  }
  if (p.slot === "IR") return null;
  const opts = STARTERS.filter((t) => plan[t] != null && canPlay(p, t)).map((t) => ({ slot: t, o: by.get(plan[t]!)! }))
    .sort((a, b) => (isDead(a.o) ? -1 : a.o.espn_week ?? 0) - (isDead(b.o) ? -1 : b.o.espn_week ?? 0));
  return opts.length ? { other: opts[0].o, slot: opts[0].slot, role: "starter" } : null;
}
