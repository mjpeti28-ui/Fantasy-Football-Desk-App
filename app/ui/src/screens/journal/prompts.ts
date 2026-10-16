import type { JDraft, Journal, Player, Shifts, TeamWeek, WaiverBoard } from "../../types";
import { f1, lastName, pct, signed } from "../../format";
import { pBeats } from "../../odds";
import { isDead } from "../../components/bits";
import { type Plan, rival } from "../myteam/lineup";

export interface Prompt { key: string; kind: "lineup" | "waiver" | "note"; title: string; why: string; draft: JDraft }

/** This week's decisions worth writing down: the closest start/sit calls, the best waiver move,
 *  and the biggest role change on your roster (a hunch to check later). Skips ones already logged. */
export function journalPrompts(tw: TeamWeek, plan: Plan, ratioQ: Record<string, number[]> | undefined, wboard: WaiverBoard | null,
  shifts: Shifts | null, journal: Journal | null): Prompt[] {
  const by = new Map(tw.players.map((p) => [p.id, p]));
  const logged = (journal?.entries ?? []).filter((e) => e.week === tw.week);
  const has = (ids: number[]) => logged.some((e) => ids.every((i) => JSON.stringify(e.payload).includes(String(i))));
  const out: Prompt[] = [];
  // close start/sit calls
  const calls: { start: Player; bench: Player; slot: string; p: number }[] = [];
  for (const b of tw.players) {
    if (Object.values(plan).includes(b.id) || isDead(b) || b.slot === "IR" || !b.espn_week) continue;
    const r = rival(b, plan, by);
    if (!r || r.role !== "starter") continue;
    const p = pBeats(r.other, b, ratioQ);
    if (p == null || p > 0.68) continue;
    calls.push({ start: r.other, bench: b, slot: r.slot, p });
  }
  for (const c of calls.sort((x, y) => Math.abs(x.p - 0.5) - Math.abs(y.p - 0.5)).slice(0, 2)) {
    if (has([c.start.id, c.bench.id])) continue;
    out.push({
      key: `l${c.start.id}-${c.bench.id}`, kind: "lineup",
      title: `${lastName(c.start.name)} or ${lastName(c.bench.name)} at ${c.slot}`,
      why: `Your plan starts ${lastName(c.start.name)} (${f1(c.start.espn_week)}); he outscores ${lastName(c.bench.name)} (${f1(c.bench.espn_week)}) ${pct(c.p)} of the time. Close enough that your read matters.`,
      draft: { kind: "lineup", week: tw.week, title: `Start ${c.start.name} over ${c.bench.name} at ${c.slot}`, payload: { start_id: c.start.id, bench_id: c.bench.id, slot: c.slot }, model_p: c.p,
        context: `ESPN ${f1(c.start.espn_week)} vs ${f1(c.bench.espn_week)} this week. Model: ${lastName(c.start.name)} outscores ${lastName(c.bench.name)} ${pct(c.p)} of the time.` },
    });
  }
  // the best waiver move
  const w = wboard ? [...wboard.candidates].sort((a, b) => (b.gain ?? 0) - (a.gain ?? 0))[0] : undefined;
  if (w && w.drop_id != null && !has([w.id])) {
    const real = (w.gain ?? 0) >= (wboard?.noise ?? 10);
    out.push({
      key: `w${w.id}`, kind: "waiver", title: `Add ${lastName(w.name)} for ${w.drop_name ? lastName(w.drop_name) : "nobody"}?`,
      why: `The best move on the wire: ${signed(w.gain)} season points, ${real ? "clear of the noise" : `inside the ±${wboard?.noise} noise`}. Whether you make it or pass, the running total says who was right.`,
      draft: { kind: "waiver", week: tw.week, title: `Add ${w.name}, drop ${w.drop_name ?? "nobody"}`, payload: { add_ids: [w.id], drop_ids: w.drop_id ? [w.drop_id] : [] }, model: { gain: w.gain },
        context: `Model: ${signed(w.gain)} season points (noise ±${wboard?.noise}). ${w.why}` },
    });
  }
  // the biggest role change on your roster
  const mine = (shifts?.rows ?? []).filter((r) => r.owner?.team_id === tw.team_id && r.recent?.zone && r.recent.d_snap != null)
    .sort((a, b) => Math.abs(b.recent!.d_snap!) + Math.abs(b.recent!.d_share ?? 0) - (Math.abs(a.recent!.d_snap!) + Math.abs(a.recent!.d_share ?? 0)))[0];
  if (mine && !has([mine.id])) {
    const r = mine.recent!;
    const up = (r.d_snap ?? 0) > 0;
    out.push({
      key: `n${mine.id}`, kind: "note", title: `${mine.name}: ${up ? "role up" : "role down"}, does it stick?`,
      why: `Snaps ${Math.round(r.from_snap ?? 0)}→${Math.round(r.to_snap ?? 0)}%${r.cause?.length ? ` with ${r.cause.join(" & ")} out` : ""}. Changes like this held ${r.hold == null ? "–" : pct(r.hold)} of the time. Write down your guess before the next two games settle it.`,
      draft: { kind: "note", week: tw.week, title: `${mine.name}: I think the ${up ? "bigger" : "smaller"} role ${up ? "sticks" : "lasts"}`, payload: { player_ids: [mine.id] },
        context: `Snaps ${Math.round(r.from_snap ?? 0)}→${Math.round(r.to_snap ?? 0)}% (${r.label}). Similar changes held ${r.hold == null ? "–" : pct(r.hold)} of the time, 2016–25.` },
    });
  }
  return out;
}
