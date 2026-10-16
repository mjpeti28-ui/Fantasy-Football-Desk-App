import { useState } from "react";
import type { Player, TeamWeek } from "../../types";
import { f1, lastName, pct, signed } from "../../format";
import { POS_VAR } from "../../components/bits";
import { pBeats } from "../../odds";
import { useFile } from "../../components/FileContext";
import { type Plan, STARTERS, type Slot, canPlay } from "./lineup";
import "./insights.css";

/* Two panels under the bench, in the board's language (condensed caps, magnets on a pegboard):
   - Slot by slot: the matchup as a tug of war, one rope per lineup slot. It follows your plan
     live, so dragging a tile shows which slot the swap actually moves.
   - What each player carries: season lineup points your roster loses if he's gone, from the
     waiver engine's simulation. It's the number behind "best drop", and it shows where depth is
     already doing the work. */

const ORDER: Slot[] = ["QB", "RB", "RB/WR", "WR", "WR/TE", "TE", "FLEX", "K", "DST"];

export function SlotDuel({ tw, plan, by, ratioQ }: { tw: TeamWeek; plan: Plan; by: Map<number, Player>; ratioQ?: Record<string, number[]> }) {
  const m = tw.matchup;
  const [hov, setHov] = useState<Slot | null>(null);
  if (!m?.opp_starters?.length) return null;
  const rows = ORDER.filter((s) => STARTERS.includes(s)).map((s) => {
    const me = plan[s] != null ? by.get(plan[s]!) ?? null : null;
    const them = (m.opp_starters.find((o) => o.slot === s) ?? null) as unknown as Player | null;
    const a = me?.espn_week ?? 0, b = them?.espn_week ?? 0;
    return { s, me, them, a, b, d: a - b, p: me && them ? pBeats(me, them, ratioQ) : null };
  });
  const mx = Math.max(20, ...rows.map((r) => Math.max(r.a, r.b)));
  const tot = rows.reduce((x, r) => x + r.d, 0);
  const mine = rows.filter((r) => r.d >= 2).sort((a, b) => b.d - a.d), theirs = rows.filter((r) => r.d <= -2).sort((a, b) => a.d - b.d);
  const h = rows.find((r) => r.s === hov);
  const w = (v: number) => `${(v / mx) * 100}%`;
  return (
    <section className="ins-panel duel" aria-label={`Slot by slot against ${m.opp_name}`}>
      <h5>Slot by slot <span>vs {m.opp_name} · ESPN projections · follows your plan</span></h5>
      <div className="d-leg"><span><i className="k you" />You</span><span><i className="k them" />{m.opp_name}</span></div>
      <div className="d-rows" onMouseLeave={() => setHov(null)}>
        {rows.map((r) => (
          <div key={r.s} className={`d-row ${hov === r.s ? "hov" : ""}`} onMouseEnter={() => setHov(r.s)}>
            <span className="nm l">{r.me ? lastName(r.me.name) : "empty"}</span>
            <span className="v l num">{f1(r.a)}</span>
            <span className="bar l"><i style={{ width: w(r.a) }} /></span>
            <span className={`sl ${r.d >= 2 ? "up" : r.d <= -2 ? "down" : ""}`}><b>{r.s}</b><em className="num">{Math.abs(r.d) < 0.05 ? "even" : signed(r.d)}</em></span>
            <span className="bar r"><i style={{ width: w(r.b) }} /></span>
            <span className="v r num">{f1(r.b)}</span>
            <span className="nm r">{r.them ? lastName(r.them.name) : "empty"}</span>
          </div>
        ))}
      </div>
      <p className="d-read">
        {h ? <>{h.me ? h.me.name : "Nobody"} {f1(h.a)} vs {h.them ? h.them.name : "nobody"} {f1(h.b)}{h.p != null ? <> · yours outscores theirs <b>{pct(h.p)}</b> of the time</> : null}.</>
          : <>Net <b className="num">{signed(tot)}</b>. {mine.length ? <>Your edges: {mine.slice(0, 3).map((r) => `${r.s} ${signed(r.d)}`).join(", ")}. </> : null}
            {theirs.length ? <>Theirs: {theirs.slice(0, 3).map((r) => `${r.s} ${signed(r.d)}`).join(", ")}.</> : null} Hover a slot for the head-to-head.</>}
      </p>
    </section>
  );
}

export function Carries({ tw, cost, plan }: { tw: TeamWeek; cost: Record<string, number> | null | undefined; plan: Plan }) {
  const { open } = useFile();
  if (!cost) return (
    <section className="ins-panel carries"><h5>What each player carries</h5><p className="d-read dim">Waiting for the waiver engine (it simulates the rest of the season once, then this fills in).</p></section>
  );
  const starters = new Set(STARTERS.map((s) => plan[s]).filter((x): x is number => x != null));
  const rows = tw.players.filter((p) => cost[String(p.id)] != null).map((p) => ({ p, c: cost[String(p.id)] })).sort((a, b) => b.c - a.c);
  const mx = Math.max(20, ...rows.map((r) => r.c));
  const NOISE = 5;
  const covered = rows.filter((r) => r.c < NOISE && starters.has(r.p.id));
  return (
    <section className="ins-panel carries" aria-label="What each player carries">
      <h5>What each player carries <span>season lineup points lost without him</span></h5>
      <div className="c-rows">
        <span className="c-band" style={{ width: `calc((100% - 200px) * ${(NOISE / mx).toFixed(4)})` }} aria-hidden="true" />
        {rows.map(({ p, c }) => (
          <button key={p.id} type="button" className="c-row" onClick={() => open(p.id, rows.map((r) => r.p.id))}
            title={`${p.name}: your lineups score about ${f1(c)} fewer points over the rest of the season without him (his slot goes to the next man up or a waiver-level fill).`}>
            <span className="nm"><i className="pchip" style={{ background: POS_VAR[p.pos], color: p.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{p.pos}</i>{lastName(p.name)}{starters.has(p.id) ? null : <em> bench</em>}</span>
            <span className="trk"><i style={{ width: `${Math.max(0.5, (c / mx) * 100)}%`, background: POS_VAR[p.pos] }} /></span>
            <span className="v num">{f1(c)}</span>
          </button>
        ))}
      </div>
      <p className="d-read">
        Grey band = under {NOISE} points, inside the simulation's noise: your bench or the wire already covers him.
        {covered.length ? <> Starting but covered: <b>{covered.map((r) => lastName(r.p.name)).join(", ")}</b>. {covered.length > 1 ? "They cost" : "He costs"} your lineup least to trade away (what he's worth to someone else is a separate question).</> : null}
      </p>
    </section>
  );
}

/** Above the lineup: ESPN's best lineup when your plan isn't it, and the moves to make in ESPN
 *  when your plan differs from what's set there (Desk can't write to ESPN yet). */
export function LineupCheck({ tw, plan, by, ratioQ, setPlan, onCopy }: {
  tw: TeamWeek; plan: Plan; by: Map<number, Player>; ratioQ?: Record<string, number[]>; setPlan: (p: Plan) => void; onCopy?: (t: string) => void;
}) {
  const sugg: Plan = {}, cur: Plan = {};
  for (const s of STARTERS) { if (tw.lineup.suggested[s] != null) sugg[s] = tw.lineup.suggested[s]; if (tw.lineup.current[s] != null) cur[s] = tw.lineup.current[s]; }
  const pts = (p: Plan) => STARTERS.reduce((a, s) => a + (p[s] != null ? by.get(p[s]!)?.espn_week ?? 0 : 0), 0);
  const inSet = (p: Plan) => new Set(STARTERS.map((s) => p[s]).filter((x): x is number => x != null));
  const mine = inSet(plan), best = inSet(sugg);
  const ins = [...best].filter((id) => !mine.has(id)).map((id) => by.get(id)!).filter(Boolean);
  const outs = [...mine].filter((id) => !best.has(id)).map((id) => by.get(id)!).filter(Boolean);
  const gain = pts(sugg) - pts(plan);
  const todo = STARTERS.filter((s) => plan[s] !== cur[s]).map((s) => ({ s, in: plan[s] != null ? by.get(plan[s]!) : undefined, out: cur[s] != null ? by.get(cur[s]!) : undefined }));
  if (!(gain > 0.05 && ins.length) && !todo.length) return null;
  const pair = ins.length === 1 && outs.length === 1 ? pBeats(ins[0], outs[0], ratioQ) : null;
  // apply it with as few slot changes as possible: incoming players take the slots the outgoing
  // ones leave, so a pure WR / WR-TE reshuffle doesn't show up as extra moves to make in ESPN
  const minimal = (): Plan => {
    const next: Plan = { ...plan }, free = [...ins];
    for (const sl of STARTERS) {
      const id = plan[sl];
      if (id != null && best.has(id)) continue;
      const k = free.findIndex((p) => canPlay(p, sl));
      if (k < 0) { if (id == null) continue; return sugg; }
      next[sl] = free[k].id; free.splice(k, 1);
    }
    return free.length ? sugg : next;
  };
  const todoText = todo.map((t) => `${t.s}: start ${t.in?.name ?? "nobody"}${t.out ? ` (instead of ${t.out.name})` : ""}`).join("\n");
  return (
    <section className="lu-check" aria-label="Lineup check">
      {gain > 0.05 && ins.length > 0 && (
        <div className="lu-row">
          <span className="lu-k">Best by ESPN</span>
          <span className="lu-t">Start <b>{ins.map((p) => p.name).join(", ")}</b> over <b>{outs.map((p) => `${p.name}${p.injury && !/ACTIVE/i.test(p.injury) ? ` (${p.injury.replace(/_/g, " ").toLowerCase()})` : ""}`).join(", ")}</b>
            : <span className="num">{signed(gain)}</span> projected{pair != null ? <>, and he outscores {lastName(outs[0].name)} {pct(pair)} of the time{pair < 0.58 ? " (close: your read can beat this)" : ""}</> : null}.</span>
          <button className="tbtn" type="button" onClick={() => setPlan(minimal())}>Use it</button>
        </div>
      )}
      {todo.length > 0 && (
        <div className="lu-row">
          <span className="lu-k">To do in ESPN</span>
          <span className="lu-t">{todo.map((t) => <span key={t.s} className="mv"><em>{t.s}</em> start <b>{t.in ? lastName(t.in.name) : "nobody"}</b>{t.out ? <> for {lastName(t.out.name)}</> : null}</span>)}</span>
          {onCopy && <button className="tbtn" type="button" onClick={() => onCopy(todoText)}>Copy</button>}
        </div>
      )}
    </section>
  );
}
