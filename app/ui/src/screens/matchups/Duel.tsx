import { type MutableRefObject, useEffect, useMemo, useState } from "react";
import type { Player, Pos, RolesBoard, TeamWeek } from "../../types";
import { Face, POS_VAR, Status, isDead } from "../../components/bits";
import { f1, lastName, pct } from "../../format";
import { type Pt, pAtLeast, pBeats, pUnder, phi } from "../../odds";
import { ELIG, STARTERS, type Slot, canPlay } from "../myteam/lineup";
import { Band, Grade, ord, signed1 } from "./Card";
import type { MxCard } from "./types";

/* Start/sit. Two or three players for one slot, side by side. The number that decides is ESPN's; the odds
   come from the same spread of past results the lineup board uses. The matchup rows say how much of each
   number is the defense, so the choice is never made on a matchup ESPN already counted. */

interface Cand { id: number; name: string; pos: Pos; team: string | null; headshot: string | null; slot: string | null; espn: number | null; injury: string; bye: boolean; opp: string | null; home: boolean | null; day: string | null }
const SKILL = ["QB", "RB", "WR", "TE"];
/** What counts as a big week and a dud for the slot. Descriptive cut-offs, not a model: a QB needs more to matter, a TE less. */
const CUT: Record<string, [number, number]> = { QB: [20, 10], TE: [12, 4] };
const cutFor = (slot: string): [number, number] => CUT[slot] ?? [15, 5];
const fromPlayer = (p: Player): Cand => ({ id: p.id, name: p.name, pos: p.pos, team: p.team, headshot: p.headshot, slot: p.slot, espn: p.espn_week, injury: p.injury, bye: !!p.game?.bye, opp: p.game?.opp ?? null, home: p.game?.home ?? null, day: p.game?.day ?? null });
const pt = (c: Cand, card?: MxCard | "err"): Pt => {
  const k = card && card !== "err" ? card : null;
  return { pos: c.pos, espn_week: c.espn ?? k?.proj.espn ?? null, injury: k?.proj.injury ?? c.injury, game: { bye: c.bye || !!k?.proj.bye } };
};

function closestCall(tw: TeamWeek, q?: Record<string, number[]>): { slot: Slot; ids: number[] } {
  const by = new Map(tw.players.map((p) => [p.id, p]));
  const slots = STARTERS.filter((s) => s !== "K" && s !== "DST" && tw.lineup.current[s] != null);
  let best: { slot: Slot; ids: number[]; d: number } | null = null;
  for (const s of slots) {
    const occ = by.get(tw.lineup.current[s]);
    if (!occ || isDead(occ)) continue;
    for (const b of tw.players) {
      if (b.slot !== "BE" || !canPlay(b, s) || isDead(b)) continue;
      const p = pBeats(occ, b, q);
      if (p == null) continue;
      const d = Math.abs(p - 0.5);
      if (!best || d < best.d) best = { slot: s, ids: [occ.id, b.id], d };
    }
  }
  if (best) return { slot: best.slot, ids: best.ids };
  const s = slots[0] ?? ("QB" as Slot);
  const occ = tw.lineup.current[s];
  return { slot: s, ids: occ != null ? [occ] : [] };
}

/** P(each candidate scores the most), exactly, over the position quantile grids (independent outcomes). */
function pTop(cs: Pt[], q?: Record<string, number[]>): (number | null)[] {
  if (!q || cs.some((c) => !q[c.pos] || !c.espn_week)) return cs.map(() => null);
  const g = cs.map((c) => q[c.pos].map((r) => r * c.espn_week!));
  const win = cs.map(() => 0);
  let n = 0;
  const rec = (i: number, vals: number[]) => {
    if (i === g.length) { const m = Math.max(...vals); const ties = vals.filter((v) => v === m).length; vals.forEach((v, k) => { if (v === m) win[k] += 1 / ties; }); n++; return; }
    for (const v of g[i]) rec(i + 1, [...vals, v]);
  };
  rec(0, []);
  return win.map((w) => w / n);
}


export function Duel({ tw, ratioQ, roles, fetchCard, copyRef, onOpen }: {
  tw: TeamWeek; ratioQ?: Record<string, number[]>; roles: RolesBoard | null; fetchCard: (id: number) => Promise<MxCard | undefined>;
  copyRef: MutableRefObject<(() => string) | null>; onOpen: (id: number) => void;
}) {
  const start = useMemo(() => closestCall(tw, ratioQ), [tw, ratioQ]);
  const [slot, setSlot] = useState<Slot>(start.slot);
  const [ids, setIds] = useState<number[]>(start.ids);
  const [extras, setExtras] = useState<Cand[]>([]);
  const [q, setQ] = useState("");
  const [cards, setCards] = useState<Record<number, MxCard | "err">>({});
  const by = useMemo(() => new Map(tw.players.map((p) => [p.id, p])), [tw]);
  const slots = useMemo(() => STARTERS.filter((s) => s !== "K" && s !== "DST" && tw.lineup.current[s] != null), [tw]);

  useEffect(() => {
    let live = true;
    for (const id of ids) {
      if (id in cards) continue;
      fetchCard(id).then((c) => { if (live) setCards((m) => ({ ...m, [id]: c ?? "err" })); }, () => { if (live) setCards((m) => ({ ...m, [id]: "err" })); });
    }
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, fetchCard]);

  const eligible = useMemo(() => tw.players.filter((p) => SKILL.includes(p.pos) && canPlay(p, slot)).sort((a, b) => (b.espn_week ?? 0) - (a.espn_week ?? 0)), [tw, slot]);
  const all: Cand[] = useMemo(() => [...eligible.map(fromPlayer), ...extras.filter((e) => ELIG[slot].includes(e.pos) && !eligible.some((x) => x.id === e.id))], [eligible, extras, slot]);
  const sel = ids.map((id) => all.find((c) => c.id === id) ?? extras.find((c) => c.id === id)).filter(Boolean) as Cand[];

  const pickSlot = (s: Slot) => {
    setSlot(s);
    const occ = tw.lineup.current[s];
    const bench = tw.players.filter((p) => p.id !== occ && SKILL.includes(p.pos) && canPlay(p, s) && !isDead(p)).sort((a, b) => (b.espn_week ?? 0) - (a.espn_week ?? 0))[0];
    setIds([occ, bench?.id].filter((x): x is number => x != null));
  };
  const toggle = (id: number) => setIds((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : cur.length >= 3 ? [...cur.slice(0, 2), id] : [...cur, id]));
  const found = q.trim().length >= 2 && roles
    ? Object.entries(roles.players).filter(([k, v]) => SKILL.includes(v.pos) && ELIG[slot].includes(v.pos) && !by.has(Number(k)) && v.name.toLowerCase().includes(q.trim().toLowerCase())).slice(0, 6)
    : [];
  const addExtra = (id: number, name: string, pos: Pos, team: string | null) => {
    setExtras((e) => (e.some((x) => x.id === id) ? e : [...e, { id, name, pos, team, headshot: `https://a.espncdn.com/i/headshots/nfl/players/full/${id}.png`, slot: null, espn: null, injury: "ACTIVE", bye: false, opp: null, home: null, day: null }]));
    setIds((cur) => (cur.includes(id) ? cur : cur.length >= 3 ? [...cur.slice(0, 2), id] : [...cur, id]));
    setQ("");
  };

  const cs = sel.map((c) => ({ c, card: cards[c.id] }));
  const pts = cs.map(({ c, card }) => pt(c, card));
  const dead = pts.map((p) => isDead(p));
  const alive = cs.map((_, i) => i).filter((i) => !dead[i]);
  const tops = alive.length >= 2 ? pTop(alive.map((i) => pts[i]), ratioQ) : alive.length === 1 ? [1] : [];
  const top: (number | null)[] = pts.map((_, i) => (dead[i] ? 0 : tops[alive.indexOf(i)] ?? null));
  const mu = tw.matchup;
  const occ = by.get(tw.lineup.current[slot]);
  const base = mu ? mu.my_proj - (occ?.espn_week ?? 0) : null;
  const win = pts.map((p) => (mu && base != null && p.espn_week != null ? phi((base + (dead[pts.indexOf(p)] ? 0 : p.espn_week) - mu.opp_proj) / mu.margin_sd) : null));

  const call = callOf(cs, pts, dead, top, win, ratioQ, slot);
  const [hi, lo] = cutFor(slot);
  useEffect(() => {
    copyRef.current = () => duelMarkdown(tw.week, slot, cs, pts, dead, top, win, call, ratioQ);
  });

  const cols = `152px repeat(${Math.max(1, cs.length)}, minmax(0, 1fr))`;
  const anyRb = cs.some(({ c }) => c.pos === "RB");
  const Cell = ({ children, cls = "" }: { children: React.ReactNode; cls?: string }) => <div className={`dc ${cls}`}>{children}</div>;
  const row = (label: string, f: (x: { c: Cand; card?: MxCard | "err" }, i: number) => React.ReactNode, cls = "") => (
    <div className={`dr ${cls}`} style={{ gridTemplateColumns: cols }}><div className="dk">{label}</div>{cs.map((x, i) => <Cell key={x.c.id} cls={dead[i] ? "dead" : ""}>{f(x, i)}</Cell>)}</div>
  );
  const K = (x: { card?: MxCard | "err" }) => (x.card && x.card !== "err" ? x.card : null);

  return (
    <div className="mx-duel">
      <div className="mx-dctl">
        <div className="seg" role="group" aria-label="Slot">
          {slots.map((s) => <button key={s} type="button" aria-pressed={slot === s} onClick={() => pickSlot(s)} title={`Now: ${by.get(tw.lineup.current[s])?.name ?? "empty"}`}>{s}</button>)}
        </div>
        <div className="mx-dchips" role="group" aria-label="Candidates">
          {all.map((c) => (
            <button key={c.id} type="button" className={`mx-dchip ${ids.includes(c.id) ? "on" : ""}`} aria-pressed={ids.includes(c.id)} onClick={() => toggle(c.id)}>
              <i style={{ background: POS_VAR[c.pos] }} />{lastName(c.name)}<small className="num">{c.espn != null ? f1(c.espn) : ""}</small>{c.slot && c.slot !== "BE" && c.slot !== "IR" ? <em>{c.slot}</em> : null}
            </button>
          ))}
          <input className="mx-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Add anyone (free agent, trade target)…" aria-label="Add a player" />
        </div>
        {found.length > 0 && <div className="mx-found">{found.map(([k, v]) => (
          <button key={k} type="button" className="mx-row" onClick={() => addExtra(Number(k), v.name, v.pos, v.team)}><span className="posband" style={{ background: POS_VAR[v.pos] }} /><span className="nm"><b>{v.name}</b><small>{v.pos} · {v.team ?? "FA"}</small></span></button>
        ))}</div>}
      </div>

      {cs.length < 2 ? <p className="mx-empty">Pick two or three players for {slot} above.</p> : (
        <>
          <div className="mx-dtbl">
            <div className="dr head" style={{ gridTemplateColumns: cols }}>
              <div className="dk" />
              {cs.map(({ c }, i) => (
                <div key={c.id} className={`dc hd ${dead[i] ? "dead" : ""}`}>
                  <Face p={{ headshot: c.headshot, pos: c.pos, name: c.name }} />
                  <div><b>{c.name}</b><small>{c.pos} · {c.team ?? "FA"}{c.slot && c.slot !== "BE" && c.slot !== "IR" ? ` · now ${c.slot}` : c.slot === "BE" ? " · bench" : ""}</small></div>
                  <button type="button" className="tbtn sm" onClick={() => onOpen(c.id)} title="Read him against his defense">Read →</button>
                </div>
              ))}
            </div>
            {row("ESPN, week " + tw.week, ({ c, card }, i) => <span className="big num">{dead[i] ? "0.0" : f1(c.espn ?? K({ card })?.proj.espn)}<Status p={pts[i]} /></span>, "lead")}
            {row("Model", ({ c, card }) => <span className="dim num">{f1(K({ card })?.proj.model ?? by.get(c.id)?.est_ppg ?? null)}</span>)}
            {row("Plays", ({ c, card }, i) => <span className="num">{c.opp ?? K({ card })?.opp ? `${c.home === false ? "@" : "vs"} ${c.opp ?? K({ card })?.opp}${c.day ? ` · ${c.day}` : ""}` : dead[i] ? "–" : "…"}</span>)}
            {row("Pass defense", ({ card }) => { const k = K({ card }); return k?.defense ? <Grade g={k.defense.pass} /> : <span className="dim">{card === "err" ? "n/a" : "…"}</span>; })}
            {anyRb && row("Run defense", ({ c, card }) => { const k = K({ card }); return c.pos === "RB" && k?.defense ? <Grade g={k.defense.run} /> : <span className="dim">–</span>; })}
            {row("Matchup in the number", ({ card }) => { const k = K({ card }); return k?.priced && k.miss_sd ? <span className="mxmini"><b className="num">{signed1(k.priced.pts)}</b><Band pts={k.priced.pts} miss={k.miss_sd} /></span> : <span className="dim">–</span>; }, "priced")}
            {row("Coverage mix", ({ card }) => { const r = K({ card })?.rows.find((x) => x.key === "lens"); return r ? <span className="num">{r.effect}</span> : <span className="dim">–</span>; })}
            {row("Role", ({ card }) => { const r = K({ card })?.rows.find((x) => x.key === "role"); return r ? <span className="rl">{r.him}{r.them ? <small>{r.them}</small> : null}</span> : <span className="dim">–</span>; })}
            {row(`Scores ${hi}+`, (_x, i) => <span className="num">{dead[i] ? "–" : pct(pAtLeast(pts[i], hi, ratioQ))}</span>)}
            {row(`Scores under ${lo}`, (_x, i) => <span className="num">{dead[i] ? "–" : pct(pUnder(pts[i], lo, ratioQ))}</span>)}
            {row(cs.length === 2 ? "Outscores the other" : "Outscores both others", (_x, i) => <span className="big num">{top[i] == null ? "–" : pct(top[i])}</span>, "lead")}
            {mu && row("Your win chance with him", (_x, i) => <span className="num">{win[i] == null ? "–" : pct(win[i])}</span>)}
          </div>
          <section className={`mx-call ${call.tone}`} aria-label="The call">
            <h3>{call.headline}</h3>
            {call.lines.map((l, i) => <p key={i}>{l}</p>)}
          </section>
        </>
      )}
      <p className="mx-note">Odds use ESPN’s projection times the spread of how past results compared with their projections at that position (2021–25), treating players as independent. Win chance uses your projected margin this week against {mu?.opp_name ?? "your opponent"} with the slot filled by each player. “In the number” is how much of each projection is the defense, measured in 2023–25; it is already counted, so it is shown to explain the numbers, never to move them.</p>
    </div>
  );
}

interface Call { tone: "clear" | "lean" | "toss" | "none"; headline: string; lines: string[] }
type Entry = { c: Cand; card?: MxCard | "err" };

function callOf(cs: Entry[], pts: Pt[], dead: boolean[], top: (number | null)[], win: (number | null)[], q: Record<string, number[]> | undefined, slot: string): Call {
  const alive = cs.map((_, i) => i).filter((i) => !dead[i]);
  if (cs.length < 2) return { tone: "none", headline: "Pick at least two players.", lines: [] };
  if (alive.length === 0) return { tone: "none", headline: "Nobody here can play this week.", lines: ["Every player is out, on a bye or has no ESPN projection."] };
  const why = (i: number) => (pts[i].game?.bye ? "is on a bye" : ["OUT", "INJURY_RESERVE", "SUSPENSION"].includes(pts[i].injury) ? "is out" : "has no ESPN projection");
  const lines: string[] = [];
  const out = cs.map((_, i) => i).filter((i) => dead[i]);
  if (alive.length === 1) {
    const w = alive[0];
    return { tone: "clear", headline: `Start ${lastName(cs[w].c.name)} at ${slot}.`, lines: out.map((i) => `${lastName(cs[i].c.name)} ${why(i)}.`) };
  }
  const order = [...alive].sort((a, b) => (pts[b].espn_week ?? 0) - (pts[a].espn_week ?? 0));
  const a = order[0], b = order[1];
  const gap = (pts[a].espn_week ?? 0) - (pts[b].espn_week ?? 0);
  const p = pBeats(pts[a], pts[b], q);
  const nm = (i: number) => lastName(cs[i].c.name);
  let tone: Call["tone"] = "toss", headline: string;
  if (p == null) { tone = "none"; headline = `No odds for ${nm(a)} vs ${nm(b)}.`; }
  else if (p >= 0.7) { tone = "clear"; headline = `Start ${nm(a)} at ${slot}.`; lines.push(`ESPN has him ${f1(gap)} ahead of ${nm(b)}, and he outscores him ${pct(p)} of the time.`); }
  else if (p >= 0.58) { tone = "lean"; headline = `Lean ${nm(a)} at ${slot}.`; lines.push(`${pct(p)} to outscore ${nm(b)}, ${f1(gap)} points ahead. A normal week misses by more than that, so this is a lean, not a call.`); }
  else { headline = `Toss-up at ${slot}: ${nm(a)} by a hair.`; lines.push(`${f1(gap)} points apart is inside a normal week’s miss (he outscores ${nm(b)} ${pct(p)} of the time). Decide on what ESPN cannot see, such as injury news or a role change, not on the defenses.`); }
  const ka = cs[a].card, kb = cs[b].card;
  if (ka && kb && ka !== "err" && kb !== "err" && ka.priced && kb.priced) {
    const d = ka.priced.pts - kb.priced.pts;
    lines.push(Math.abs(d) < 0.5
      ? `The two defenses are worth about the same inside ESPN’s numbers (${nm(a)} ${signed1(ka.priced.pts)}, ${nm(b)} ${signed1(kb.priced.pts)}), so the matchup is not a tiebreaker.`
      : `Inside ESPN’s numbers the matchup is worth ${signed1(ka.priced.pts)} to ${nm(a)} and ${signed1(kb.priced.pts)} to ${nm(b)}. That is already counted in the gap above, so don’t count it twice.`);
  }
  if (win[a] != null && win[b] != null) lines.push(`Your win chance this week: ${pct(win[a])} with ${nm(a)}, ${pct(win[b])} with ${nm(b)} (${(((win[a] ?? 0) - (win[b] ?? 0)) * 100 >= 0 ? "+" : "−")}${Math.abs(((win[a] ?? 0) - (win[b] ?? 0)) * 100).toFixed(1)} points).`);
  for (const i of out) lines.push(`${nm(i)} ${why(i)}.`);
  void top;
  return { tone, headline, lines };
}

function duelMarkdown(week: number, slot: string, cs: Entry[], pts: Pt[], dead: boolean[], top: (number | null)[], win: (number | null)[], call: Call, q?: Record<string, number[]>): string {
  const [hi, lo] = cutFor(slot);
  if (cs.length < 2) return `# Start/sit at ${slot}, week ${week}\n\nPick two or three players for ${slot} first.`;
  const out = [`# Start/sit at ${slot}, week ${week}`, "", `**${call.headline}**`, ...call.lines.map((l) => `- ${l}`), ""];
  const head = ["", ...cs.map(({ c }) => c.name)];
  const row = (label: string, f: (e: Entry, i: number) => string) => `| ${[label, ...cs.map(f)].join(" | ")} |`;
  const kk = (e: Entry) => (e.card && e.card !== "err" ? e.card : null);
  out.push(`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`,
    row("ESPN", (_e, i) => (dead[i] ? "0.0" : f1(pts[i].espn_week))),
    row("Model", (e) => f1(kk(e)?.proj.model ?? null)),
    row("Opponent", (e) => kk(e)?.opp ?? e.c.opp ?? "–"),
    row("Pass defense", (e) => { const k = kk(e); return k?.defense ? `${k.defense.pass.label} (${ord(k.defense.pass.rank)} of ${k.defense.pass.of})` : "–"; }),
    row("Matchup inside ESPN’s number (pts)", (e) => { const k = kk(e); return k?.priced ? signed1(k.priced.pts) : "–"; }),
    row(`Scores ${hi}+`, (_e, i) => (dead[i] ? "–" : pct(pAtLeast(pts[i], hi, q)))),
    row(`Scores under ${lo}`, (_e, i) => (dead[i] ? "–" : pct(pUnder(pts[i], lo, q)))),
    row("Outscores the others", (_e, i) => (top[i] == null ? "–" : pct(top[i]))),
    row("Your win chance with him", (_e, i) => (win[i] == null ? "–" : pct(win[i]))));
  out.push("", "_Odds: ESPN projection times the 2021–25 spread of actual/projection at the position, independent outcomes. “Matchup inside ESPN’s number” is measured (2023–25) and already counted; it never changes the projection._");
  return out.join("\n");
}
