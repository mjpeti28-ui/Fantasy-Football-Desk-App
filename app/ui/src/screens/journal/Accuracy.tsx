import { type ReactNode, useEffect, useState } from "react";
import type { AccBin, Accuracy as Acc } from "../../types";
import { accuracyApi } from "../../data";
import { f1, signed } from "../../format";

/* How we're doing: every number Desk shows before kickoff, scored on this season's games. Each line
   is measured against what it promised (an 80% range should hold 80%; a 60% favourite should win
   60%) with a 95% range, so "outside" means outside the noise, not a reason to retune on six weeks. */

type Tone = "ok" | "off" | "few";
const tone = (lo: number | null | undefined, hi: number | null | undefined, target: number, n: number, few = 30): Tone =>
  n < few || lo == null || hi == null ? "few" : lo <= target && target <= hi ? "ok" : "off";
const WORD: Record<Tone, string> = { ok: "on target", off: "outside its range", few: "too few to tell" };
const pp = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);

/** One horizontal 0–100% track: measured value with its 95% range, and the target as a tick. */
function Track({ v, ci, target, lo = 0, hi = 1 }: { v: number; ci: [number | null, number | null]; target: number; lo?: number; hi?: number }) {
  const x = (t: number) => `${((Math.max(lo, Math.min(hi, t)) - lo) / (hi - lo)) * 100}%`;
  return (
    <span className="ac-trk">
      {ci[0] != null && ci[1] != null && <i className="ci" style={{ left: x(ci[0]), width: `calc(${x(ci[1])} - ${x(ci[0])})` }} />}
      <i className="tgt" style={{ left: x(target) }} title={`target ${pp(target)}`} />
      <i className="v" style={{ left: x(v) }} />
    </span>
  );
}

/** Predicted vs what happened, by bin: on the diagonal = says what it means. */
function Calib({ bins, lo = 0, hi = 1, label }: { bins: AccBin[]; lo?: number; hi?: number; label: string }) {
  const W = 220, H = 150, L = 28, B = 22, T = 8, R = 8;
  const x = (c: number) => L + ((c - lo) / (hi - lo)) * (W - L - R);
  const y = (h: number) => T + (1 - (h - lo) / (hi - lo)) * (H - T - B);
  const ticks = [lo, (lo + hi) / 2, hi];
  return (
    <svg className="ac-cal" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {ticks.map((t) => <g key={t}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" /><text x={L - 5} y={y(t) + 3.5} textAnchor="end">{Math.round(t * 100)}</text>
        <text x={x(t)} y={H - 6} textAnchor="middle">{Math.round(t * 100)}</text></g>)}
      <line x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} className="diag" />
      {bins.map((b, i) => <g key={i}><title>{`Said ${pp(b.pred)}: happened ${pp(b.actual)} (${b.n} cases, range ${pp(b.ci[0])}–${pp(b.ci[1])})`}</title>
        <line x1={x(b.pred)} x2={x(b.pred)} y1={y(Math.max(lo, b.ci[0]))} y2={y(Math.min(hi, b.ci[1]))} className="whisk" />
        <circle cx={x(b.pred)} cy={y(Math.max(lo, Math.min(hi, b.actual)))} r={2.5 + Math.min(5, Math.sqrt(b.n) / 3)} className="pt" /></g>)}
    </svg>
  );
}

function Line({ label, t, value, note, children }: { label: string; t: Tone; value: string; note: string; children?: ReactNode }) {
  return (
    <div className={`ac-line ${t}`}>
      <span className="lab">{label}</span>
      <b className="val num">{value}</b>
      {children ?? <span />}
      <span className="vd"><em>{WORD[t]}</em> {note}</span>
    </div>
  );
}

export function AccuracyPanel() {
  const [a, setA] = useState<Acc | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { accuracyApi.get().then(setA, (e) => setErr(String(e?.message ?? e))); }, []);
  if (err) return <section className="ac"><h3>How we're doing</h3><p className="dim">{err}</p></section>;
  if (!a) return <section className="ac"><h3>How we're doing</h3><p className="dim">Scoring this season…</p></section>;
  const r = a.range, b15 = a.boom?.["15"], b20 = a.boom?.["20"], h = a.h2h, w = a.winp, L = a.lineup;
  const backs = Object.entries(a.back).filter(([, v]) => v.n > 0);
  return (
    <section className="ac">
      <h3>How we're doing <small>weeks 1–{a.through_week}, {a.n_players.toLocaleString()} rostered player-games</small></h3>
      <p className="ac-sub">Each number Desk shows before kickoff, checked against what happened. The bar is this season with its 95% range; the tick is what the number promises.
        “Outside its range” means the gap is bigger than noise, so worth a look. It's not a reason to retune off a few weeks.</p>
      <div className="ac-lines">
        {r && <Line label="80% outcome ranges" t={tone(r.ci[0], r.ci[1], r.expect, r.n)} value={pp(r.rate)}
          note={`of scores landed inside (should be 80%). By position: ${Object.entries(r.by_pos).map(([p, v]) => `${p} ${pp(v.rate)}`).join(", ")}.`}>
          <Track v={r.rate} ci={r.ci} target={r.expect} lo={0.5} hi={1} /></Line>}
        {h && <Line label="Head-to-head odds" t={tone(h.ci[0], h.ci[1], h.pred, h.n)} value={pp(h.actual)}
          note={`of favourites won; the odds said ${pp(h.pred)} on average (${h.n.toLocaleString()} same-team, same-position pairs).`}>
          <Track v={h.actual} ci={h.ci} target={h.pred} lo={0.5} hi={1} /></Line>}
        {b15 && <Line label="Chance of 15+" t={tone(b15.ci[0], b15.ci[1], b15.pred, b15.n)} value={pp(b15.actual)}
          note={`scored 15+; predicted ${pp(b15.pred)}.`}><Track v={b15.actual} ci={b15.ci} target={b15.pred} /></Line>}
        {b20 && <Line label="Chance of 20+" t={tone(b20.ci[0], b20.ci[1], b20.pred, b20.n)} value={pp(b20.actual)}
          note={`scored 20+; predicted ${pp(b20.pred)}.`}><Track v={b20.actual} ci={b20.ci} target={b20.pred} /></Line>}
        {w.n > 0 && w.ci && <Line label="Weekly win odds" t={tone(w.ci[0], w.ci[1], w.pred!, w.n)} value={pp(w.actual)}
          note={`of league favourites won; predicted ${pp(w.pred)} (${w.n} games). Brier ${w.brier} vs ${w.brier_coin} for a coin flip, lower is better.`}>
          <Track v={w.actual!} ci={w.ci} target={w.pred!} lo={0.3} hi={1} /></Line>}
        {L.vs_espn && L.vs_espn.n > 0 && <Line label="Your lineup vs ESPN's" t={L.vs_espn.n < 8 ? "few" : tone(L.vs_espn.ci[0], L.vs_espn.ci[1], 0, 99)} value={signed(L.vs_espn.mean)}
          note={`pts a week against the lineup ESPN's projections would have started (league average ${signed(L.league_vs_espn?.mean)}; 2021–25 managers averaged ${signed(L.expect_vs_espn)}). ${f1(L.left)} pts left on your bench in total.`}>
          <span className="ac-wk">{L.weeks.map((x) => <i key={x.week} className={x.pts >= x.espn ? "up" : "dn"} style={{ height: `${Math.min(100, 12 + Math.abs(x.pts - x.espn) * 3)}%` }} title={`Week ${x.week}: ${f1(x.pts)} vs ESPN's lineup ${f1(x.espn)} (best possible ${f1(x.opt)})`} />)}</span></Line>}
      </div>
      <div className="ac-figs">
        {h && <figure><Calib bins={h.bins} lo={0.5} label="Head-to-head calibration" /><figcaption>Head-to-head: said (across) vs favourite won (up). On the dashed line = means what it says.</figcaption></figure>}
        {b15 && <figure><Calib bins={b15.bins} label="15+ calibration" /><figcaption>Chance of 15+: said vs happened, by bin. Whiskers are the 95% range.</figcaption></figure>}
        {backs.length > 0 && <figure className="ac-back"><table><thead><tr><th>First game back</th><th>vs ESPN</th><th>expected</th><th>n</th></tr></thead>
          <tbody>{backs.map(([p, v]) => <tr key={p}><th>{p}</th><td className="num">{signed(v.mean)}</td><td className="num dim">{signed(v.expect)}</td><td className="num dim">{v.n}</td></tr>)}</tbody></table>
          <figcaption>Points vs ESPN's projection in a player's first game back, this season, against the 2021–25 measure the start screen uses. Too few cases yet to confirm or reject it.</figcaption></figure>}
        {a.sources.length > 0 && <figure className="ac-back"><table><thead><tr><th>Projection source</th><th>error</th><th>rank corr</th><th>weeks</th></tr></thead>
          <tbody>{[...a.sources].sort((x, y) => x.mae - y.mae).map((s) => <tr key={s.source}><th>{s.source.replace("_weekly", "").replace("espn", "ESPN").replace("fantasypros", "FantasyPros").replace("pff", "PFF").replace("blend", "our blend")}</th>
            <td className="num">{s.mae.toFixed(2)}</td><td className="num">{s.rank_corr.toFixed(3)}</td><td className="num dim">{s.weeks}</td></tr>)}</tbody></table>
          <figcaption>Average miss in points per player-week (lower is better) from the weekly run's scoring.</figcaption></figure>}
      </div>
    </section>
  );
}
