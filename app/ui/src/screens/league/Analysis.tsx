import { useState } from "react";
import type { LeagueAnalysis, Pos, PowerTeam } from "../../types";
import { f1, pct, signed } from "../../format";
import "./analysis.css";

const POS_VAR: Record<string, string> = { QB: "var(--qb)", RB: "var(--rb)", WR: "var(--wr)", TE: "var(--te)", K: "var(--k)", DST: "var(--dst)" };
const POS_LABEL: Record<string, string> = { QB: "QB", RB: "RB", WR: "WR", TE: "TE", K: "K", DST: "D/ST" };
const SKILL: Pos[] = ["QB", "RB", "WR", "TE"];
const short = (n: string) => (n.length > 14 ? n.split(" ").slice(0, 2).join(" ").slice(0, 14) : n);

interface Props { a: LeagueAnalysis; teams: PowerTeam[]; order: number[]; myId: number; oppId: number | null; onSelect: (id: number) => void; onTrade?: (id: number) => void }

const role = (id: number, myId: number, oppId: number | null) => (id === myId ? "me" : id === oppId ? "opp" : "");

/* 1 ─ positional strength: starter points per week at each position vs the league average */
function Strength({ a, teams, order, myId, oppId, onSelect, onTrade }: Props) {
  const pos = a.positions;
  const mean: Record<string, number> = Object.fromEntries(pos.map((p) => [p, order.reduce((s, t) => s + a.pos_ppw[t][p], 0) / order.length]));
  const dev = (t: number, p: string) => a.pos_ppw[t][p] - mean[p];
  const span = Math.max(...order.flatMap((t) => pos.map((p) => Math.abs(dev(t, p)))), 1);
  const name = (t: number) => teams.find((x) => x.id === t)!.name;

  // trade fits: their surplus where you're short + your surplus where they're short (skill spots)
  const fits = order.filter((t) => t !== myId).map((t) => {
    let give: { p: string; v: number } | null = null, get: { p: string; v: number } | null = null, score = 0;
    for (const p of SKILL) {
      const mine = dev(myId, p), theirs = dev(t, p);
      if (mine < 0 && theirs > 0) { const v = Math.min(-mine, theirs); score += v; if (!get || v > get.v) get = { p, v }; }
      if (mine > 0 && theirs < 0) { const v = Math.min(mine, -theirs); score += v; if (!give || v > give.v) give = { p, v }; }
    }
    return { t, score, give, get };
  }).filter((f) => f.give && f.get).sort((x, y) => y.score - x.score).slice(0, 3);

  return (
    <section className="an-card wide">
      <header><h4>Where the points come from</h4><p>Starter points per week by position, rest of the regular season, against the league average. From the same simulation as the odds.</p></header>
      <div className="an-stack">
        <div className="mx-wrap">
          <table className="mx">
            <thead><tr><th />{pos.map((p) => <th key={p}><i className="sw" style={{ background: POS_VAR[p] }} />{POS_LABEL[p]}</th>)}<th>Total</th></tr></thead>
            <tbody>
              {order.map((t) => {
                const tot = pos.reduce((s, p) => s + a.pos_ppw[t][p], 0);
                const totMean = pos.reduce((s, p) => s + mean[p], 0);
                return (
                  <tr key={t} className={role(t, myId, oppId)} onClick={() => onSelect(t)}>
                    <th className="tn">{short(name(t))}</th>
                    {pos.map((p) => {
                      const v = dev(t, p), alpha = Math.min(1, Math.abs(v) / span);
                      const streams = a.stream_rate[t][p] >= 0.3;
                      return (
                        <td key={p} className={`${v >= 0 ? "pos" : "neg"} ${streams ? "streams" : ""}`} style={{ ["--a" as string]: (0.12 + 0.6 * alpha).toFixed(2) }}
                          title={`${name(t)} ${p}: ${f1(a.pos_ppw[t][p])} per week (league ${f1(mean[p])})${streams ? `; a waiver streamer starts ${pct(a.stream_rate[t][p])} of weeks` : ""}`}>
                          {signed(v)}
                        </td>
                      );
                    })}
                    <td className="tot">{signed(tot - totMean)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="an-note">Hatched cell: a waiver-level streamer starts there in 30%+ of simulated weeks (thin depth, or a starter the model rates below the waiver wire).</p>
        </div>
        <div className="fits">
          <h5>Trade fits</h5>
          <p className="an-note">Teams strong where you're short and short where you're strong. A starting point for Trades, not an evaluation.</p>
          {fits.length === 0 && <p className="an-note">No team lines up both ways right now.</p>}
          {fits.map((f) => (
            <button key={f.t} type="button" className="fit" onClick={() => (onTrade ? onTrade(f.t) : onSelect(f.t))}>
              <b>{name(f.t)} {onTrade && <span className="go">Open in Trades →</span>}</b>
              <span className="fl"><i className="sw" style={{ background: POS_VAR[f.get!.p] }} />They have <em>{POS_LABEL[f.get!.p]}</em> to spare <span className="num">{signed(dev(f.t, f.get!.p))}</span>, you're {signed(dev(myId, f.get!.p))}</span>
              <span className="fl"><i className="sw" style={{ background: POS_VAR[f.give!.p] }} />You have <em>{POS_LABEL[f.give!.p]}</em> to spare <span className="num">{signed(dev(myId, f.give!.p))}</span>, they're {signed(dev(f.t, f.give!.p))}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

/* 2 ─ seed odds: probability of each final regular-season rank */
function Seeds({ a, teams, order, myId, oppId, onSelect }: Props) {
  const n = order.length, cut = a.playoff_teams;
  return (
    <section className="an-card">
      <header><h4>Where each team finishes</h4><p>Chance of each final seed. Seeding is by total points; 1–2 face the 7–8 seeds, 9–10 miss.</p></header>
      <div className="seeds" style={{ gridTemplateColumns: `minmax(0, 1.25fr) repeat(${n}, minmax(0, 1fr))` }}>
        <span />
        {Array.from({ length: n }, (_, k) => <span key={k} className={`sh ${k === 1 ? "cut2" : ""} ${k === cut - 1 ? "cutp" : ""}`}>{k + 1}</span>)}
        {order.map((t) => {
          const d = a.seed_dist[t], r = role(t, myId, oppId);
          return [
            <button key={`n${t}`} type="button" className={`tn ${r}`} onClick={() => onSelect(t)}>{short(teams.find((x) => x.id === t)!.name)}</button>,
            ...d.map((p, k) => (
              <span key={`${t}-${k}`} className={`cell ${r} ${k === 1 ? "cut2" : ""} ${k === cut - 1 ? "cutp" : ""}`} style={{ ["--a" as string]: Math.min(1, p * 2.2).toFixed(2) }}
                title={`${teams.find((x) => x.id === t)!.name}: seed ${k + 1}, ${pct(p)}`}>{p >= 0.1 ? Math.round(p * 100) : ""}</span>
            )),
          ];
        })}
      </div>
      <p className="an-note">Numbers are percent, shown at 10% and up. The solid rule marks the top-2 seeds, the dashed rule the playoff line.</p>
    </section>
  );
}

/* 3 ─ luck vs strength */
function Scatter({ a, teams, order, myId, oppId, onSelect }: Props) {
  const W = 420, H = 300, L = 46, R = 16, T = 18, B = 40;
  const regIdx = a.weeks.map((w, i) => (w <= a.reg_end ? i : -1)).filter((i) => i >= 0);
  const pts = order.map((t) => {
    const tm = teams.find((x) => x.id === t)!;
    const strength = regIdx.reduce((s, i) => s + a.week_pts[t][i], 0) / regIdx.length;
    return { t, name: short(tm.name), x: strength, y: (tm.luck ?? 0) / Math.max(1, tm.weekly.length) };
  });
  const [x0, x1] = [Math.min(...pts.map((p) => p.x)) - 2, Math.max(...pts.map((p) => p.x)) + 2];
  const ym = Math.max(...pts.map((p) => Math.abs(p.y)), 5) * 1.15;
  const X = (v: number) => L + ((W - L - R) * (v - x0)) / (x1 - x0), Y = (v: number) => T + ((H - T - B) * (ym - v)) / (2 * ym);
  const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const [hover, setHover] = useState<number | null>(null);
  const labelY = new Map<number, number>();
  const ticks = [Math.ceil(x0 / 5) * 5, Math.round((x0 + x1) / 10) * 5, Math.floor(x1 / 5) * 5].filter((v, i, arr) => arr.indexOf(v) === i);
  return (
    <section className="an-card">
      <header><h4>Strength ahead vs luck so far</h4><p>Right = stronger projected lineup the rest of the way. Up = scored above projection so far. In past seasons only the horizontal carried forward.</p></header>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="sc" role="img" aria-label="Scatter of projected strength against luck so far">
        <line x1={L} x2={W - R} y1={Y(0)} y2={Y(0)} className="ax" />
        <line x1={X(mx)} x2={X(mx)} y1={T} y2={H - B} className="ax dash" />
        {ticks.map((v) => <text key={v} x={X(v)} y={H - B + 16} textAnchor="middle" className="tk">{v}</text>)}
        {[-Math.round(ym * 0.66), 0, Math.round(ym * 0.66)].map((v) => <text key={v} x={L - 8} y={Y(v) + 4} textAnchor="end" className="tk">{v > 0 ? `+${v}` : v}</text>)}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="lab">projected points per week, rest of regular season</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="lab" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>above projection per week so far</text>
        <text x={W - R - 4} y={T + 10} textAnchor="end" className="q">strong and lucky</text>
        <text x={W - R - 4} y={H - B - 6} textAnchor="end" className="q">strong, unlucky</text>
        <text x={L + 4} y={T + 10} className="q">lucky, weaker</text>
        <text x={L + 4} y={H - B - 6} className="q">weaker, unlucky</text>
        {(() => { // nudge labels that would overlap: same side, close in x, within 12px in y
          const placed: { x: number; y: number; right: boolean }[] = [];
          for (const p of [...pts].sort((a, b) => Y(a.y) - Y(b.y))) {
            const lx = X(p.x), right = lx < W - 110;
            let ly = Y(p.y) + 4;
            for (const q of placed) if (q.right === right && Math.abs(q.x - lx) < 90 && Math.abs(q.y - ly) < 12) ly = q.y + 12;
            placed.push({ x: lx, y: ly, right });
            labelY.set(p.t, ly);
          }
          return null;
        })()}
        {pts.map((p) => {
          const r = role(p.t, myId, oppId), lx = X(p.x), ly = Y(p.y), right = lx < W - 110;
          return (
            <g key={p.t} className={`pt ${r} ${hover === p.t ? "hv" : ""}`} onMouseEnter={() => setHover(p.t)} onMouseLeave={() => setHover(null)} onClick={() => onSelect(p.t)}>
              <circle cx={lx} cy={ly} r={14} className="hit" />
              <circle cx={lx} cy={ly} r={r ? 6 : 4.5} className="dot" />
              <text x={right ? lx + 9 : lx - 9} y={labelY.get(p.t) ?? ly + 4} textAnchor={right ? "start" : "end"} className="nm">{p.name}</text>
              <title>{`${p.name}: ${f1(p.x)} projected per week, ${signed(p.y)} per week vs projection so far`}</title>
            </g>
          );
        })}
      </svg>
    </section>
  );
}

/* 0 ─ the points race: seeding is by total points, so this is the standings that matter.
   Cumulative points against the league average each week, played weeks solid and the simulated
   average dashed; hover a line to pull it forward. */
function Race({ a, teams, order, myId, oppId, onSelect }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const played = Math.max(...teams.map((t) => t.weekly.length));
  const weeks = Array.from({ length: a.reg_end }, (_, i) => i + 1);
  const raw = new Map(order.map((t) => {
    const tm = teams.find((x) => x.id === t)!;
    let c = 0;
    const ys = weeks.map((w) => {
      if (w <= played) c += tm.weekly[w - 1] ?? 0;
      else { const i = a.weeks.indexOf(w); c += i >= 0 ? a.week_pts[t]?.[i] ?? 0 : 0; }
      return c;
    });
    return [t, ys];
  }));
  const avg = weeks.map((_, i) => [...raw.values()].reduce((s, ys) => s + ys[i], 0) / raw.size);
  const rel = new Map([...raw].map(([t, ys]) => [t, ys.map((v, i) => v - avg[i])]));
  const all = [...rel.values()].flat();
  const ym = Math.max(20, ...all.map(Math.abs)) * 1.08;
  const W = 900, H = 300, L = 44, R = 170, T = 14, B = 30;
  const X = (w: number) => L + ((W - L - R) * (w - 1)) / (weeks.length - 1), Y = (v: number) => T + ((H - T - B) * (ym - v)) / (2 * ym);
  const path = (ys: number[], from: number, to: number) => ys.slice(from - 1, to).map((v, i) => `${i ? "L" : "M"}${X(from + i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  // end labels, nudged apart
  const ends = [...rel].map(([t, ys]) => ({ t, y: Y(ys[ys.length - 1]), v: ys[ys.length - 1] })).sort((p, q) => p.y - q.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  const nameOf = (t: number) => short(teams.find((x) => x.id === t)!.name);
  const fin = [...rel].map(([t, ys]) => ({ t, v: ys[ys.length - 1], now: ys[played - 1] ?? 0 })).sort((p, q) => q.v - p.v);
  const me = fin.find((f) => f.t === myId), myRank = fin.findIndex((f) => f.t === myId) + 1;
  const third = fin[2], second = fin[1];
  const ticks = [-Math.round(ym * 0.6 / 10) * 10, 0, Math.round(ym * 0.6 / 10) * 10].filter((v, i, arr) => arr.indexOf(v) === i);
  const cls = (t: number) => `ln ${role(t, myId, oppId)} ${hover === t ? "hv" : ""} ${hover != null && hover !== t ? "dim" : ""}`;
  return (
    <section className="an-card wide race">
      <header><h4>The points race</h4><p>Seeding is by total points, so this is the table that decides seeds: each team's season total against the league average, week by week. Solid = played; dashed = the simulation's average for the rest of the regular season.</p></header>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="rc" role="img" aria-label="Cumulative points above the league average by week, played and projected" onMouseLeave={() => setHover(null)}>
        <rect x={X(played)} y={T} width={X(weeks.length) - X(played)} height={H - T - B} className="future" />
        <text x={X(played) + 6} y={T + 12} className="tk">projected →</text>
        {ticks.map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className={v === 0 ? "ax" : "grid"} /><text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tk">{v > 0 ? `+${v}` : v}</text></g>)}
        <text x={L - 6} y={Y(0) - 6} textAnchor="end" className="tk">avg</text>
        {weeks.map((w) => <text key={w} x={X(w)} y={H - 10} textAnchor="middle" className="tk">{w}</text>)}
        {[...rel].sort(([p], [q]) => (role(p, myId, oppId) ? 1 : 0) - (role(q, myId, oppId) ? 1 : 0) || (p === hover ? 1 : 0) - (q === hover ? 1 : 0)).map(([t, ys]) => (
          <g key={t} className={cls(t)} onMouseEnter={() => setHover(t)} onClick={() => onSelect(t)}>
            <path d={path(ys, 1, played)} className="past" />
            <path d={path(ys, played, weeks.length)} className="proj" />
            <path d={path(ys, 1, weeks.length)} className="hit" />
            <circle cx={X(played)} cy={Y(ys[played - 1])} r={3.5} className="now" />
          </g>
        ))}
        {ends.map((e) => (
          <text key={e.t} x={W - R + 8} y={e.y + 4} className={`end ${role(e.t, myId, oppId)} ${hover === e.t ? "hv" : ""}`} onMouseEnter={() => setHover(e.t)} onClick={() => onSelect(e.t)}>
            {e.v >= 0 ? "+" : "−"}{Math.abs(Math.round(e.v))} {nameOf(e.t)}</text>
        ))}
        {hover != null && (() => { const ys = rel.get(hover)!; return <text x={X(played)} y={Math.max(T + 26, Y(ys[played - 1]) - 10)} textAnchor="middle" className="hvl">{nameOf(hover)}: {Math.round(ys[played - 1]) >= 0 ? "+" : "−"}{Math.abs(Math.round(ys[played - 1]))} now → {Math.round(ys[ys.length - 1]) >= 0 ? "+" : "−"}{Math.abs(Math.round(ys[ys.length - 1]))} after week {a.reg_end}</text>; })()}
      </svg>
      <p className="an-note">
        {me ? <>You're {Math.round(me.now) >= 0 ? "+" : "−"}{Math.abs(Math.round(me.now))} against the average now and finish about {Math.round(me.v) >= 0 ? "+" : "−"}{Math.abs(Math.round(me.v))} on the simulation's averages (#{myRank}). </> : null}
        {myRank <= 2 && third ? <>The #3 team ends about {Math.round((me?.v ?? 0) - third.v)} points behind you: that's the cushion for a top-2 seed. </> : second && me ? <>You'd need about {Math.round(second.v - me.v)} more points than projected to reach #2. </> : null}
        {a.reg_end > played ? <>One team's weekly score swings about ±23 points, so {a.reg_end - played} weeks of luck can move a line roughly ±{Math.round(22.6 * Math.sqrt(a.reg_end - played) / 5) * 5}: read the dashed part as the middle of the outcomes, not the final order.</> : null}
      </p>
    </section>
  );
}

/* 4 ─ season ahead: expected points each remaining week */
function Calendar({ a, teams, order, myId, oppId, onSelect }: Props) {
  const all = order.flatMap((t) => a.week_pts[t]);
  const [lo, hi] = [Math.min(...all), Math.max(...all)];
  const mine = a.week_pts[myId];
  const worst = mine.indexOf(Math.min(...mine.slice(0, a.weeks.filter((w) => w <= a.reg_end).length)));
  return (
    <section className="an-card wide">
      <header><h4>The season ahead</h4>
        <p>Expected lineup points each week after byes and injury risk; heavier fill is a stronger week. Your softest regular-season week is week {a.weeks[worst]} ({f1(mine[worst])}), outlined. Weeks {a.reg_end + 1}–{a.weeks[a.weeks.length - 1]} are the playoffs.</p></header>
      <div className="cal" style={{ gridTemplateColumns: `minmax(110px, 1.3fr) repeat(${a.weeks.length}, minmax(0, 1fr))` }}>
        <span />
        {a.weeks.map((w) => <span key={w} className={`ch ${w > a.reg_end ? "po" : ""}`}>{w}</span>)}
        {order.map((t) => {
          const r = role(t, myId, oppId);
          return [
            <button key={`n${t}`} type="button" className={`tn ${r}`} onClick={() => onSelect(t)}>{short(teams.find((x) => x.id === t)!.name)}</button>,
            ...a.week_pts[t].map((v, i) => {
              const z = (v - lo) / Math.max(1, hi - lo);
              return <span key={`${t}-${i}`} className={`c ${a.weeks[i] > a.reg_end ? "po" : ""} ${t === myId && i === worst ? "worst" : ""}`}
                style={{ ["--z" as string]: z.toFixed(3) }} title={`${teams.find((x) => x.id === t)!.name}, week ${a.weeks[i]}: ${f1(v)} expected`}>{Math.round(v)}</span>;
            }),
          ];
        })}
      </div>
    </section>
  );
}

/* 5 ─ points so far by position */
function Composition({ a, teams, order, myId, oppId, onSelect }: Props) {
  const pos = a.positions;
  return (
    <section className="an-card wide">
      <header><h4>Where the points came from so far</h4><p>Share of each team's starting-lineup points by position, weeks 1–{teams[0].weekly.length}.</p></header>
      <div className="legend">{pos.map((p) => <span key={p}><i className="sw" style={{ background: POS_VAR[p] }} />{POS_LABEL[p]}</span>)}</div>
      <div className="comp">
        {order.map((t) => {
          const so = a.pos_so_far[t] ?? {}, tot = pos.reduce((s, p) => s + (so[p] ?? 0), 0) || 1, r = role(t, myId, oppId);
          return (
            <div key={t} className={`crow ${r}`} onClick={() => onSelect(t)}>
              <span className="tn">{short(teams.find((x) => x.id === t)!.name)}</span>
              <span className="stack">
                {pos.map((p) => {
                  const share = (so[p] ?? 0) / tot;
                  return share > 0 ? <i key={p} style={{ width: `${share * 100}%`, background: POS_VAR[p] }} className={p === "WR" ? "wr" : ""}
                    title={`${POS_LABEL[p]}: ${f1(so[p] ?? 0)} pts (${pct(share)})`}>{share >= 0.1 ? `${POS_LABEL[p]} ${Math.round(share * 100)}` : ""}</i> : null;
                })}
              </span>
              <span className="num tt">{f1(tot)}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function Analysis(p: Props) {
  return (
    <div className="analysis">
      <h3 className="an-h">Analysis</h3>
      <Race {...p} />
      <Strength {...p} />
      <div className="an-row"><Seeds {...p} /><Scatter {...p} /></div>
      <Calendar {...p} />
      <Composition {...p} />
    </div>
  );
}
