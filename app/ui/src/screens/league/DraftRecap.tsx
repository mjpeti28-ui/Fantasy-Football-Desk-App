import { useMemo, useState } from "react";
import type { DPick, DraftRecap as DR, DTeam } from "../../types";
import { f1, lastName, signed, teamShort, tightName } from "../../format";
import { Ev } from "../../components/Analytics";
import { POS_VAR } from "../../components/bits";
import { useFile } from "../../components/FileContext";
import "./draft.css";

/* League › Draft recap (app/engine/draft.py). Every pick is valued over replacement (points minus games
   played × the first non-starter's per-game rate, floored at 0) and set against what that pick slot
   usually returns in this league by the same week (2021–25 drafts). Rest of season uses our forecast
   against this year's picks around the same slot. All of it is what happened, not a skill score: the
   history card says how little these grades carry. */

interface Props { d: DR; myId: number; oppId: number | null; onSelect: (id: number) => void }
const role = (id: number, myId: number, oppId: number | null) => (id === myId ? "me" : id === oppId ? "opp" : "");
const r2 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(2).replace("-", "−"));
const POS_LABEL: Record<string, string> = { QB: "QB", RB: "RB", WR: "WR", TE: "TE", K: "K", DST: "D/ST" };
const POS = ["QB", "RB", "WR", "TE", "K", "DST"];
const mean = (xs: (number | null)[]) => { const v = xs.filter((x): x is number => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const pickTip = (p: DPick, team: string) =>
  `${p.name} (${p.pos}), round ${p.round}, pick ${p.pick} by ${team}. ${f1(p.pts)} pts in ${p.gp} of ${p.possible} games; ` +
  `${signed(p.v_now)} over replacement vs the slot so far${p.v_absence < -0.5 ? ` (${signed(p.v_absence)} of it from missed games)` : ""}; ` +
  `${signed(p.v_ros)} rest of season. ${p.adp != null ? `PFF ADP ${f1(p.adp)}. ` : ""}` +
  (p.status === "kept" ? "Still on the team." : p.status === "traded" ? "Traded away." : `Dropped${p.left_after ? ` after week ${p.left_after}` : ""}${p.started_elsewhere ? `; ${f1(p.started_elsewhere)} pts started for other teams since` : ""}.`);

/* 1 ─ the grade table: so far (split per-game vs missed games) + rest of season = total */
function Grades({ d, myId, oppId, onSelect, name }: Props & { name: (id: number) => string }) {
  const order = [...d.teams].sort((a, b) => b.v_total - a.v_total);
  const mx = Math.max(...order.map((t) => Math.max(Math.abs(t.v_now) + Math.abs(t.v_ros), 1)));
  const seg = (v: number) => `${(50 * Math.abs(v)) / mx}%`;
  return (
    <section className="an-card wide">
      <header><h4>Who drafted best, so far <Ev l="descriptive" tip="What happened, scored against this league's own draft history. Not a repeatable-skill score (see the history card)." /></h4>
        <p>Each pick's value over replacement, minus what that pick slot usually returns in this league by week {d.through_week}, summed per team.
          Solid = points so far; light = our rest-of-season forecast vs this year's picks around the same slot. Zero is an average draft.</p></header>
      <table className="dr-grades">
        <thead><tr><th>#</th><th className="l">Team</th><th title="Value over the slot so far">So far</th>
          <th title="Part of 'so far' lost to games missed (injury, suspension, benching). Mostly luck.">Missed games</th>
          <th title={`Forecast value from week ${d.ros_from_week} on, vs this year's picks around the same slot`}>Rest of season</th>
          <th className="l">Total</th>
          <th title={`Average picks taken after PFF's preseason ADP (positive = players fell to them), relative to the league average. The whole league drafted ${f1(Math.abs(d.league_vs_adp ?? 0))} picks ahead of PFF's ADP on average.`}>vs ADP</th>
          <th title="Draft picks still on the roster / dropped / traded">Kept</th></tr></thead>
        <tbody>
          {order.map((t, i) => (
            <tr key={t.id} className={role(t.id, myId, oppId)} onClick={() => onSelect(t.id)}>
              <td className="num dim">{i + 1}</td>
              <td className="l tn">{name(t.id)}</td>
              <td className={`num ${t.v_now >= 0 ? "above" : "below"}`}>{signed(t.v_now)}</td>
              <td className="num dim">{signed(t.v_absence)}</td>
              <td className={`num ${t.v_ros >= 0 ? "above" : "below"}`}>{signed(t.v_ros)}</td>
              <td className="l">
                <span className="dv" title={`${name(t.id)}: ${signed(t.v_now)} so far, ${signed(t.v_ros)} rest of season`}>
                  <b className="mid" />
                  {[["now", t.v_now], ["ros", t.v_ros]].reduce<{ el: JSX.Element[]; pos: number; neg: number }>((acc, [k, v]) => {
                    const val = v as number, w = (50 * Math.abs(val)) / mx;
                    const left = val >= 0 ? 50 + acc.pos : 50 - acc.neg - w;
                    acc.el.push(<i key={k as string} className={`${k} ${val >= 0 ? "p" : "n"}`} style={{ left: `${left}%`, width: seg(val) }} />);
                    if (val >= 0) acc.pos += w; else acc.neg += w;
                    return acc;
                  }, { el: [], pos: 0, neg: 0 }).el}
                </span>
                <span className={`num tot ${t.v_total >= 0 ? "above" : "below"}`}>{signed(t.v_total)}</span>
              </td>
              <td className="num dim">{signed(t.vs_adp_rel)}</td>
              <td className="num dim">{t.kept}<small>/{t.dropped}/{t.traded}</small></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="an-note">Value is points over replacement: a late QB's raw points don't count as a steal, because a replacement QB scores a lot too. A bust costs only its slot's expectation, since you'd bench or cut a player below replacement.</p>
    </section>
  );
}

/* 2 ─ quadrant: so far (x) vs rest of season (y) */
function Quadrant({ d, myId, oppId, onSelect, name }: Props & { name: (id: number) => string }) {
  const W = 420, H = 326, L = 46, R = 16, T = 30, B = 60;
  const xm = Math.max(...d.teams.map((t) => Math.abs(t.v_now)), 20) * 1.15, ym = Math.max(...d.teams.map((t) => Math.abs(t.v_ros)), 20) * 1.15;
  const X = (v: number) => L + ((W - L - R) * (v + xm)) / (2 * xm), Y = (v: number) => T + ((H - T - B) * (ym - v)) / (2 * ym);
  const [hover, setHover] = useState<number | null>(null);
  const labelY = new Map<number, number>();
  const placed: { x: number; y: number; right: boolean }[] = [];
  for (const t of [...d.teams].sort((a, b) => Y(a.v_ros) - Y(b.v_ros))) {
    const lx = X(t.v_now), right = lx < W - 110;
    let ly = Y(t.v_ros) + 4;
    for (const q of placed) if (q.right === right && Math.abs(q.x - lx) < 90 && Math.abs(q.y - ly) < 12) ly = q.y + 12;
    placed.push({ x: lx, y: ly, right });
    labelY.set(t.id, ly);
  }
  const tk = (m: number) => { const s = m > 120 ? 50 : m > 50 ? 25 : 10; const v = Math.floor((m * 0.8) / s) * s; return [-v, 0, v]; };
  return (
    <section className="an-card">
      <header><h4>Hot start or built to last <Ev l="descriptive" tip="Right/up = above an average draft for those slots." /></h4>
        <p>Right = picks have beaten their slots so far. Up = the forecast likes them more than other picks from the same slots the rest of the way.</p></header>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="sc" role="img" aria-label="Scatter of draft value so far against rest-of-season draft value">
        <line x1={L} x2={W - R} y1={Y(0)} y2={Y(0)} className="ax" />
        <line x1={X(0)} x2={X(0)} y1={T} y2={H - B} className="ax" />
        {tk(xm).map((v) => <text key={v} x={X(v)} y={H - B + 16} textAnchor="middle" className="tk">{v > 0 ? `+${v}` : v}</text>)}
        {tk(ym).map((v) => <text key={v} x={L - 8} y={Y(v) + 4} textAnchor="end" className="tk">{v > 0 ? `+${v}` : v}</text>)}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="lab">value vs slot so far (pts over replacement)</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="lab" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>rest of season vs slot</text>
        <text x={W - R - 4} y={T - 12} textAnchor="end" className="q">hit and holding ↗</text>
        <text x={W - R - 4} y={H - B + 32} textAnchor="end" className="q">fast start, fading ↘</text>
        <text x={L + 4} y={T - 12} className="q">↖ slow start, loaded</text>
        <text x={L + 4} y={H - B + 32} className="q">↙ in a hole</text>
        {d.teams.map((t) => {
          const r = role(t.id, myId, oppId), lx = X(t.v_now), ly = Y(t.v_ros), right = lx < W - 110;
          return (
            <g key={t.id} className={`pt ${r} ${hover === t.id ? "hv" : ""}`} onMouseEnter={() => setHover(t.id)} onMouseLeave={() => setHover(null)} onClick={() => onSelect(t.id)}>
              <circle cx={lx} cy={ly} r={14} className="hit" />
              <circle cx={lx} cy={ly} r={r ? 6 : 4.5} className="dot" />
              <text x={right ? lx + 9 : lx - 9} y={labelY.get(t.id) ?? ly + 4} textAnchor={right ? "start" : "end"} className="nm">{name(t.id)}</text>
              <title>{`${t.name}: ${signed(t.v_now)} so far, ${signed(t.v_ros)} rest of season`}</title>
            </g>
          );
        })}
      </svg>
    </section>
  );
}

/* 3 ─ how much of this is luck: the league's own history */
function History({ d }: { d: DR }) {
  const h = d.history, hold = mean(h.hold), nx = mean(h.next_year), vp = mean(h.vs_points);
  const bar = (v: number | null, k: string) => (
    <span key={k} className="hb" title={`${k}: r ${r2(v)}`}>
      <b className="mid" />{v != null && <i className={v >= 0 ? "p" : "n"} style={{ left: v >= 0 ? "50%" : `${50 + 50 * v}%`, width: `${50 * Math.abs(v)}%` }} />}
    </span>
  );
  const row = (label: string, xs: (number | null)[], seasons: string[], avg: number | null, note: string) => (
    <div className="hr">
      <div className="hl"><b>{label}</b><small>{note}</small></div>
      <div className="hbars">{xs.map((v, i) => <div key={i} className="hc">{bar(v, seasons[i])}<small>{seasons[i]}</small></div>)}</div>
      <span className="num">{r2(avg)}</span>
    </div>
  );
  return (
    <section className="an-card">
      <header><h4>How much of this is luck <Ev l="noise" tip="Draft value by team slot didn't repeat year to year in 2021–25." /></h4>
        <p>The same draft grade, computed for this league's 2021–25 drafts. Rank correlations, one bar per season; the number is the average.</p></header>
      <div className="dr-hist">
        {row(`Week ${h.through} grade → final grade`, h.hold, h.seasons.map(String), hold, "does an early ranking hold up?")}
        {row("Final grade → regular-season points", h.vs_points, h.seasons.map(String), vp, "does drafting well win?")}
        {row("Grade → same team's grade next year", h.next_year, h.seasons.slice(1).map((s) => `${s - 1}→${String(s).slice(2)}`), nx, "is it a repeatable skill?")}
      </div>
      <p className="an-note">
        Opinion, from those numbers: the draft matters a lot to a season ({r2(vp)}), and an early read like this one holds up roughly {hold != null && hold > 0.6 ? "well" : "half the time"} ({r2(hold)}).
        But a team slot's draft grade barely carries to the next year ({r2(nx)} on average, from {r2(Math.min(...h.next_year.filter((x): x is number => x != null)))} to {r2(Math.max(...h.next_year.filter((x): x is number => x != null)))} across seasons, ten teams each; owners may have changed). So read this page mostly as who got lucky or unlucky with their picks, not who is good at drafting.
        Missed games are split out in the grade table because injuries are mostly luck.
      </p>
    </section>
  );
}

/* 4 ─ the board, colored by value vs slot */
function Board({ d, myId, oppId, name }: Omit<Props, "onSelect"> & { name: (id: number) => string }) {
  const { open } = useFile();
  const cols = d.picks.filter((p) => p.round === 1).sort((a, b) => a.pick - b.pick).map((p) => p.team_id);
  const at = new Map(d.picks.map((p) => [`${p.team_id}-${p.round}`, p]));
  const rounds = Array.from({ length: d.rounds }, (_, i) => i + 1);
  const scale = Math.max(30, ...d.picks.map((p) => Math.abs(p.v_total))) * 0.7;
  const ids = [...d.picks].sort((a, b) => a.pick - b.pick).map((p) => p.id);
  return (
    <section className="an-card wide">
      <header><h4>The board <Ev l="descriptive" tip="What happened, plus the forecast for the rest of the season." /></h4>
        <p>Every pick in draft order. Green beat its slot, red fell short, and deeper means further, so far and rest of season combined. The stripe is the position.
          Struck-through picks have been dropped. Click a player to open his file.</p></header>
      <div className="mx-wrap">
        <table className="dr-board">
          <thead><tr><th />{cols.map((t) => <th key={t} className={`tn ${role(t, myId, oppId)}`} title={name(t)}>{name(t)}</th>)}</tr></thead>
          <tbody>
            {rounds.map((r) => (
              <tr key={r}>
                <th className="rd">{r}</th>
                {cols.map((t) => {
                  const p = at.get(`${t}-${r}`);
                  if (!p) return <td key={t} />;
                  const a = Math.min(1, Math.abs(p.v_total) / scale) * 0.7;   // capped so the name stays readable
                  return (
                    <td key={t} className={`${p.v_total >= 0 ? "pos" : "neg"} ${p.status !== "kept" ? "gone" : ""} ${role(t, myId, oppId)}`}
                      style={{ ["--a" as string]: a * 0.85, ["--pc" as string]: POS_VAR[(p.pos ?? "QB") as keyof typeof POS_VAR] ?? "var(--noise)" }}
                      title={pickTip(p, name(t))} onClick={() => open(p.id, ids)}>
                      <span className="pn">{p.name ? tightName(lastName(p.name), 11) : "?"}</span>
                      <small>{p.pick} · {signed(p.v_total)}</small>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* 5 ─ steals and busts, and the best/worst pick each round */
function Outliers({ d, myId, oppId, name, part }: Omit<Props, "onSelect"> & { name: (id: number) => string; part: "steals" | "rounds" }) {
  const { open } = useFile();
  const s = [...d.picks].sort((a, b) => b.v_total - a.v_total);
  const row = (p: DPick) => (
    <li key={p.id} className={role(p.team_id, myId, oppId)} title={pickTip(p, name(p.team_id))} onClick={() => open(p.id)}>
      <span className="nm"><i className="sw" style={{ background: POS_VAR[(p.pos ?? "QB") as keyof typeof POS_VAR] }} />{p.name}</span>
      <span className="dim">{name(p.team_id)} · R{p.round}</span>
      <span className={`num ${p.v_total >= 0 ? "above" : "below"}`}>{signed(p.v_total)}</span>
    </li>
  );
  const rounds = Array.from({ length: d.rounds }, (_, i) => i + 1);
  return (
    part === "steals" ? (
      <section className="an-card wide">
        <header><h4>Steals and busts <Ev l="descriptive" tip="So far plus rest of season, vs the slot." /></h4>
          <p>Biggest gaps between what a pick has given (plus the forecast for the rest of the season) and what that slot usually returns.</p></header>
        <div className="sf-draft">
          <div><h5>Steals</h5><ul>{s.slice(0, 8).map(row)}</ul></div>
          <div><h5>Busts</h5><ul>{s.slice(-8).reverse().map(row)}</ul></div>
        </div>
      </section>
    ) : (
      <section className="an-card">
        <header><h4>Best and worst pick each round</h4><p>Within each round, the pick furthest above and below its slot.</p></header>
        <table className="dr-rounds">
          <thead><tr><th>Rd</th><th className="l">Best</th><th className="l">Worst</th></tr></thead>
          <tbody>
            {rounds.map((r) => {
              const g = d.picks.filter((p) => p.round === r).sort((a, b) => b.v_total - a.v_total);
              if (!g.length) return null;
              const c = (p: DPick) => (
                <td className={`l ${role(p.team_id, myId, oppId)}`} title={pickTip(p, name(p.team_id))} onClick={() => open(p.id)}>
                  <b>{p.name ? tightName(p.name, 16) : "?"}</b> <span className="dim">{name(p.team_id)}</span> <span className={`num ${p.v_total >= 0 ? "above" : "below"}`}>{signed(p.v_total)}</span>
                </td>
              );
              return <tr key={r}><td className="num dim">{r}</td>{c(g[0])}{c(g[g.length - 1])}</tr>;
            })}
          </tbody>
        </table>
      </section>
    )
  );
}

/* 6 ─ reach vs market: picks against PFF's preseason ADP */
function Market({ d, myId, oppId, name }: Omit<Props, "onSelect"> & { name: (id: number) => string }) {
  const { open } = useFile();
  const pts = d.picks.filter((p) => p.vs_adp != null);
  const W = 420, H = 326, L = 46, R = 16, T = 22, B = 46;
  const xm = 40, ym = Math.max(...pts.map((p) => Math.abs(p.v_total)), 20) * 1.1;
  const cx = (v: number) => Math.max(-xm, Math.min(xm, v));
  const X = (v: number) => L + ((W - L - R) * (cx(v) + xm)) / (2 * xm), Y = (v: number) => T + ((H - T - B) * (ym - v)) / (2 * ym);
  const off = d.league_vs_adp ?? 0;
  const [hover, setHover] = useState<number | null>(null);
  const hp = pts.find((p) => p.id === hover);
  return (
    <section className="an-card">
      <header><h4>Reaches vs fallers <Ev l="descriptive" tip="PFF's preseason ADP is the only draft-day market Desk saved; ESPN's ADP wasn't kept." /></h4>
        <p>Left = taken ahead of PFF's preseason ADP (a reach); right = fell past it. Up = beat the slot. Dashed line = where this league drafted on average ({f1(Math.abs(off))} picks ahead of ADP).
          Kickers and defenses left out. Correlation {r2(d.reach_corr)}: {d.reach_corr != null && d.reach_corr > 0.15 ? "players who fell did better than reaches (opinion: the market knew something the reachers didn't)" : "reaching vs waiting made little difference"}.</p></header>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="sc dr-mk" role="img" aria-label="Scatter of each pick against PFF ADP and value vs slot" onMouseLeave={() => setHover(null)}>
        <line x1={L} x2={W - R} y1={Y(0)} y2={Y(0)} className="ax" />
        <line x1={X(0)} x2={X(0)} y1={T} y2={H - B} className="ax" />
        <line x1={X(off)} x2={X(off)} y1={T} y2={H - B} className="ax dash" />
        {[-Math.round(ym * 0.6), Math.round(ym * 0.6)].map((v) => <text key={`y${v}`} x={L - 8} y={Y(v) + 4} textAnchor="end" className="tk">{v > 0 ? `+${v}` : v}</text>)}
        {[-40, -20, 0, 20, 40].map((v) => <text key={v} x={X(v)} y={H - B + 16} textAnchor="middle" className="tk">{v === -40 ? "≤−40" : v === 40 ? "40+" : v > 0 ? `+${v}` : v}</text>)}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="lab">picks after ADP (left = reach)</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="lab" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>value vs slot</text>
        {pts.map((p) => {
          const r = role(p.team_id, myId, oppId);
          return (
            <circle key={p.id} cx={X(p.vs_adp!)} cy={Y(p.v_total)} r={r ? 5 : 3.5} className={`pk ${r} ${hover === p.id ? "hv" : ""}`}
              style={{ fill: POS_VAR[(p.pos ?? "QB") as keyof typeof POS_VAR] }} onMouseEnter={() => setHover(p.id)} onClick={() => open(p.id)}>
              <title>{pickTip(p, name(p.team_id))}</title>
            </circle>
          );
        })}
        {hp && <text x={Math.min(X(hp.vs_adp!) + 8, W - 150)} y={Math.max(Y(hp.v_total) - 8, T + 10)} className="hvl">{hp.name} · {name(hp.team_id)}</text>}
      </svg>
      <div className="legend">{POS.slice(0, 4).map((p) => <span key={p}><i className="sw" style={{ background: POS_VAR[p as keyof typeof POS_VAR] }} />{POS_LABEL[p]}</span>)}<span>Bigger dots: you and this week's opponent</span></div>
    </section>
  );
}

/* 7 ─ draft DNA: position by round, and what each position returned */
function Strategy({ d, myId, oppId, onSelect, name }: Props & { name: (id: number) => string }) {
  const order = [...d.teams].sort((a, b) => b.v_total - a.v_total);
  const byTeam = useMemo(() => {
    const m = new Map<number, DPick[]>();
    for (const p of [...d.picks].sort((a, b) => a.pick - b.pick)) m.set(p.team_id, [...(m.get(p.team_id) ?? []), p]);
    return m;
  }, [d]);
  const scale = Math.max(20, ...d.teams.flatMap((t) => Object.values(t.by_pos).map(Math.abs))) * 0.8;
  return (
    <section className="an-card wide">
      <header><h4>Draft shape <Ev l="descriptive" tip="What happened." /></h4>
        <p>Left: each team's picks in order, colored by position, so you can see who went RB-early or waited on QB. Right: value vs slot by position, so far plus rest of season.</p></header>
      <div className="mx-wrap">
        <table className="dr-dna">
          <thead><tr><th />{Array.from({ length: d.rounds }, (_, i) => <th key={i}>{i + 1}</th>)}<th className="gap" />{POS.map((p) => <th key={p}><i className="sw" style={{ background: POS_VAR[p as keyof typeof POS_VAR] }} />{POS_LABEL[p]}</th>)}</tr></thead>
          <tbody>
            {order.map((t) => (
              <tr key={t.id} className={role(t.id, myId, oppId)} onClick={() => onSelect(t.id)}>
                <th className="tn">{name(t.id)}</th>
                {(byTeam.get(t.id) ?? []).map((p) => (
                  <td key={p.id} className={`dna ${p.pos === "WR" ? "wr" : ""}`} style={{ background: POS_VAR[(p.pos ?? "QB") as keyof typeof POS_VAR] }} title={pickTip(p, name(t.id))}>{POS_LABEL[p.pos ?? ""] ?? "?"}</td>
                ))}
                <td className="gap" />
                {POS.map((p) => {
                  const v = t.by_pos[p];
                  if (v == null) return <td key={p} className="tot">–</td>;
                  return <td key={p} className={`val ${v >= 0 ? "pos" : "neg"}`} style={{ ["--a" as string]: Math.min(1, Math.abs(v) / scale) * 0.85 }} title={`${t.name}, ${POS_LABEL[p]}: ${signed(v)} vs slot`}>{Math.round(v)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* 8 ─ dropped picks that are scoring for someone else */
function Away({ d, myId, oppId, name }: Omit<Props, "onSelect"> & { name: (id: number) => string }) {
  const { open } = useFile();
  const away = d.picks.filter((p) => p.status !== "kept" && p.pos !== "K" && p.pos !== "DST" && (p.started_elsewhere > 0 || (p.ros ?? 0) > 0)).sort((a, b) => b.started_elsewhere - a.started_elsewhere || (b.v_ros ?? 0) - (a.v_ros ?? 0)).slice(0, 10);
  return (
    <section className="an-card wide">
      <header><h4>The ones that got away <Ev l="descriptive" tip="What happened." /></h4>
        <p>Draft picks no longer with the team that took them, ordered by points they've since scored in other teams' starting lineups; the second number is their forecast value over replacement from here. Only players who've scored or are forecast to matter; kickers and defenses (streamed) left out.</p></header>
      {away.length ? (
        <div className="sf-draft one"><ul>
          {away.map((p) => (
            <li key={p.id} className={role(p.team_id, myId, oppId)} title={pickTip(p, name(p.team_id))} onClick={() => open(p.id)}>
              <span className="nm"><i className="sw" style={{ background: POS_VAR[(p.pos ?? "QB") as keyof typeof POS_VAR] }} />{p.name}</span>
              <span className="dim">{name(p.team_id)} R{p.round} → {p.owner_now != null ? name(p.owner_now) : "free agent"}</span>
              <span className="num">{f1(p.started_elsewhere)} <span className="dim">· {f1(p.ros)}</span></span>
            </li>
          ))}
        </ul></div>
      ) : <p className="an-note">No dropped or traded pick has done anything elsewhere yet.</p>}
    </section>
  );
}

export function DraftRecap(p: Props) {
  const all = p.d.teams.map((t: DTeam) => t.name);
  const name = (id: number) => teamShort(p.d.teams.find((t) => t.id === id)?.name ?? String(id), all);
  const best = [...p.d.teams].sort((a, b) => b.v_total - a.v_total);
  const me = p.d.teams.find((t) => t.id === p.myId);
  return (
    <div className="analysis sf dr">
      <h3 className="an-h">Draft recap <small>through week {p.d.through_week} · points over replacement vs what each pick slot usually returns in this league</small></h3>
      <p className="dr-take">
        Best draft so far: <b>{best[0].name}</b> ({signed(best[0].v_total)}). Worst: <b>{best[best.length - 1].name}</b> ({signed(best[best.length - 1].v_total)}).
        {me && <> Yours ranks <b className="you">#{me.v_total_rank}</b> ({signed(me.v_total)}: {signed(me.v_now)} so far, {signed(me.v_ros)} ahead).</>}
      </p>
      <Grades {...p} name={name} />
      <div className="an-row"><Quadrant {...p} name={name} /><History d={p.d} /></div>
      <Board {...p} name={name} />
      <Outliers {...p} name={name} part="steals" />
      <div className="an-row"><Market {...p} name={name} /><Outliers {...p} name={name} part="rounds" /></div>
      <Away {...p} name={name} />
      <Strategy {...p} name={name} />
    </div>
  );
}
