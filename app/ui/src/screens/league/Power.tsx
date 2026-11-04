import { useState } from "react";
import type { PowerView } from "../../types";
import { f1, pct, signed, teamShort } from "../../format";
import "./power.css";

interface Props { pw: PowerView; myId: number; oppId: number | null; onSelect: (id: number) => void }

const role = (id: number, myId: number, oppId: number | null) => (id === myId ? "me" : id === oppId ? "opp" : "");

/** P(A's rest-of-season points per game beat B's) from the Power gap, shown capped (see notes.pairwise). */
export function pAhead(pw: PowerView, gap: number) {
  const p = 1 / (1 + Math.exp(-pw.logit_b * gap));
  return p >= pw.p_cap ? `${Math.round(pw.p_cap * 100)}%+` : p <= 1 - pw.p_cap ? `<${Math.round((1 - pw.p_cap) * 100)}%` : `${Math.round(p * 100)}%`;
}

/* 1 ─ one scale: every team's Power Score, tier bands behind */
function Strip({ pw, myId, oppId, onSelect }: Props) {
  const W = 760, H = 124, pad = 26;
  const vals = pw.teams.map((t) => t.power);
  const lo = Math.floor(Math.min(...vals) - 2), hi = Math.ceil(Math.max(...vals) + 2);
  const x = (v: number) => pad + ((v - lo) / (hi - lo)) * (W - 2 * pad);
  const names = pw.teams.map((t) => t.name);
  const tiers = [...new Set(pw.teams.map((t) => t.tier))].map((k) => {
    const g = pw.teams.filter((t) => t.tier === k);
    return { k, lo: Math.min(...g.map((t) => t.power)), hi: Math.max(...g.map((t) => t.power)) };
  });
  // alternate labels above/below the axis so neighbours don't collide
  const sorted = [...pw.teams].sort((a, b) => a.power - b.power);
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / 5) * 5; v <= hi; v += 5) ticks.push(v);
  return (
    <svg className="pw-strip" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Power Scores: ${pw.teams.map((t) => `${t.name} ${f1(t.power)}`).join(", ")}`}>
      {tiers.map((t) => (
        <g key={t.k}>
          <rect x={x(t.lo) - 9} y={30} width={x(t.hi) - x(t.lo) + 18} height={46} rx={8} className={`band t${t.k % 2}`} />
          <text x={(x(t.lo) + x(t.hi)) / 2} y={92} textAnchor="middle" className="tierlab">Tier {t.k}</text>
        </g>
      ))}
      <line x1={pad} x2={W - pad} y1={53} y2={53} className="axis" />
      {ticks.map((v) => <g key={v}><line x1={x(v)} x2={x(v)} y1={50} y2={56} className="axis" /><text x={x(v)} y={H - 4} textAnchor="middle" className="tk">{v}</text></g>)}
      {sorted.map((t, i) => {
        const up = i % 2 === 0, r = role(t.id, myId, oppId);
        return (
          <g key={t.id} className={`pt ${r}`} onClick={() => onSelect(t.id)}>
            <title>{`${t.name}: ${f1(t.power)} pts/wk, tier ${t.tier}`}</title>
            <line x1={x(t.power)} x2={x(t.power)} y1={up ? 40 : 53} y2={up ? 53 : 66} className="stem" />
            <circle cx={x(t.power)} cy={53} r={r === "me" ? 6 : 4.5} className="dot" />
            <text x={x(t.power)} y={up ? 37 : 76} textAnchor="middle" className="nm">{teamShort(t.name, names)}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* 2 ─ Power over time, you in cobalt, everyone else muted */
function Trend({ pw, myId, oppId, onSelect }: Props) {
  const [hov, setHov] = useState<number | null>(null);
  const W = 520, H = 230, L = 34, R = 92, T = 10, B = 24;
  const weeks = pw.teams[0].history.map((h) => h.week);
  const all = pw.teams.flatMap((t) => t.history.map((h) => h.power));
  const lo = Math.floor(Math.min(...all) - 2), hi = Math.ceil(Math.max(...all) + 2);
  const x = (w: number) => L + ((w - weeks[0]) / Math.max(1, weeks[weeks.length - 1] - weeks[0])) * (W - L - R);
  const y = (v: number) => T + ((hi - v) / (hi - lo)) * (H - T - B);
  const names = pw.teams.map((t) => t.name);
  // end labels: nudge apart so they stay legible
  const ends = [...pw.teams].sort((a, b) => b.power - a.power).map((t) => ({ t, y: y(t.history[t.history.length - 1].power) }));
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 11);
  const order = [...pw.teams].sort((a, b) => (a.id === myId ? 1 : 0) - (b.id === myId ? 1 : 0) || (a.id === hov ? 1 : 0) - (b.id === hov ? 1 : 0));
  return (
    <svg className="pw-trend" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Power Score by week for every team">
      {[lo, (lo + hi) / 2, hi].map((v) => <g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid" /><text x={L - 5} y={y(v) + 3} textAnchor="end" className="tk">{Math.round(v)}</text></g>)}
      {weeks.map((w) => <text key={w} x={x(w)} y={H - 6} textAnchor="middle" className="tk">{w}</text>)}
      {order.map((t) => {
        const r = role(t.id, myId, oppId), on = hov === t.id;
        const d = t.history.map((h, i) => `${i ? "L" : "M"}${x(h.week).toFixed(1)},${y(h.power).toFixed(1)}`).join("");
        return (
          <g key={t.id} className={`ln ${r} ${on ? "on" : ""}`} onMouseEnter={() => setHov(t.id)} onMouseLeave={() => setHov(null)} onClick={() => onSelect(t.id)}>
            <path d={d} className="hit" />
            <path d={d} className="path" />
            {t.history.map((h) => <circle key={h.week} cx={x(h.week)} cy={y(h.power)} r={r === "me" || on ? 2.6 : 0} className="dot"><title>{`${t.name}, week ${h.week}: ${f1(h.power)}`}</title></circle>)}
          </g>
        );
      })}
      {ends.map(({ t, y: ly }) => (
        <text key={t.id} x={W - R + 6} y={ly + 3} className={`end ${role(t.id, myId, oppId)} ${hov === t.id ? "on" : ""}`}>{teamShort(t.name, names)} {f1(t.power)}</text>
      ))}
    </svg>
  );
}

/* a diverging mini-bar: base (grey) + delta (green/red), drawn on one shared scale */
function Parts({ base, delta, max, unit }: { base: number; delta: number; max: number; unit: string }) {
  const W = 84, s = W / max;
  const b = Math.max(0, base) * s, dd = delta * s;
  return (
    <svg className="parts" width={W + 34} height={14} viewBox={`0 0 ${W + 34} 14`} aria-label={`${f1(base)} ${unit} expected, ${signed(delta)} luck`}>
      <rect x={0} y={3} width={b} height={8} className="base" />
      <rect x={dd >= 0 ? b : b + dd} y={3} width={Math.abs(dd)} height={8} className={dd >= 0 ? "up" : "down"} />
      <line x1={b} x2={b} y1={0} y2={14} className="mark" />
    </svg>
  );
}

export function Power(p: Props) {
  const { pw, myId, oppId, onSelect } = p;
  const me = pw.teams.find((t) => t.id === myId)!;
  const above = pw.teams.filter((t) => t.rank === me.rank - 1)[0];
  const below = pw.teams.filter((t) => t.rank === me.rank + 1)[0];
  const maxW = Math.max(...pw.teams.map((t) => Math.max(t.ledger.wins, t.ledger.allplay_wins)), 1);
  const maxPf = Math.max(...pw.teams.map((t) => Math.max(t.pf_g, t.ledger.proj_g)), 1);
  const tierOf = (k: number) => pw.teams.filter((t) => t.tier === k);
  const myTier = tierOf(me.tier).filter((t) => t.id !== myId);
  const n = pw.notes;

  const take = (() => {
    const lead = above ? <>You're <b>#{me.rank}</b>, {f1(above.power - me.power)} pts/wk behind {above.name} (they finish ahead {pAhead(pw, above.power - me.power)} of the time)</>
      : <>You're <b>#1</b>{below ? <>, {f1(me.power - below.power)} pts/wk ({f1((me.power - below.power) / pw.good_trade)} good trades) ahead of {below.name}</> : null}</>;
    return <>{lead}. {myTier.length ? <>Same tier: {myTier.map((t) => t.name).join(", ")}. </> : <>Nobody else in your tier. </>}
      Your points are {signed(me.ledger.over_g)} a week {me.ledger.over_g >= 0 ? "above" : "below"} your starters' projections; that part hasn't carried over in past seasons.</>;
  })();

  return (
    <section className="power">
      <header className="pw-hd">
        <div>
          <h3>Power <span className="num">{f1(me.power)}</span><small>pts/wk · tier {me.tier}</small></h3>
          <p className="take">{take}</p>
        </div>
        <p className="pw-def" title={n.power}>
          Power Score = expected points per week from here on: {Math.round(pw.weight * 100)}% roster strength (best lineup on ESPN's projections, 3-week average) + {Math.round((1 - pw.weight) * 100)}% points per game so far. {n.power.replace(/^Expected points per week from here on\. /, "")}
        </p>
      </header>

      <div className="pw-card wide">
        <h4>One scale <small>Power Score, pts/wk</small></h4>
        <Strip {...p} />
        <p className="pw-note">{n.tiers}</p>
      </div>

      <div className="pw-tbl">
        <table>
          <thead><tr>
            <th className="r">#</th><th className="l">Team</th><th className="r" title={n.power}>Power</th>
            <th className="r" title="Rank change since last week, and whether the roster or the scoring moved it">Δ</th>
            <th className="r" title="Best-lineup projection, 3-week average">Roster</th><th className="r">Pts/g</th>
            <th className="r" title={n.pairwise}>You ahead</th>
            <th className="r" title={n.trades}>Gap · trades</th>
            <th className="r">Playoffs</th>
            <th className="r" title={n.title}>Title · per +1/wk</th>
          </tr></thead>
          <tbody>
            {[...new Set(pw.teams.map((t) => t.tier))].map((k) => [
              <tr key={`t${k}`} className="tier"><td colSpan={10}>Tier {k}<span> · within {pw.tier_gap} pts/wk of {tierOf(k)[0].name}</span></td></tr>,
              ...tierOf(k).map((t) => {
                const gap = me.power - t.power;
                return (
                  <tr key={t.id} className={role(t.id, myId, oppId)} onClick={() => onSelect(t.id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onSelect(t.id)}>
                    <td className="r num rk">{t.rank}</td>
                    <td className="l nm">{t.name}{t.id === myId && <em>you</em>}{t.id === oppId && <em className="o">this week</em>}</td>
                    <td className="r num big">{f1(t.power)}</td>
                    <td className="r num">{t.rank_change ? <span className={t.rank_change > 0 ? "above" : "below"}>{t.rank_change > 0 ? "▲" : "▼"}{Math.abs(t.rank_change)}<small> {t.driver}</small></span> : <span className="dim">–</span>}</td>
                    <td className="r num dim">{f1(t.strength)}</td>
                    <td className="r num dim">{f1(t.pf_g)}</td>
                    <td className="r num">{t.id === myId ? <span className="dim">–</span> : pAhead(pw, gap)}</td>
                    <td className="r num">{t.id === myId ? <span className="dim">–</span> : <>{f1(Math.abs(gap) / pw.good_trade)} <small>{gap >= 0 ? "you lead" : "they lead"}</small></>}</td>
                    <td className="r num">{pct(t.odds.playoffs)}</td>
                    <td className="r num">{pct(t.odds.title)} <small>{t.odds.title_per_pt != null ? `+${(t.odds.title_per_pt * 100).toFixed(1)}` : ""}</small></td>
                  </tr>
                );
              }),
            ])}
          </tbody>
        </table>
      </div>

      <div className="pw-row">
        <div className="pw-card">
          <h4>Power by week</h4>
          <Trend {...p} />
          <p className="pw-note">Each week's score uses only what was known then. Hover a line to pick it out. {n.trajectory}</p>
        </div>
        <div className="pw-card">
          <h4>Luck ledger</h4>
          <div className="ledger-wrap"><table className="ledger">
            <thead><tr><th className="l">Team</th><th className="l" title={n.luck}>Wins = expected + luck</th><th className="l" title={n.luck}>Pts/g = projected + over</th>
              <th className="r" title="How much opponents beat their own average when facing this team (pts/wk)">Opp. vs them</th>
              <th className="r" title={n.lineup_cost}>Lineup vs ESPN-best</th></tr></thead>
            <tbody>
              {pw.teams.map((t) => (
                <tr key={t.id} className={role(t.id, myId, oppId)} onClick={() => onSelect(t.id)}>
                  <td className="l nm">{teamShort(t.name, pw.teams.map((x) => x.name))}</td>
                  <td className="l"><Parts base={t.ledger.allplay_wins} delta={t.ledger.sched_luck} max={maxW} unit="wins" />
                    <span className="num">{t.ledger.wins} = {f1(t.ledger.allplay_wins)} <b className={t.ledger.sched_luck >= 0 ? "above" : "below"}>{signed(t.ledger.sched_luck)}</b></span></td>
                  <td className="l"><Parts base={t.ledger.proj_g} delta={t.ledger.over_g} max={maxPf} unit="pts/g" />
                    <span className="num">{f1(t.pf_g)} = {f1(t.ledger.proj_g)} <b className={t.ledger.over_g >= 0 ? "above" : "below"}>{signed(t.ledger.over_g)}</b></span></td>
                  <td className="r num dim">{signed(t.ledger.opp_over)}</td>
                  <td className={`r num ${t.ledger.lineup_cost_g != null && t.ledger.lineup_cost_g < 0 ? "below" : ""}`}>{signed(t.ledger.lineup_cost_g)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
          <p className="pw-note">{n.luck} {n.lineup_cost}</p>
        </div>
      </div>

      <div className="pw-card wide">
        <h4>What's behind each roster</h4>
        <div className="ctx">
          {pw.teams.map((t) => (
            <button key={t.id} type="button" className={`cx ${role(t.id, myId, oppId)}`} onClick={() => onSelect(t.id)}>
              <b>{t.name}</b>
              <span className="fl">Top starter <em>{t.fragility.player ?? "–"}</em>: losing him costs {Math.round(t.fragility.share * 100)}% of the lineup</span>
              <span className="fl">Roster {t.trajectory == null ? "–" : <em className={t.trajectory >= 0 ? "above" : "below"}>{signed(t.trajectory)} pts/wk</em>} over 3 weeks
                {t.moves.length ? <>: {t.moves.map((m, i) => <span key={i}>{i ? ", " : " "}{m.dir === "in" ? "+" : "−"}{m.name} ({f1(m.v)})</span>)}</> : null}</span>
            </button>
          ))}
        </div>
        <p className="pw-note">{n.fragility} {n.trajectory}</p>
      </div>
    </section>
  );
}

export function powerMarkdown(pw: PowerView, myId: number) {
  const me = pw.teams.find((t) => t.id === myId)!;
  const L = [`## Power after week ${pw.through_week} (from Desk)`, "",
    `Power Score = expected pts/wk from here on (${Math.round(pw.weight * 100)}% roster strength, ${Math.round((1 - pw.weight) * 100)}% points per game). Tiers: within ${pw.tier_gap} pts/wk of the tier's top team.`, "",
    "| # | Tier | Team | Power | Roster | Pts/g | You finish ahead | Wins = exp + luck | Pts/g over proj | Lineup vs ESPN-best | Playoffs | Title |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|"];
  for (const t of pw.teams)
    L.push(`| ${t.rank} | ${t.tier} | ${t.name}${t.id === myId ? " (you)" : ""} | ${f1(t.power)} | ${f1(t.strength)} | ${f1(t.pf_g)} | ${t.id === myId ? "–" : pAhead(pw, me.power - t.power)} | ${t.ledger.wins} = ${f1(t.ledger.allplay_wins)} ${signed(t.ledger.sched_luck)} | ${signed(t.ledger.over_g)} | ${signed(t.ledger.lineup_cost_g)} | ${pct(t.odds.playoffs)} | ${pct(t.odds.title)} |`);
  L.push("", ...Object.values(pw.notes).map((x) => `- ${x}`));
  return L.join("\n");
}
