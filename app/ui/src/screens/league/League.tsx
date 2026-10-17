import { useMemo, useState } from "react";
import type { ALeague, LeagueLive, LiveOdds, OutItem, Power, PowerTeam, WhatIfResult } from "../../types";
import { SoFar } from "./SoFar";
import { f1, pct, signed } from "../../format";
import { Analysis } from "./Analysis";
import "./league.css";

type SortKey = "proj" | "pf" | "top2" | "playoffs" | "luck" | "bench" | "vsespn";
type OddsKey = "seed1" | "top2" | "playoffs" | "title";

interface Props {
  power: Power;
  myId: number;
  oppId: number | null;
  live: LeagueLive | null;
  liveErr: string | null;
  engineUp: boolean;
  whatif: WhatIfResult | null;
  scenario: OutItem[];
  busy: boolean;
  selected: number | null;
  onSelect: (id: number) => void;
  onTrade?: (id: number) => void;
  onRemove: (id: number) => void;
  onClear: () => void;
  /** season-so-far tables (GET /api/analytics/league) */
  alg?: ALeague | null;
}

function Weekly({ t, max }: { t: PowerTeam; max: number }) {
  const W = 96, H = 28, n = Math.max(t.weekly.length, 1), bw = W / n;
  const y = (v: number) => H - (H * v) / max;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="wk" role="img"
      aria-label={`weekly points ${t.weekly.map((v, i) => `${f1(v)} vs ${f1(t.weekly_proj[i])} projected`).join(", ")}`}>
      {t.weekly.map((v, i) => {
        const p = t.weekly_proj[i] ?? null, x = i * bw + bw * 0.25, w = bw * 0.5;
        return (
          <g key={i}>
            <title>{`Week ${i + 1}: ${f1(v)} scored, ${f1(p)} projected`}</title>
            <rect x={x} y={y(v)} width={w} height={H - y(v)} rx={1.5} className="bar" />
            {p != null && <line x1={x - 2} x2={x + w + 2} y1={y(p)} y2={y(p)} className={v >= p ? "tick above" : "tick below"} />}
          </g>
        );
      })}
    </svg>
  );
}

export function League(p: Props) {
  const [sort, setSort] = useState<SortKey>("proj");
  const teams = p.power.teams;
  const maxWeek = Math.max(...teams.flatMap((t) => [...t.weekly, ...t.weekly_proj.map((x) => x ?? 0)]), 1) * 1.04;
  const maxLuck = Math.max(...teams.map((t) => Math.abs(t.luck ?? 0)), 1);

  // which odds to show: scenario > live sim > Tuesday's ledger
  const odds = (t: PowerTeam): (LiveOdds | null) =>
    p.whatif?.scenario[t.id] ?? p.live?.odds[t.id] ?? (t.odds ? { ...(t.odds as LiveOdds), reg_pts: t.proj_final_pts ?? 0 } : null);
  const delta = (t: PowerTeam) => p.whatif?.delta[t.id] ?? null;
  const noise = (p.whatif?.noise_pp ?? p.live?.noise_pp ?? 1.1) / 100;

  const rows = useMemo(() => {
    const key: Record<SortKey, (t: PowerTeam) => number> = {
      proj: (t) => odds(t)?.reg_pts ?? t.proj_final_pts ?? t.pf, pf: (t) => t.pf, luck: (t) => t.luck ?? 0,
      bench: (t) => p.alg?.teams.find((x) => x.id === t.id)?.left ?? 0,
      vsespn: (t) => p.alg?.teams.find((x) => x.id === t.id)?.vs_espn ?? 0,
      top2: (t) => odds(t)?.top2 ?? 0, playoffs: (t) => odds(t)?.playoffs ?? 0,
    };
    return [...teams].sort((a, b) => key[sort](b) - key[sort](a));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teams, sort, p.live, p.whatif, p.alg]);

  const me = teams.find((t) => t.id === p.myId)!;
  const top2Race = [...teams].sort((a, b) => (odds(b)?.top2 ?? 0) - (odds(a)?.top2 ?? 0)).slice(0, 3);
  const lead = me.pf - Math.max(...teams.filter((t) => t.id !== me.id).map((t) => t.pf));
  const src = p.whatif ? `What-if · ${p.whatif.sims.toLocaleString()} simulated seasons` : p.live ? `Live · ${p.live.sims.toLocaleString()} simulated seasons, current rosters`
    : `From Tuesday's run (week ${p.power.odds_as_of_week})${p.engineUp ? " · live odds loading…" : ""}`;

  const oddsCell = (t: PowerTeam, k: OddsKey, muted = false, meter = true) => {
    const o = odds(t), d = delta(t);
    const v = o ? o[k] : null, dv = d ? d[k] : null;
    return (
      <td className={`r odds ${muted ? "muted" : ""}`}>
        <span className="ov num">{pct(v)}</span>
        {!muted && meter && <span className="meter" aria-hidden="true"><i style={{ width: `${Math.round((v ?? 0) * 100)}%` }} /></span>}
        {dv != null && (Math.abs(dv) < 2 * noise
          ? <span className="dl flat" title="Inside simulation noise">≈</span>
          : <span className={`dl ${dv > 0 ? "up" : "down"}`}>{dv > 0 ? "▲" : "▼"}{Math.round(Math.abs(dv) * 100)}</span>)}
      </td>
    );
  };
  const th = (k: SortKey | null, label: string, cls = "r", title?: string) => (
    <th className={`${cls} ${k ? "sortable" : ""} ${sort === k ? "on" : ""}`} title={title} onClick={k ? () => setSort(k) : undefined}
      aria-sort={sort === k ? "descending" : undefined}>{label}</th>
  );

  return (
    <div className="league">
      <header className="lg-hd">
        <div>
          <h2>League</h2>
          <p className="take">
            You lead the league by <b className="num">{f1(lead)}</b> points. Seeding is by total points.
            Top-2 race: {top2Race.map((t, i) => <span key={t.id}>{i ? ", " : " "}<b className={t.id === p.myId ? "you" : ""}>{t.id === p.myId ? "you" : t.name}</b> {pct(odds(t)?.top2)}</span>)}.
          </p>
        </div>
        <span className={`src ${p.whatif ? "wi" : ""}`}>{src}</span>
      </header>

      {(p.scenario.length > 0 || p.busy) && (
        <div className="scenario" role="status">
          <span className="lab">What if</span>
          {p.scenario.map((o) => (
            <span key={o.id} className="chip">{o.name} out {o.weeks ? `${o.weeks} wk` : "for the season"}
              <button type="button" aria-label={`Remove ${o.name}`} onClick={() => p.onRemove(o.id)}>×</button></span>
          ))}
          {p.busy ? <span className="busy">Simulating…</span> : <button type="button" className="clear" onClick={p.onClear}>Clear</button>}
          <span className="note">Arrows are changes in percentage points; ≈ means inside simulation noise (about ±{Math.round(2 * noise * 100)}).</span>
        </div>
      )}

      <nav className="lg-jump" aria-label="League sections">
        {([["lg-table", "Table"], ...(p.live?.analysis ? [["lg-analysis", "Analysis"]] : []), ...(p.alg ? [["lg-sofar", "Season so far"]] : [])] as [string, string][]).map(([id, l]) =>
          <button key={id} type="button" onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" })}>{l}</button>)}
      </nav>

      <div className="tbl" id="lg-table">
        <table>
          <thead><tr>
            {th(null, "#")}
            {th(null, "Team", "l")}
            {th(null, "W–L")}
            {th("pf", "Points")}
            {th(null, `Weeks 1–${p.power.through_week}`, "l", "Bars: points scored. Ticks: ESPN's projection for the lineup started (green beat it, red missed).")}
            {th("luck", "vs projection", "l", "Points scored minus projection for the lineups used. Early luck hasn't carried over in past seasons.")}
            {p.alg && th("bench", "Bench left", "r", "Points the best lineup from the same roster would have added, season so far. Didn't repeat in 2021–25 (r .11): what happened, not a habit.")}
            {p.alg && th("vsespn", "vs ESPN-best lineup", "r", "Points scored minus what the lineup with the highest ESPN projections (same roster) actually scored, season so far. In 2021–25 every manager in this league trailed it, by 1.2 points a week on average.")}
            {th("proj", "Proj. final")}
            {th(null, "#1 seed")}
            {th("top2", "Top-2")}
            {th("playoffs", "Playoffs")}
          </tr></thead>
          <tbody>
            {rows.map((t, i) => {
              const L = t.luck ?? 0, w = (30 * Math.abs(L)) / maxLuck;
              const o = odds(t), d = delta(t);
              return (
                <tr key={t.id} className={`${t.id === p.myId ? "me" : ""} ${t.id === p.oppId ? "opp" : ""} ${p.selected === t.id ? "sel" : ""}`}
                  onClick={() => p.onSelect(t.id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && p.onSelect(t.id)} aria-selected={p.selected === t.id}>
                  <td className="r rk num">{i + 1}</td>
                  <td className="l nm">{t.name}{t.id === p.myId && <em>you</em>}{t.id === p.oppId && <em className="o">this week</em>}</td>
                  <td className="r num">{t.record.w}–{t.record.l}</td>
                  <td className="r num">{f1(t.pf)} <small>#{t.pf_rank}</small></td>
                  <td className="l"><Weekly t={t} max={maxWeek} /></td>
                  <td className="l"><span className="luck"><span className="lb" aria-hidden="true"><i className={L >= 0 ? "p" : "n"} style={{ width: `${w}px` }} /></span><span className={`num ${L >= 0 ? "above" : "below"}`}>{signed(L)}</span></span></td>
                  {p.alg && <td className="r num dim">{f1(p.alg.teams.find((x) => x.id === t.id)?.left ?? null)}</td>}
                  {p.alg && <td className="r num dim">{signed(p.alg.teams.find((x) => x.id === t.id)?.vs_espn ?? null)}</td>}
                  <td className="r num">{f1(o?.reg_pts ?? null)}{d && Math.abs(d.reg_pts) >= 1 && <small className={d.reg_pts > 0 ? "above" : "below"}> {signed(d.reg_pts)}</small>}</td>
                  {oddsCell(t, "seed1", false, false)}
                  {oddsCell(t, "top2")}
                  {oddsCell(t, "playoffs", false, false)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fig">
        Click a team to see its roster and test injuries. Proj. final = expected regular-season points (seeding is by points).
        {p.liveErr && ` Live odds failed: ${p.liveErr}.`} Seed and playoff odds start meaning something around week 6. Title odds are in the side panel only: three single-week playoff games make them close to a lottery.
      </p>
      {p.live?.analysis && <div id="lg-analysis" className="lg-sec"><Analysis a={p.live.analysis} teams={teams} order={rows.map((t) => t.id)} myId={p.myId} oppId={p.oppId} onSelect={p.onSelect} onTrade={p.onTrade} /></div>}
      {!p.live?.analysis && p.engineUp && <p className="fig">Analysis appears once the live simulation finishes.</p>}
      {p.alg && <div id="lg-sofar" className="lg-sec"><SoFar lg={p.alg} myId={p.myId} oppId={p.oppId} onSelect={p.onSelect} /></div>}
    </div>
  );
}

export function leagueMarkdown(power: Power, live: LeagueLive | null, whatif: WhatIfResult | null, scenario: OutItem[], alg?: ALeague | null) {
  const odds = (t: PowerTeam) => whatif?.scenario[t.id] ?? live?.odds[t.id] ?? (t.odds as LiveOdds | null);
  const L = [`## League after week ${power.through_week} (from Desk)`, "",
    scenario.length ? `What-if: ${scenario.map((o) => `${o.name} out ${o.weeks ? o.weeks + " wk" : "for the season"}`).join("; ")}` : "Current rosters.",
    "", "| Team | W-L | Points | vs projection | Proj. final | #1 seed | Top-2 | Playoffs | Title |", "|---|---|---|---|---|---|---|---|---|"];
  for (const t of [...power.teams].sort((a, b) => (odds(b)?.reg_pts ?? 0) - (odds(a)?.reg_pts ?? 0))) {
    const o = odds(t), d = whatif?.delta[t.id];
    const c = (k: OddsKey) => `${pct(o?.[k])}${d ? ` (${d[k] >= 0 ? "+" : "−"}${Math.round(Math.abs(d[k]) * 100)})` : ""}`;
    L.push(`| ${t.name} | ${t.record.w}-${t.record.l} | ${f1(t.pf)} | ${signed(t.luck)} | ${f1(o?.reg_pts ?? null)} | ${c("seed1")} | ${c("top2")} | ${c("playoffs")} | ${c("title")} |`);
  }
  if (alg) {
    L.push("", `Season so far through week ${alg.through_week} (descriptive; lineup efficiency and luck didn't repeat in 2021-25):`, "",
      "| Team | Points | Best possible | Left on bench | Share from added players | Best week | Worst week |", "|---|---|---|---|---|---|---|");
    for (const t of [...alg.teams].sort((a, b) => b.pts - a.pts))
      L.push(`| ${t.name} | ${f1(t.pts)} | ${f1(t.opt)} | ${f1(t.left)} | ${t.pts ? Math.round(100 * (1 - t.drafted / t.pts)) : 0}% | ${f1(t.best)} | ${f1(t.worst)} |`);
  }
  L.push("", `Odds: ${whatif ? `${whatif.sims} simulated seasons, same draws as the base` : live ? `${live.sims} simulated seasons on current rosters` : "Tuesday ledger"}. Seeding is by total points; title odds are close to a lottery.`);
  return L.join("\n");
}
