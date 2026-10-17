import { type ReactNode, useEffect, useMemo, useState } from "react";
import type { APlayer, Analytics, FpMetric, Player, PlayerFile as PF } from "../../types";
import { EvFor, FlagChip, TierStrip, pctTxt } from "../../components/Analytics";
import { f1, lastName, pct, signed } from "../../format";
import { Face, POS_VAR, Spark } from "../../components/bits";
import { pBeats } from "../../odds";
import { holdText } from "../../components/Role";
import { matchupsApi } from "../../data";
import type { MxCompare } from "../matchups/types";
import "./compare.css";

/* C — Compare: a spec sheet. Players across, evidence down, the better value in each row filled —
   unless the gap is inside the noise, in which case the row says so instead of picking. The
   question you're asking (this week vs rest of season) reorders the sheet and changes the chart
   on top: head-to-head odds for a start/sit, overlapping outcome ranges for a hold/add/trade. */

type Mode = "week" | "ros";
const LET = ["A", "B", "C", "D"];
const asPlayer = (f: PF) => ({ pos: f.pos, espn_week: f.espn_week, game: f.game ?? null, injury: (f.owner?.injury ?? f.ledger?.status ?? "ACTIVE") || "ACTIVE" } as unknown as Player);
const statusOf = (f: PF) => (f.owner?.injury ?? f.ledger?.status ?? "ACTIVE") || "ACTIVE";
/** Not playing this week: bye, out, IR or suspended. */
const sitting = (f: PF) => (f.game?.bye ? "on bye" : /^(OUT|INJURY_RESERVE|IR|SUSPENSION)$/i.test(statusOf(f)) ? statusOf(f).replace(/_/g, " ").toLowerCase().replace("injury reserve", "on IR") : null);
const edge = (p: number) => (p >= 0.65 ? "clear" : p >= 0.55 ? "lean" : "coin flip");
const fmtv = (fmt: string | undefined, v: number | null | undefined) =>
  v == null ? "–" : fmt === "pct100" ? `${Math.round(v)}%` : fmt === "pct" ? `${Math.round(v * 100)}%` : fmt === "num2" ? v.toFixed(2) : v.toFixed(1);

/** P(a's rest-of-season > b's), pairing every comp of one with every comp of the other. */
function pRos(a: PF, b: PF): number | null {
  const A = a.comps?.all.map((x) => x.ros_ppg), B = b.comps?.all.map((x) => x.ros_ppg);
  if (!A?.length || !B?.length) return null;
  let w = 0;
  for (const x of A) for (const y of B) w += x > y ? 1 : x === y ? 0.5 : 0;
  return w / (A.length * B.length);
}

interface Row { label: string; ev?: ReactNode; hint?: string; cell: (f: PF, i: number) => ReactNode; val?: (f: PF) => number | null; better?: "hi" | "lo"; tie?: number; noisy?: (f: PF) => boolean }
interface Group { key: string; title: string; rows: Row[] }
type Mx = Record<number, MxCompare | null | undefined>;

const TAG = { priced: "in ESPN", small: "tiny", context: "context", blind: "ESPN blind spot" } as const;
const Tag = ({ t }: { t: keyof typeof TAG }) => <em className={`mxtag ${t}`}>{TAG[t]}</em>;
const MX_ORDER = ["pass", "run", "allowed", "depth", "trench", "seen", "lens", "pressure", "style_run"];
/** ESPN over-projects players coming back from missed games (2021-25, vs the same position's other games; pff/creative.md E10). */
function backPts(f: PF): number | null {
  const b = f.back;
  if (!b || f.game?.bye) return null;
  if (b.game === 2) return f.pos === "WR" ? -0.8 : null;
  return f.pos === "WR" ? -1.3 : f.pos === "QB" ? -1.5 : f.pos === "RB" ? (b.missed >= 2 ? -1.5 : -0.7) : null;
}

function mxRows(files: PF[], mx: Mx): Row[] {
  const out: Row[] = [];
  const has = (f: PF) => mx[f.id];
  out.push({ label: "The read", hint: "How tough this defense is for him, and how much of ESPN's number that is", cell: (f) => {
    const m = has(f);
    return m === undefined ? <span className="dim">loading…</span> : <span className="mxread">{m?.read ?? "No matchup data for him."}</span>;
  } });
  if (files.some((f) => f.back)) out.push({ label: "Back from missed games", ev: <Tag t="blind" />,
    hint: "ESPN over-projects players in their first game back (WR -1.3, QB -1.5, RB -0.7, or -1.5 after 2+ missed; TE nothing) and WRs in their second (-0.8). 2021-25, 4 of 5 seasons. 'Missed' includes healthy scratches.",
    cell: (f) => { const b = f.back, p = backPts(f);
      if (!b) return <span className="dim">–</span>;
      const txt = `${b.game === 1 ? "1st" : "2nd"} game back after ${b.missed} missed`;
      return <span>{txt}<small className={p != null ? "mxsub warn-n" : "mxsub dim"}>{p != null ? `ESPN usually ~${Math.abs(p).toFixed(1)} pts high here` : "no measured ESPN miss here"}</small></span>; } });
  if (files.some((f) => f.game?.implied_total != null)) out.push({ label: "Vegas team total", ev: <Tag t="small" />,
    hint: "Points the betting line implies for his team. Favourites beat ESPN slightly (QB/RB about +0.4 pts per sd of margin, 2021-25); mostly already in ESPN's number.",
    cell: (f) => { const g = f.game; if (!g || g.implied_total == null) return <span className="dim">–</span>;
      const m = g.game_total != null ? 2 * g.implied_total - g.game_total : null;
      return <span className="num">{g.implied_total}<small className="mxsub dim">{m == null ? "" : Math.abs(m) < 0.5 ? "pick'em" : m > 0 ? `favored by ${m.toFixed(1)}` : `underdog by ${(-m).toFixed(1)}`}{g.game_total != null ? ` · total ${g.game_total}` : ""}</small></span>; } });
  const keys = MX_ORDER.filter((k) => files.some((f) => has(f)?.rows.some((r) => r.key === k)));
  for (const k of keys) {
    const r0 = files.map((f) => has(f)?.rows.find((r) => r.key === k)).find(Boolean)!;
    out.push({ label: r0.label, ev: <Tag t={r0.tag} />, hint: r0.tip ?? undefined, cell: (f) => {
      const r = has(f)?.rows.find((x) => x.key === k);
      return r ? <span title={r.tip ?? undefined}>{r.text}{r.sub && <small className="mxsub dim">{r.sub}</small>}</span> : <span className="dim">–</span>;
    } });
  }
  return out;
}

function groups(files: PF[], ratioQ?: Record<string, number[]>, an?: Analytics | null, mx: Mx = {}): Group[] {
  const sameType = new Set(files.map((f) => (f.pos === "QB" ? "QB" : "SKILL"))).size === 1;
  const commonFp = files.length ? files[0].fingerprint.filter((m) => files.every((f) => f.fingerprint.some((x) => x.key === m.key)) && files.every((f) => f.pos === files[0].pos || ["ppr_g", "snap", "td_g"].includes(m.key))) : [];
  const fpOf = (f: PF, k: string) => f.fingerprint.find((m) => m.key === k);
  const week: Row[] = [
    { label: "ESPN projection", val: (f) => f.espn_week, better: "hi", tie: 1, cell: (f) => <b className="num">{f1(f.espn_week)}</b> },
    { label: "Game", cell: (f) => <span>{f.game ? (f.game.bye ? "Bye" : `${f.game.home ? "vs" : "@"} ${f.game.opp} · ${f.game.day ?? ""}`) : "–"}</span> },
    { label: "Status", cell: (f) => { const s = f.game?.bye ? "BYE" : statusOf(f); return <span className={/^(active|normal)$/i.test(s) ? "" : "warn"}>{s.replace(/_/g, " ").toLowerCase()}</span>; } },
  ];
  if (files.length >= 2 && ratioQ) {
    week.push({ label: "Outscores the field", hint: "Chance he outscores every other player here this week (ESPN projection × the position's actual/forecast spread)",
      val: (f) => { const me = asPlayer(f); let p = 1; for (const g of files) if (g !== f) { const x = pBeats(me, asPlayer(g), ratioQ); if (x == null) return null; p *= x; } return p; },
      better: "hi", tie: 0.06, cell: (f) => { const me = asPlayer(f); let p: number | null = 1; for (const g of files) if (g !== f) { const x = pBeats(me, asPlayer(g), ratioQ); p = x == null || p == null ? null : p * x; } return <span className="num">{pct(p)}</span>; } });
  }
  const ros: Row[] = [
    { label: "Our model, rest of season", hint: "PPR points per game when he plays", val: (f) => f.ledger?.est_ppg ?? null, better: "hi", tie: 0.75, cell: (f) => <b className="num">{f1(f.ledger?.est_ppg)}</b> },
    { label: "Plays rest of season", val: (f) => f.ledger?.avail_ros ?? null, better: "hi", tie: 0.05, cell: (f) => <span className="num">{pct(f.ledger?.avail_ros)}</span> },
    { label: "Players like him: median", hint: "40 most similar player-seasons 2016–25 at the same week; their rest-of-season PPR/g", val: (f) => f.comps?.q.p50 ?? null, better: "hi", tie: 1, cell: (f) => <b className="num">{f.comps ? f.comps.q.p50 : "–"}</b> },
    { label: "Players like him: 80% range", cell: (f) => <span className="num dim">{f.comps ? `${f.comps.q.p10}–${f.comps.q.p90}` : "–"}</span> },
    { label: "Kept their snap share", val: (f) => f.comps?.kept_role ?? null, better: "hi", tie: 0.08, cell: (f) => <span className="num">{pct(f.comps?.kept_role)}</span> },
    { label: "Weeks of 15+ (comps)", val: (f) => f.comps?.weeks.ge15 ?? null, better: "hi", tie: 0.06, cell: (f) => <span className="num">{pct(f.comps?.weeks.ge15)}</span> },
    { label: "Weeks under 5 (comps)", val: (f) => f.comps?.weeks.lt5 ?? null, better: "lo", tie: 0.06, cell: (f) => <span className="num">{pct(f.comps?.weeks.lt5)}</span> },
    { label: "This season, PPR/g", val: (f) => f.season_line?.ppr_g ?? null, better: "hi", tie: 1.5, cell: (f) => <span className="num">{f1(f.season_line?.ppr_g)} <small className="dim">{f.season_line?.rank ? `${f.pos}${f.season_line.rank}` : ""}</small></span> },
    { label: "Last season, PPR/g", val: (f) => f.last_season?.ppr_g ?? null, better: "hi", tie: 1.5, cell: (f) => <span className="num">{f1(f.last_season?.ppr_g)}</span> },
  ];
  const role: Row[] = [
    { label: "Style", cell: (f) => f.role.style ? <span className="sty"><span className="mix">{Object.entries(f.role.style.probs).sort((a, b) => b[1] - a[1]).map(([k, v], i) =>
      <i key={k} style={{ width: `${v * 100}%`, ...(i === 0 ? { background: POS_VAR[f.pos] } : {}) }} title={`${k} ${Math.round(v * 100)}%`} />)}</span>{f.role.style.name} <small className="dim">{Math.round(f.role.style.p * 100)}%</small></span> : "–" },
    { label: "Volume tier", cell: (f) => <span>{f.role.tier}</span> },
    { label: "Route share (PFF)", val: (f) => f.role.route_sh?.v ?? null, better: "hi", tie: 5, cell: (f) => f.role.route_sh ? <span className="num">{f.role.route_sh.v}% <small className="dim">{f.role.route_sh.rank} of {f.role.route_sh.of}</small></span> : <span className="dim">–</span> },
    { label: "Traits", cell: (f) => <span className="tr">{f.role.traits.length ? f.role.traits.map((t) => <i key={t.label} title={t.text}>{t.label}</i>) : <span className="dim">–</span>}</span> },
    { label: "Latest change", cell: (f) => { const t = f.role.trends.find((x) => !x.season) ?? f.role.trends[0]; return t ? <span className="chg" title={holdText(t)}><b>{t.label}</b> <small className="dim">{t.hold != null ? `held ${pct(t.hold)}` : ""}</small></span> : <span className="dim">none flagged</span>; } },
  ];
  const real: Row[] = commonFp.map((m0) => ({
    label: m0.label, hint: `Settles after ~${Math.max(1, Math.round(m0.stab_games))} games`,
    val: (f) => { const m = fpOf(f, m0.key); return m?.pct == null ? null : m.lower_better ? -m.pct : m.pct; }, better: "hi" as const, tie: 10,
    noisy: (f) => (fpOf(f, m0.key)?.rel ?? 0) < 0.35,
    cell: (f) => { const m = fpOf(f, m0.key) as FpMetric | undefined; return m ? <span className={`fp ${(m.rel ?? 0) < 0.35 ? "noisy" : ""}`} title={`raw ${fmtv(m.fmt, m.raw)}; signal ${Math.round((m.rel ?? 0) * 100)}%`}>
      <b className="num">{fmtv(m.fmt, m.est)}</b><span className="pt"><i style={{ left: `${m.pct ?? 50}%`, background: POS_VAR[f.pos] }} /></span><small className="num dim">{m.pct == null ? "" : `${Math.round(m.pct)}th`}</small></span> : "–"; },
  }));
  const shape: Row[] = [
    { label: "Weekly points", cell: (f) => <Spark w={f.gamelog.filter((r) => !r.upcoming).map((r) => (r.played ? r.espn_pts ?? r.ppr : null))} bw={6} h={20} color={POS_VAR[f.pos]} /> },
    { label: "Beat ESPN's projection", val: (f) => { const r = f.gamelog.filter((x) => x.played && !x.upcoming && x.espn_proj != null); return r.length ? r.filter((x) => (x.espn_pts ?? x.ppr ?? 0) >= (x.espn_proj ?? 0)).length / r.length : null; },
      cell: (f) => { const r = f.gamelog.filter((x) => x.played && !x.upcoming && x.espn_proj != null); return <span className="num">{r.length ? `${r.filter((x) => (x.espn_pts ?? x.ppr ?? 0) >= (x.espn_proj ?? 0)).length} of ${r.length}` : "–"}</span>; } },
  ];
  const logi: Row[] = [
    { label: "Bye week", cell: (f) => <span className="num">{f.ledger?.bye ?? "–"}</span> },
    { label: "Rostered by", cell: (f) => <span>{f.owner?.name ?? "free agent"}</span> },
    { label: "Age", val: (f) => f.age, better: "lo", tie: 2, cell: (f) => <span className="num">{f.age ? Math.floor(f.age) : "–"}</span> },
  ];
  // season analytics (labels from the 2021-25 backtest; only rows that held up get a "better" fill)
  const A = (f: PF): APlayer | undefined => an?.players.find((p) => p.id === f.id);
  const M = an ? (files.length && files.every((f) => f.pos === files[0].pos) ? an.metrics_pos[files[0].pos] ?? an.metrics : an.metrics) : undefined;
  const ana: Row[] = an ? [
    { label: "Usage, expected pts/g", ev: <EvFor m={M} k="xfp" />, hint: "What his targets, carries and field position usually produce", val: (f) => A(f)?.xfp ?? null, better: "hi", tie: 1.5,
      cell: (f) => <span className="num">{f1(A(f)?.xfp)}</span> },
    { label: "Scored over usage", ev: <EvFor m={M} k="fpoe" />, hint: "About 9% of this carries forward; never filled",
      cell: (f) => { const a = A(f); const fl = a?.flags.find((x) => x.key === "hot" || x.key === "cold"); return fl ? <FlagChip f={fl} /> : <span className="num">{signed(a?.fpoe)}</span>; } },
    { label: "Finishes", ev: <EvFor m={M} k="start" />, cell: (f) => { const a = A(f); return a ? <TierStrip w={a.weekly} pos={a.pos} /> : <span className="dim">–</span>; } },
    { label: "Startable weeks", ev: <EvFor m={M} k="start" />, val: (f) => A(f)?.start ?? null, better: "hi", tie: 0.2,
      cell: (f) => { const a = A(f); return <span className="num">{a ? `${a.starts} of ${a.games}` : "–"}</span>; } },
    { label: "Dud weeks", ev: <EvFor m={M} k="dud" />, hint: "Didn't hold out of sample; never filled", cell: (f) => { const a = A(f); return <span className="num">{a ? `${a.duds} of ${a.games}` : "–"}</span>; } },
    { label: "Swing (sd ÷ avg)", ev: <EvFor m={M} k="cv" />, hint: "Repeats moderately, doesn't predict points; never filled",
      cell: (f) => { const a = A(f); return <span className="num">{a?.cv != null ? a.cv.toFixed(2) : "–"}{a?.pct.cv != null ? <small className="dim"> {Math.round(a.pct.cv)}th</small> : null}</span>; } },
    ...(files.some((f) => f.pos !== "QB") ? [{ label: "Snap share, last 3 vs season", ev: <EvFor m={M} k="snap_tr" />, val: (f: PF) => A(f)?.snap_tr ?? null, better: "hi" as const, tie: 0.12, hint: "Small edge: only filled when the gap is worth about half a point a game or more",
      cell: (f: PF) => { const a = A(f); return <span className="num">{a && a.snap != null && a.snap_tr != null ? <>{pctTxt(a.snap + a.snap_tr)} <small className="dim">vs {pctTxt(a.snap)}</small></> : "–"}</span>; } }] : []),
    { label: `Model value, wk ${an.est_weeks[0] ?? "–"}→${an.est_weeks[1] ?? "–"}`, ev: <EvFor m={M} k="est_tr" />, hint: "Small edge: only filled when the gap is worth about half a point a game or more", val: (f) => A(f)?.est_tr ?? null, better: "hi", tie: 2.5,
      cell: (f) => <span className="num">{signed(A(f)?.est_tr)}</span> },
  ] : [];
  return [
    { key: "an", title: `Usage vs results, week to week (through week ${an?.through_week ?? "–"}; chips = how much 2021–25 trusts each row)`, rows: ana },
    { key: "week", title: "This week", rows: week },
    { key: "mx", title: "Matchup this week (describes the game; the defense is already in ESPN's number, only “ESPN blind spot” rows moved points in backtests)", rows: mxRows(files, mx) },
    { key: "ros", title: "Rest of season", rows: ros },
    { key: "role", title: "Role", rows: role },
    { key: "real", title: sameType ? "What's real (best guess, percentile vs position)" : "What's real (shared stats only: different positions)", rows: real },
    { key: "shape", title: "Week to week", rows: shape },
    { key: "logi", title: "Logistics", rows: logi },
  ];
}

function bestOf(files: PF[], r: Row): { best: number | null; tie: boolean } {
  if (!r.val || files.length < 2) return { best: null, tie: false };
  const v = files.map((f) => r.val!(f));
  const ok = v.map((x, i) => ({ x, i })).filter((o) => o.x != null) as { x: number; i: number }[];
  if (ok.length < 2) return { best: null, tie: false };
  ok.sort((a, b) => (r.better === "lo" ? a.x - b.x : b.x - a.x));
  if (Math.abs(ok[0].x - ok[1].x) < (r.tie ?? 0)) return { best: null, tie: true };
  if (r.noisy && r.noisy(files[ok[0].i])) return { best: null, tie: true };
  return { best: ok[0].i, tie: false };
}

function H2H({ files, ratioQ }: { files: PF[]; ratioQ?: Record<string, number[]> }) {
  if (files.length < 2 || !ratioQ) return null;
  const P = files.map((a) => files.map((b) => (a === b ? null : pBeats(asPlayer(a), asPlayer(b), ratioQ))));
  if (files.length === 2) {
    const out = files.map(sitting);
    if (out[0] || out[1]) return <p className="cap">{files.map((f, i) => (out[i] ? `${lastName(f.name)} is ${out[i]}.` : "")).join(" ")} Nothing to weigh this week.</p>;
    const p = P[0][1];
    return (
      <div className="cmp-h2h">
        <div className="lbls"><span><i className="let">A</i>{lastName(files[0].name)} {f1(files[0].espn_week)}</span><span>{f1(files[1].espn_week)} {lastName(files[1].name)}<i className="let">B</i></span></div>
        {p != null ? <div className="split"><i className={p >= 0.5 ? "lead" : ""} style={{ width: `${p * 100}%` }}><b>{pct(p)}</b></i><i className={p < 0.5 ? "lead" : ""} style={{ width: `${(1 - p) * 100}%` }}><b>{pct(1 - p)}</b></i></div> : <p className="dim">No ESPN projection for one of them.</p>}
        <p className="cap">Chance each outscores the other this week: ESPN's projection times how far actual scores at each position have landed from forecasts (2021–25), treated as independent. {p != null ? `${edge(Math.max(p, 1 - p))[0].toUpperCase()}${edge(Math.max(p, 1 - p)).slice(1)}.` : ""}</p>
      </div>
    );
  }
  const out = files.map(sitting);
  return <Matrix files={files} P={P} out={out} cap="Row beats column this week (ESPN projection × position spread)." />;
}

function Matrix({ files, P, cap, out }: { files: PF[]; P: (number | null)[][]; cap: string; out?: (string | null)[] }) {
  return (
    <div className="cmp-mx">
      <table>
        <thead><tr><th />{files.map((f, j) => <th key={j}><i className="let">{LET[j]}</i>{lastName(f.name)}</th>)}</tr></thead>
        <tbody>{files.map((f, i) => <tr key={i}><th><i className="let">{LET[i]}</i>{lastName(f.name)}</th>
          {files.map((_, j) => { const p = P[i][j]; const o = out?.[i] ?? out?.[j];
            if (p != null && o) return <td key={j} className="off">{out?.[i] ? out[i]!.replace(/^on /, "") : "–"}</td>;
            return <td key={j} className={p == null ? "self" : p >= 0.55 ? "w" : p <= 0.45 ? "l" : ""} style={p != null ? { ["--p" as string]: `${Math.round(Math.abs(p - 0.5) * 140)}%` } : undefined}>{p == null ? "" : pct(p)}</td>; })}</tr>)}</tbody>
      </table>
      <p className="cap">{cap}</p>
    </div>
  );
}

function Ranges({ files }: { files: PF[] }) {
  const withC = files.filter((f) => f.comps);
  if (!withC.length) return <p className="dim">Comps need two games this season.</p>;
  const xmax = Math.max(20, ...withC.map((f) => f.comps!.q.p90 + 2), ...files.map((f) => (f.ledger?.est_ppg ?? 0) + 2));
  const x = (v: number) => `${(Math.max(0, Math.min(xmax, v)) / xmax) * 100}%`;
  const P = files.map((a) => files.map((b) => (a === b ? null : pRos(a, b))));
  const ticks = [0, 5, 10, 15, 20, 25, 30].filter((t) => t <= xmax);
  return (
    <div className="cmp-ranges">
      {files.map((f, i) => (
        <div key={f.id} className="rr">
          <span className="nm"><i className="let">{LET[i]}</i>{lastName(f.name)}</span>
          <span className="trk">
            {f.comps && <><i className="b80" style={{ left: x(f.comps.q.p10), width: `calc(${x(f.comps.q.p90)} - ${x(f.comps.q.p10)})` }} />
              <i className="b50" style={{ left: x(f.comps.q.p25), width: `calc(${x(f.comps.q.p75)} - ${x(f.comps.q.p25)})` }} />
              <i className="med" style={{ left: x(f.comps.q.p50), background: POS_VAR[f.pos] }} title={`comps median ${f.comps.q.p50}`} /></>}
            {f.ledger?.est_ppg != null && <i className="mdl" style={{ left: x(f.ledger.est_ppg) }} title={`our model ${f1(f.ledger.est_ppg)}`} />}
          </span>
          <span className="vals num">{f.comps ? `${f.comps.q.p50}` : "–"} <small className="dim">{f.comps ? `${f.comps.q.p10}–${f.comps.q.p90}` : ""}</small></span>
        </div>
      ))}
      <div className="rr axis"><span /><span className="trk">{ticks.map((t) => <em key={t} style={{ left: x(t) }}>{t}</em>)}</span><span /></div>
      <p className="cap"><i className="k80" /> 80% and <i className="k50" /> 50% of what the 40 most similar player-seasons scored per game the rest of the way; colored tick = their median; ▲ = our model.
        Ranges like these held the real outcome about 80% of the time in backtests.</p>
      {files.length >= 3 && <Matrix files={files} P={P} cap="Row ahead of column: share of pairings of their comps' rest-of-season outcomes. Above 65% is a clear edge, 55–65% a lean, under 55% a coin flip." />}
    </div>
  );
}

function verdict(files: PF[], mode: Mode, ratioQ?: Record<string, number[]>): string {
  if (files.length < 2) return "Add another player to compare.";
  if (mode === "week") {
    const sit = files.filter((f) => sitting(f));
    const play = files.filter((f) => !sitting(f));
    const pre = sit.map((f) => `${lastName(f.name)} is ${sitting(f)}.`).join(" ");
    if (sit.length && play.length < 2) return play.length ? `${pre} This week: ${play[0].name} by default.` : `${pre} Nobody here plays this week.`;
    if (sit.length) return `${pre} ${verdict(play, mode, ratioQ)}`;
    const s = files.map((a) => { let p = 1; for (const b of files) if (a !== b) { const x = pBeats(asPlayer(a), asPlayer(b), ratioQ); if (x == null) return { f: a, p: -1 }; p *= x; } return { f: a, p }; }).sort((a, b) => b.p - a.p);
    if (s[0].p < 0) return "One of them has no ESPN projection this week.";
    const top = s[0].f, nxt = s[1].f, p = pBeats(asPlayer(top), asPlayer(nxt), ratioQ) ?? 0.5;
    return `This week: ${top.name}. He outscores ${lastName(nxt.name)} ${pct(p)} of the time — ${edge(p)}${p < 0.55 ? ", so start whoever fits your roster" : ""}.`;
  }
  const s = [...files].sort((a, b) => (b.ledger?.est_ppg ?? b.comps?.q.p50 ?? 0) - (a.ledger?.est_ppg ?? a.comps?.q.p50 ?? 0));
  const top = s[0], nxt = s[1], p = pRos(top, nxt);
  const d = (top.ledger?.est_ppg ?? 0) - (nxt.ledger?.est_ppg ?? 0);
  return `Rest of season: ${top.name}, ${f1(top.ledger?.est_ppg)} vs ${f1(nxt.ledger?.est_ppg)} per game by our model (${signed(d)}).` +
    (p == null ? "" : p >= 0.5 ? ` Players like him finished ahead of players like ${lastName(nxt.name)} in ${pct(p)} of pairings — ${edge(p)}.`
      : ` But players like ${lastName(nxt.name)} finished ahead in ${pct(1 - p)} of comp pairings, so the model's edge is the only edge here — ${edge(1 - p) === "coin flip" ? "call it close" : "treat it as contested"}.`);
}

export function compareMarkdown(files: PF[], mode: Mode, ratioQ?: Record<string, number[]>, an?: Analytics | null): string {
  const A = (f: PF) => an?.players.find((p) => p.id === f.id);
  const rows: [string, (f: PF) => string][] = [
    ["Pos / team / rostered by", (f) => `${f.pos} ${f.pro_team ?? f.team ?? ""} / ${f.owner?.name ?? "FA"}`],
    ["ESPN projection this week", (f) => f1(f.espn_week)],
    ["Our model, rest of season (PPR/g)", (f) => f1(f.ledger?.est_ppg)],
    ["Plays rest of season", (f) => pct(f.ledger?.avail_ros)],
    ["Comps: rest-of-season median (80% range)", (f) => (f.comps ? `${f.comps.q.p50} (${f.comps.q.p10}–${f.comps.q.p90})` : "–")],
    ["Comps: kept snap share / weeks 15+ / weeks <5", (f) => (f.comps ? `${pct(f.comps.kept_role)} / ${pct(f.comps.weeks.ge15)} / ${pct(f.comps.weeks.lt5)}` : "–")],
    ["This season PPR/g (rank)", (f) => `${f1(f.season_line?.ppr_g)}${f.season_line?.rank ? ` (${f.pos}${f.season_line.rank})` : ""}`],
    ["Last season PPR/g", (f) => f1(f.last_season?.ppr_g)],
    ["Style / tier", (f) => `${f.role.style ? `${f.role.style.name} ${Math.round(f.role.style.p * 100)}%` : "–"} / ${f.role.tier}`],
    ["Traits", (f) => f.role.traits.map((t) => t.label).join(", ") || "–"],
    ["Latest change", (f) => { const t = f.role.trends.find((x) => !x.season) ?? f.role.trends[0]; return t ? `${t.label}${t.hold != null ? ` (held ${pct(t.hold)})` : ""}` : "–"; }],
    ["Bye", (f) => String(f.ledger?.bye ?? "–")],
    ...(an ? [
      ["Usage xFP/g / scored over usage (fades; ~9% carries)", (f: PF) => `${f1(A(f)?.xfp)} / ${signed(A(f)?.fpoe)}`],
      ["Startable / smash / dud weeks", (f: PF) => { const a = A(f); return a ? `${a.starts} / ${a.smashes} / ${a.duds} of ${a.games}` : "–"; }],
      ["Snap share last 3 vs season (small edge)", (f: PF) => { const a = A(f); return a && a.snap != null && a.snap_tr != null ? `${pctTxt(a.snap + a.snap_tr)} vs ${pctTxt(a.snap)}` : "–"; }],
      ["Model value 3-week move (small edge)", (f: PF) => signed(A(f)?.est_tr)],
    ] as [string, (f: PF) => string][] : []),
  ];
  const L = [`## Compare: ${files.map((f) => f.name).join(" vs ")} (from Desk, week ${files[0]?.week ?? ""})`, "", verdict(files, mode, ratioQ), "",
    `| | ${files.map((f) => f.name).join(" | ")} |`, `|---|${files.map(() => "---").join("|")}|`,
    ...rows.map(([l, fn]) => `| ${l} | ${files.map(fn).join(" | ")} |`)];
  if (files.length >= 2 && ratioQ) {
    const p = pBeats(asPlayer(files[0]), asPlayer(files[1]), ratioQ), r = pRos(files[0], files[1]);
    L.push("", `${files[0].name} outscores ${files[1].name} this week: ${pct(p)}. Comps pairing, rest of season: ${pct(r)}.`);
  }
  return L.join("\n");
}

export function CompareView({ files, loading, err, ratioQ, candidates, onAdd, onRemove, onClose, onOpen, onCopy, an, suggest }: {
  an?: Analytics | null;
  /** with one player open: who to put next to him (your players at his position, top free agents) */
  suggest?: (id: number, pos: string) => { id: number; name: string; why: string }[];
  files: PF[]; loading: boolean; err: string | null; ratioQ?: Record<string, number[]>;
  candidates: { id: number; name: string; pos: string; team: string | null }[];
  onAdd: (id: number) => void; onRemove: (id: number) => void; onClose: () => void; onOpen: (id: number) => void; onCopy: (md: string) => void;
}) {
  const [mode, setMode] = useState<Mode>("week");
  const [q, setQ] = useState("");
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape" && !(e.target as HTMLElement)?.closest?.("input")) onClose(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const [mx, setMx] = useState<Mx>({});
  const ids = files.map((f) => f.id).join(",");
  useEffect(() => {
    let live = true;
    for (const f of files) if (!(f.id in mx)) matchupsApi.player(f.id).then((p) => p.compare ?? null, () => null).then((c) => live && setMx((m) => ({ ...m, [f.id]: c })));
    return () => { live = false; };
  }, [ids]); // eslint-disable-line react-hooks/exhaustive-deps
  const G = useMemo(() => groups(files, ratioQ, an, mx), [files, ratioQ, an, mx]);
  const order = mode === "week" ? ["week", "mx", "ros", "role", "an", "real", "shape", "logi"] : ["ros", "an", "role", "real", "shape", "week", "mx", "logi"];
  const matches = q.length >= 2 ? candidates.filter((c) => c.name.toLowerCase().includes(q.toLowerCase()) && !files.some((f) => f.id === c.id)).slice(0, 8) : [];
  const cols = `minmax(190px, 230px) repeat(${Math.max(files.length, 1)}, minmax(0, 290px))`;
  return (
    <div className="cmp-wrap" role="dialog" aria-modal="true" aria-label="Compare players">
      <div className="cmp-scroll">
        <header className="cmp-hd">
          <h1>Compare</h1>
          <div className="seg" role="group" aria-label="Question">
            <button type="button" aria-pressed={mode === "week"} onClick={() => setMode("week")}>Who to start this week</button>
            <button type="button" aria-pressed={mode === "ros"} onClick={() => setMode("ros")}>Who to hold or get</button>
          </div>
          <span className="sp" />
          {files.length < 4 && (
            <div className="add">
              <input type="search" placeholder="Add a player…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Add a player" />
              {matches.length > 0 && <ul className="menu">{matches.map((c) => <li key={c.id}><button type="button" onClick={() => { onAdd(c.id); setQ(""); }}>{c.name} <small>{c.pos} · {c.team ?? "FA"}</small></button></li>)}</ul>}
            </div>
          )}
          <button className="tbtn" type="button" onClick={() => onCopy(compareMarkdown(files, mode, ratioQ, an))} disabled={!files.length}>Copy for Claude</button>
          <button className="tbtn" type="button" onClick={onClose}>Close <kbd>esc</kbd></button>
        </header>
        {err && <p className="cmp-err">{err}</p>}
        {loading && !files.length && <p className="dim cmp-pad">Opening files…</p>}
        {files.length > 0 && <>
          <div className="cmp-verdict"><b>{verdict(files, mode, ratioQ)}</b></div>
          {files.length === 1 && suggest && (() => { const sg = suggest(files[0].id, files[0].pos); return sg.length > 0 && (
            <div className="cmp-suggest"><span className="dim">Compare with</span>
              {sg.map((x) => <button key={x.id} type="button" className="tbtn" onClick={() => onAdd(x.id)}>{x.name} <small>{x.why}</small></button>)}</div>); })()}
          <div className="cmp-cols" style={{ gridTemplateColumns: cols }}>
            <span />
            {files.map((f, i) => (
              <div key={f.id} className="col-hd">
                <i className="let big">{LET[i]}</i>
                <Face p={{ headshot: f.headshot ?? null, pos: f.pos, name: f.name }} className="round" />
                <div><button type="button" className="nm" onClick={() => onOpen(f.id)} title="Open player file">{f.name}</button>
                  <small><i className="pchip" style={{ background: POS_VAR[f.pos], color: f.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{f.pos}</i> {f.pro_team ?? f.team ?? "FA"} · {f.owner?.name ?? "free agent"}</small></div>
                <button type="button" className="x" onClick={() => onRemove(f.id)} aria-label={`Remove ${f.name}`}>×</button>
              </div>
            ))}
          </div>
          <section className="cmp-chart">
            {mode === "week" ? <><h2>Head to head this week</h2><H2H files={files} ratioQ={ratioQ} /></> : <><h2>Rest of season: what players like them did</h2><Ranges files={files} /></>}
          </section>
          {order.map((k) => G.find((g) => g.key === k)!).filter((g) => g.rows.length).map((g) => (
            <section key={g.key} className={`cmp-grp ${(mode === "week" && g.key === "week") || (mode === "ros" && g.key === "ros") ? "lead" : ""}`}>
              <h3>{g.title}</h3>
              {g.rows.map((r) => {
                const b = bestOf(files, r);
                return (
                  <div key={r.label} className="cmp-row" style={{ gridTemplateColumns: cols, maxWidth: 230 + files.length * 302 }}>
                    <span className="lab" title={r.hint}>{r.label}{r.ev && <> {r.ev}</>}{b.tie && <em className="tie" title="The gap is inside the noise">≈ toss-up</em>}</span>
                    {files.map((f, i) => <span key={f.id} className={`cell ${b.best === i ? "best" : ""}`}>{r.cell(f, i)}</span>)}
                  </div>
                );
              })}
            </section>
          ))}
          <p className="cmp-foot">Matchup tags: <Tag t="priced" /> ESPN's projection already moves with it; <Tag t="small" /> real but worth a fraction of a point; <Tag t="context" /> describes the game, tested null as a predictor once ESPN's number is known; <Tag t="blind" /> ESPN measurably misses it. Hover a row for how it was measured.</p>
        <p className="cmp-foot">Filled cells are better by more than the noise for that row (projection 1 pt, model 0.75 ppg, percentiles 10 points, rates 5–8 points); “≈ toss-up” means the gap is smaller than that, or the leader's number is still mostly noise.</p>
        </>}
      </div>
    </div>
  );
}

/** Floating tray: the players queued for comparison, from any screen. */
export function CompareTray({ items, onRemove, onOpen, onClear }: { items: { id: number; name: string }[]; onRemove: (id: number) => void; onOpen: () => void; onClear: () => void }) {
  if (!items.length) return null;
  return (
    <div className="cmp-tray" role="region" aria-label="Compare tray">
      <span className="t">Compare</span>
      {items.map((x, i) => <span key={x.id} className="chip"><i className="let">{LET[i]}</i>{x.name}<button type="button" aria-label={`Remove ${x.name}`} onClick={() => onRemove(x.id)}>×</button></span>)}
      {items.length < 4 && <span className="hint">⌘-click players to add</span>}
      <button className="tbtn pri" type="button" onClick={onOpen} disabled={items.length < 1}>{items.length < 2 ? "Open" : `Compare ${items.length}`}</button>
      <button className="tbtn" type="button" onClick={onClear}>Clear</button>
    </div>
  );
}
