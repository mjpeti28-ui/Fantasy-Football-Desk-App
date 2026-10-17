import { useMemo } from "react";
import type { ALeague, APlayer, Analytics, TeamWeek } from "../../types";
import { f1, lastName, signed } from "../../format";
import { useFile } from "../../components/FileContext";
import { Dumbbell, Ev, EvFor, FlagChip, PctCell, TierLegend, TierStrip, evTip, pctTxt, signed2, useWidth } from "../../components/Analytics";
import "./manalysis.css";

/* My Team › Analysis: the season so far for your players, with every stat labelled by how much the
   2021–25 backtest trusts it. Same pegboard surface as the board; panels are the board's cards. */

const SKILL = ["QB", "RB", "WR", "TE"];

interface Props {
  tw: TeamWeek; a: Analytics | null; lg: ALeague | null; err: string | null;
  selected: number | null; onSelect: (id: number) => void;
}

function useMine(tw: TeamWeek, a: Analytics | null) {
  return useMemo(() => {
    if (!a) return { rows: [] as APlayer[], missing: [] as string[] };
    const by = new Map(a.players.map((p) => [p.id, p]));
    const mine = tw.players.filter((p) => SKILL.includes(p.pos));
    const rows = mine.map((p) => by.get(p.id)).filter((p): p is APlayer => !!p).sort((x, y) => (y.est_ppg ?? 0) - (x.est_ppg ?? 0));
    return { rows, missing: mine.filter((p) => !by.has(p.id)).map((p) => p.name) };
  }, [tw, a]);
}

export function MyAnalysis({ tw, a, lg, err, selected, onSelect }: Props) {
  const { open } = useFile();
  const { rows, missing } = useMine(tw, a);
  if (!a) return <div className="board mta"><p className="mta-empty">{err ?? "Building the season analytics…"}</p></div>;
  const M = a.metrics;
  const ids = rows.map((p) => p.id);
  const pick = (id: number) => onSelect(id);
  const dbl = (id: number) => open(id, ids);
  const byGap = [...rows].filter((p) => p.fpoe != null && p.games >= 3).sort((x, y) => Math.abs(y.fpoe!) - Math.abs(x.fpoe!));
  const big = byGap[0];
  const sticks = Math.round(100 * (M.fpoe?.sticks ?? 0.09));
  return (
    <div className="board mta">
      <header className="mta-hd">
        <h4>Season so far <span>through week {a.through_week} · {rows.length} of your players</span></h4>
        <p>Every panel carries how much 2021–25 trusts it:
          <Ev l="edge" tip="Beat the model's own number in past seasons and held up out of sample. Still small." />held up beyond the model,
          <Ev l="priced" tip="Predicts later points, but the model already uses it." />already in the model,
          <Ev l="trait" tip="Repeats week to week but doesn't predict points." />repeats,
          <Ev l="noise" tip="Didn't repeat in past seasons." />noise. Hover a chip for the numbers.</p>
      </header>

      <div className="mta-two">
        <section className="ins-panel">
          <h5>Scoring vs usage <EvFor m={M} k="fpoe" /><span>luck vs role</span></h5>
          <p className="mta-cap">Ring = what his targets, carries and field position usually produce (expected points). Dot = what he scored per game.
            About {sticks}% of a gap like this has carried forward in past seasons, and the model's numbers already discount it.</p>
          <Dumbbell rows={[...rows].sort((x, y) => (y.xfp ?? 0) - (x.xfp ?? 0))} onPick={pick} sel={selected} />
          {big && Math.abs(big.fpoe!) >= 2 && <p className="d-read"><b>{lastName(big.name)}</b> is scoring {f1(Math.abs(big.fpoe!))} a game {big.fpoe! > 0 ? "over" : "under"} his usage ({f1(big.ppg)} vs {f1(big.xfp)}).
            Expect most of that to {big.fpoe! > 0 ? "fade" : "come back"}; the model has him at {f1(big.est_ppg)} for the rest of the season.</p>}
        </section>

        <section className="ins-panel">
          <h5>Week by week <EvFor m={M} k="start" /><span>finish at his position each week</span></h5>
          <TierLegend a={a} />
          <table className="mta-wk">
            <thead><tr><th /><th>Weeks 1–{a.through_week}</th><th title="Startable weeks">Start</th><th title="Smash weeks">Smash</th><th title="Dud weeks">Dud</th><th>Swing <EvFor m={M} k="cv" /></th></tr></thead>
            <tbody>
              {rows.map((p) => {
                const sw = p.flags.find((f) => f.key === "volatile" || f.key === "steady");
                return (
                  <tr key={p.id} className={selected === p.id ? "sel" : ""} onClick={() => pick(p.id)} onDoubleClick={() => dbl(p.id)}>
                    <th className="nm">{lastName(p.name)}</th>
                    <td><TierStrip w={p.weekly} pos={p.pos} /></td>
                    <td className="num">{p.starts}/{p.games}</td><td className="num">{p.smashes || ""}</td><td className="num">{p.duds || ""}</td>
                    <td className="sw"><span className="dim num" title="Coefficient of variation: weekly sd ÷ average">{p.cv != null ? p.cv.toFixed(2) : "–"}</span>{sw && <> <FlagChip f={sw} /></>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mta-cap">Startable and smash rates repeat because they track how good a player is, which the model already knows. Swing (weekly spread ÷ average) repeats moderately but doesn't predict points; it matters only for how much risk you want in a given week.</p>
        </section>
      </div>

      <section className="ins-panel">
        <h5>Snap share by week <EvFor m={M} k="snap_tr" /><span>one scale, 0–100%; line = his season average</span></h5>
        <div className="mta-sm">
          {rows.filter((p) => p.pos !== "QB").map((p) => <SnapMini key={p.id} p={p} weeks={a.through_week} sel={selected === p.id} onPick={() => pick(p.id)} onOpen={() => dbl(p.id)} />)}
        </div>
        <p className="mta-cap">About half of a snap-share change over three games sticks. Rising snaps have beaten the model by {signed2(M.snap_tr?.beyond?.v ?? null)} ppg per standard deviation, and that held up out of sample. It is the steadiest edge in the backtest and still small: a few points over a season.</p>
      </section>

      <section className="ins-panel">
        <h5>Against the position <span>percentile among the top {a.pool.QB} QB/TE and top {a.pool.RB} RB/WR by points per game</span></h5>
        <div className="mta-scroll">
          <table className="mta-pct">
            <thead><tr><th />
              {PCT_COLS.map((c) => <th key={c.k} title={evTip(M[c.k])}>{c.label}{M[c.k] && M[c.k].label !== "priced" && <><br /><EvFor m={M} k={c.k} /></>}</th>)}
            </tr></thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id} className={selected === p.id ? "sel" : ""} onClick={() => pick(p.id)} onDoubleClick={() => dbl(p.id)}>
                  <th className="nm">{lastName(p.name)} <em>{p.pos}</em></th>
                  {PCT_COLS.map((c) => (c.skip?.includes(p.pos) ? <td key={c.k} className="pc none">·</td>
                    : <PctCell key={c.k} v={p.pct[c.k]} title={`${p.name}: ${c.label} ${c.fmt(p)} (${p.pct[c.k] ?? "–"}th percentile among ${p.pos}s)`} />))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mta-cap">Darker = higher, with no good or bad meaning (high swing is neither). These rank where he stands. Every column is already in the model unless it carries a chip saying otherwise.</p>
      </section>

      <div className="mta-two">
        <section className="ins-panel"><TeamWeeks tw={tw} lg={lg} /></section>
        <section className="ins-panel"><ValueMoves rows={rows} a={a} pick={pick} sel={selected} /></section>
      </div>
      {missing.length > 0 && <p className="foot">No games yet this season: {missing.join(", ")}.</p>}
    </div>
  );
}

const PCT_COLS: { k: string; label: string; fmt: (p: APlayer) => string; skip?: string[] }[] = [
  { k: "ppg", label: "Pts/g", fmt: (p) => f1(p.ppg) },
  { k: "xfp", label: "Usage", fmt: (p) => f1(p.xfp) },
  { k: "fpoe", label: "Over usage", fmt: (p) => signed(p.fpoe) },
  { k: "snap", label: "Snaps", fmt: (p) => pctTxt(p.snap), skip: ["QB"] },
  { k: "tsh", label: "Tgt share", fmt: (p) => pctTxt(p.tsh), skip: ["QB"] },
  { k: "start", label: "Startable", fmt: (p) => pctTxt(p.start) },
  { k: "cv", label: "Swing", fmt: (p) => (p.cv != null ? p.cv.toFixed(2) : "–") },
];

function SnapMini({ p, weeks, sel, onPick, onOpen }: { p: APlayer; weeks: number; sel: boolean; onPick: () => void; onOpen: () => void }) {
  const W = 150, H = 44, pad = 4;
  const xs = (i: number) => pad + ((W - 2 * pad) * i) / Math.max(1, weeks - 1);
  const ys = (v: number) => H - 6 - (H - 12) * v;
  const pts = p.weekly.map((w, i) => (w && w[3] != null ? [xs(i), ys(w[3])] as [number, number] : null));
  const segs: [number, number][][] = [];
  let cur: [number, number][] = [];
  for (const q of pts) { if (q) cur.push(q); else if (cur.length) { segs.push(cur); cur = []; } }
  if (cur.length) segs.push(cur);
  const fl = p.flags.find((f) => f.key === "snaps_up" || f.key === "snaps_down");
  const last3 = p.snap != null && p.snap_tr != null ? p.snap + p.snap_tr : null;
  return (
    <button type="button" className={`sm ${sel ? "sel" : ""}`} onClick={onPick} onDoubleClick={onOpen}>
      <span className="sm-h"><b>{lastName(p.name)}</b><span className="num">{pctTxt(last3)}<small> last 3</small></span></span>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${p.name} snap share by week: ${p.weekly.map((w, i) => `week ${i + 1} ${w && w[3] != null ? pctTxt(w[3]) : "no game"}`).join(", ")}`}>
        <line className="base" vectorEffect="non-scaling-stroke" x1={pad} x2={W - pad} y1={ys(0)} y2={ys(0)} />
        {p.snap != null && <line className="avg" vectorEffect="non-scaling-stroke" x1={pad} x2={W - pad} y1={ys(p.snap)} y2={ys(p.snap)} />}
        {segs.map((s, i) => <polyline key={i} className="ln" vectorEffect="non-scaling-stroke" points={s.map(([x, y]) => `${x},${y}`).join(" ")} />)}
        {pts.map((q, i) => q && <circle key={i} className="pt" cx={q[0]} cy={q[1]} r={3}><title>{`Week ${i + 1}: ${pctTxt(p.weekly[i]![3])} of snaps`}</title></circle>)}
      </svg>
      <span className="sm-f">{fl ? <FlagChip f={fl} /> : <span className="dim">{p.snap_tr != null ? `${signed(100 * p.snap_tr).replace(".0", "")} pts vs season` : "–"}</span>}</span>
    </button>
  );
}

const ordinal = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th"}`;

function TeamWeeks({ tw, lg }: { tw: TeamWeek; lg: ALeague | null }) {
  const [ref, W0] = useWidth<HTMLDivElement>(420);
  return <div ref={ref} className="tw-w">{lg ? <TeamWeeksIn tw={tw} lg={lg} W0={W0} /> : <p className="mta-cap">League table loading…</p>}</div>;
}

function TeamWeeksIn({ tw, lg, W0 }: { tw: TeamWeek; lg: ALeague; W0: number }) {
  const mine = lg.cells.filter((c) => c.team_id === tw.team_id).sort((x, y) => x.week - y.week);
  const t = lg.teams.find((x) => x.id === tw.team_id);
  if (!mine.length || !t) return null;
  const mx = Math.max(...mine.map((c) => Math.max(c.opt, c.proj))) * 1.08;
  const W = Math.min(W0, 560), H = 160, L = 8, B = 20, bw = Math.min(24, (W - L) / mine.length - 14);
  const x = (i: number) => L + ((W - L) * (i + 0.5)) / mine.length;
  const y = (v: number) => H - B - ((H - B - 8) * v) / mx;
  const leftRank = [...lg.teams].sort((a, b) => b.left - a.left).findIndex((r) => r.id === tw.team_id) + 1;
  return (
    <>
      <h5>Your weeks <Ev l="noise" tip={`Lineup efficiency repeated at r ${lg.labels.eff?.repeat?.toFixed(2)} and luck at r ${lg.labels.luck?.repeat?.toFixed(2)} in 2021–25: both noise.`} /><span>what you scored, what you could have, what ESPN expected</span></h5>
      <div><svg className="mta-tw" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Your points by week against your best possible lineup and ESPN's projection">
        <line className="base" x1={0} x2={W} y1={y(0)} y2={y(0)} />
        {mine.map((c, i) => (
          <g key={c.week}>
            <rect className="opt" x={x(i) - bw / 2} y={y(c.opt)} width={bw} height={y(c.pts) - y(c.opt)} rx={2} />
            <rect className="got" x={x(i) - bw / 2} y={y(c.pts)} width={bw} height={y(0) - y(c.pts)} rx={4} />
            <line className="proj" x1={x(i) - bw / 2 - 4} x2={x(i) + bw / 2 + 4} y1={y(c.proj)} y2={y(c.proj)} />
            <text className="wk" x={x(i)} y={H - 6} textAnchor="middle">{c.week}</text>
            <text className="rk" x={x(i)} y={y(c.opt) - 4} textAnchor="middle">#{c.rank}</text>
            <title>{`Week ${c.week}: ${f1(c.pts)} (#${c.rank} of ${lg.n_teams}); best possible ${f1(c.opt)}, ${f1(c.left)} left on the bench; ESPN projected your starters for ${f1(c.proj)}`}</title>
          </g>
        ))}
      </svg></div>
      <p className="mta-leg"><span><i className="k got" />Scored</span><span><i className="k opt" />Left on the bench</span><span><i className="k proj" />ESPN's projection of your starters</span><span className="dim">#n = your rank that week</span></p>
      <p className="d-read">You left <b>{f1(t.left)}</b> on the bench (the {ordinal(leftRank)}-most of {lg.n_teams} teams) and beat ESPN's projection of your starters by <b>{signed(t.luck)}</b>. Seeding is by total points, so both count, but neither has carried forward in past seasons.</p>
      <p className="d-read">vs ESPN-best lineup: <b>{signed(t.vs_espn)}</b> pts. Starting the highest ESPN projections on your roster would have scored {f1(t.espn)}{t.flip_won + t.flip_lost > 0 && <> and flipped {t.flip_won} loss{t.flip_won === 1 ? "" : "es"} to wins and {t.flip_lost} win{t.flip_lost === 1 ? "" : "s"} to losses</>}. Every manager here trailed it in 2021–25 (1.2 pts a week on average).</p>
    </>
  );
}

function ValueMoves({ rows, a, pick, sel }: { rows: APlayer[]; a: Analytics; pick: (id: number) => void; sel: number | null }) {
  const mv = rows.filter((p) => p.est_tr != null && Math.abs(p.est_tr) >= 0.5).sort((x, y) => (y.est_tr ?? 0) - (x.est_tr ?? 0));
  const [w0, w1] = a.est_weeks;
  if (!mv.length || w0 == null) return null;
  return (
    <div className="mta-moves">
      <h5>Model value, week {w0} → {w1} <EvFor m={a.metrics} k="est_tr" /></h5>
      {mv.map((p) => (
        <button key={p.id} type="button" className={`mv ${sel === p.id ? "sel" : ""}`} onClick={() => pick(p.id)}>
          <span className="nm">{lastName(p.name)}</span>
          <span className="num">{f1((p.est_ppg ?? 0) - (p.est_tr ?? 0))} → <b>{f1(p.est_ppg)}</b></span>
          <span className={`num d ${p.est_tr! > 0 ? "up" : "down"}`}>{signed(p.est_tr)}</span>
        </button>
      ))}
      <p className="mta-cap">Moves in the model's rest-of-season number have kept going a little ({signed2(a.metrics.est_tr?.beyond?.v ?? null)} ppg per standard deviation, out of sample too). Small.</p>
    </div>
  );
}

export function analysisMarkdown(tw: TeamWeek, a: Analytics | null, lg: ALeague | null): string {
  if (!a) return "";
  const by = new Map(a.players.map((p) => [p.id, p]));
  const rows = tw.players.map((p) => by.get(p.id)).filter((p): p is APlayer => !!p);
  const L = [`## ${tw.team_name}: season analytics through week ${a.through_week}`, "",
    "| Player | Pos | Pts/g | Usage xFP | Over usage | Startable | Duds | Snap last 3 vs season | Model ROS | Flags |", "|---|---|---|---|---|---|---|---|---|---|"];
  for (const p of rows) {
    L.push(`| ${p.name} | ${p.pos} | ${f1(p.ppg)} | ${f1(p.xfp)} | ${signed(p.fpoe)} | ${p.starts}/${p.games} | ${p.duds} | ${p.snap_tr != null ? signed(100 * p.snap_tr) : "–"} | ${f1(p.est_ppg)} | ${p.flags.map((f) => f.short).join("; ") || "–"} |`);
  }
  L.push("", "What the 2021–25 backtest says about these stats:");
  for (const k of ["fpoe", "xfp", "snap_tr", "est_tr", "cv", "start"]) if (a.metrics[k]) L.push(`- ${a.metrics[k].name} (${a.metrics[k].label_text}): ${a.metrics[k].reading}`);
  const t = lg?.teams.find((x) => x.id === tw.team_id);
  if (t) L.push("", `Team: ${f1(t.pts)} points, ${f1(t.left)} left on the bench, ${signed(t.luck)} vs ESPN's projection of the starters used (neither repeated in 2021–25).`);
  return L.join("\n");
}
