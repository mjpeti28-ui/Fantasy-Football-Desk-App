import type { AMetric, APlayer, Analytics, Pos } from "../../types";
import { f1, signed } from "../../format";
import { POS_VAR } from "../../components/bits";
import { Ev, EvFor, FlagChip, TierLegend, TierStrip, evTip, pctTxt, signed2, useWidth } from "../../components/Analytics";
import "./fanalysis.css";

/* Player file › Season analytics: usage vs results, finishes and spread, last three games vs the
   season, and where he ranks at the position — each stat with its 2021–25 evidence chip. */

type Col = { k: string; label: string; v: (p: APlayer) => string; skip?: Pos[] };
const RANKS: Col[] = [
  { k: "ppg", label: "Points per game", v: (p) => f1(p.ppg) },
  { k: "xfp", label: "Usage (expected pts/g)", v: (p) => f1(p.xfp) },
  { k: "fpoe", label: "Over usage", v: (p) => signed(p.fpoe) },
  { k: "opp", label: "Targets + carries /g", v: (p) => f1(p.opp) },
  { k: "snap", label: "Snap share", v: (p) => pctTxt(p.snap), skip: ["QB"] },
  { k: "tsh", label: "Target share", v: (p) => pctTxt(p.tsh), skip: ["QB"] },
  { k: "start", label: "Startable weeks", v: (p) => pctTxt(p.start) },
  { k: "smash", label: "Smash weeks", v: (p) => pctTxt(p.smash) },
  { k: "cv", label: "Swing (sd ÷ avg)", v: (p) => (p.cv != null ? p.cv.toFixed(2) : "–") },
  { k: "resid", label: "vs ESPN's projection", v: (p) => signed(p.resid) },
];

export function FileAnalytics({ ar, a }: { ar: APlayer | undefined; a: Analytics | null | undefined }) {
  if (!a) return <p className="pf-cap">Season analytics are still building.</p>;
  if (!ar) return <p className="pf-cap">Not in the season analytics (needs a game this season and some fantasy relevance).</p>;
  const M: Record<string, AMetric> = a.metrics_pos[ar.pos] ?? a.metrics;
  const sticks = Math.round(100 * (M.fpoe?.sticks ?? 0.09));
  const last = (cur: number | null, tr: number | null) => (cur != null && tr != null ? cur + tr : null);
  const trends: { k: string; label: string; from: string; to: string; d: string; up: boolean | null }[] = [
    ...(ar.pos !== "QB" ? [
      { k: "snap_tr", label: "Snap share", from: pctTxt(ar.snap), to: pctTxt(last(ar.snap, ar.snap_tr)), d: ar.snap_tr != null ? `${signed(100 * ar.snap_tr)} pts` : "–", up: ar.snap_tr == null ? null : ar.snap_tr > 0 },
      { k: "tsh_tr", label: "Target share", from: pctTxt(ar.tsh), to: pctTxt(last(ar.tsh, ar.tsh_tr)), d: ar.tsh_tr != null ? `${signed(100 * ar.tsh_tr)} pts` : "–", up: ar.tsh_tr == null ? null : ar.tsh_tr > 0 },
    ] : []),
    { k: "xfp_tr", label: "Usage (xFP/g)", from: f1(ar.xfp), to: f1(last(ar.xfp, ar.xfp_tr)), d: signed(ar.xfp_tr), up: ar.xfp_tr == null ? null : ar.xfp_tr > 0 },
    { k: "opp_tr", label: "Targets + carries", from: f1(ar.opp), to: f1(last(ar.opp, ar.opp_tr)), d: signed(ar.opp_tr), up: ar.opp_tr == null ? null : ar.opp_tr > 0 },
    { k: "est_tr", label: `Model value, wk ${a.est_weeks[0] ?? "–"}→${a.est_weeks[1] ?? "–"}`, from: f1(last(ar.est_ppg, ar.est_tr != null ? -ar.est_tr : null)), to: f1(ar.est_ppg), d: signed(ar.est_tr), up: ar.est_tr == null ? null : ar.est_tr > 0 },
  ];
  return (
    <div className="pf-an">
      {ar.flags.length > 0 && <ul className="pf-an-flags">{ar.flags.map((fl) => <li key={fl.key}><FlagChip f={fl} /><Ev l={fl.label} tip={evTip(M[fl.metric])} /><span>{fl.text}</span></li>)}</ul>}
      <div className="pf-an-grid">
        <div className="pf-an-card">
          <h3>Scoring vs usage <EvFor m={M} k="fpoe" /></h3>
          <div className="pf-an-big">
            <span><b>{f1(ar.ppg)}</b><small>scored /g</small></span>
            <span><b>{f1(ar.xfp)}</b><small>usage says</small></span>
            <span className={ar.fpoe == null ? "" : Math.abs(ar.fpoe) < 1 ? "" : ar.fpoe > 0 ? "up" : "down"}><b>{signed(ar.fpoe)}</b><small>gap</small></span>
          </div>
          <p className="pf-cap">Usage says = what his targets, carries and field position usually produce. About {sticks}% of a gap like this has carried forward in past seasons; the model ({f1(ar.est_ppg)} rest of season) already discounts it.</p>
        </div>
        <div className="pf-an-card">
          <h3>Finishes <EvFor m={M} k="start" /></h3>
          <TierStrip w={ar.weekly} pos={ar.pos} />
          <p className="pf-an-ct"><b>{ar.starts}</b> startable · <b>{ar.smashes}</b> smash · <b>{ar.duds}</b> dud <span className="dim">in {ar.games} games</span></p>
          <Spread ar={ar} />
          <p className="pf-cap">Swing {ar.cv != null ? ar.cv.toFixed(2) : "–"} (weekly sd ÷ average){ar.pct.cv != null ? `, ${Math.round(ar.pct.cv)}th percentile among ${ar.pos}s` : ""}. <EvFor m={M} k="cv" /> It repeats moderately and doesn't predict points; it tells you how wide his weeks run.</p>
        </div>
        <div className="pf-an-card">
          <h3>Last 3 games vs the season</h3>
          <table className="pf-an-tr">
            <tbody>{trends.map((t) => (
              <tr key={t.k}><th>{t.label} <EvFor m={M} k={t.k} /></th><td className="num dim">{t.from}</td><td className="ar">→</td><td className="num"><b>{t.to}</b></td>
                <td className={`num d ${t.up == null ? "" : t.k === "snap_tr" || t.k === "est_tr" ? (t.up ? "up" : "down") : ""}`}>{t.d}</td></tr>))}
            </tbody>
          </table>
          <p className="pf-cap">About {Math.round(100 * (M.snap_tr?.sticks ?? 0.49))}% of a snap-share change sticks, and only snap share and the model's value moves have beaten the model out of sample ({signed2(M.snap_tr?.beyond?.v ?? null)} and {signed2(M.est_tr?.beyond?.v ?? null)} ppg per standard deviation): small. Other changes mostly drift back.</p>
        </div>
      </div>
      <h3 className="pf-an-h">Against the position <small>percentile among the top {a.pool[ar.pos]} {ar.pos}s by points per game (3+ games)</small></h3>
      <div className="pf-an-rk">
        {RANKS.filter((c) => !c.skip?.includes(ar.pos)).map((c) => {
          const v = ar.pct[c.k];
          return (
            <div key={c.k} className="rk" title={evTip(M[c.k])}>
              <span className="lab">{c.label} <EvFor m={M} k={c.k} /></span>
              <span className="trk"><i className="mid" />{v != null && <i className="dot" style={{ left: `${v}%`, background: POS_VAR[ar.pos] }} />}</span>
              <span className="num val">{c.v(ar)}</span><span className="num dim pc">{v == null ? "" : `${Math.round(v)}th`}</span>
            </div>
          );
        })}
      </div>
      <TierLegend a={a} pos={[ar.pos]} />
    </div>
  );
}

/** Weekly points on one axis: each game a dot, the middle 60% shaded, his average marked. */
function Spread({ ar }: { ar: APlayer }) {
  const [ref, W] = useWidth<HTMLDivElement>(300);
  const pts = ar.weekly.map((w) => w?.[0]).filter((x): x is number => x != null);
  if (!pts.length) return null;
  const mx = Math.max(25, ...pts) * 1.05, L = 6, R = 6, H = 34, y = 16;
  const x = (v: number) => L + ((W - L - R) * Math.max(0, v)) / mx;
  const ticks = [0, 10, 20, 30, 40].filter((t) => t <= mx);
  return (
    <div ref={ref} className="pf-an-sp">
      <svg width={W} height={H} role="img" aria-label={`Weekly points: ${pts.map((p) => f1(p)).join(", ")}`}>
        {ticks.map((t) => <g key={t}><line className="gl" x1={x(t)} x2={x(t)} y1={4} y2={H - 10} /><text x={x(t)} y={H - 1} textAnchor="middle">{t}</text></g>)}
        {ar.p20 != null && ar.p80 != null && <rect className="band" x={x(ar.p20)} y={y - 7} width={Math.max(2, x(ar.p80) - x(ar.p20))} height={14} rx={3}><title>{`Middle 60% of his weeks: ${f1(ar.p20)}–${f1(ar.p80)}`}</title></rect>}
        {ar.ppg != null && <line className="avg" x1={x(ar.ppg)} x2={x(ar.ppg)} y1={y - 9} y2={y + 9} />}
        {ar.weekly.map((w, i) => (w && w[0] != null ? <circle key={i} className={`d ${w[7]}`} cx={x(w[0])} cy={y} r={4.5}><title>{`Week ${i + 1}: ${f1(w[0])}`}</title></circle> : null))}
      </svg>
    </div>
  );
}

/** Markdown lines for Copy for Claude. */
export function analyticsLines(ar: APlayer | undefined, a: Analytics | null | undefined): string[] {
  if (!ar || !a) return [];
  return ["", `Season analytics through week ${a.through_week} (labels from a 2021–25 backtest):`,
    `- Scored ${f1(ar.ppg)}/g vs ${f1(ar.xfp)} his usage usually produces (${signed(ar.fpoe)}); about ${Math.round(100 * (a.metrics.fpoe?.sticks ?? 0.09))}% of such gaps carry forward.`,
    `- Finishes: ${ar.starts} startable, ${ar.smashes} smash, ${ar.duds} dud in ${ar.games} games; swing ${ar.cv != null ? ar.cv.toFixed(2) : "–"}.`,
    ...(ar.pos !== "QB" ? [`- Snap share last 3 vs season: ${pctTxt(ar.snap != null && ar.snap_tr != null ? ar.snap + ar.snap_tr : null)} vs ${pctTxt(ar.snap)} (small edge beyond the model when it moves).`] : []),
    ...(ar.flags.length ? [`- Flags: ${ar.flags.map((x) => `${x.short} (${x.label})`).join("; ")}`] : [])];
}
