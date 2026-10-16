import { type ReactNode, useLayoutEffect, useRef, useState } from "react";
import type { ALabel, AFlag, AMetric, APlayer, AWeek, Analytics } from "../types";
import { f1, lastName, signed } from "../format";
import "./analytics.css";

/* Shared pieces for the analytics views. Every stat on screen carries the label the 2021–25 backtest
   gave it (backtest/analytics_metrics.py): whether it repeats, whether it predicts later points, and
   whether it knows anything the model doesn't. Noise keeps the shared grey-dashed look. */

export const EV_SHORT: Record<ALabel, string> = {
  adds: "adds", edge: "small edge", tilt: "didn't hold", priced: "in the model", trait: "repeats",
  weak: "weak", noise: "noise", descriptive: "what happened",
};

/** Evidence chip: how much the backtest trusts a stat. */
export function Ev({ l, tip }: { l: ALabel; tip?: string }) {
  return <span className={`ev ev-${l}`} title={tip} aria-label={`${EV_SHORT[l]}${tip ? `: ${tip}` : ""}`}>{EV_SHORT[l]}</span>;
}

export const evTip = (m?: AMetric) => (m ? `${m.name}. ${m.label_text}. ${m.reading}` : undefined);

/** Evidence chip for a stat key, read from the registry. */
export function EvFor({ m, k }: { m?: Record<string, AMetric>; k: string }) {
  const x = m?.[k];
  return x ? <Ev l={x.label} tip={evTip(x)} /> : null;
}

const ARROW: Record<string, string> = { up: "↑", down: "↓", info: "" };
/** Inline flag: hot/cold vs usage (noise, fades), snaps or model value moving (small edge), boom-bust/steady (trait). */
export function FlagChip({ f }: { f: AFlag }) {
  return <span className={`aflag t-${f.tone} l-${f.label}`} title={f.text}>{ARROW[f.tone]}{f.short.replace(/^(Hot|Cold) /, "$1 ")}</span>;
}

export const pctTxt = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
export const TIER_TXT: Record<string, string> = { smash: "Smash", start: "Startable", mid: "Bench-level", dud: "Dud" };

/** One week cell per week: tier fill, finish rank printed; empty when he didn't play. */
export function TierStrip({ w, from = 1, pos }: { w: AWeek[]; from?: number; pos: string }) {
  return (
    <span className="tstrip" role="img" aria-label={`weekly finishes: ${w.map((x, i) => (x ? `week ${from + i} ${TIER_TXT[x[7]]} ${pos}${x[6]}` : `week ${from + i} no game`)).join(", ")}`}>
      {w.map((x, i) => (x
        ? <i key={i} className={`tc ${x[7]}`} title={`Week ${from + i}: ${f1(x[0])} pts, ${pos}${x[6]} (${TIER_TXT[x[7]]})${x[1] != null ? ` · ESPN had ${f1(x[1])}` : ""}${x[2] != null ? ` · usage ${f1(x[2])}` : ""}`}>{x[6]}</i>
        : <i key={i} className="tc none" title={`Week ${from + i}: no game`} />))}
    </span>
  );
}

export function TierLegend({ a, pos }: { a: Analytics; pos?: string[] }) {
  const ps = pos ?? ["QB", "RB", "WR", "TE"];
  const rng = (k: "smash" | "start" | "dud") => ps.map((p) => `${p} ${a.tiers[k][p]}`).join(", ");
  return (
    <span className="tlegend">
      <span><i className="tc smash" />Smash: top {rng("smash")}</span>
      <span><i className="tc start" />Startable: top {rng("start")}</span>
      <span><i className="tc mid" />Bench-level</span>
      <span><i className="tc dud" />Dud: outside {rng("dud")}</span>
    </span>
  );
}

/** Width of an element, kept current, so SVG charts draw at 1:1 (text never scales). */
export function useWidth<T extends HTMLElement>(init = 480) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(init);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(Math.round(el.getBoundingClientRect().width) || init);
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width) || init));
    ro.observe(el);
    return () => ro.disconnect();
  }, [init]);
  return [ref, w] as const;
}

export const signed2 = (x: number | null | undefined) => (x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(2));

/** Usage vs results: hollow ring = points his usage usually produces (xFP), dot = what he scored. */
export function Dumbbell({ rows, onPick, sel }: { rows: APlayer[]; onPick?: (id: number) => void; sel?: number | null }) {
  const [ref, W] = useWidth<HTMLDivElement>(480);
  const R = rows.filter((p) => p.ppg != null && p.xfp != null);
  const L = 104, RT = 52, H = 24, top = 18;
  const mx = Math.max(10, ...R.flatMap((p) => [p.ppg!, p.xfp!])) + 2;
  const x = (v: number) => L + ((W - L - RT) * v) / mx;
  const ticks = Array.from({ length: Math.floor(mx / 5) + 1 }, (_, i) => i * 5);
  const hgt = top + R.length * H + 6;
  return (
    <div ref={ref} className="dumb-w"><svg className="dumb" width={W} height={hgt} viewBox={`0 0 ${W} ${hgt}`} role="img" aria-label="Points per game against what each player's usage usually produces">
      {ticks.map((t) => <g key={t}><line className="grid" x1={x(t)} x2={x(t)} y1={top - 4} y2={hgt - 4} /><text className="tk" x={x(t)} y={10} textAnchor="middle">{t}</text></g>)}
      {R.map((p, i) => {
        const y = top + i * H + H / 2, gap = p.ppg! - p.xfp!;
        const cls = Math.abs(gap) < 1 ? "even" : gap > 0 ? "up" : "down";
        return (
          <g key={p.id} className={`row ${sel === p.id ? "sel" : ""}`} onClick={() => onPick?.(p.id)}>
            <rect className="hit" x={0} y={y - H / 2} width={W} height={H} />
            <text className="nm" x={L - 10} y={y + 4} textAnchor="end">{lastName(p.name)}</text>
            <line className={`link ${cls}`} x1={x(p.xfp!)} x2={x(p.ppg!)} y1={y} y2={y} />
            <circle className="xfp" cx={x(p.xfp!)} cy={y} r={5} />
            <circle className="act" cx={x(p.ppg!)} cy={y} r={5} />
            <text className={`gap ${cls}`} x={W - 4} y={y + 4} textAnchor="end">{signed(gap)}</text>
            <title>{`${p.name}: ${f1(p.ppg)} ppg vs ${f1(p.xfp)} his usage usually produces (${signed(gap)}) over ${p.games} games`}</title>
          </g>
        );
      })}
    </svg></div>
  );
}

/** Percentile cell on a neutral ink ramp (no good/bad meaning), value printed. */
export function PctCell({ v, title }: { v: number | null | undefined; title?: string }) {
  if (v == null) return <td className="pc none" title={title}>–</td>;
  const a = Math.max(0.04, Math.min(1, v / 100));
  return <td className={`pc ${v >= 65 ? "hi" : ""}`} style={{ ["--a" as string]: a }} title={title}>{Math.round(v)}</td>;
}

/** Inspector section: this player's flags with their evidence, usage vs results, finishes. */
export function Signals({ a, p, note }: { a: Analytics | null; p: { id: number; pos: string }; note?: ReactNode }) {
  const r = a?.players.find((x) => x.id === p.id);
  if (!a || !r || !["QB", "RB", "WR", "TE"].includes(p.pos)) return null;
  const m = a.metrics_pos[r.pos] ?? a.metrics;
  return (
    <section className="sig">
      <h6>Signals <small>through week {a.through_week} · labels from 2021–25</small></h6>
      {r.flags.length > 0 && <ul className="sig-l">{r.flags.map((f) => <li key={f.key}><FlagChip f={f} /> <Ev l={f.label} tip={evTip(m[f.metric])} /><span>{f.text}</span></li>)}</ul>}
      <dl className="kv">
        <dt>Scored / usage says <EvFor m={m} k="fpoe" /></dt><dd>{f1(r.ppg)} / {f1(r.xfp)} <span className="dim">({signed(r.fpoe)})</span></dd>
        <dt>Finishes <EvFor m={m} k="start" /></dt><dd><TierStrip w={r.weekly} pos={r.pos} /></dd>
        <dt>Startable weeks</dt><dd>{r.starts} of {r.games}{r.duds ? ` · ${r.duds} dud${r.duds > 1 ? "s" : ""}` : ""}</dd>
        {r.pos !== "QB" && <><dt>Snap share, last 3 <EvFor m={m} k="snap_tr" /></dt><dd>{pctTxt(r.snap != null && r.snap_tr != null ? r.snap + r.snap_tr : null)} <span className="dim">vs {pctTxt(r.snap)}</span></dd></>}
      </dl>
      {note}
    </section>
  );
}
