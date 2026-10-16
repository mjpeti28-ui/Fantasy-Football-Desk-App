import type { ReactNode } from "react";
import type { Shell } from "./types";
import { SHELLS, SHELL_LABEL } from "./types";

/* Small drawing parts for the Matchups screen. Every bar is a "tape": a fill from zero, an optional
   tick for the reference (the position's norm, the league, or his own overall rate), a value, and the
   sample behind it. A dashed outline marks a number resting on too little to lean on. */

export const lastOf = <T,>(a: T[] | undefined): T | undefined => (a && a.length ? a[a.length - 1] : undefined);
export const p0 = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? "–" : `${Math.round(x * 100)}%`);
export const p1 = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(1)}%`);
const sgn = (x: number, shown: string) => (Number(shown) === 0 ? "" : x > 0 ? "+" : "−");
export const pp = (x: number | null | undefined, d = 1) => { if (x == null || Number.isNaN(x)) return "–"; const t = Math.abs(x * 100).toFixed(d); return `${sgn(x, t)}${t}`; };
export const sg2 = (x: number | null | undefined, d = 2) => { if (x == null || Number.isNaN(x)) return "–"; const t = Math.abs(x).toFixed(d); return `${sgn(x, t)}${t}`; };
export const n2 = (x: number | null | undefined, d = 2) => (x == null || Number.isNaN(x) ? "–" : (x < 0 ? "−" : "") + Math.abs(x).toFixed(d));

type Tone = "ink" | "you" | "them";
const TONE: Record<Tone, string> = { ink: "var(--ink-2)", you: "var(--you)", them: "var(--them)" };

/** label · track(fill, tick) · value · note */
export function Meter({ label, v, tick, max, tone = "ink", thin = false, val, note, aux, sub, title }: {
  label: ReactNode; v: number | null; tick?: number | null; max: number; tone?: Tone; thin?: boolean; val?: ReactNode; note?: ReactNode; aux?: ReactNode; sub?: ReactNode; title?: string;
}) {
  const w = v == null ? 0 : Math.max(0, Math.min(1, v / max));
  const r = tick == null ? null : Math.max(0, Math.min(1, tick / max));
  return (
    <div className={`mx-meter ${thin ? "thin" : ""} ${aux !== undefined ? "aux" : ""}`} title={title}>
      <span className="ml">{label}{sub && <small>{sub}</small>}</span>
      <span className="trk"><i style={{ width: `${w * 100}%`, background: TONE[tone] }} />{r != null && <b style={{ left: `${r * 100}%` }} />}</span>
      <span className="mv num">{val ?? p1(v)}</span>
      <span className="mn num">{note}</span>
      {aux !== undefined && <span className="ma num">{aux}</span>}
    </div>
  );
}

/** A value on a centered axis (0 in the middle): fill goes left or right, whisker shows ± one standard error. */
export function Centered({ label, v, se, span, tone = "ink", thin = false, val, note, aux, sub }: {
  label: ReactNode; v: number | null; se?: number | null; span: number; tone?: Tone; thin?: boolean; val?: ReactNode; note?: ReactNode; aux?: ReactNode; sub?: ReactNode;
}) {
  const c = (x: number) => 50 + (Math.max(-span, Math.min(span, x)) / span) * 50;
  const a = v == null ? 50 : c(v);
  return (
    <div className={`mx-meter cen ${thin ? "thin" : ""} ${aux !== undefined ? "aux" : ""}`}>
      <span className="ml">{label}{sub && <small>{sub}</small>}</span>
      <span className="trk">
        <i style={{ left: `${Math.min(50, a)}%`, width: `${Math.abs(a - 50)}%`, background: TONE[tone] }} />
        {v != null && se != null && <u style={{ left: `${c(v - se)}%`, width: `${Math.max(1, c(v + se) - c(v - se))}%` }} />}
        <b style={{ left: "50%" }} />
      </span>
      <span className="mv num">{val ?? sg2(v)}</span>
      <span className="mn num">{note}</span>
      {aux !== undefined && <span className="ma num">{aux}</span>}
    </div>
  );
}

const RAMP = ["var(--mx-s1)", "var(--mx-s2)", "var(--mx-s3)", "var(--mx-s4)", "var(--mx-s5)"];

/** A mix that sums to 100% (coverage shells, depth of target, location): one stacked bar. */
export function Stack({ label, parts, sub }: { label: ReactNode; parts: { k: string; name: string; v: number }[]; sub?: ReactNode }) {
  const tot = parts.reduce((a, b) => a + (b.v || 0), 0) || 1;
  return (
    <div className="mx-stack">
      <span className="ml">{label}{sub && <small>{sub}</small>}</span>
      <span className="bar" role="img" aria-label={parts.map((p) => `${p.name} ${Math.round((p.v / tot) * 100)}%`).join(", ")}>
        {parts.map((p, i) => (
          <i key={p.k} style={{ width: `${(p.v / tot) * 100}%`, background: RAMP[i % RAMP.length] }} title={`${p.name} ${Math.round((p.v / tot) * 100)}%`}>
            {p.v / tot >= 0.09 && <em>{Math.round((p.v / tot) * 100)}</em>}
          </i>
        ))}
      </span>
    </div>
  );
}
export function StackKey({ names }: { names: string[] }) {
  return <div className="mx-skey">{names.map((n, i) => <span key={n}><i style={{ background: RAMP[i % RAMP.length] }} />{n}</span>)}</div>;
}
export const shellParts = (m: Record<Shell, number>) => SHELLS.map((s) => ({ k: s, name: SHELL_LABEL[s], v: m[s] ?? 0 }));

/** "how much of this gap is signal" as a small gauge: 0–100%, dashed when weak. */
export function Signal({ rel, label = "signal" }: { rel: number | null | undefined; label?: string }) {
  const r = rel == null ? 0 : Math.max(0, Math.min(1, rel));
  const weak = r < 0.35;
  return (
    <span className={`mx-sig ${weak ? "weak" : ""}`} title="Share of a gap this size (between players, same sample) that is real rather than luck, measured on 2023–25">
      <span className="g"><i style={{ width: `${r * 100}%` }} /></span>{Math.round(r * 100)}% {label}
    </span>
  );
}

export function Tile({ k, v, sub }: { k: string; v: ReactNode; sub?: ReactNode }) {
  return <div className="mx-tile"><small>{k}</small><b className="num">{v}</b>{sub != null && <span className="num">{sub}</span>}</div>;
}

export function Sec({ title, aside, children, id }: { title: string; aside?: ReactNode; children: ReactNode; id?: string }) {
  return <section className="mx-sec" id={id}><h5>{title}{aside != null && <span>{aside}</span>}</h5>{children}</section>;
}

/** Man rate by season for a defense against the league's own number each season, plus where 2026 is expected. */
export function ManLine({ hist, league, expected, base }: {
  hist: { season: number; man: number }[]; league: Record<string, number>; expected: number | null; base: number;
}) {
  const W = 300, H = 92, L = 34, R = 46, T = 10, B = 18;
  const xs = [...hist.map((h) => h.season), (lastOf(hist)?.season ?? 2025) + 1];
  const vals = [...hist.map((h) => h.man), ...hist.map((h) => league[String(h.season)]), expected ?? base, base];
  const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.05) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.05) * 20) / 20);
  const sx = (i: number) => L + (i / (xs.length - 1)) * (W - L - R);
  const sy = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const team = hist.map((h, i) => `${i ? "L" : "M"}${sx(i)},${sy(h.man)}`).join(" ");
  const lg = [...hist.map((h) => league[String(h.season)]), base].map((v, i) => `${i ? "L" : "M"}${sx(i)},${sy(v)}`).join(" ");
  const last = hist.length - 1;
  return (
    <svg className="mx-line" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Man coverage rate by season against the league">
      {[lo, (lo + hi) / 2, hi].map((v) => <g key={v}><line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} className="grid" /><text x={L - 5} y={sy(v) + 3} textAnchor="end">{Math.round(v * 100)}%</text></g>)}
      <path d={lg} className="lg" />
      <path d={team} className="tm" />
      {expected != null && last >= 0 && <path d={`M${sx(last)},${sy(hist[last].man)} L${sx(last + 1)},${sy(expected)}`} className="tm exp" />}
      {hist.map((h, i) => <circle key={h.season} cx={sx(i)} cy={sy(h.man)} r="3.2" className="pt" />)}
      {expected != null && <circle cx={sx(xs.length - 1)} cy={sy(expected)} r="4" className="pt exp" />}
      {xs.map((x, i) => <text key={x} x={sx(i)} y={H - 4} textAnchor="middle">{i === xs.length - 1 ? `${x}e` : String(x).slice(2)}</text>)}
      {expected != null && <text x={sx(xs.length - 1) + 7} y={sy(expected) + 3.5} className="lab">{Math.round(expected * 100)}%</text>}
      <text x={sx(xs.length - 1) + 7} y={sy(base) + (expected != null && Math.abs(sy(base) - sy(expected)) < 12 ? (base >= expected ? -4 : 13) : 3.5)} className="lab lgl">lg {Math.round(base * 100)}%</text>
    </svg>
  );
}
