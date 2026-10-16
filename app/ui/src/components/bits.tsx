import { type ReactNode, useState } from "react";
import type { Player, Pos } from "../types";
import { f1, pct } from "../format";

export const POS_VAR: Record<Pos, string> = { QB: "var(--qb)", RB: "var(--rb)", WR: "var(--wr)", TE: "var(--te)", K: "var(--k)", DST: "var(--dst)" };

export function Face({ p, className = "" }: { p: { headshot: string | null; pos: Pos; name: string }; className?: string }) {
  if (!p.headshot) return <span className={`face ${className}`} aria-hidden="true" />;
  return <img className={`face ${p.pos === "DST" ? "logo" : ""} ${className}`} src={p.headshot} alt="" loading="lazy" draggable={false} />;
}

export function statusOf(p: Pick<Player, "game" | "injury" | "espn_week">): { cls: string; label: string } {
  if (p.game?.bye) return { cls: "bye", label: "Bye" };
  const s = (p.injury || "ACTIVE").toUpperCase();
  if (s === "ACTIVE") return { cls: "ok", label: "Active" };
  if (s.includes("RESERVE")) return { cls: "ir", label: "IR" };
  if (s === "OUT" || s === "SUSPENSION") return { cls: "out", label: "Out" };
  if (s === "QUESTIONABLE") return { cls: "q", label: "Questionable" };
  if (s === "DOUBTFUL") return { cls: "out", label: "Doubtful" };
  return { cls: "q", label: s.replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase()) };
}

export function Status({ p }: { p: Pick<Player, "game" | "injury" | "espn_week"> }) {
  const s = statusOf(p);
  return <span className={`status ${s.cls}`}>{s.label}</span>;
}

/** Is this player unable to score this week? */
export const isDead = (p: Pick<Player, "game" | "injury" | "espn_week">) => !!p.game?.bye || ["OUT", "INJURY_RESERVE", "SUSPENSION"].includes(p.injury) || !p.espn_week;

/** Point first; on hover/focus, the decision context passed in `tip` (shared component). */
export function Proj({ p, tip, className = "" }: { p: Player; tip?: ReactNode; className?: string }) {
  const [flip, setFlip] = useState(false);
  if (!p.espn_week) return <span className={`num ${className}`}>0.0</span>;
  if (!tip) return <span className={`num ${className}`}>{f1(p.espn_week)}</span>;
  // Open toward whichever side has room inside the scrolling content area.
  const place = (e: React.SyntheticEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const box = (e.currentTarget.closest(".content") ?? document.body).getBoundingClientRect();
    setFlip(r.right - 290 < box.left);
  };
  return (
    <span className={`rng num ${flip ? "flip" : ""} ${className}`} tabIndex={0} onMouseEnter={place} onFocus={place}>
      {f1(p.espn_week)}
      <span className="tip" role="tooltip">{tip}</span>
    </span>
  );
}

/** Head-to-head against the real alternative: the number that matters for a swap. */
export function H2H({ p, other, slot, role, prob }: { p: Player; other: Player; slot: string; role: "bench" | "starter"; prob: number | null }) {
  const w = prob == null ? null : Math.round(prob * 100);
  return (
    <span className="h2h">
      <b>{w == null ? "No comparison" : w >= 100 ? `${lastNameOf(other.name)} can't score this week` : `Outscores ${lastNameOf(other.name)} in ${w}% of weeks`}</b>
      {w != null && w < 100 && w > 0 && (
        <span className="split" aria-hidden="true"><i style={{ width: `${w}%` }} /><i style={{ width: `${100 - w}%` }} /></span>
      )}
      <span className="sub">{role === "bench" ? `Best bench option for ${slot}: ${other.name}, ${f1(other.espn_week)}` : `Starter he'd replace at ${slot}: ${other.name}, ${f1(other.espn_week)}`}</span>
      <span className="sub">ESPN {f1(p.espn_week)} this week · rest of season {f1(p.est_ppg)} ppg · plays {pct(p.avail_now)}</span>
    </span>
  );
}

const lastNameOf = (n: string) => (n.endsWith("D/ST") ? n : n.split(" ").slice(1).join(" ") || n);

export function Spark({ w, h = 16, bw = 5, color = "currentColor", from }: { w: (number | null)[]; h?: number; bw?: number; color?: string; from?: number }) {
  const mx = Math.max(25, ...w.filter((x): x is number => x != null));
  const W = w.length * (bw + 2);
  return (
    <svg width={W} height={h} viewBox={`0 0 ${W} ${h}`} aria-label={`weekly points ${w.map((x) => (x == null ? "none" : f1(x))).join(", ")}`}>
      {w.map((x, i) =>
        x == null ? <rect key={i} x={i * (bw + 2)} y={h - 1} width={bw} height={1} fill={color} opacity={0.4}>{from != null && <title>{`Week ${from + i}: none`}</title>}</rect>
          : <rect key={i} x={i * (bw + 2)} y={h - Math.max(1.5, (h * x) / mx)} width={bw} height={Math.max(1.5, (h * x) / mx)} rx={1} fill={color}>{from != null && <title>{`Week ${from + i}: ${f1(x)}`}</title>}</rect>)}
    </svg>
  );
}
