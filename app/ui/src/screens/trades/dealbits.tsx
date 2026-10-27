/* Pieces the trade finder, the Trade ideas panel and the deal desk share, so a deal reads the
   same wherever it shows up: one valuation (stored per viewer), the same gain / agreement /
   partner / odds cells, filler variants folded into one deal, and three ranks per player. */
import type { FinderOpts, FinderPlayer, FinderRow, Ranks } from "../../types";
import { signed } from "../../format";

export const POS_VAR: Record<string, string> = { QB: "var(--qb)", RB: "var(--rb)", WR: "var(--wr)", TE: "var(--te)" };
export const HZ: Record<FinderOpts["horizon"], string> = { ros: "rest of season", reg: "regular season", playoffs: "playoff weeks",
  title: "season, weighted by title odds (in playoff points)" };
export const HZ_SHORT: Record<FinderOpts["horizon"], string> = { ros: "rest of season", reg: "regular season", playoffs: "playoff weeks", title: "title-weighted" };
export const n0 = (x: number) => { const r = Math.round(x); return `${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(r)}`; };

/* The valuation part of the finder's options, remembered so ideas and the deal desk score the same way. */
export type Valuation = Pick<FinderOpts, "horizon" | "desk_weight" | "their_view" | "min_their" | "sort">;
const VKEY = "desk.trades.valuation";
export const VAL_DEFAULT: Valuation = { horizon: "ros", desk_weight: 0.5, their_view: "espn", min_their: 0, sort: "gain" };
export function loadValuation(): Valuation {
  try { return { ...VAL_DEFAULT, ...JSON.parse(localStorage.getItem(VKEY) ?? "{}") }; } catch { return VAL_DEFAULT; }
}
export function saveValuation(o: FinderOpts) {
  try { localStorage.setItem(VKEY, JSON.stringify({ horizon: o.horizon, desk_weight: o.desk_weight, their_view: o.their_view, min_their: o.min_their, sort: o.sort })); } catch { /* private window */ }
}
export const playoffScale = (h?: string) => h === "playoffs" || h === "title";

/** How the two views read a deal for you: agree, or most of the gain lives in one of them. */
export function read(r: Pick<FinderRow, "your_gain" | "desk_gain" | "espn_gain">, po: boolean): { t: string; tone: string; tip: string } {
  // playoff-week gains are ~1/3 the size of rest-of-season ones, so the cutoffs scale down
  const gap = r.desk_gain - r.espn_gain, small = po ? 6 : 10, big = po ? 5 : 15;
  if (r.your_gain < small) return { t: "Small edge", tone: "dim", tip: `Under ${small} points: in the 2021-25 backtest edges this size had the right sign only a bit more than half the time.` };
  if (gap >= big) return { t: "Mostly our model", tone: "warn", tip: `Desk sees ${n0(gap)} more than ESPN. A bet that our model is right where ESPN isn't; Desk has beaten ESPN only narrowly.` };
  if (gap <= -big) return { t: "Mostly ESPN", tone: "mid", tip: `ESPN sees ${n0(-gap)} more than Desk. Looks great on the app; our model is less sold.` };
  return { t: "Both agree", tone: "good", tip: `Desk and ESPN both have you gaining ${small}+ points, within ${big} of each other.` };
}

/** Partner's side on what their app shows: our only (weak) proxy for whether they'd accept. */
export function theirRead(x: number, po: boolean): { t: string; tone: string } {
  if (x >= (po ? 2 : 5)) return { t: "fair on their app", tone: "good" };
  if (x >= 0) return { t: "barely fair", tone: "mid" };
  return { t: "looks bad to them", tone: "warn" };
}

export const oddsTone = (p: number) => (p >= 0.65 ? "good" : p >= 0.58 ? "mid" : "dim");

/* Three views of a player's rest of season, as position ranks. They split when the sources
   disagree by a lot (6+ ranks and a third of the middle one): worth a look before you deal. */
const SRC: [keyof Ranks, string, string][] = [
  ["fp", "FP", "FantasyPros rest-of-season consensus rank (expert rankings)"],
  ["espn", "ESPN", "Rank by ESPN's projections (what their app's numbers imply; not ESPN's own ROS ranking)"],
  ["desk", "Desk", "Rank by Desk's expected rest-of-season points (our blend, availability and byes)"],
];
export function RankLine({ pos, r }: { pos: string; r?: Ranks }) {
  const xs = SRC.map(([k]) => r?.[k]).filter((x): x is number => x != null);
  if (!xs.length) return null;
  const mid = [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const split = xs.length > 1 && Math.max(...xs) - Math.min(...xs) >= Math.max(6, mid / 3);
  return (
    <span className={`rk3 ${split ? "split" : ""}`} title={`${pos} rank, rest of season (1 = best)${split ? ". The sources disagree on him." : ""}\n${SRC.map(([k, l, t]) => `${l} ${r?.[k] ?? "–"}: ${t}`).join("\n")}`}>
      {SRC.map(([k, l]) => <span key={k}><em>{l}</em>{r?.[k] != null ? `${pos}${r[k]}` : "–"}</span>)}
    </span>
  );
}

export function PlayerChip({ p, name }: { p: FinderPlayer; name: string }) {
  return (
    <span className="fp"><span className="fp1"><i className="pd" style={{ background: POS_VAR[p.pos] }} /><b title={p.name}>{name}</b>
      <small>{p.pos}{p.ppg != null ? ` · ${p.ppg.toFixed(1)} ppg` : ` · ${Math.round(p.ros)} pts`}</small></span>
      <RankLine pos={p.pos} r={p.ranks} /></span>
  );
}

/* Deals that differ only in the filler coming back fold into one: the best-scoring variant leads. */
export interface DealGroup { lead: FinderRow; variants: FinderRow[]; main: FinderPlayer; fillers: FinderPlayer[]; bare: boolean }
export function groupRows(rows: FinderRow[]): DealGroup[] {
  const by = new Map<string, FinderRow[]>();
  rows.forEach((r) => { const k = r.group ?? `${r.partner}:${r.give_ids.join(",")}:${r.get_ids.join(",")}`; by.set(k, [...(by.get(k) ?? []), r]); });
  return [...by.values()].map((vs) => {
    const lead = vs.reduce((a, b) => (b.your_gain > a.your_gain ? b : a));
    const main = lead.get.reduce((a, b) => (b.ros > a.ros ? b : a));
    const fillers = vs.flatMap((v) => v.get.filter((p) => p.id !== main.id));
    return { lead, variants: vs, main, fillers: fillers.filter((p, i) => fillers.findIndex((q) => q.id === p.id) === i), bare: vs.some((v) => v.get.length === 1) };
  });
}
export const range = (xs: number[], f = signed) => {
  const lo = Math.min(...xs), hi = Math.max(...xs);
  return Math.abs(hi - lo) < 0.5 ? f(hi) : `${f(lo)} to ${f(hi)}`;
};
/** "+ 1 filler: A, B (5–9 ppg)" under the main player, or nothing when the deal has none. */
export function fillerText(g: DealGroup): string | null {
  if (!g.fillers.length) return null;
  const ppg = g.fillers.map((p) => p.ppg ?? 0);
  const lo = Math.min(...ppg), hi = Math.max(...ppg);
  const n = Math.max(...g.variants.map((v) => v.get.length)) - 1;
  return `+ ${g.bare ? "optional " : ""}${n} filler${n > 1 ? "s" : ""}: ${g.fillers.map((p) => p.name.split(" ").slice(-1)[0]).join(", ")} (${lo === hi ? lo.toFixed(1) : `${lo.toFixed(1)}–${hi.toFixed(1)}`} ppg)`;
}
