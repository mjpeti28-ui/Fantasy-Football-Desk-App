import type { ALabel, APlayer, Pos, RoleProfile, RolesBoard } from "../../types";
import { f1, signed } from "../../format";

/* The Lab's stat catalog: the season analytics (usage, scoring, weekly shape, trends), the PFF alignment stats,
   and every fingerprint metric the player file computes (raw, or noise-adjusted toward the position average). */

export type Fmt = "num" | "num2" | "pct" | "pp" | "sgn" | "rank";
export interface Stat {
  k: string; label: string; short: string; fmt: Fmt; group: string;
  get: (p: APlayer, r: RoleProfile | undefined, adj: boolean) => number | null;
  ev?: string; fixed?: ALabel; note?: string; fp?: boolean;
}
const pctOf = (x: number | null) => (x == null ? null : 100 * x);

const BASE: Stat[] = [
  { k: "ppg", label: "Points per game", short: "Pts/g", fmt: "num", group: "Scoring", get: (p) => p.ppg, ev: "ppg" },
  { k: "xfp", label: "Expected points per game (usage)", short: "Usage xFP", fmt: "num", group: "Scoring", get: (p) => p.xfp, ev: "xfp" },
  { k: "fpoe", label: "Scored over usage, per game", short: "Over usage", fmt: "sgn", group: "Scoring", get: (p) => p.fpoe, ev: "fpoe" },
  { k: "est_ppg", label: "Model, rest of season (ppg)", short: "Model ROS", fmt: "num", group: "Scoring", get: (p) => p.est_ppg, fixed: "descriptive", note: "The model's own number: what everything else is measured against." },
  { k: "resid", label: "Points vs ESPN's projection, per game", short: "vs ESPN", fmt: "sgn", group: "Scoring", get: (p) => p.resid, ev: "resid" },
  { k: "opp", label: "Targets + carries per game", short: "Opps/g", fmt: "num", group: "Usage", get: (p) => p.opp, ev: "opp" },
  { k: "snap", label: "Snap share", short: "Snap %", fmt: "pct", group: "Usage", get: (p) => pctOf(p.snap), ev: "snap" },
  { k: "tsh", label: "Target share", short: "Tgt share", fmt: "pct", group: "Usage", get: (p) => pctOf(p.tsh), ev: "tsh" },
  { k: "ays", label: "Air-yards share", short: "Air share", fmt: "pct", group: "Usage", get: (p) => pctOf(p.ays), ev: "ays" },
  { k: "wopr", label: "WOPR", short: "WOPR", fmt: "num2", group: "Usage", get: (p) => p.wopr, ev: "wopr" },
  { k: "csh", label: "Team carry share", short: "Carry share", fmt: "pct", group: "Usage", get: (p) => pctOf(p.csh), ev: "csh" },
  { k: "start", label: "Startable-week rate", short: "Startable", fmt: "pct", group: "Weekly shape", get: (p) => pctOf(p.start), ev: "start" },
  { k: "smash", label: "Smash-week rate", short: "Smash", fmt: "pct", group: "Weekly shape", get: (p) => pctOf(p.smash), ev: "smash" },
  { k: "dud", label: "Dud-week rate", short: "Dud", fmt: "pct", group: "Weekly shape", get: (p) => pctOf(p.dud), ev: "dud" },
  { k: "cv", label: "Swing (weekly sd ÷ average)", short: "Swing", fmt: "num2", group: "Weekly shape", get: (p) => p.cv, ev: "cv" },
  { k: "snap_tr", label: "Snap share, last 3 minus season (pts)", short: "Snap trend", fmt: "pp", group: "Trend", get: (p) => pctOf(p.snap_tr), ev: "snap_tr" },
  { k: "xfp_tr", label: "Usage xFP, last 3 minus season", short: "Usage trend", fmt: "sgn", group: "Trend", get: (p) => p.xfp_tr, ev: "xfp_tr" },
  { k: "est_tr", label: "Model value, 3-week change", short: "Value move", fmt: "sgn", group: "Trend", get: (p) => p.est_tr, ev: "est_tr" },
  { k: "route_sh", label: "Route share (PFF)", short: "Route share", fmt: "pct", group: "PFF", get: (_p, r) => r?.route_sh?.v ?? null, fixed: "descriptive", note: "Routes run ÷ his team's dropbacks, WR/TE. The one PFF stat that held up in 2021–25 backtests, mainly for players with 3 or fewer games; not a model input." },
  { k: "slot", label: "Slot rate (PFF), % of pass snaps", short: "Slot %", fmt: "pct", group: "PFF", get: (_p, r) => r?.pff?.slot ?? null, fixed: "descriptive", note: "WR/TE alignment. Describes the role; it didn't improve the model." },
  { k: "inline", label: "Inline rate (PFF), % of pass snaps", short: "Inline %", fmt: "pct", group: "PFF", get: (_p, r) => r?.pff?.inline ?? null, fixed: "descriptive", note: "Mostly a TE stat: in-line next to the tackle vs flexed out." },
  { k: "adot", label: "Average depth of target (PFF), yards", short: "aDOT (PFF)", fmt: "num", group: "PFF", get: (_p, r) => r?.pff?.adot ?? null, fixed: "descriptive", note: "How far downfield his targets go. Describes the role, not a predictor." },
  { k: "yprr", label: "Yards per route run (PFF)", short: "YPRR", fmt: "num2", group: "PFF", get: (_p, r) => r?.pff?.yprr ?? null, fixed: "noise", note: "Efficiency. In 2021–25 backtests PFF efficiency stats added nothing over usage." },
  { k: "ecr", label: "Experts' rank at position", short: "ECR", fmt: "rank", group: "Market", get: (p) => p.ecr_rank, fixed: "descriptive", note: "FantasyPros consensus; the model already blends it in." },
];

// fingerprint metrics already covered above by the season analytics
const FP_SKIP = new Set(["snap", "tgt_sh", "ay_sh", "ppr_g", "car_sh"]);
const FP_SHORT: Record<string, string> = {
  opp_sh: "Opp share", rec_mix: "Catch mix", pd_sh: "Passing-down tgts", rz_car_sh: "RZ carries", gl_car_sh: "Goal-line carries",
  td_g: "TDs/g", ypc: "Yds/carry", ld_tgt_sh: "Late-down tgts", adot: "aDOT", deep_r: "Deep tgts", behind_r: "Behind-line tgts",
  rz_tgt_sh: "RZ targets", yac_rec: "YAC/catch", catch_r: "Catch rate", ypt: "Yds/target", cont_r: "Contested", tgt_per_snap: "Tgts/snap",
  des_rush_g: "Designed runs", scr_rate: "Scramble rate", p_adot: "Throw depth", p_deep: "Deep throws", sack_r: "Sack rate", ypa: "Yds/attempt",
};

/** The full catalog for this roles board (fingerprint stats appear only once the engine sends them). */
export function catalog(roles?: RolesBoard | null): Stat[] {
  const meta = roles?.fp_meta ?? {};
  const seen = new Map<string, { label: string; fmt: string; group: string; stab: [Pos, number][] }>();
  for (const [pos, ms] of Object.entries(meta)) {
    for (const [k, m] of Object.entries(ms)) {
      if (FP_SKIP.has(k)) continue;
      const s = seen.get(k) ?? { label: m.label, fmt: m.fmt, group: m.group, stab: [] };
      s.stab.push([pos as Pos, m.stab]);
      seen.set(k, s);
    }
  }
  const fps: Stat[] = [...seen.entries()].map(([k, m]) => {
    const pct = m.fmt === "pct";
    const half = m.stab.map(([p, g]) => `${p} ~${g >= 20 ? "20+" : Math.round(g)}`).join(", ");
    return {
      k: `fp_${k}`, label: m.label, short: FP_SHORT[k] ?? m.label, fmt: pct || m.fmt === "pct100" ? "pct" : m.fmt === "num2" ? "num2" : "num", group: m.group, fp: true,
      get: (_p, r, adj) => { const v = r?.fp?.[k]?.[adj ? 1 : 0]; return v == null ? null : pct ? 100 * v : v; },
      fixed: m.group === "Efficiency" ? "noise" : "descriptive",
      note: `From play-by-play (player file fingerprint). Games until half of it is signal: ${half}. ${m.group === "Efficiency" ? "Efficiency: mostly noise this early in a season." : "Describes the role; not a model input."}`,
    };
  });
  return [...BASE, ...fps];
}

export const GROUP_ORDER = ["Scoring", "Usage", "Role", "Style", "Weekly shape", "Trend", "PFF", "Efficiency", "Market"];

export const fmt = (s: Stat, v: number | null) => (v == null || !Number.isFinite(v) ? "–" : s.fmt === "pct" ? `${v.toFixed(0)}%` : s.fmt === "pp" ? `${signed(v).replace(".0", "")} pts`
  : s.fmt === "sgn" ? signed(v) : s.fmt === "rank" ? `#${Math.round(v)}` : s.fmt === "num2" ? v.toFixed(2) : f1(v));

/** Presets: a pair for the scatter, a stat set for clustering. Keys missing from the catalog or the population are skipped. */
export const PRESETS: { label: string; x: string; y: string; pos?: "QB" | "RB" | "WR" | "TE"; why: string }[] = [
  { label: "Luck vs role", x: "xfp", y: "ppg", why: "Above the line = scoring more than his usage usually produces. About 9% of that has carried forward." },
  { label: "Season vs model", x: "ppg", y: "est_ppg", why: "Below the line = the model expects less than he has scored. Mostly efficiency it expects to fade." },
  { label: "Snaps moving", x: "snap_tr", y: "est_tr", why: "The two small edges: snap-share trend and model value moves, each worth a fraction of a point a game." },
  { label: "Volume vs swing", x: "opp", y: "cv", why: "Swing falls as volume rises. Swing repeats a little but doesn't predict points." },
  { label: "Routes vs targets", x: "route_sh", y: "tsh", pos: "WR", why: "WR/TE. Right and low = running routes without being targeted yet; left and high = targets on a part-time role. Descriptive." },
  { label: "Deep vs reliable", x: "fp_adot", y: "fp_catch_r", pos: "WR", why: "Depth costs catch rate. Off the line = catching more (or less) than his depth predicts; catch rate takes ~17 games to be half signal." },
  { label: "Red zone → TDs", x: "fp_rz_tgt_sh", y: "fp_td_g", pos: "WR", why: "Do touchdowns follow red-zone share? Far above the line = TDs his red-zone role doesn't explain yet." },
  { label: "Pass-catching backs", x: "fp_rec_mix", y: "ppg", pos: "RB", why: "RB. How much of his work is catches vs points per game." },
  { label: "Goal-line role", x: "fp_gl_car_sh", y: "fp_td_g", pos: "RB", why: "RB. Goal-line carry share against TDs per game." },
  { label: "QB legs", x: "fp_des_rush_g", y: "ppg", pos: "QB", why: "QB. Designed runs per game against points per game: rushing is the QB cheat code." },
];

export const CLUSTER_SETS: Record<string, { label: string; keys: string[]; why: string }[]> = {
  WR: [
    { label: "Route style", keys: ["fp_adot", "fp_deep_r", "fp_behind_r", "slot", "fp_cont_r", "fp_yac_rec"], why: "How he's used, not how much: depth, screens, slot, contested balls, YAC." },
    { label: "Role size", keys: ["snap", "tsh", "ays", "route_sh", "fp_ld_tgt_sh", "fp_rz_tgt_sh"], why: "How big the job is. Expect tiers more than types." },
    { label: "Weekly shape", keys: ["start", "smash", "dud", "cv"], why: "Steady vs boom-bust. Descriptive: shape repeats a little but doesn't predict points." },
  ],
  TE: [
    { label: "TE type", keys: ["inline", "slot", "fp_adot", "fp_tgt_per_snap", "fp_behind_r", "route_sh"], why: "Blocker, move tight end or big slot: alignment plus how often he's a target." },
    { label: "Role size", keys: ["snap", "tsh", "route_sh", "fp_rz_tgt_sh", "fp_ld_tgt_sh"], why: "How big the job is." },
  ],
  RB: [
    { label: "Back type", keys: ["fp_rec_mix", "fp_pd_sh", "csh", "fp_rz_car_sh", "fp_gl_car_sh", "tsh"], why: "Early-down grinder, passing-down back, goal-line hammer or three-down workhorse." },
    { label: "Role size", keys: ["snap", "opp", "csh", "tsh", "fp_opp_sh"], why: "How big the job is." },
  ],
  QB: [
    { label: "QB style", keys: ["fp_des_rush_g", "fp_scr_rate", "fp_p_adot", "fp_p_deep", "fp_sack_r"], why: "Runner vs pocket passer, gunslinger vs check-down." },
  ],
  All: [
    { label: "Usage", keys: ["snap", "opp", "tsh", "csh", "ays"], why: "Across positions this mostly splits by position, which is a check that the method works more than a finding." },
    { label: "Weekly shape", keys: ["start", "smash", "dud", "cv"], why: "Steady vs boom-bust, across positions." },
  ],
};
