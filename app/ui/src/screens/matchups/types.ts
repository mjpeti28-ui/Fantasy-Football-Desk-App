import type { Pos } from "../../types";

export type Shell = "C01" | "C2" | "C3" | "C46" | "OTH";
export const SHELLS: Shell[] = ["C01", "C2", "C3", "C46", "OTH"];
export const SHELL_LABEL: Record<Shell, string> = { C01: "Cover 0/1", C2: "Cover 2", C3: "Cover 3", C46: "Cover 4/6", OTH: "Other" };

export interface MxGrade { r: number; z: number; rank: number; of: number; label: "Tough" | "Middle" | "Soft"; w_now: number; plays: number }
export interface MxTeamBrief { man: number | null; blitz: number | null; epa_db: number | null; run_epa: number | null; games: number; pressure: number | null; pass_grade?: MxGrade; run_grade?: MxGrade }
export interface MxLeagueSeason { man: number; blitz: number; pressure: number; pa: number; pr5: number; shell: Record<Shell, number> }
export interface MxLeague {
  man: number; blitz: number; pa: number; pressure: number; pr5: number; ttt: number; shell: Record<Shell, number>;
  base: { man: number; shell: Record<Shell, number>; season: number };
  cur: { blitz: number; pa: number; pr5: number; epa_db: number; success: number; sack: number; comp: number; n: number };
  cur_run: { epa: number; success: number; ypc: number; explosive: number; n: number };
  [season: string]: unknown;
}

export interface MxBrief {
  pos: Pos;
  // receivers and backs
  games?: number; tgt?: number; share?: number; adot?: number; epa_t?: number;
  tr_man?: number | null; tr_zone?: number | null; diff?: number | null; diff_shrunk?: number | null; rel?: number; hist_db?: number;
  // quarterbacks
  db?: number; epa_db?: number; pa?: number; epa_man?: number; epa_zone?: number;
  n_man?: number; n_zone?: number;
}
export interface MxLens {
  opp: string; opp_man_exp: number; league_man: number; d_targets: number; d_ppr: number; tr_rel: number; ppr_rel: number; game_sd: number | null; tgt_game: number;
}
export interface MxRow {
  id: number; gsis: string | null; name: string; pos: Pos; team: string | null; opp: string | null; home: boolean | null; day: string | null; date: string | null;
  time: string | null; bye: boolean; slot: string; side: "me" | "opp"; injury: string | null; headshot: string | null; brief: MxBrief | null; lens: MxLens | null;
}
export interface MxBoard {
  season: number; hist: number[]; teams: Record<string, MxTeamBrief>; league: MxLeague; attribution: string; alias: Record<string, string>;
  week: number; schedule: Record<string, { opp: string; home: boolean; date: string; time: string | null }>;
  mine: MxRow[]; theirs: MxRow[]; opp_name: string | null; team_name: string | null;
}

export interface MxHistRow {
  season: number; n: number; man: number; shell: Record<Shell, number>; blitz: number; pr5: number; pressure: number; pa: number; ttt: number; epa_db: number; success: number;
}
export interface MxAllowed { tgt: number; share: number; league_share: number; ppr_t: number; league_ppr_t: number; raw_ppr_t: number | null }
export interface MxDepth { tgt: number; epa_t: number; league_epa_t: number; comp: number; league_comp: number }
export interface MxDefense {
  abbr: string; history: MxHistRow[]; expected: { man: number; shell: Record<Shell, number> } | null; persist: Record<string, number>;
  current: { games: number; n: number; epa_db: number; success: number; comp: number; ypa: number; sack: number; hit: number; blitz: number; pr5: number; pa: number; screen: number; blitz_exp: number } | null;
  run: { n: number; epa: number; success: number; ypc: number; explosive: number } | null;
  allowed: Record<"RB" | "WR" | "TE", MxAllowed>; depth: Record<"behind" | "short" | "mid" | "deep", MxDepth>;
  faced: { gsis: string; name: string; pos: string; tgt: number; rec: number; yds: number; td: number; ppr: number }[];
  league: MxLeague; shell_label: Record<Shell, string>;
}
export interface MxPlayMaker { gsis: string; name: string; pos: string; sacks: number; hit: number; tfl: number; int: number; pbu: number; tkl: number; ff: number; still?: boolean | null }
export interface MxOnOff { gsis: string; name: string; pos: string | null; on: number; off: number; pct: number; on_epa: number; off_epa: number | null; se: number | null; still: boolean | null }
export interface MxDefenders { abbr: string; current: MxPlayMaker[]; last: MxPlayMaker[]; onoff?: MxOnOff[]; onoff_dropbacks?: number; onoff_season?: number }
export interface MxDefenseView { defense: MxDefense; defenders: MxDefenders; pff?: MxPffDefense | null }

/* PFF matchup context (matchups_pff.py). Descriptive only: none of it beat ESPN's number in backtests. */
export interface MxPffUnit { grade: number; rank: number; of: number }
export type MxPffUnits = Partial<Record<"coverage" | "pass_rush" | "run_defense" | "pass_block" | "run_block", MxPffUnit>>;
export interface MxPffDef {
  pff_id: number; name: string; pos: string; games: number; corner: number; slot: number; box: number; cov_snaps: number; cov_grade: number | null;
  tgt: number; rec: number; yds: number; td: number; int: number; yds_cov: number | null; rush_snaps: number; rush_grade: number | null; pressures: number;
  sacks: number | null; prp: number | null; run_snaps: number; run_grade: number | null; stops: number; share: number | null;
}
export interface MxPffDefense { abbr: string; season: number; through: number | null; units: MxPffUnits; outside: MxPffDef[]; slot: MxPffDef[]; rushers: MxPffDef[]; run: MxPffDef[] }
export interface MxPffLine { name: string; at: string; snaps: number; pass_grade: number | null; run_grade: number | null; pressures: number; sacks: number }
export interface MxPffPlayer {
  opp: string; season: number; through: number | null; pos: Pos; them: MxPffUnits; us: MxPffUnits; line: MxPffLine[];
  align?: { slot: number; wide: number; inline: number; routes: number } | null; likely?: "slot" | "outside" | "inline";
  outside?: MxPffDef[]; slot?: MxPffDef[]; rushers?: MxPffDef[]; run?: MxPffDef[];
  covered_by?: { name: string | null; team: string | null; pos: string | null; tgt: number; rec: number; yds: number; td: number; theirs: boolean }[];
}

export interface MxSplit { db: number; tgt: number; tr: number | null; ypt: number | null; ppr_t: number | null; epa_t: number | null; comp: number | null }
export interface MxBase {
  adot: number; epa_t: number; catch_rate: number; catchable: number; pa_share: number; screen_share: number; blitz_share: number;
  depth: Record<"behind" | "short" | "mid" | "deep", number>; loc: Record<"left" | "middle" | "right", number>;
}
export interface MxCurrent {
  team: string; games: number; tgt: number; team_tgt: number; share: number; rec: number; yds: number; td: number; adot: number; ppr: number;
  depth: Record<"behind" | "short" | "mid" | "deep", number>; loc: Record<"left" | "middle" | "right", number>;
  pa_share: number; team_pa_share: number; screen_share: number; team_screen_share: number; blitz_share: number; team_blitz_share: number;
  contested: number; drops: number; catchable: number; catch_rate: number; epa_t: number; base: MxBase | null;
}
export interface MxRoute { route: string; label: string; n: number; share: number; league_share: number; ppr_t: number; epa_t: number }
export interface MxHist {
  seasons: Record<string, number>; total: MxSplit; man: MxSplit; zone: MxSplit; tr_diff: number | null; tr_diff_shrunk: number | null; tr_rel: number;
  ppr_diff: number | null; ppr_diff_shrunk: number | null; ppr_rel: number; shells: Record<Shell, MxSplit>;
  other: { blitz: MxSplit; no_blitz: MxSplit; pa: MxSplit; no_pa: MxSplit }; routes: MxRoute[]; pos_mean_tr: { man: number; zone: number };
}
export interface MxQbSplit { db: number; epa_db: number; se: number | null; comp: number | null; ypa: number | null; sack: number; pressure: number }
export interface MxQbCurrent { db: number; epa_db: number; success: number; comp: number; ypa: number; sack: number; adot: number; pa: number; blitz_epa: number | null; noblitz_epa: number | null; blitz_n: number; td: number; int: number }
export interface MxQbHist {
  total: MxQbSplit; man: MxQbSplit; zone: MxQbSplit; shells: Record<Shell, MxQbSplit>; blitz: MxQbSplit; no_blitz: MxQbSplit; pa: MxQbSplit; no_pa: MxQbSplit; seasons: Record<string, number>;
}
export type MxTag = "priced" | "small" | "context" | "role";
export interface MxCardRow { key: string; label: string; him: string | null; them: string | null; effect: string | null; tag: MxTag; tip: string }
export interface MxCardPart { kind: "pass" | "run"; pts: number; per_sd: number; ci: [number, number]; pct?: number | null; pct_ci?: [number, number] | null; left: number; left_ci: [number, number]; z: number }
export interface MxCard {
  espn_id: number; name: string; pos: Pos; team: string | null; opp: string | null; week: number; season: number; dead: boolean; miss_sd: number | null;
  through?: boolean; sched?: string | null; // a defense he is not playing this week ("Look through"); sched = who he really plays
  proj: { espn: number | null; model: number | null; injury: string | null; bye: boolean; listed: boolean };
  defense?: { abbr: string; pass: MxGrade; run: MxGrade };
  priced: { pts: number; parts: MxCardPart[] } | null;
  rows: MxCardRow[]; role: { tier: string | null; style: string | null; trend: string | null; trend_text: string | null } | null;
  bottom: { tone: "go" | "out" | "bye" | "whatif" | "none"; headline: string; lines: string[] };
}
export interface MxPlayer {
  card?: MxCard;
  gsis: string; name: string; pos: Pos; espn_id: number; team: string | null; opp: string | null;
  game: { opp: string; home: boolean; date: string; time: string | null } | null;
  current: MxCurrent | MxQbCurrent | null; hist: MxHist | MxQbHist | null; lens?: MxLens | null;
  rush?: { car: number; yds: number; ypc: number; epa: number; success: number; explosive: number; td: number };
  pff?: MxPffPlayer | null;
  compare?: MxCompare;
}
/** Short rows for the Compare screen. tag: priced = inside ESPN's number, small = real but tiny, context = descriptive, tested null. */
export interface MxCompare {
  opp: string | null; read: string | null; error?: string;
  rows: { key: string; label: string; text: string; sub: string | null; tag: "priced" | "small" | "context"; tip: string | null }[];
}
export const isQb = (p: MxPlayer): p is MxPlayer & { current: MxQbCurrent | null; hist: MxQbHist | null } => p.pos === "QB";
