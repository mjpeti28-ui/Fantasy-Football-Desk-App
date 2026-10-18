export interface Game {
  bye: boolean;
  opp?: string;
  home?: boolean;
  day?: string;
  date?: string;
  time?: string;
  implied_total?: number | null;
  game_total?: number | null;
}

export type Pos = "QB" | "RB" | "WR" | "TE" | "K" | "DST";

export interface Player {
  id: number;
  name: string;
  pos: Pos;
  team: string | null;
  headshot: string | null;
  slot: string;
  injury: string;
  espn_week: number | null;
  week_range: [number, number] | null;
  est_ppg: number | null;
  avail_now: number | null;
  avail_ros: number | null;
  ecr_rank: number | null;
  status: string | null;
  bye: number | null;
  game: Game | null;
  weekly: (number | null)[];
  ppg: number | null;
  games: number;
}

export interface OppStarter {
  id: number;
  name: string;
  pos: Pos;
  team: string | null;
  headshot: string | null;
  espn_week: number | null;
  slot: string;
  game: Game | null;
  injury: string;
}

export interface Matchup {
  opp_id: number;
  opp_name: string;
  my_proj: number;
  opp_proj: number;
  p_win: number;
  margin_sd: number;
  opp_best?: number;
  p_win_best?: number;
  opp_holes?: string[];
  opp_starters: OppStarter[];
}

export interface TeamWeek {
  season: number;
  week: number;
  team_id: number;
  team_name: string;
  players: Player[];
  lineup: {
    current: Record<string, number>;
    suggested: Record<string, number>;
    current_pts: number;
    suggested_pts: number;
    swaps: { start: string; bench: string; gain: number }[];
  };
  matchup: Matchup | null;
  notes: Record<string, string>;
}

export interface Odds { seed1: number | null; top2: number | null; playoffs: number | null; title: number | null }

export interface PowerTeam {
  id: number;
  name: string;
  record: { w: number; l: number; t: number; pa: number };
  pf: number;
  pa: number;
  weekly: number[];
  weekly_proj: (number | null)[];
  proj_pf: number | null;
  luck: number | null;
  ros_value: number | null;
  proj_final_pts: number | null;
  odds: Odds | null;
  odds_history: ({ week: number } & Odds)[];
  pf_rank: number;
  proj_rank: number;
}

export interface Power {
  season: number;
  through_week: number;
  weeks_played: number;
  odds_as_of_week: number;
  teams: PowerTeam[];
  notes: Record<string, string>;
}

export interface Snapshot {
  generated_at: number;
  season: number;
  week: number;
  data_age: Record<string, number | null>;
  team_week: TeamWeek;
  power: Power;
  /** per-position quantiles of actual/forecast points, used for head-to-head odds */
  ratio_q?: Record<string, number[]>;
  team_colors?: Record<string, { name: string; color: string; alt: string }>;
  engine_version?: number;
}

export interface LiveOdds { seed1: number; top2: number; playoffs: number; title: number; reg_pts: number }
export interface RosterRow { id: number; name: string; pos: Pos; est_ppg: number | null; avail: number | null; status: string; ir: boolean; headshot: string | null; team?: string | null }
export interface LeagueAnalysis {
  weeks: number[]; positions: Pos[]; reg_end: number; playoff_teams: number;
  seed_dist: Record<string, number[]>; week_pts: Record<string, number[]>;
  pos_ppw: Record<string, Record<string, number>>; stream_rate: Record<string, Record<string, number>>;
  pos_so_far: Record<string, Record<string, number>>;
}
export interface LeagueLive { sims: number; week: number; odds: Record<string, LiveOdds>; rosters: Record<string, RosterRow[]>; noise_pp: number; analysis?: LeagueAnalysis }
export interface WhatIfResult { sims: number; base: Record<string, LiveOdds>; scenario: Record<string, LiveOdds>; delta: Record<string, LiveOdds>; noise_pp: number }
export interface OutItem { id: number; weeks: number | null; name: string; team: number }

export interface TradeSide {
  name: string; gives: string[]; gets: string[]; would_drop: string[];
  delta_regular_season: number; delta_playoffs: number; delta_total: number; delta_total_calibrated: number;
  p_direction_right: number; edge_label: string; why: Record<string, string>;
}
export interface TradeEval { model: { week: number; teams: Record<string, TradeSide> }; espn: { week: number; teams: Record<string, TradeSide> } }
export type OddsSide = LiveOdds & { seed_med?: number };
export interface TradeOdds { sims: number; noise_pp: number; week?: number; teams: Record<string, { before: OddsSide; after: OddsSide; delta: OddsSide }> }
export interface TradeHistory { sent: number; accepted: number; declined: number; canceled: number }
export interface IdeaRow {
  with: string; give: string; get: string; you_drop: string; your_gain: number; your_gain_calibrated: number;
  your_gain_playoffs: number; p_right: number; their_gain: number; trust: string; why?: string; partner: number; give_ids: number[]; get_ids: number[];
}
export interface IdeasState { state: "idle" | "running" | "done" | "failed"; partner: number | null; started: number | null; rows: IdeaRow[]; error: string | null }

export interface Usage { weeks: number[]; targets: (number | null)[]; carries: (number | null)[]; target_share: (number | null)[]; snap_pct: (number | null)[] }
export interface WaiverRow {
  id: number; name: string; pos: Pos; team: string | null; headshot: string | null;
  espn_wk: number | null; est_ppg: number | null; avail: number | null; ros: number | null; ecr_rank: number | null;
  p_start?: number | null; p_boom?: number | null;
  tag: string; why: string; gain: number | null; drop_id: number | null; drop_name: string | null; usage: Usage | null;
}
export interface DepthRow { id: number; name: string; est_ppg: number | null; avail: number | null; ir: boolean; team: string | null; usage: Usage | null; drop_cost?: number | null; kept?: boolean }
export interface WaiverBoard {
  week: number; candidates: WaiverRow[]; depth: Record<string, DepthRow[]>; replacement: Record<string, number>;
  left_off: { id: number; name: string; pos: Pos; tag: string; est_ppg: number | null }[]; noise: number;
  /** season lineup points lost without each of your players (engine v3) */
  drop_cost?: Record<string, number>; pool_size?: Record<string, number>; computed_at?: number;
  /** ids of your players you won't drop (saved by the engine; the drop search skips them) */
  keep?: number[];
  /** next week's byes on your roster and the free agents that add most to that one week's lineup */
  next_week?: { week: number; byes: { id: number; name: string; pos: Pos }[]; lineup: number;
    adds: { id: number; name: string; pos: Pos; team: string | null; pts: number; gain: number; tag: string }[] } | null;
}

export type JKind = "lineup" | "waiver" | "trade" | "note";
export type JStatus = "done" | "offered" | "skipped";
export interface JOutcome {
  state: "pending" | "final" | "running" | "none";
  start_pts?: number; bench_pts?: number; right?: boolean; weeks?: [number, number]; in_pts?: number; out_pts?: number;
}
export interface JEntry {
  id: string; created_at: number; week: number; kind: JKind; title: string; why: string; confidence: number | null;
  payload: Record<string, unknown>; model: Record<string, unknown>; model_p: number | null; status: JStatus; outcome: JOutcome;
}
export interface JSummary {
  logged: number; resolved: number; hit_rate: number | null; your_conf: number | null; model_p: number | null;
  buckets: { lo: number; hi: number; n: number; hit: number | null }[];
}
export interface Journal { entries: JEntry[]; summary: JSummary; week: number }
export interface JDraft {
  kind: JKind; title: string; why?: string; confidence?: number | null; payload?: Record<string, unknown>;
  model?: Record<string, unknown>; model_p?: number | null; week?: number; status?: JStatus;
  /** shown in the log sheet, not stored */
  context?: string;
}

export interface RoleTrait { label: string; text: string; pct: number; side: "hi" | "lo"; key?: string; rel?: number }
export interface RoleTrend {
  kind: string; label: string; text: string; hold: number | null; hold_n?: number; cause?: string[]; hold_if_still_out?: number | null;
  next_vs_espn?: number | null; next_lo?: number | null; next_hi?: number | null; next_n?: number; season?: boolean;
}
export interface RoleStyle { name: string; p: number; alt: string; alt_p: number; probs: Record<string, number>; from_last_season?: boolean; pff?: boolean }
export interface RoleProfile {
  name: string; pos: Pos; team: string | null; games: number; tier: string;
  style: RoleStyle | null; traits: RoleTrait[]; trends: RoleTrend[];
  /** WR/TE: PFF routes run / team dropbacks in his games, season to date, with his rank at the position */
  route_sh?: { v: number; rank: number; of: number } | null;
  /** WR/TE: PFF slot and inline rates (% of pass snaps), average depth of target, yards per route run */
  pff?: { slot: number | null; inline: number | null; adot: number | null; yprr: number | null } | null;
  series: { weeks: number[]; snap: number[]; share: (number | null)[]; rz: (number | null)[]; share_label: string } | null;
  /** fingerprint metric -> [raw this season, noise-adjusted estimate, reliability 0-1] */
  fp?: Record<string, [number | null, number | null, number | null]>;
}
export interface RolesBoard {
  season: number; through_week: number; partial?: boolean; fit_seasons: [number, number];
  styles: Record<string, { names: string[]; yoy_same: number; early_acc?: Record<string, { acc: number; conf: number; n: number }> }>;
  styles_free?: Record<string, { names: string[]; yoy_same: number; early_acc?: Record<string, { acc: number; conf: number; n: number }> }>;
  pff_through?: number | null;
  /** position -> fingerprint metric -> label, format, group, games until half signal */
  fp_meta?: Record<string, Record<string, { label: string; fmt: "pct" | "pct100" | "num1" | "num2"; group: string; stab: number }>>;
  players: Record<string, RoleProfile>;
}

export interface FpMetric {
  key: string; label: string; fmt: "pct" | "pct100" | "num1" | "num2"; group: "Role" | "Style" | "Scoring" | "Efficiency";
  raw: number | null; est: number; prev: number | null; pct: number | null; raw_pct: number | null; rel: number; stab_games: number; lower_better: boolean; rank?: number; of?: number;
}
export interface MapGroup { name: string; x: number; y: number; cov: [[number, number], [number, number]] }
export interface MapPeer { id: number | null; x: number; y: number; vol: number | null; name: string | null; owner: { team_id: number; name: string } | null }
export interface LogRow {
  week: number; played: boolean; upcoming: boolean; opp: string | null; snap: number | null; tgt: number | null; rec: number | null; rec_yd: number | null; rec_td: number | null;
  car: number | null; rush_yd: number | null; rush_td: number | null; cmp: number | null; att: number | null; pass_yd: number | null; pass_td: number | null;
  ints: number | null; fl: number | null; ppr: number | null; espn_pts: number | null; espn_proj: number | null;
}
export interface CompRow {
  name: string; season: number; team: string; sim: number; ppg_then: number; snap_then: number; share_then: number | null; age: number | null;
  prev_ppg: number | null; ros_ppg: number; ros_games: number; ros_team_games: number; ros_snap: number | null; traj: (number | null)[];
}
export interface Comps {
  W: number; k: number; pool: number; q: { p10: number; p25: number; p50: number; p75: number; p90: number };
  avail: number; kept_role: number; beat_now: number; weeks: { n: number; q: number[]; ge15: number; lt5: number; hist: number[] };
  dist_ppg: number[]; list: CompRow[]; all: { name: string; season: number; ros_ppg: number; ppg_then: number }[];
  me: { ppg: number; age: number | null; exp: number | null; prev_ppg: number | null };
  bt: { mae: number; mae_sofar: number; mae_lin: number; cov80: number; cov50: number; rho: number; n: number } | null; starter_ppg: number | null;
}
export interface PlayerFile {
  id: number; name: string; pos: Pos; team: string | null; pro_team?: string | null; headshot?: string | null; season: number; through_week: number; week: number;
  age: number | null; draft: { year: number; round: number | null; pick: number | null } | null;
  owner: { team_id: number; name: string; injury?: string | null } | null;
  ledger: { est_ppg: number | null; avail_now: number | null; avail_ros: number | null; ecr_rank: number | null; status: string | null; bye: number | null } | null;
  espn_week: number | null; game?: Game | null;
  /** returning from games his team played without him: which game back this is, and how many he missed */
  back?: { game: 1 | 2; missed: number } | null;
  season_line: { games: number; ppr_g: number; snap: number; espn_g: number | null; rank: number | null; of: number } | null;
  last_season: { games: number; ppr_g: number; snap: number } | null;
  role: RoleProfile; fingerprint: FpMetric[];
  style_info: { names: string[]; centroids: Record<string, Record<string, number>>; transitions: Record<string, Record<string, number>>;
    meaning: Record<string, { ppg: number; n: number; snap: number }>; early_acc: Record<string, { acc: number; conf: number; n: number }>; yoy_same: number; kappa: number; seasons?: [number, number]; pff?: boolean };
  map: { x: string; y: string; xlabel: string; ylabel: string; groups: MapGroup[]; peers: MapPeer[]; trail: { week: number; x: number; y: number }[];
    last_season: { x: number; y: number } | null; fmt: Record<string, string>; seasons?: [number, number] };
  gamelog: LogRow[]; comps: Comps | null;
  read: { kind: "role" | "signal" | "caution" | "change" | "outlook"; anchor: string; text: string }[];
}

export interface ShiftWin {
  from_snap: number | null; to_snap: number | null; from_share: number | null; to_share: number | null; label: string;
  d_snap: number | null; d_share: number | null; zone: string | null; hold?: number | null; hold_if_still_out?: number | null;
  cause?: string[]; espn?: { rel: number; lo: number; hi: number } | null;
}
export interface ShiftRow {
  id: number; name: string; pos: Pos; team: string | null; headshot: string | null; owner: { team_id: number; name: string; injury?: string | null } | null;
  games: number; tier: string; style: string | null; series: { weeks: number[]; snap: number[]; share: (number | null)[]; share_label: string };
  recent: ShiftWin | null; season: ShiftWin | null; ledger: { est_ppg: number | null; avail: number | null; status: string | null } | null;
  espn_week: number | null; injury: string | null;
  /** WR/TE PFF route share and slot rate, last two weeks vs before (needs the extra PFF window) */
  pff?: { label: string; from_route: number; to_route: number; d_route: number; from_slot: number; to_slot: number; d_slot: number; games: number; move: boolean } | null;
}
export interface Shifts {
  season: number; through_week: number; partial?: boolean; week: number; rows: ShiftRow[];
  zones: Record<string, Record<string, { hold: number | null; n: number | null; espn?: { rel: number; lo: number; hi: number } | null }> & { season: Record<string, { hold: number; n: number }> }>;
  thresholds: { snap: number; share: number; snap_with: number; route?: number; slot?: number };
}

/* ---------- season analytics (GET /api/analytics, /api/analytics/league) ---------- */
/** Backtest label for a stat (backtest/analytics_metrics.py, 2021–25). */
export type ALabel = "adds" | "edge" | "tilt" | "priced" | "trait" | "noise" | "weak" | "descriptive";
export interface AEst { v: number; lo: number; hi: number; n: number; seasons_same_sign?: number; seasons?: number }
export interface AMetric {
  name: string; label: ALabel; label_text: string; repeat: number | null; sticks: number | null; predict: number | null;
  beyond: AEst | null; weekly: AEst | null; reading: string;
}
export interface AFlag { key: string; tone: "up" | "down" | "info"; metric: string; short: string; text: string; label: ALabel; edge?: number | null }
/** [pts, ESPN proj, xFP, snap share, target share, opportunities, rank at position, tier] */
export type AWeek = [number | null, number | null, number | null, number | null, number | null, number | null, number, "smash" | "start" | "mid" | "dud"] | null;
export interface APlayer {
  id: number; name: string; pos: Pos; team: string | null; owner: number | null; games: number; pool: boolean;
  est_ppg: number | null; ecr_rank: number | null;
  ppg: number | null; xfp: number | null; fpoe: number | null; opp: number | null; snap: number | null; tsh: number | null; ays: number | null;
  wopr: number | null; csh: number | null; cv: number | null; spread: number | null; sd: number | null; p20: number | null; p80: number | null;
  smash: number | null; start: number | null; dud: number | null; resid: number | null; beat: number | null;
  xfp_tr: number | null; snap_tr: number | null; tsh_tr: number | null; opp_tr: number | null; est_tr: number | null;
  pct: Record<string, number | null>; duds: number; smashes: number; starts: number;
  weekly: AWeek[]; flags: AFlag[];
  breakout: { prev_ppg: number; gain: number; usage_pct: number; eff_pct: number } | null;
}
export interface Analytics {
  season: number; through_week: number; week: number; n_teams: number;
  tiers: { start: Record<string, number>; smash: Record<string, number>; dud: Record<string, number> };
  pool: Record<string, number>; est_weeks: [number | null, number | null];
  metrics: Record<string, AMetric>; metrics_pos: Record<string, Record<string, AMetric>>; players: APlayer[];
  /** NFL schedule ahead (nflverse abbreviations): team -> week -> opponent */
  schedule: Record<string, Record<string, { opp: string; home: boolean }>>; sched_weeks: number[];
  sched_ev: { fpa_playoffs: AEst | null; pass_playoffs: AEst | null; run_playoffs: AEst | null; pass_next4: AEst | null; run_next4: AEst | null; label: ALabel };
}
export interface ALeagueCell { team_id: number; week: number; pts: number; opt: number; proj: number; drafted: number; missed: number[]; z: number; rank: number; eff: number; left: number; luck: number }
export interface ALeagueTeam { id: number; name: string; pts: number; opt: number; proj: number; left: number; espn: number; vs_espn: number; flip_won: number; flip_lost: number; luck: number; drafted: number; sd: number; best: number; worst: number; top3: number; eff: number; opt_rank: number; pts_rank: number }
export interface ADraftPick { id: number; name: string; pos: Pos; team_id: number; round: number; overall: number; pos_pick: number; pos_rank_now: number | null; pts: number | null; keeper: boolean }
export interface ALeague {
  season: number; through_week: number; n_teams: number; teams: ALeagueTeam[]; cells: ALeagueCell[]; draft: ADraftPick[];
  labels: Record<string, { name: string; label: ALabel; repeat: number | null }>;
}

/** /api/draft: draft recap (app/engine/draft.py). v_* = value over replacement vs what that pick slot usually returns. */
export interface DPick {
  id: number; name: string | null; pos: Pos | null; team_id: number; round: number; pick: number;
  pts: number; vor: number; gp: number; possible: number; exp_now: number; v_now: number; v_rate: number; v_absence: number;
  ros: number | null; exp_ros: number | null; v_ros: number | null; v_total: number;
  adp: number | null; vs_adp: number | null; board_rank: number | null;
  status: "kept" | "dropped" | "traded"; owner_now: number | null; left_after: number | null;
  started_for_drafter: number; started_elsewhere: number;
}
export interface DTeam {
  id: number; name: string; v_now: number; v_rate: number; v_absence: number; v_ros: number; v_total: number; pts: number;
  started_for_drafter: number; started_elsewhere: number; vs_adp: number | null; vs_adp_rel: number | null; adp_n: number;
  kept: number; dropped: number; traded: number; by_pos: Record<string, number>; v_now_rank: number; v_ros_rank: number; v_total_rank: number;
}
export interface DraftRecap {
  season: number; through_week: number; ros_from_week: number; n_teams: number; rounds: number; picks: DPick[]; teams: DTeam[];
  reach_corr: number | null; league_vs_adp: number | null; adp_matched: number;
  history: { through: number; seasons: number[]; hold: (number | null)[]; next_year: (number | null)[]; vs_points: (number | null)[] };
}

/** /api/accuracy: this season's pre-kickoff numbers scored against results (app/engine/accuracy.py). ci = 95% range. */
export interface AccBin { lo: number; hi: number; n: number; pred: number; actual: number; ci: [number, number] }
export interface AccMean { n: number; mean: number | null; ci: [number | null, number | null] }
export interface Accuracy {
  season: number; through_week: number; n_players: number;
  range?: { n: number; rate: number; ci: [number, number]; expect: number; by_pos: Record<string, { n: number; rate: number; ci: [number, number] }> };
  boom?: Record<"15" | "20", { n: number; pred: number; actual: number; ci: [number, number]; bins: AccBin[] }>;
  h2h?: { n: number; pred: number; actual: number; ci: [number, number]; bins: AccBin[] };
  proj_err?: { mae: number; bias: number; n: number };
  back: Record<string, AccMean & { expect: number }>;
  lineup: { weeks: { week: number; pts: number; espn: number; opt: number }[]; vs_espn?: AccMean; league_vs_espn?: AccMean; left?: number; expect_vs_espn?: number };
  winp: { n: number; pred?: number; actual?: number; ci?: [number, number]; brier?: number; brier_coin?: number };
  sources: { source: string; weeks: number; n: number; mae: number; bias: number; rank_corr: number }[];
}
