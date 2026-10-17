import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { copyText, runWhatIf, useLeague, useSnapshot } from "./data";
import { League, leagueMarkdown } from "./screens/league/League";
import { WhatIfPanel } from "./screens/league/WhatIf";
import type { OutItem, RosterRow, TradeHistory, WhatIfResult } from "./types";
import { type Deal, IdeasPanel, Trades } from "./screens/trades/Trades";
import { tradeApi } from "./data";
import { age, f1, lastName, phi } from "./format";
import { PlayerInspector, TeamInspector } from "./components/Inspector";
import { Board } from "./screens/myteam/Board";
import { GameDay } from "./screens/myteam/GameDay";
import { type Plan, STARTERS, canPlay, move, planFrom, planMarkdown, planPoints, rival, slotOf } from "./screens/myteam/lineup";
import { H2H } from "./components/bits";
import { pBeats } from "./odds";
import type { ColorMode } from "./teamColor";

const MODES: { id: ColorMode; label: string }[] = [
  { id: "position", label: "Position" }, { id: "stripe", label: "Stripe" }, { id: "tint", label: "Tint" }, { id: "card", label: "Card" },
];
const loadMode = (): ColorMode => { try { const v = localStorage.getItem("desk.colorMode"); return (MODES.some((m) => m.id === v) ? v : "stripe") as ColorMode; } catch { return "stripe"; } };
import { Waivers, WaiverInspector, waiversMarkdown } from "./screens/waivers/Waivers";
import { Journal, journalMarkdown } from "./screens/journal/Journal";
import { LogSheet } from "./components/LogSheet";
import { journalApi, useRoles, waiverApi } from "./data";
import { RoleSection, RosterShifts } from "./components/Role";
import { FileCtx } from "./components/FileContext";
import { PlayerFileView, fileMarkdown } from "./screens/file/PlayerFile";
import { fileApi } from "./data";
import { toggleFullscreen } from "./windowCtl";
import { Palette, type PalAction, type PalPlayer } from "./components/Palette";
import { journalPrompts } from "./screens/journal/prompts";
import { ShiftsScreen } from "./screens/shifts/Shifts";
import { LiveScreen, liveMarkdown, type LiveBoard } from "./screens/live/Live";
import { MatchupsScreen } from "./screens/matchups/Matchups";
import type { MxBoard } from "./screens/matchups/types";
import { CompareTray, CompareView } from "./screens/compare/Compare";
import { analyticsApi, matchupsApi, shiftsApi, useWarm, warmed, changesApi, kickoffMs, liveApi, type Changes } from "./data";
import { MyAnalysis, analysisMarkdown } from "./screens/myteam/Analysis";
import { Signals } from "./components/Analytics";
import { LabScreen } from "./screens/lab/Lab";
import type { ALeague, Analytics } from "./types";
import { ChangesPanel } from "./components/Changes";
import { Shortcuts } from "./components/Shortcuts";
import type { Shifts as ShiftsT } from "./types";
import type { PlayerFile } from "./types";
import type { JDraft, Journal as JournalT, WaiverBoard } from "./types";

type Screen = "team" | "league" | "trades" | "waivers" | "journal" | "shifts" | "live" | "matchups" | "lab";
type Sel = { kind: "player"; id: number } | { kind: "team"; id: number } | null;
const SCREENS: { id: Screen; label: string }[] = [
  { id: "team", label: "My Team" }, { id: "league", label: "League" }, { id: "trades", label: "Trades" }, { id: "waivers", label: "Waivers" }, { id: "journal", label: "Journal" }, { id: "shifts", label: "Shifts" }, { id: "live", label: "Live" }, { id: "matchups", label: "Matchups" }, { id: "lab", label: "Lab" },
];
const STALE_H = 12;

export default function App() {
  const { snap, source, error, refresh, startRefresh } = useSnapshot(3);
  const [screen, setScreenState] = useState<Screen>(() => { try { const v = localStorage.getItem("desk.screen"); return (SCREENS.some((x) => x.id === v) ? v : "team") as Screen; } catch { return "team"; } });
  const setScreen = useCallback((x: Screen) => { setScreenState(x); try { localStorage.setItem("desk.screen", x); } catch { /* not persisted */ } }, []);
  // One scroll container serves every screen, so a screen used to open wherever the last one was
  // scrolled to (League landed mid-page). Remember the position per screen instead; new screens start at the top.
  const mainEl = useRef<HTMLElement>(null);
  const scrollAt = useRef<Record<string, number>>({});
  const [keysOpen, setKeysOpen] = useState(false);
  const [mode, setMode] = useState<"board" | "gameday" | "analysis">("board");
  const scrollKey = screen === "team" ? `team-${mode}` : screen;
  useLayoutEffect(() => { const m = mainEl.current; if (m) m.scrollTop = scrollAt.current[scrollKey] ?? 0; }, [scrollKey]);
  const [plan, setPlan] = useState<Plan>({});
  const [sel, setSel] = useState<Sel>(null);
  const [colorMode, setColorModeState] = useState<ColorMode>(loadMode);
  const setColorMode = (m: ColorMode) => { setColorModeState(m); try { localStorage.setItem("desk.colorMode", m); } catch { /* not persisted */ } };
  const [toast, setToast] = useState<string | null>(null);
  const autoRefreshed = useRef(false);
  const [scenario, setScenario] = useState<OutItem[]>([]);
  const [whatif, setWhatif] = useState<WhatIfResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [leagueSel, setLeagueSel] = useState<number | null>(null);
  // The engine precomputes the slow views as it starts; fetch each as soon as it's ready (so
  // switching screens is instant), or right away if you open that screen first.
  const warm = useWarm(source === "engine", refresh.state);
  const dataKey = snap ? `${snap.data_age.rosters}|${snap.data_age.espn_proj}|${snap.data_age.ledger}` : "";
  const { live, err: liveErr } = useLeague(source === "engine" && (screen === "league" || screen === "trades" || warmed(warm, "league")), dataKey);
  const [deal, setDeal] = useState<Deal>({ partner: null, give: [], get: [] });
  const [tHistory, setTHistory] = useState<Record<string, TradeHistory>>({});
  useEffect(() => { if (source === "engine") tradeApi.history().then(setTHistory, () => undefined); }, [source]);
  const roles = useRoles(source === "engine", snap?.generated_at);
  const [wboard, setWboard] = useState<WaiverBoard | null>(null);
  const [wErr, setWErr] = useState<string | null>(null);
  const [wSel, setWSel] = useState<number | null>(null);
  const [wBusy, setWBusy] = useState(false);
  const wKey = useRef<string | null>(null);
  const wReq = useRef(0);
  /** Players you won't drop: saved in the engine, then the board is rebuilt without them (a few seconds). */
  const setKeep = useCallback((ids: number[]) => {
    const n = ++wReq.current;
    setWBusy(true);
    setWboard((b) => (b ? { ...b, keep: ids } : b));
    waiverApi.setKeep(ids).then(() => waiverApi.board()).then(
      (b) => { if (wReq.current === n) { setWboard(b); setWErr(null); } },
      (e) => { if (wReq.current === n) setWErr(`Couldn't rebuild the waiver board: ${e}`); },
    ).finally(() => { if (wReq.current === n) setWBusy(false); });
  }, []);
  useEffect(() => {  // preloaded on open: the engine starts building it the moment it boots
    if (source !== "engine" || !dataKey || wKey.current === dataKey) return;
    wKey.current = dataKey;
    waiverApi.board().then((b) => { setWboard(b); setWErr(null); }, (e) => { wKey.current = null; setWErr(`Couldn't build the waiver board: ${e}`); });
  }, [source, dataKey]);
  const [journal, setJournal] = useState<JournalT | null>(null);
  const [jErr, setJErr] = useState<string | null>(null);
  const loadJournal = useCallback(() => journalApi.list().then((j) => { setJournal(j); setJErr(null); }, (e) => setJErr(`Couldn't read the journal: ${e}`)), []);
  useEffect(() => { if (screen === "journal" && source === "engine") void loadJournal(); }, [screen, source, loadJournal]);
  const [logDraft, setLogDraft] = useState<JDraft | null>(null);
  // player file (full-screen dossier), openable from any screen
  const [fileNav, setFileNav] = useState<{ id: number; list: number[] } | null>(null);
  const [pfile, setPfile] = useState<PlayerFile | null>(null);
  const [pfErr, setPfErr] = useState<string | null>(null);
  const [pfBusy, setPfBusy] = useState(false);
  const openFile = useCallback((id: number, list?: number[]) => setFileNav({ id, list: list && list.includes(id) ? list : [id] }), []);
  useEffect(() => {
    if (!fileNav) return;
    let live = true;
    setPfBusy(true); setPfErr(null);
    fileApi.get(fileNav.id).then((f) => { if (live) { setPfile(f); setPfBusy(false); } },
      (e) => { if (live) { setPfErr(String(e).includes("404") ? "No usage data for this player this season or last (kickers and defenses don't have files)." : `Couldn't open the file: ${e}`); setPfBusy(false); setPfile(null); } });
    return () => { live = false; };
  }, [fileNav]);
  // league-wide usage shifts
  const [shifts, setShifts] = useState<ShiftsT | null>(null);
  const [shErr, setShErr] = useState<string | null>(null);
  const shKey = useRef<string | null>(null);
  useEffect(() => {
    if (source !== "engine" || !dataKey || shKey.current === dataKey || !(screen === "shifts" || screen === "waivers" || warmed(warm, "shifts"))) return;
    shKey.current = dataKey;
    shiftsApi.get().then((x) => { setShifts(x); setShErr(null); }, (e) => { shKey.current = null; setShErr(`Couldn't read usage shifts: ${e}`); });
  }, [screen, source, dataKey, warm]);

  // season analytics (every stat labelled by the 2021–25 backtest): preloaded once the engine has built it
  const [an, setAn] = useState<Analytics | null>(null);
  const [anLg, setAnLg] = useState<ALeague | null>(null);
  const [anErr, setAnErr] = useState<string | null>(null);
  const anKey = useRef<string | null>(null);
  useEffect(() => {
    if (source !== "engine" || !dataKey || anKey.current === dataKey || !((screen === "team" && mode === "analysis") || screen === "league" || screen === "trades" || screen === "matchups" || screen === "lab" || !!fileNav || warmed(warm, "analytics"))) return;
    anKey.current = dataKey;
    analyticsApi.players().then((x) => { setAn(x); setAnErr(null); },
      (e) => { anKey.current = null; setAnErr(/404/.test(String(e)) ? "The engine running now predates the analytics. Quit Desk and open it again once." : `Couldn't build the season analytics: ${e}`); });
    analyticsApi.league().then(setAnLg, () => undefined);
  }, [screen, mode, source, dataKey, warm, fileNav]);

  // matchups: coverage splits, defenses and defenders from free charted data (built by the engine; ~20 s the first time)
  const [mx, setMx] = useState<MxBoard | null>(null);
  const [mxErr, setMxErr] = useState<string | null>(null);
  const mxKey = useRef<string | null>(null);
  const mxCopy = useRef<(() => string) | null>(null);
  useEffect(() => {
    if (source !== "engine" || !dataKey || mxKey.current === dataKey || !(screen === "matchups" || warmed(warm, "matchups"))) return;
    mxKey.current = dataKey;
    matchupsApi.board().then((b) => { setMx(b); setMxErr(null); },
      (e) => { mxKey.current = null; setMxErr(/404/.test(String(e)) ? "The engine running now predates Matchups. Quit Desk and open it again once." : /503/.test(String(e)) ? "Matchup data isn\u2019t built yet. Run: python3 -m app.engine.matchups_data 2023 2024 2025 2026" : `Couldn\u2019t build the matchup tables: ${e}`); });
  }, [screen, source, dataKey, warm]);

  // live games: every few seconds on the Live screen while a game is on, slowly in the background
  // (so the sidebar can show that something is live). The engine caches ESPN's feeds itself.
  const [liveB, setLiveB] = useState<LiveBoard | null>(null);
  const [liveE, setLiveE] = useState<string | null>(null);
  useEffect(() => {
    if (source !== "engine") return;
    let stop = false, t: number | undefined;
    const tick = async () => {
      try {
        const b = await liveApi.get();
        if (stop) return;
        setLiveB(b); setLiveE(null);
        const on = b.games.some((g) => g.state === "in");
        t = window.setTimeout(tick, screen === "live" ? (on ? b.poll_s * 1000 : 60_000) : (on ? 60_000 : 300_000));
      } catch (e) {
        if (stop) return;
        setLiveE(/404/.test(String(e)) ? "The engine running now predates Live. Quit Desk and open it again once (the engine picks up the new route on start; after that Live updates itself)." : `Couldn't reach ESPN's live scores (${String(e).match(/"detail":"([^"]*)"/)?.[1] ?? String(e)}). Trying again every 30 seconds.`);
        t = window.setTimeout(tick, 30_000);
      }
    };
    void tick();
    return () => { stop = true; if (t) window.clearTimeout(t); };
  }, [source, screen]);
  const liveNow = !!liveB && [...liveB.me, ...liveB.opp].some((p) => p.state === "in");
  // compare: up to four players, queued from anywhere
  const [cmpIds, setCmpIds] = useState<number[]>([]);
  const [cmpOpen, setCmpOpen] = useState(false);
  const [cmpFiles, setCmpFiles] = useState<PlayerFile[]>([]);
  const [cmpErr, setCmpErr] = useState<string | null>(null);
  const toggleCompare = useCallback((id: number) => setCmpIds((xs) => (xs.includes(id) ? xs.filter((x) => x !== id) : xs.length >= 4 ? [...xs.slice(1), id] : [...xs, id])), []);
  const openCompare = useCallback((id?: number) => { if (id != null) setCmpIds((xs) => (xs.includes(id) ? xs : xs.length >= 4 ? [...xs.slice(1), id] : [...xs, id])); setFileNav(null); setCmpOpen(true); }, []);
  useEffect(() => {
    if (!cmpOpen) return;
    let live = true;
    setCmpErr(null);
    Promise.all(cmpIds.map((id) => fileApi.get(id).catch(() => null))).then((fs) => {
      if (!live) return;
      setCmpFiles(fs.filter((f): f is PlayerFile => !!f));
      if (fs.some((f) => !f)) setCmpErr("Some players have no usage data (kickers and defenses can't be compared here).");
    });
    return () => { live = false; };
  }, [cmpOpen, cmpIds]);
  const [palOpen, setPalOpen] = useState(false);
  // what's new since you last looked (seen = the newest pull you opened the panel on)
  const [chg, setChg] = useState<Changes | null>(null);
  const [chgOpen, setChgOpen] = useState(false);
  const [seen, setSeenState] = useState<number | null>(() => { try { const v = Number(localStorage.getItem("desk.seenState")); return v || null; } catch { return null; } });
  const setSeen = (v: number | null) => { setSeenState(v); try { if (v) localStorage.setItem("desk.seenState", String(v)); } catch { /* not persisted */ } };
  const chgKey = useRef<string | null>(null);
  useEffect(() => {
    if (source !== "engine" || !dataKey || chgOpen || chgKey.current === `${dataKey}|${seen}`) return;
    chgKey.current = `${dataKey}|${seen}`;
    changesApi.get(seen).then((c) => {
      setChg((prev) => { if (prev && c.to !== prev.to && c.items.length) flash(`${c.items.length} change${c.items.length > 1 ? "s" : ""} since you last looked: see What's new`); return c; });
    }, () => { chgKey.current = null; });
  }, [source, dataKey, seen, chgOpen]); // eslint-disable-line react-hooks/exhaustive-deps
  const fileStep = (d: number) => setFileNav((n) => { if (!n) return n; const i = n.list.indexOf(n.id) + d; return i >= 0 && i < n.list.length ? { ...n, id: n.list[i] } : n; });
  const saveLog = async (d: JDraft) => {
    await journalApi.add(d);
    setLogDraft(null); flash("Logged to journal");
    if (journal) void loadJournal();
  };
  const applyScenario = useCallback(async (next: OutItem[]) => {
    setScenario(next);
    if (!next.length) { setWhatif(null); return; }
    setBusy(true);
    try { setWhatif(await runWhatIf(next)); } catch { setWhatif(null); } finally { setBusy(false); }
  }, []);

  const tw = snap?.team_week;
  const by = useMemo(() => new Map((tw?.players ?? []).map((p) => [p.id, p])), [tw]);
  useEffect(() => { if (tw) setPlan(planFrom(tw.lineup.current)); }, [tw]);

  // Refresh once on open when the roster pull is stale.
  useEffect(() => {
    if (!snap || source !== "engine" || autoRefreshed.current) return;
    const t = snap.data_age.rosters;
    if (t && (Date.now() / 1000 - t) / 3600 > STALE_H) { autoRefreshed.current = true; void startRefresh(); }
  }, [snap, source, startRefresh]);

  // Automatic quick refreshes (rosters + projections): every 8 hours, every 3 hours on the day of
  // your games, every 25 minutes in the 3 hours before one of them (inactives, late scratches).
  const nextKick = useMemo(() => {
    const ks = [...(tw?.players ?? []), ...(tw?.matchup?.opp_starters ?? [])].map((p) => kickoffMs(p.game?.date, p.game?.time)).filter((x): x is number => x != null && x > Date.now() - 20 * 60e3);
    return ks.length ? Math.min(...ks) : null;
  }, [tw]);
  const autoEvery = (now: number) => { const h = nextKick == null ? 99 : (nextKick - now) / 3600e3; return h <= 3 ? 25 : h <= 24 ? 180 : 480; };
  useEffect(() => {
    if (source !== "engine" || !snap) return;
    const tick = () => {
      const t = snap.data_age.rosters;
      if (!t || refresh.state === "running") return;
      if ((Date.now() / 1000 - t) / 60 > autoEvery(Date.now())) void startRefresh(true);
    };
    const id = window.setInterval(tick, 60e3);
    return () => window.clearInterval(id);
  }, [source, snap, refresh.state, startRefresh, nextKick]); // eslint-disable-line react-hooks/exhaustive-deps

  const flash = (m: string) => { setToast(m); window.setTimeout(() => setToast(null), m.length > 40 ? 3500 : 1800); };
  const copyPlan = useCallback(async () => {
    if (!tw) return;
    const pts = planPoints(plan, by);
    const pWin = tw.matchup ? phi((pts - tw.matchup.opp_proj) / tw.matchup.margin_sd) : null;
    if (mode === "analysis") { flash((await copyText(analysisMarkdown(tw, an, anLg))) ? "Copied as Markdown" : "Copy failed"); return; }
    flash((await copyText(planMarkdown(tw, plan, by, pWin))) ? "Copied as Markdown" : "Copy failed");
  }, [tw, plan, by, mode, an, anLg]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " " && !e.metaKey && !fileNav && !(e.target as HTMLElement)?.closest?.("input, textarea, button[aria-pressed], [contenteditable]")) {
        const id = screen === "team" && sel?.kind === "player" ? sel.id : screen === "waivers" ? wSel : null;
        if (id != null) { e.preventDefault(); openFile(id, screen === "team" ? (tw?.players ?? []).map((p) => p.id) : (wboard?.candidates ?? []).map((c) => c.id)); }
        return;
      }
      if (e.key === "?" && !e.metaKey && !(e.target as HTMLElement)?.closest?.("input, textarea, [contenteditable]")) { e.preventDefault(); setKeysOpen((o) => !o); return; }
      if (!e.metaKey) return;
      if (e.key.toLowerCase() === "k") { e.preventDefault(); setPalOpen((o) => !o); return; }
      const n = Number(e.key);
      if (n >= 1 && n <= SCREENS.length) { e.preventDefault(); setScreen(SCREENS[n - 1].id); }
      if (e.shiftKey && e.key.toLowerCase() === "c") { e.preventDefault(); void copyPlan(); }
      if (e.shiftKey && e.key.toLowerCase() === "f") { e.preventDefault(); void toggleFullscreen().then(flash); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [copyPlan, fileNav, screen, sel, wSel, tw, wboard, openFile]);

  if (!snap || !tw) return <div style={{ padding: 40 }} className="dim">{error ?? "Loading…"}</div>;

  const now = Date.now() / 1000;
  const warmStep = warm?.state === "running" ? Object.values(warm.steps).find((x) => x.state === "running") : undefined;
  const loadingScreen: Partial<Record<Screen, boolean>> = source === "engine"
    ? { waivers: !wboard && !wErr, league: !live && !liveErr, shifts: !shifts && !shErr, matchups: !mx && !mxErr } : {};
  const stale = (snap.data_age.rosters && (now - snap.data_age.rosters) / 3600 > STALE_H) || false;
  const failedSteps = refresh.state === "failed" ? (refresh.log ?? []).filter((l) => !l.ok) : [];
  // FantasyPros free tier: the client stops at limit - reserve calls per UTC day; a full refresh spends ~25
  const fp = refresh.fp, fpLeft = fp ? fp.limit - fp.reserve - fp.used : null, fpLow = fpLeft !== null && fpLeft < 25;
  const teams = [...snap.power.teams].sort((a, b) => b.pf - a.pf);
  const selPlayer = sel?.kind === "player" ? by.get(sel.id) : undefined;
  const selTeam = sel?.kind === "team" ? snap.power.teams.find((t) => t.id === sel.id) : snap.power.teams.find((t) => t.id === tw.team_id);
  const moves = selPlayer && selPlayer.slot !== "IR"
    ? [...STARTERS.filter((s) => canPlay(selPlayer, s)).map((s) => ({ label: s, to: s as string, here: slotOf(plan, selPlayer.id) === s })),
       { label: "Bench", to: "BENCH", here: !slotOf(plan, selPlayer.id) }]
    : [];
  const doMove = (to: string) => { if (!selPlayer) return; const next = move(plan, by, selPlayer.id, to as never); if (next) setPlan(next); };
  const r = selPlayer ? rival(selPlayer, plan, by) : null;
  const h2h = selPlayer && r ? <H2H p={selPlayer} other={r.other} slot={r.slot} role={r.role} prob={pBeats(selPlayer, r.other, snap.ratio_q)} />
    : <span className="h2h"><b>No healthy alternative on the roster</b></span>;
  const lineupDraft = (): JDraft | null => {
    if (!selPlayer || !r) return null;
    const start = r.role === "bench" ? selPlayer : r.other, bench = r.role === "bench" ? r.other : selPlayer;
    const p = pBeats(start, bench, snap.ratio_q);
    return { kind: "lineup", week: tw.week, title: `Start ${start.name} over ${bench.name} at ${r.slot}`,
      payload: { start_id: start.id, bench_id: bench.id, slot: r.slot }, model_p: p,
      context: `ESPN ${f1(start.espn_week)} vs ${f1(bench.espn_week)} this week. Model: ${lastName(start.name)} outscores ${lastName(bench.name)} ${p == null ? "–" : Math.round(p * 100) + "%"} of the time.` };
  };
  const setOut = (team: number) => (r: RosterRow, weeks: number | null | undefined) => {
    const rest = scenario.filter((o) => o.id !== r.id);
    void applyScenario(weeks === undefined ? rest : [...rest, { id: r.id, weeks, name: r.name, team }]);
  };
  const leagueTeam = snap.power.teams.find((t) => t.id === (leagueSel ?? tw.team_id))!;
  const playerCopy = async () => { if (selPlayer) flash((await copyText(`${selPlayer.name}: ESPN wk ${tw.week} ${f1(selPlayer.espn_week)}, ROS ${f1(selPlayer.est_ppg)} ppg, plays ${Math.round((selPlayer.avail_now ?? 0) * 100)}%`)) ? "Copied" : "Copy failed"); };

  const palPlayers = (): PalPlayer[] => {
    const names = new Map(snap.power.teams.map((t) => [t.id, t.name]));
    const m = new Map<number, PalPlayer>();
    // ownership is only claimed when a source knows it: rosters (League) or the shifts board
    const put = (p: PalPlayer) => { const o = m.get(p.id); m.set(p.id, o ? { ...o, ...Object.fromEntries(Object.entries(p).filter(([k, v]) => v !== undefined && v !== "" && !(k === "mine" && v === false && o.mine))) } as PalPlayer : p); };
    for (const [k, v] of Object.entries(roles?.players ?? {})) put({ id: Number(k), name: v.name, pos: v.pos, team: v.team, mine: false, note: v.style?.name ?? v.tier });
    for (const r of shifts?.rows ?? []) put({ id: r.id, name: r.name, pos: r.pos, team: r.team, owner: r.owner?.name ?? null, mine: r.owner?.team_id === tw.team_id });
    for (const [t, rows] of Object.entries(live?.rosters ?? {})) for (const r of rows) put({ id: r.id, name: r.name, pos: r.pos, team: r.team ?? null, owner: names.get(Number(t)) ?? null, mine: Number(t) === tw.team_id });
    for (const c of wboard?.candidates ?? []) put({ id: c.id, name: c.name, pos: c.pos, team: c.team ?? null, owner: null, mine: false });
    for (const p of tw.players) put({ id: p.id, name: p.name, pos: p.pos, team: p.team ?? null, owner: tw.team_name, mine: true });
    return [...m.values()];
  };
  const palActions = (): PalAction[] => [
    ...SCREENS.map((x, i) => ({ id: `go-${x.id}`, group: "Go to" as const, label: x.label, hint: `⌘${i + 1}`, run: () => { setScreen(x.id); setFileNav(null); setCmpOpen(false); } })),
    { id: "go-gameday", group: "Go to", label: "Game day", hint: "My Team, by kickoff", run: () => { setScreen("team"); setMode("gameday"); setFileNav(null); } },
    ...(cmpIds.length ? [{ id: "go-compare", group: "Go to" as const, label: `Compare (${cmpIds.length})`, run: () => openCompare() }] : []),
    { id: "refresh", group: "Actions", label: "Refresh data from ESPN", hint: "rosters, projections, usage", run: () => void startRefresh() },
    { id: "refresh-full", group: "Actions", label: "Full refresh", hint: `weekly routine · FantasyPros ${fp ? `${fp.used}/${fp.limit} calls today` : "~25 calls"}`, run: () => void startRefresh(false, true) },
    { id: "fs", group: "Actions", label: "Full screen", hint: "⇧⌘F", run: () => void toggleFullscreen().then(flash) },
    ...(cmpIds.length ? [{ id: "cmp-clear", group: "Actions" as const, label: "Clear compare", run: () => setCmpIds([]) }] : []),
    ...snap.power.teams.flatMap((t) => [
      { id: `team-${t.id}`, group: "Teams" as const, label: t.name, hint: t.id === tw.team_id ? "you · League" : "League", run: () => { setScreen("league"); setLeagueSel(t.id); setFileNav(null); } },
      ...(t.id === tw.team_id ? [] : [{ id: `trade-${t.id}`, group: "Teams" as const, label: `Trade with ${t.name}`, hint: "Trades", run: () => { setDeal({ partner: t.id, give: [], get: [] }); setScreen("trades"); setFileNav(null); } }]),
    ]),
  ];

  return (
    <FileCtx.Provider value={{ open: openFile, compare: toggleCompare, inCompare: (id: number) => cmpIds.includes(id), openCompare }}>
    <div className="app">
      <header className="titlebar" data-tauri-drag-region>
        <div className="brand" data-tauri-drag-region>Desk</div>
        <div className="tools" data-tauri-drag-region>
          {screen === "waivers" && wboard && <button className="tbtn" type="button" onClick={async () => flash((await copyText(waiversMarkdown(wboard))) ? "Copied as Markdown" : "Copy failed")}>Copy for Claude</button>}
          {screen === "journal" && journal && <button className="tbtn" type="button" onClick={async () => flash((await copyText(journalMarkdown(journal))) ? "Copied as Markdown" : "Copy failed")}>Copy for Claude</button>}
          {screen === "matchups" && mx && <button className="tbtn" type="button" onClick={async () => flash((await copyText(mxCopy.current?.() ?? "")) ? "Copied as Markdown" : "Copy failed")}>Copy for Claude</button>}
          {screen === "live" && liveB && <button className="tbtn" type="button" onClick={async () => flash((await copyText(liveMarkdown(liveB))) ? "Copied as Markdown" : "Copy failed")}>Copy for Claude</button>}
          {screen === "league" && <button className="tbtn" type="button" onClick={async () => flash((await copyText(leagueMarkdown(snap.power, live, whatif, scenario, anLg))) ? "Copied as Markdown" : "Copy failed")}>Copy for Claude</button>}
          {screen === "team" && <>
            <div className="seg" role="group" aria-label="My Team view">
              <button type="button" aria-pressed={mode === "board"} onClick={() => setMode("board")}>Board</button>
              <button type="button" aria-pressed={mode === "gameday"} onClick={() => setMode("gameday")}>Game day</button>
              <button type="button" aria-pressed={mode === "analysis"} onClick={() => setMode("analysis")}>Analysis</button>
            </div>
            {mode === "board" && <div className="seg" role="group" aria-label="Tile color">
              {MODES.map((m) => <button key={m.id} type="button" aria-pressed={colorMode === m.id} onClick={() => setColorMode(m.id)}>{m.label}</button>)}
            </div>}
            <button className="tbtn" type="button" onClick={copyPlan}>Copy for Claude <kbd>⇧⌘C</kbd></button>
          </>}
        </div>
        <div className="right">
          <button className="tbtn chg-btn" type="button" onClick={() => { if (chgOpen) { setChgOpen(false); if (chg?.to) setSeen(chg.to); } else setChgOpen(true); }} disabled={!chg} title="Injury tags, projections and roster moves since you last looked">
            What's new{chg && chg.items.length > 0 && chg.to !== seen ? <span className="badge">{chg.items.length}</span> : null}</button>
          <button className="tbtn" type="button" onClick={() => setPalOpen(true)} title="Find a player, team or command">Find <kbd>⌘K</kbd></button>
          <button className="tbtn icon" type="button" onClick={() => void toggleFullscreen().then(flash)} title="Full screen (⇧⌘F)" aria-label="Toggle full screen">
            <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M1.5 4.5v-3h3M8.5 1.5h3v3M11.5 8.5v3h-3M4.5 11.5h-3v-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
        </div>
      </header>
      <div className={`body ${screen === "journal" || screen === "shifts" || screen === "live" || screen === "matchups" || screen === "lab" ? "no-insp" : ""}`}>
        <nav className="sidebar" aria-label="Desk">
          <h6>Desk</h6>
          {SCREENS.map((s, i) => (
            <button key={s.id} className="sitem" type="button" aria-current={screen === s.id} onClick={() => setScreen(s.id)}>
              <span>{s.label}{loadingScreen[s.id] && <i className="busy" title="Loading in the background" />}{s.id === "live" && liveNow && <i className="live-dot" title="A game with your starters or theirs is on" />}</span><small>⌘{i + 1}</small></button>
          ))}
          <h6>Teams · points</h6>
          {teams.map((t) => (
            <button key={t.id} className="sitem" type="button" aria-current={screen === "league" && (leagueSel ?? tw.team_id) === t.id}
              onClick={() => { if (t.id === tw.team_id) { setScreen("team"); setSel(null); } else { setScreen("league"); setLeagueSel(t.id); } }}>
              <span>{t.id === tw.team_id ? <span className="you-dot" /> : t.id === tw.matchup?.opp_id ? <span className="them-dot" /> : null}{t.name}</span><small>{f1(t.pf)}</small>
            </button>
          ))}
          <div className="side-data">
            <h6>Data</h6>
            <span className={`stamp ${stale || failedSteps.length ? "warn" : ""}`}
              title={failedSteps.length ? failedSteps.map((l) => `${l.step}: ${(l.tail ?? []).join(" | ")}`).join("\n") : "Age of the last ESPN roster pull"}>
              {source !== "engine" ? (source === "starting" ? "Starting engine…" : "Sample data") : refresh.state === "running" ? (refresh.quick ? "Checking ESPN…" : refresh.full ? `Full refresh: ${refresh.step ?? "starting"}…` : "Refreshing…")
              : failedSteps.length ? `Refresh: ${failedSteps.length} step${failedSteps.length > 1 ? "s" : ""} failed`
              : warmStep ? `Precomputing ${warmStep.label.toLowerCase()}…` : `Rosters ${age(now, snap.data_age.rosters)} · projections ${age(now, snap.data_age.espn_proj)}`}
            </span>
            {failedSteps.map((l) => <small key={l.step} className="side-fail" title={(l.tail ?? []).join("\n")}><b>{l.step}</b> {(l.tail ?? []).map((t) => {  // "[FAIL] step (1s) :: traceback | … | Error: msg" -> "step (1s): Error: msg"
              const [h, rest] = t.split(" :: "); return rest ? `${h.replace("[FAIL] ", "")}: ${rest.split(" | ").slice(-1)[0]}` : t; }).join(" · ")}</small>)}
            <div className="side-btns">
              <button className="tbtn" type="button" onClick={() => void startRefresh()} disabled={source !== "engine" || refresh.state === "running"}
              title={`Pull rosters, projections and usage now. Automatic quick refreshes: every ${autoEvery(Date.now()) >= 60 ? `${autoEvery(Date.now()) / 60} hours` : `${autoEvery(Date.now())} minutes`} right now.`}>Refresh</button>
              <button className="tbtn" type="button" onClick={() => void startRefresh(false, true)} disabled={source !== "engine" || refresh.state === "running"}
              title={`Run the whole weekly routine: every source (ESPN, FantasyPros, nflverse), ledger and season odds. Takes several minutes and uses ~25 FantasyPros API calls${fpLeft !== null ? `; ${fpLeft} usable left today` : ""}.`}>Full refresh</button>
            </div>
            {fp && <span className={`stamp ${fpLow ? "warn" : ""}`}
              title={`FantasyPros API calls today (UTC day ${fp.day}): ${fp.used} of ${fp.limit}. Desk stops at ${fp.limit - fp.reserve} to keep ${fp.reserve} in reserve.${fpLow ? " Not enough left for another full refresh today; the FantasyPros step will fail but the rest still runs." : ""}`}>
              FantasyPros API: {fp.used}/{fp.limit} today</span>}
          </div>
        </nav>
        <main className="content" ref={mainEl} onScroll={(e) => { scrollAt.current[scrollKey] = e.currentTarget.scrollTop; }}>
          {error && <div className="banner-src">{error}</div>}
          {screen === "team" && mode === "analysis" && <MyAnalysis tw={tw} a={source === "engine" ? an : null} lg={anLg} err={source !== "engine" ? "Start the engine to see the season analytics." : anErr}
            selected={selPlayer?.id ?? null} onSelect={(id) => setSel({ kind: "player", id })} />}
          {screen === "team" && mode !== "analysis" && (mode === "board"
            ? <Board tw={tw} plan={plan} setPlan={setPlan} selected={selPlayer?.id ?? null} onSelect={(id) => setSel({ kind: "player", id })} ratioQ={snap.ratio_q} colorMode={colorMode} teamColors={snap.team_colors} roles={roles} dropCost={wboard?.drop_cost ?? null} freeAgents={wboard?.candidates ?? null} onWaivers={() => setScreen("waivers")} onCopy={async (t) => flash((await copyText(t)) ? "Copied the moves" : "Copy failed")} />
            : <GameDay tw={tw} plan={plan} live={liveB} selected={selPlayer?.id ?? null} onSelect={(id) => setSel({ kind: "player", id })} />)}
          {screen === "league" && <League alg={source === "engine" ? anLg : null} power={snap.power} myId={tw.team_id} oppId={tw.matchup?.opp_id ?? null} live={live} liveErr={liveErr}
            engineUp={source === "engine"} whatif={whatif} scenario={scenario} busy={busy} selected={leagueSel}
            onSelect={setLeagueSel} onTrade={(t) => { setDeal({ partner: t, give: [], get: [] }); setScreen("trades"); }} onRemove={(id) => void applyScenario(scenario.filter((o) => o.id !== id))} onClear={() => void applyScenario([])} />}
          {screen === "trades" && <Trades an={source === "engine" ? an : null} me={tw.team_id} teams={snap.power.teams} live={live} engineUp={source === "engine"} teamColors={snap.team_colors}
            history={tHistory} deal={deal} setDeal={setDeal} week={tw.week} onLog={setLogDraft} onCopy={async (md) => flash((await copyText(md)) ? "Copied as Markdown" : "Copy failed")} />}
          {screen === "waivers" && <Waivers board={source === "engine" ? wboard : null} err={source !== "engine" ? "Start the engine to see the waiver board." : wErr} sel={wSel} onSel={setWSel} players={tw.players} roles={roles} shifts={shifts} onKeep={setKeep} busy={wBusy} />}
          {screen === "shifts" && <ShiftsScreen data={source === "engine" ? shifts : null} err={source !== "engine" ? "Start the engine to see league-wide usage shifts." : shErr} myTeam={tw.team_id}
            onCopy={async (md) => flash((await copyText(md)) ? "Copied as Markdown" : "Copy failed")} inCompare={(id) => cmpIds.includes(id)} onCompare={toggleCompare} />}
          {screen === "matchups" && <MatchupsScreen an={source === "engine" ? an : null} data={source === "engine" ? mx : null} err={source !== "engine" ? "Start the engine to see matchups." : mxErr} roles={roles} copyRef={mxCopy} tw={tw} ratioQ={snap.ratio_q} />}
          {screen === "lab" && <LabScreen a={source === "engine" ? an : null} err={source !== "engine" ? "Start the engine to use the Lab." : anErr} tw={tw} teams={snap.power.teams} roles={roles}
            onCopy={async (md) => flash((await copyText(md)) ? "Copied as Markdown" : "Copy failed")} />}
          {screen === "live" && <LiveScreen data={source === "engine" ? liveB : null} err={source !== "engine" ? "Start the engine to follow live games." : liveE} />}
          {screen === "journal" && <Journal journal={journal} err={jErr} engineUp={source === "engine"} onAdd={saveLog} onLog={setLogDraft}
            prompts={journalPrompts(tw, plan, snap.ratio_q, wboard, shifts, journal)} names={roles ? Object.fromEntries(Object.entries(roles.players).map(([k, v]) => [k, v.name])) : {}}
            onUpdate={async (id, f) => { await journalApi.update(id, f); await loadJournal(); }} onDelete={async (id) => { await journalApi.remove(id); await loadJournal(); }} />}
        </main>
        {screen === "waivers" && (
          <aside className="inspector" aria-label="Pickup">
            {(() => {
              const w = wboard?.candidates.find((c) => c.id === wSel) ?? (wboard ? [...wboard.candidates].sort((a, b) => (b.gain ?? 0) - (a.gain ?? 0))[0] : undefined);
              if (!w || !wboard) return <p className="ins-note">Pick a free agent to see him against the drop.</p>;
              return <WaiverInspector r={w} board={wboard} roles={roles} onKeep={setKeep}
                onCopy={async () => flash((await copyText(`${w.name} (${w.pos}, ${w.team}): add for ${w.drop_name}, season gain ${w.gain} (noise ±${wboard.noise}). ${w.why}`)) ? "Copied" : "Copy failed")}
                onLog={() => setLogDraft({ kind: "waiver", week: tw.week, title: `Add ${w.name}, drop ${w.drop_name ?? "nobody"}`,
                  payload: { add_ids: [w.id], drop_ids: w.drop_id ? [w.drop_id] : [] }, model: { gain: w.gain },
                  context: `Model: ${w.gain != null && w.gain >= 0 ? "+" : ""}${w.gain} season points (noise ±${wboard.noise}). ${w.why}` })} />;
            })()}
          </aside>
        )}
        {screen === "trades" && (
          <aside className="inspector" aria-label="Trade ideas">
            <IdeasPanel partner={deal.partner} pname={snap.power.teams.find((t) => t.id === deal.partner)?.name ?? ""} engineUp={source === "engine"}
              onLoad={(partner, give, get) => setDeal({ partner, give, get })} />
          </aside>
        )}
        {screen === "league" && (
          <aside className="inspector" aria-label="What if">
            <WhatIfPanel team={leagueTeam} isMe={leagueTeam.id === tw.team_id} isOpp={leagueTeam.id === tw.matchup?.opp_id} live={live}
              engineUp={source === "engine"} whatif={whatif} scenario={scenario} busy={busy} onSet={setOut(leagueTeam.id)} />
          </aside>
        )}
        {screen === "team" && (
          <aside className="inspector" aria-label="Inspector">
            {selPlayer ? <PlayerInspector p={selPlayer} signals={<Signals a={an} p={selPlayer} />} role={<RoleSection r={roles?.players[String(selPlayer.id)]} board={roles} pos={selPlayer.pos} id={selPlayer.id} list={tw.players.map((p) => p.id)} />} onCopy={playerCopy} onLog={r ? () => setLogDraft(lineupDraft()) : undefined} moves={moves} onMove={doMove} h2h={h2h} />
              : selTeam ? <TeamInspector t={selTeam} isMe={selTeam.id === tw.team_id} isOpp={selTeam.id === tw.matchup?.opp_id}
                shifts={selTeam.id === tw.team_id ? <RosterShifts players={tw.players} board={roles} onSelect={(id) => setSel({ kind: "player", id })} /> : undefined} /> : null}
          </aside>
        )}
      </div>
      {cmpOpen && <CompareView an={an} files={cmpIds.map((id) => cmpFiles.find((f) => f.id === id)).filter((f): f is PlayerFile => !!f)} loading={cmpFiles.length < cmpIds.length} err={cmpErr}
        ratioQ={snap.ratio_q} candidates={roles ? Object.entries(roles.players).map(([k, v]) => ({ id: Number(k), name: v.name, pos: v.pos, team: v.team })) : []}
        onAdd={(id) => setCmpIds((xs) => (xs.includes(id) || xs.length >= 4 ? xs : [...xs, id]))} onRemove={(id) => setCmpIds((xs) => xs.filter((x) => x !== id))}
        onClose={() => setCmpOpen(false)} onOpen={(id) => { setCmpOpen(false); openFile(id, cmpIds); }}
        onCopy={async (md) => flash((await copyText(md)) ? "Copied as Markdown" : "Copy failed")} />}
      {!cmpOpen && !fileNav && <CompareTray items={cmpIds.map((id) => ({ id, name: roles?.players[String(id)]?.name ?? by.get(id)?.name ?? cmpFiles.find((f) => f.id === id)?.name ?? `#${id}` }))}
        onRemove={(id) => setCmpIds((xs) => xs.filter((x) => x !== id))} onOpen={() => setCmpOpen(true)} onClear={() => setCmpIds([])} />}
      {logDraft && <LogSheet draft={logDraft} onCancel={() => setLogDraft(null)} onSave={saveLog} />}
      {fileNav && <PlayerFileView file={pfile && pfile.id === fileNav.id ? pfile : null} err={pfErr} loading={pfBusy}
        nameHint={roles?.players[String(fileNav.id)]?.name ?? by.get(fileNav.id)?.name} myTeam={tw.team_id} colors={snap.team_colors}
        dark={window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false}
        hasPrev={fileNav.list.indexOf(fileNav.id) > 0} hasNext={fileNav.list.indexOf(fileNav.id) < fileNav.list.length - 1}
        onPrev={() => fileStep(-1)} onNext={() => fileStep(1)} onClose={() => setFileNav(null)} onOpen={(id) => openFile(id, [id])}
        an={an} onCopy={async (f) => flash((await copyText(fileMarkdown(f, an?.players.find((p) => p.id === f.id), an))) ? "Copied as Markdown" : "Copy failed")}
        onLog={(f) => setLogDraft({ kind: "note", week: tw.week, title: `${f.name}: `, payload: { player_ids: [f.id] }, context: f.read[0]?.text })}
        onTrade={(f) => { if (f.owner) { setDeal({ partner: f.owner.team_id, give: [], get: [f.id] }); setScreen("trades"); setFileNav(null); } }}
        onWaivers={(f) => { setScreen("waivers"); setWSel(f.id); setFileNav(null); }} />}
      {palOpen && <Palette players={palPlayers()} actions={palActions()} onClose={() => setPalOpen(false)}
        onOpen={(id) => { setCmpOpen(false); openFile(id, [id]); }} onCompare={(id) => { toggleCompare(id); flash("Added to compare"); }} />}
      {keysOpen && <Shortcuts onClose={() => setKeysOpen(false)} />}
      {chgOpen && chg && <ChangesPanel data={chg} teamName={(id) => snap.power.teams.find((t) => t.id === id)?.name ?? "A team"}
        onOpen={(id) => openFile(id, [id])} onClose={() => { setChgOpen(false); if (chg.to) setSeen(chg.to); }} />}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
    </FileCtx.Provider>
  );
}
