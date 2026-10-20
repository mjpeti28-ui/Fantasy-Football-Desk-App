import { useCallback, useEffect, useRef, useState } from "react";
import type { IdeasState, JDraft, PlayerFile, RolesBoard, Shifts, JEntry, Journal, WaiverBoard, LeagueLive, OutItem, Snapshot, TradeEval, TradeHistory, TradeOdds, WhatIfResult } from "./types";

export const ENGINE = "http://127.0.0.1:8765";
export const ENGINE_VERSION = 3;

export type Source = "engine" | "sample" | "starting";
/** FantasyPros API calls spent today (UTC day), from the client's own ledger. The client stops at limit - reserve. */
export interface FpBudget { day: string; used: number; limit: number; reserve: number }
export interface RefreshState { state: "idle" | "running" | "done" | "failed"; quick?: boolean; full?: boolean; step?: string | null; fp?: FpBudget; log?: { step: string; ok: boolean; seconds: number; tail?: string[] }[] }

async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

const BOOT_WAIT_MS = 20000;   // the engine imports pandas and the model stack; give it time
const POLL_MS = 5000;         // while on the sample, keep checking for the engine

/** Loads from the local engine. While it boots, shows the bundled sample; switches to live
 *  data automatically as soon as the engine answers. */
export function useSnapshot(team = 3) {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [source, setSource] = useState<Source>("starting");
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState<RefreshState>({ state: "idle" });
  const live = useRef(false);

  const sample = useRef<Snapshot | null>(null);
  const tryEngine = useCallback(async () => {
    try {
      const s = await getJSON<Snapshot>(`${ENGINE}/api/snapshot?team=${team}`);
      live.current = true;
      const old = (s.engine_version ?? 1) < ENGINE_VERSION;
      // An engine left running from an older Desk lacks newer fields: borrow them from the sample.
      if (old && sample.current) { s.ratio_q ??= sample.current.ratio_q; s.team_colors ??= sample.current.team_colors; }
      setSnap(s); setSource("engine");
      getJSON<RefreshState>(`${ENGINE}/api/refresh`).then((r) => setRefresh((x) => ({ ...x, fp: r.fp }))).catch(() => { /* older engine */ });
      setError(old ? "An older Desk engine is still running. Quit Desk and start it again with app/run.sh to update it." : null);
      return true;
    } catch { return false; }
  }, [team]);

  useEffect(() => {
    let stop = false;
    const t0 = Date.now();
    (async () => {
      try { const s = await getJSON<Snapshot>("./sample-snapshot.json"); sample.current = s; if (!live.current) setSnap(s); } catch { /* no sample */ }
      while (!stop && !live.current) {
        if (await tryEngine()) return;
        const booting = Date.now() - t0 < BOOT_WAIT_MS;
        setSource(booting ? "starting" : "sample");
        setError(booting ? "Starting the engine… showing the saved sample until it answers."
          : "Engine not running: showing the saved sample. Check the terminal that launched Desk for a Python error.");
        await new Promise((r) => setTimeout(r, booting ? 1000 : POLL_MS));
      }
    })();
    return () => { stop = true; };
  }, [tryEngine]);

  const startRefresh = useCallback(async (quick = false, full = false) => {
    if (source !== "engine") return;
    try {
      setRefresh(await getJSON<RefreshState>(`${ENGINE}/api/refresh${quick ? "?quick=true" : full ? "?full=true" : ""}`, { method: "POST" }));
      const poll = setInterval(async () => {
        const s = await getJSON<RefreshState>(`${ENGINE}/api/refresh`);
        setRefresh(s);
        if (s.state !== "running") { clearInterval(poll); void tryEngine(); }
      }, 2000);
    } catch {
      setRefresh({ state: "failed" });
    }
  }, [source, tryEngine]);

  return { snap, source, error, refresh, startRefresh, reload: tryEngine };
}

export async function copyText(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { return false; }
}

/** Live odds and every roster (the engine simulates on first call; a few seconds). */
export function useLeague(enabled: boolean, stamp: unknown) {
  const [live, setLive] = useState<LeagueLive | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const got = useRef<unknown>(undefined);
  useEffect(() => {
    if (!enabled || got.current === stamp) return;
    got.current = stamp;
    getJSON<LeagueLive>(`${ENGINE}/api/league`).then((l) => { setLive(l); setErr(null); }, (e) => { got.current = undefined; setErr(String(e)); });
  }, [enabled, stamp]);
  return { live, err };
}

export async function runWhatIf(out: OutItem[]): Promise<WhatIfResult> {
  return getJSON<WhatIfResult>(`${ENGINE}/api/whatif`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ out: out.map((o) => ({ id: o.id, weeks: o.weeks })) }),
  });
}

const post = <T,>(path: string, body: unknown) => getJSON<T>(`${ENGINE}${path}`, {
  method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
});
export const tradeApi = {
  history: () => getJSON<Record<string, TradeHistory>>(`${ENGINE}/api/trades/history`),
  evaluate: (partner: number, give: number[], get: number[]) => post<TradeEval>("/api/trades/evaluate", { partner, give, get }),
  odds: (partner: number, give: number[], get: number[]) => post<TradeOdds>("/api/trades/odds", { partner, give, get }),
  startIdeas: (partner: number | null, player: number | null = null) => post<IdeasState>("/api/trades/ideas", { partner, player }),
  ideas: () => getJSON<IdeasState>(`${ENGINE}/api/trades/ideas`),
};

export const waiverApi = {
  board: () => getJSON<WaiverBoard>(`${ENGINE}/api/waivers?top=6`),
  setKeep: (ids: number[]) => post<{ keep: number[] }>("/api/waivers/keep", { ids }),
};

const send = <T,>(path: string, method: string, body?: unknown) => getJSON<T>(`${ENGINE}${path}`, {
  method, headers: { "content-type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body),
});
export const journalApi = {
  list: () => getJSON<Journal>(`${ENGINE}/api/journal`),
  add: (d: JDraft) => { const { context: _c, ...body } = d; return send<JEntry>("/api/journal", "POST", body); },
  update: (id: string, f: Partial<Pick<JEntry, "why" | "confidence" | "status" | "title">>) => send<JEntry>(`/api/journal/${id}`, "PATCH", f),
  remove: (id: string) => send<{ ok: boolean }>(`/api/journal/${id}`, "DELETE"),
};

/** Role profiles for every player who has played this season (engine builds them in ~5 s the first time). */
export function useRoles(enabled: boolean, stamp: unknown) {
  const [roles, setRoles] = useState<RolesBoard | null>(null);
  useEffect(() => {
    if (!enabled) return;
    getJSON<RolesBoard>(`${ENGINE}/api/roles`).then(setRoles, () => undefined);
  }, [enabled, stamp]);
  return roles;
}

const fileCache = new Map<number, PlayerFile>();
export const fileApi = {
  get: async (id: number, fresh = false): Promise<PlayerFile> => {
    if (!fresh && fileCache.has(id)) return fileCache.get(id)!;
    const f = await getJSON<PlayerFile>(`${ENGINE}/api/player/${id}`);
    fileCache.set(id, f);
    return f;
  },
  clear: () => fileCache.clear(),
};

export const liveApi = { get: () => getJSON<import("./screens/live/Live").LiveBoard>(`${ENGINE}/api/live`) };
export const accuracyApi = { get: () => getJSON<import("./types").Accuracy>(`${ENGINE}/api/accuracy`) };
export const analyticsApi = {
  players: () => getJSON<import("./types").Analytics>(`${ENGINE}/api/analytics`),
  league: () => getJSON<import("./types").ALeague>(`${ENGINE}/api/analytics/league`),
  draft: () => getJSON<import("./types").DraftRecap>(`${ENGINE}/api/draft`),
};
export const shiftsApi = { get: () => getJSON<Shifts>(`${ENGINE}/api/shifts`) };
export const matchupsApi = {
  board: () => getJSON<import("./screens/matchups/types").MxBoard>(`${ENGINE}/api/matchups`),
  defense: (abbr: string) => getJSON<import("./screens/matchups/types").MxDefenseView>(`${ENGINE}/api/matchups/defense/${abbr}`),
  player: (id: number, opp?: string) => getJSON<import("./screens/matchups/types").MxPlayer>(`${ENGINE}/api/matchups/player/${id}${opp ? `?opp=${opp}` : ""}`),
};

export interface WarmStep { label: string; state: "waiting" | "running" | "ready" | "failed"; seconds?: number; error?: string }
export interface Warm { state: "idle" | "running" | "done"; started: number | null; finished: number | null; steps: Record<string, WarmStep> }

/** The engine precomputes the slow views as it starts (and again after a refresh). Polls its
 *  progress while that runs; `kick` restarts polling (pass something that changes on refresh). */
export function useWarm(enabled: boolean, kick: unknown) {
  const [warm, setWarm] = useState<Warm | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let stop = false;
    (async () => {
      for (let i = 0; !stop && i < 900; i++) {
        try {
          const w = await getJSON<Warm>(`${ENGINE}/api/warm`);
          if (stop) return;
          setWarm(w);
          if (w.state === "done") return;
        } catch { /* engine busy or older engine without /api/warm */ if (i > 3) return; }
        await new Promise((r) => setTimeout(r, 1200));
      }
    })();
    return () => { stop = true; };
  }, [enabled, kick]);
  return warm;
}
/** True once a warm-up step has finished. */
export const warmed = (w: Warm | null, step: string) => !!w && (w.state === "done" || w.steps[step]?.state === "ready" || w.steps[step]?.state === "failed");

export interface ChangeItem { kind: "mine" | "opp" | "move"; id: number; name: string | null; pos: string | null; team: number | null; text: string; tone: "up" | "down" | "" }
export interface Changes { from: number | null; to: number | null; at_from?: number | null; at_to?: number | null; from_source?: string; week?: number; opp?: number | null; opp_name?: string | null; items: ChangeItem[]; note?: string }
export const changesApi = { get: (since?: number | null) => getJSON<Changes>(`${ENGINE}/api/changes${since ? `?since=${since}` : ""}`) };

/** Kickoff time (ms) of an NFL game given ESPN's ET date and clock. */
export function kickoffMs(date?: string | null, time?: string | null): number | null {
  if (!date || !time) return null;
  for (const off of ["-04:00", "-05:00"]) {
    const t = Date.parse(`${date}T${time.length === 5 ? time : time.padStart(5, "0")}:00${off}`);
    if (Number.isNaN(t)) return null;
    const z = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", timeZoneName: "short" }).format(new Date(t));
    if ((off === "-04:00" && z.includes("EDT")) || (off === "-05:00" && z.includes("EST"))) return t;
  }
  return null;
}
