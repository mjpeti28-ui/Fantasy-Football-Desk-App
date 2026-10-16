import { type MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RolesBoard, TeamWeek } from "../../types";
import { matchupsApi } from "../../data";
import { lastName } from "../../format";
import { Face, POS_VAR } from "../../components/bits";
import { Centered, ManLine, Meter, lastOf, Sec, Signal, Stack, StackKey, Tile, n2, p0, p1, pp, sg2, shellParts } from "./parts";
import { Grade, MatchupCard, ord } from "./Card";
import { Duel } from "./Duel";
import { mxMarkdown } from "./read";
import type { MxBoard, MxCard, MxCurrent, MxDefenseView, MxHist, MxPlayer, MxQbCurrent, MxQbHist, MxRow } from "./types";
import { Ahead, aheadMarkdown } from "./Ahead";
import type { Analytics } from "../../types";
import { SHELLS, SHELL_LABEL } from "./types";
import "./matchups.css";

/* N — Matchups. A scouting-room board: the left rail is the slate (your players, their starters, any player you
   look up); the middle is a player and a defense side by side with the net read above them. Everything the
   free play-by-play and FTN charting can say, with the sample under each number and a plain label wherever
   the number is mostly luck. No PFF grades here (terms bar feeding them to a model) and no PFR-sourced numbers. */

const DEPTH = [["behind", "Behind LOS"], ["short", "0–9"], ["mid", "10–19"], ["deep", "20+"]] as const;
const LOC = [["left", "Left"], ["middle", "Middle"], ["right", "Right"]] as const;
const dayLabel = (d: string | null, t: string | null) => {
  if (!d) return "";
  const dt = new Date(`${d}T12:00:00`);
  const wd = dt.toLocaleDateString("en-US", { weekday: "short" });
  if (!t) return wd;
  const [h, m] = t.split(":").map(Number);
  return `${wd} ${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")}${h < 12 ? "a" : "p"}`;
};
const useStored = <T extends string>(key: string, init: T, ok: T[]) => {
  const [v, setV] = useState<T>(() => { try { const x = localStorage.getItem(key) as T | null; return x && ok.includes(x) ? x : init; } catch { return init; } });
  return [v, (x: T) => { setV(x); try { localStorage.setItem(key, x); } catch { /* not persisted */ } }] as const;
};

export function MatchupsScreen({ data, err, roles, copyRef, tw, ratioQ, an }: {
  an?: Analytics | null;
  data: MxBoard | null; err: string | null; roles: RolesBoard | null; copyRef: MutableRefObject<(() => string) | null>; tw: TeamWeek; ratioQ?: Record<string, number[]>;
}) {
  const [mode, setMode] = useStored<"week" | "duel" | "league" | "ahead">("desk.mx.mode", "week", ["week", "duel", "league", "ahead"]);
  const [pick, setPick] = useState<number | null>(null);
  const [through, setThrough] = useState<string | null>(null);
  const [dsel, setDsel] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const pcache = useRef(new Map<string, MxPlayer>());
  const dcache = useRef(new Map<string, MxDefenseView>());
  const [pl, setPl] = useState<MxPlayer | null>(null);
  const [dv, setDv] = useState<MxDefenseView | null>(null);
  const [perr, setPerr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { pcache.current.clear(); }, [data]);
  /** One player’s card for the Start/sit table, through the same cache the week view uses. */
  const fetchCard = useCallback(async (id: number): Promise<MxCard | undefined> => {
    const key = `${id}|`;
    let p = pcache.current.get(key);
    if (!p) { p = await matchupsApi.player(id); pcache.current.set(key, p); }
    return p.card;
  }, []);

  const rows = useMemo(() => (data ? [...data.mine, ...data.theirs] : []), [data]);
  const byId = useMemo(() => new Map(rows.map((r) => [r.id, r])), [rows]);
  // default pick: your first starter who has a game
  useEffect(() => {
    if (!data || pick != null) return;
    const r = data.mine.find((x) => !x.bye && !["BE", "IR"].includes(x.slot)) ?? data.mine[0];
    if (r) setPick(r.id);
  }, [data, pick]);
  const row = pick != null ? byId.get(pick) ?? null : null;
  const oppOf = through ?? row?.opp ?? null;

  useEffect(() => {
    if (!data || pick == null || mode !== "week") return;
    const key = `${pick}|${through ?? ""}`;
    const hit = pcache.current.get(key);
    if (hit) { setPl(hit); setPerr(null); return; }
    let live = true;
    setBusy(true);
    matchupsApi.player(pick, through ?? row?.opp ?? undefined).then((p) => { if (!live) return; pcache.current.set(key, p); setPl(p); setPerr(null); setBusy(false); },
      (e) => { if (!live) return; setPl(null); setBusy(false); setPerr(/404/.test(String(e)) ? "No charted data for this player (kickers, defenses and players without plays in 2023–26)." : `Couldn’t load him: ${e}`); });
    return () => { live = false; };
  }, [data, pick, through, row?.opp, mode]);

  const dAbbr = mode === "league" ? dsel : mode === "duel" ? null : (through ?? pl?.opp ?? oppOf);
  useEffect(() => {
    if (!data || !dAbbr) { setDv(null); return; }
    const hit = dcache.current.get(dAbbr);
    if (hit) { setDv(hit); return; }
    let live = true;
    matchupsApi.defense(dAbbr).then((v) => { if (!live) return; dcache.current.set(dAbbr, v); setDv(v); }, () => { if (live) setDv(null); });
    return () => { live = false; };
  }, [data, dAbbr]);

  useEffect(() => { setThrough(null); }, [pick]);
  useEffect(() => {
    if (!data) { copyRef.current = null; return; }
    if (mode === "duel") return; // the Start/sit table owns the copy while it is open
    if (mode === "ahead") { copyRef.current = () => (data ? aheadMarkdown(data, an ?? null, tw) : ""); return; }
    copyRef.current = () => mxMarkdown(data, mode === "week" ? pl : null, dv && dv.defense.abbr === dAbbr ? dv : null, mode, dsel);
  }, [data, pl, dv, mode, dsel, dAbbr, copyRef, an, tw]);

  if (err && !data) return <div className="mx"><p className="mx-empty">{err}</p></div>;
  if (!data) return <div className="mx"><p className="mx-empty">Loading matchup tables… (the first build reads three seasons of charted plays, about 20 seconds)</p></div>;

  const found = q.trim().length >= 2 && roles
    ? Object.entries(roles.players).filter(([, v]) => ["QB", "RB", "WR", "TE"].includes(v.pos) && v.name.toLowerCase().includes(q.trim().toLowerCase())).slice(0, 8).map(([k, v]) => ({ id: Number(k), name: v.name, pos: v.pos, team: v.team }))
    : [];

  return (
    <div className="mx">
      <div className="mx-top">
        <div className="seg" role="group" aria-label="Matchups view">
          <button type="button" aria-pressed={mode === "week"} onClick={() => setMode("week")}>Week {data.week}</button>
          <button type="button" aria-pressed={mode === "duel"} onClick={() => setMode("duel")}>Start/sit</button>
          <button type="button" aria-pressed={mode === "league"} onClick={() => setMode("league")}>All defenses</button>
          <button type="button" aria-pressed={mode === "ahead"} onClick={() => setMode("ahead")}>Schedule ahead</button>
        </div>
        <span className="mx-src" title={data.attribution}>Free data only: play-by-play, FTN charting via nflverse. Coverage splits 2023–{lastOf(data.hist)}.</span>
      </div>
      {mode === "week" ? (
        <div className="mx-grid">
          <Slate data={data} pick={pick} onPick={(id) => setPick(id)} q={q} setQ={setQ} found={found} />
          <div className="mx-main">
            {perr && <p className="mx-empty">{perr}</p>}
            {!perr && (!pl || !row && !pl) && <p className="mx-empty">{busy ? "Loading…" : "Pick a player from the slate."}</p>}
            {pl && <Versus pl={pl} row={row} data={data} dv={dv} through={through} setThrough={setThrough} busy={busy} />}
          </div>
        </div>
      ) : mode === "ahead" ? (
        <div className="mx-league"><Ahead data={data} an={an ?? null} tw={tw} /></div>
      ) : mode === "duel" ? (
        <div className="mx-league">
          <Duel tw={tw} ratioQ={ratioQ} roles={roles} fetchCard={fetchCard} copyRef={copyRef} onOpen={(id) => { setPick(id); setMode("week"); }} />
        </div>
      ) : (
        <div className="mx-league">
          {dsel && dv && dv.defense.abbr === dsel
            ? <div className="mx-dwrap"><button type="button" className="tbtn back" onClick={() => setDsel(null)}>← All defenses</button><DefensePanel v={dv} data={data} wide /></div>
            : <DefenseTable data={data} rows={rows} onPick={setDsel} />}
        </div>
      )}
      <p className="mx-foot">{data.attribution}. Man/zone and coverage shells are FTN’s charting as published in nflverse’s participation file; the {data.season} file doesn’t exist until after the season, so players’ coverage splits use {data.hist[0]}–{lastOf(data.hist)} and each defense is projected from its own history.</p>
    </div>
  );
}

/* ---------------- slate ---------------- */
function Lean({ b }: { b: MxRow["brief"] }) {
  if (!b || b.diff_shrunk == null) return <span className="mx-lean none" title="No coverage split (too few 2023–25 snaps)" />;
  const v = Math.max(-0.1, Math.min(0.1, b.diff_shrunk)) / 0.1;
  const weak = (b.rel ?? 0) < 0.35;
  return (
    <span className={`mx-lean ${weak ? "weak" : ""}`} title={`Man minus zone target rate, shrunk: ${pp(b.diff_shrunk)} pts. ${p0(b.rel)} of a gap like this is signal.`}>
      <i style={v >= 0 ? { left: "50%", width: `${v * 50}%` } : { left: `${50 + v * 50}%`, width: `${-v * 50}%` }} /><b />
    </span>
  );
}

function SlateRow({ r, on, onPick }: { r: MxRow; on: boolean; onPick: (id: number) => void }) {
  const dead = r.bye || /OUT|RESERVE/i.test(r.injury ?? "");
  return (
    <button type="button" className={`mx-row ${r.side} ${dead ? "dead" : ""}`} aria-current={on} onClick={() => onPick(r.id)}>
      <span className="posband" style={{ background: POS_VAR[r.pos] }} />
      <span className="nm"><b>{r.name}</b><small>{r.bye ? "Bye" : `${r.home ? "vs" : "@"} ${r.opp ?? "–"}`}{r.slot && !["BE", "IR"].includes(r.slot) && r.side === "me" ? ` · ${r.slot}` : r.slot === "IR" ? " · IR" : ""}</small></span>
      <Lean b={r.brief} />
    </button>
  );
}

function Slate({ data, pick, onPick, q, setQ, found }: {
  data: MxBoard; pick: number | null; onPick: (id: number) => void; q: string; setQ: (s: string) => void; found: { id: number; name: string; pos: string; team: string | null }[];
}) {
  const starters = data.mine.filter((r) => !["BE", "IR"].includes(r.slot)), bench = data.mine.filter((r) => ["BE", "IR"].includes(r.slot));
  return (
    <aside className="mx-slate" aria-label="Slate">
      <input className="mx-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Look up any QB, RB, WR, TE…" aria-label="Look up a player" />
      {q.trim().length >= 2 && (
        <div className="mx-found">
          {found.length === 0 && <p className="dim">No match.</p>}
          {found.map((f) => (
            <button key={f.id} type="button" className="mx-row" aria-current={pick === f.id} onClick={() => { onPick(f.id); setQ(""); }}>
              <span className="posband" style={{ background: POS_VAR[f.pos as keyof typeof POS_VAR] }} /><span className="nm"><b>{f.name}</b><small>{f.pos} · {f.team ?? "FA"}</small></span>
            </button>
          ))}
        </div>
      )}
      <h6><i className="d you" />{data.team_name ?? "You"}</h6>
      {starters.map((r) => <SlateRow key={r.id} r={r} on={pick === r.id} onPick={onPick} />)}
      {bench.length > 0 && <h6 className="sub">Bench</h6>}
      {bench.map((r) => <SlateRow key={r.id} r={r} on={pick === r.id} onPick={onPick} />)}
      <h6><i className="d them" />{data.opp_name ?? "Opponent"}</h6>
      {data.theirs.map((r) => <SlateRow key={r.id} r={r} on={pick === r.id} onPick={onPick} />)}
      <p className="mx-key"><span className="mx-lean"><i style={{ left: "50%", width: "30%" }} /><b /></span> leans man · <span className="mx-lean"><i style={{ left: "20%", width: "30%" }} /><b /></span> leans zone · dashed = mostly luck</p>
    </aside>
  );
}

/* ---------------- the matchup ---------------- */
function Versus({ pl, row, data, dv, through, setThrough, busy }: {
  pl: MxPlayer; row: MxRow | null; data: MxBoard; dv: MxDefenseView | null; through: string | null; setThrough: (s: string | null) => void; busy: boolean;
}) {
  row = row && row.id === pl.espn_id ? row : null; // a row only describes the player it belongs to
  const mine = row ? row.side === "me" : true;
  const d = dv && dv.defense.abbr === (through ?? pl.opp) ? dv : null;
  const sched = row ?? null;
  const teams = Object.keys(data.teams).sort();
  const g = pl.game ?? (sched && sched.opp ? { opp: sched.opp, home: !!sched.home, date: sched.date ?? "", time: sched.time } : null);
  return (
    <>
      <header className={`mx-vs ${mine ? "you" : row ? "them" : ""}`}>
        <Face p={{ headshot: row?.headshot ?? `https://a.espncdn.com/i/headshots/nfl/players/full/${pl.espn_id}.png`, pos: pl.pos, name: pl.name }} />
        <div className="who"><h2>{pl.name}</h2><span><i className="chip" style={{ background: POS_VAR[pl.pos] }}>{pl.pos}</i>{pl.team ?? "FA"}{row?.slot && !["BE", "IR"].includes(row.slot) ? ` · ${row.slot}` : ""}</span></div>
        <div className="vs">
          <small>{through ? "Looked through" : "This week"}</small>
          <b>{pl.opp ? <>{g && !through ? (g.home ? "vs " : "@ ") : "vs "}{pl.opp}</> : "Bye"}</b>
          <span>{!through && g ? dayLabel(g.date, g.time) : ""}</span>
        </div>
        <label className="thr">Look through <select value={through ?? ""} onChange={(e) => setThrough(e.target.value || null)}>
          <option value="">{row?.opp ?? pl.opp ?? "scheduled opponent"} (scheduled)</option>
          {teams.map((t) => <option key={t} value={t}>{t}</option>)}
        </select></label>
      </header>
      {pl.card
        ? <MatchupCard c={pl.card} busy={busy} side={mine ? "you" : "them"} />
        : <div className="mx-read"><div className="lines"><p>The engine that is running is older than this screen and didn’t send the start/sit card. Quit and reopen Desk to load it.</p></div></div>}
      <div className="mx-cols">
        <PlayerPanel pl={pl} tone={mine ? "you" : "them"} data={data} />
        {d ? <DefensePanel v={d} data={data} /> : <div className="mx-sec"><p className="mx-empty">{pl.opp ? "Loading the defense…" : "Bye week: nothing to show on the defense side. Pick a team in “Look through”."}</p></div>}
      </div>
    </>
  );
}

/* ---------------- player side ---------------- */
function PlayerPanel({ pl, tone, data }: { pl: MxPlayer; tone: "you" | "them"; data: MxBoard }) {
  if (pl.pos === "QB") return <QbPanel pl={pl as MxPlayer & { current: MxQbCurrent | null; hist: MxQbHist | null }} tone={tone} />;
  const c = pl.current as MxCurrent | null, h = pl.hist as MxHist | null;
  const b = c?.base ?? null;
  const thin = (c?.tgt ?? 0) < 25;
  return (
    <div className="mx-side">
      <Sec title="How he’s been used" aside={c ? `${data.season} · ${c.games} games` : undefined}>
        {!c ? <p className="dim">No targets this season yet.</p> : (<>
          <div className="mx-tiles">
            <Tile k="Targets" v={c.tgt} sub={`${(c.tgt / Math.max(1, c.games)).toFixed(1)} a game`} />
            <Tile k="Team share" v={p0(c.share)} sub={`of ${c.team_tgt}`} />
            <Tile k="aDOT" v={n2(c.adot, 1)} sub={b ? `${pl.pos} avg ${n2(b.adot, 1)}` : undefined} />
            <Tile k="EPA/tgt" v={sg2(c.epa_t)} sub={b ? `avg ${sg2(b.epa_t)}` : undefined} />
          </div>
          <Stack label="Depth of target" parts={DEPTH.map(([k, n]) => ({ k, name: n, v: c.depth[k] }))} />
          {b && <Stack label={<span className="dim">{pl.pos} average</span>} parts={DEPTH.map(([k, n]) => ({ k, name: n, v: b.depth[k] }))} />}
          <StackKey names={DEPTH.map(([, n]) => n)} />
          <Stack label="Location" parts={LOC.map(([k, n]) => ({ k, name: n, v: c.loc[k] }))} />
          {b && <Stack label={<span className="dim">{pl.pos} average</span>} parts={LOC.map(([k, n]) => ({ k, name: n, v: b.loc[k] }))} />}
          <StackKey names={LOC.map(([, n]) => n)} />
          <h6>Against his team’s mix (tick = team)</h6>
          <Meter label="Play-action" v={c.pa_share} tick={c.team_pa_share} max={0.5} tone={tone} thin={thin} note={`team ${p0(c.team_pa_share)}`} val={p0(c.pa_share)} />
          <Meter label="Screens" v={c.screen_share} tick={c.team_screen_share} max={0.4} tone={tone} thin={thin} note={`team ${p0(c.team_screen_share)}`} val={p0(c.screen_share)} />
          <Meter label="Into a blitz" v={c.blitz_share} tick={c.team_blitz_share} max={0.6} tone={tone} thin={thin} note={`team ${p0(c.team_blitz_share)}`} val={p0(c.blitz_share)} />
          <h6>Quality of what he sees (tick = {pl.pos} average)</h6>
          <Meter label="Catchable" v={c.catchable} tick={b?.catchable} max={1} tone={tone} thin={thin} note={`${c.contested} contested`} val={p0(c.catchable)} />
          <Meter label="Catch rate" v={c.catch_rate} tick={b?.catch_rate} max={1} tone={tone} thin={thin} note={`${c.drops} drop${c.drops === 1 ? "" : "s"}`} val={p0(c.catch_rate)} />
          {thin && <p className="mx-note">Under 25 targets: every share above moves a lot with one more catch.</p>}
        </>)}
        {pl.rush && <p className="mx-note r">Rushing: {pl.rush.car} carries, {n2(pl.rush.ypc, 1)} a carry, EPA/rush {sg2(pl.rush.epa)}, {p0(pl.rush.success)} successful, {p0(pl.rush.explosive)} of carries go 10+.</p>}
      </Sec>
      <Sec title="What he does against each coverage" aside={h ? `2023–25 · ${h.total.db.toLocaleString()} dropbacks on the field` : undefined}>
        {!h ? <p className="dim">Fewer than 60 charted snaps in 2023–25, so no split. (A rookie or a player new to the league has only this season’s usage above.)</p> : <Coverage h={h} pos={pl.pos} tone={tone} />}
      </Sec>
    </div>
  );
}

function Coverage({ h, pos, tone }: { h: MxHist; pos: string; tone: "you" | "them" }) {
  const max = Math.max(0.12, ...[h.man, h.zone, ...SHELLS.map((s) => h.shells[s]), h.other.blitz, h.other.no_blitz, h.other.pa, h.other.no_pa].map((s) => s.tr ?? 0)) * 1.15;
  const m = (label: string, s: MxHist["total"], ref: number | null, key?: string) => (
    <Meter key={key ?? label} label={label} v={s.tr} tick={ref} max={max} tone={tone} thin={s.tgt < 30} val={p1(s.tr)} note={`${s.tgt}/${s.db}`} aux={n2(s.ppr_t, 1)} />
  );
  return (
    <>
      <p className="mx-q2">Targets per dropback he is on the field for. Tick = his overall rate; the man and zone ticks are the {pos} norm.</p>
      <div className="mx-hd"><span /><span /><span>rate</span><span>tgt/db</span><span>PPR/t</span></div>
      {m("Man", h.man, h.pos_mean_tr.man)}
      {m("Zone", h.zone, h.pos_mean_tr.zone)}
      <div className="mx-gap">
        <span>Man − zone: <b className="num">{pp(h.tr_diff)}</b> pts raw, <b className="num">{pp(h.tr_diff_shrunk)}</b> shrunk</span>
        <Signal rel={h.tr_rel} />
      </div>
      <h6>By coverage shell</h6>
      {SHELLS.map((s) => m(SHELL_LABEL[s], h.shells[s], h.total.tr))}
      <h6>Pressure and play-action</h6>
      {m("Blitz", h.other.blitz, h.total.tr)}
      {m("No blitz", h.other.no_blitz, h.total.tr)}
      {m("Play-action", h.other.pa, h.total.tr)}
      {m("No play-action", h.other.no_pa, h.total.tr)}
      <p className="mx-note">
        PPR per target by coverage is shown for completeness (right column) but measured at <b>{p0(h.ppr_rel)}</b> signal: who gets targeted against which coverage is partly real, how well he does with them is mostly luck at this sample. Dashed rows have under 30 targets.
      </p>
      {h.routes.length > 0 && (<>
        <h6>Routes he is targeted on (tick = {pos} average)</h6>
        {h.routes.slice(0, 6).map((r) => <Meter key={r.route} label={r.label} v={r.share} tick={r.league_share} max={0.3} tone={tone} thin={r.n < 15} val={p0(r.share)} note={`${r.n}`} aux={n2(r.ppr_t, 1)} />)}
      </>)}
    </>
  );
}

function QbPanel({ pl, tone }: { pl: MxPlayer & { current: MxQbCurrent | null; hist: MxQbHist | null }; tone: "you" | "them" }) {
  const c = pl.current, h = pl.hist;
  const row = (label: string, s: MxQbHist["total"], key?: string) => (
    <Centered key={key ?? label} label={label} v={s.epa_db} se={s.se} span={0.5} tone={tone} thin={s.db < 100} val={sg2(s.epa_db)} note={`${s.db}`} aux={p0(s.pressure)} />
  );
  return (
    <div className="mx-side">
      <Sec title="How he’s been playing" aside={c ? `${c.db} dropbacks` : undefined}>
        {!c ? <p className="dim">No dropbacks this season yet.</p> : (<>
          <div className="mx-tiles">
            <Tile k="EPA/db" v={sg2(c.epa_db)} sub={`${p0(c.success)} success`} />
            <Tile k="Comp" v={p0(c.comp)} sub={`${n2(c.ypa, 1)} Y/A`} />
            <Tile k="aDOT" v={n2(c.adot, 1)} sub={`${p0(c.pa)} PA`} />
            <Tile k="Sacked" v={p1(c.sack)} sub={`${c.td} TD · ${c.int} INT`} />
          </div>
          <p className="mx-kv"><span>EPA vs blitz</span><b className="num">{sg2(c.blitz_epa)}</b><small className="num">{c.blitz_n} dropbacks</small></p>
          <p className="mx-kv"><span>EPA vs no blitz</span><b className="num">{sg2(c.noblitz_epa)}</b><small className="num">{c.db - c.blitz_n} dropbacks</small></p>
        </>)}
      </Sec>
      <Sec title="EPA per dropback by coverage" aside={h ? "2023–25" : undefined}>
        {!h ? <p className="dim">Under 100 charted dropbacks in 2023–25.</p> : (<>
          <p className="mx-q2">Bars run from zero; the whisker is ±1 standard error. Right: share of dropbacks pressured.</p>
          <div className="mx-hd"><span /><span /><span>EPA/db</span><span>db</span><span>press</span></div>
          {row("All", h.total)}
          {row("Man", h.man)}
          {row("Zone", h.zone)}
          <h6>By coverage shell</h6>
          {SHELLS.map((s) => row(SHELL_LABEL[s], h.shells[s]))}
          <h6>Pressure and play-action</h6>
          {row("Blitz", h.blitz)}{row("No blitz", h.no_blitz)}{row("Play-action", h.pa)}{row("No play-action", h.no_pa)}
          <p className="mx-note">A quarterback’s EPA per dropback moves about ±0.1–0.2 on a few hundred dropbacks. Whiskers that cross the “All” line mean the split doesn’t separate him from his own average. These are descriptions, not forecasts.</p>
        </>)}
      </Sec>
    </div>
  );
}

/* ---------------- defense side ---------------- */
function DefensePanel({ v, data, wide }: { v: MxDefenseView; data: MxBoard; wide?: boolean }) {
  const d = v.defense, lg = d.league, base = lg.base, c = d.current, r = d.run;
  const last = lastOf(d.history);
  const [tab, setTab] = useState<"play" | "onoff">("play");
  const mineFacing = [...data.mine, ...data.theirs].filter((x) => x.opp === d.abbr);
  const gp = data.teams[d.abbr]?.pass_grade, gr = data.teams[d.abbr]?.run_grade;
  return (
    <div className={`mx-side def ${wide ? "wide" : ""}`}>
      {gp && gr && (
        <Sec title={`How tough ${d.abbr} is`} aside="one rating per defense · place in line of 32, 1 = toughest">
          <div className="mx-tiles two">
            <Tile k="Vs the pass" v={gp.label} sub={`${ord(gp.rank)} of ${gp.of} · ${gp.plays.toLocaleString()} plays`} />
            <Tile k="Vs the run" v={gr.label} sub={`${ord(gr.rank)} of ${gr.of} · ${gr.plays.toLocaleString()} plays`} />
          </div>
          <p className="mx-note r">Tough is the top 11, Soft the bottom 11. The rating is EPA allowed on every play this season (counted {p0(gp.w_now)} now, more each week) blended with last season. A defense’s rating does predict the next game’s offense, but ESPN’s projections already move with it: it explains why a number is high or low, it is not a reason to change one.</p>
        </Sec>
      )}
      <Sec title={`${d.abbr} scheme`} aside="charted coverage · their own history">
        <ManLine hist={d.history} league={Object.fromEntries(d.history.map((h) => [String(h.season), (lg[String(h.season)] as { man: number }).man]))} expected={d.expected?.man ?? null} base={base.man} />
        <p className="mx-note r">Solid: {d.abbr}’s man rate. Dashed grey: the league’s that season ({d.history.map((h) => p0((lg[String(h.season)] as { man: number }).man)).join(" → ")}, so a team is only ever read against its own year). Hollow: expected for {data.season}, keeping about {p0(d.persist.man)} of last year’s distance from the league.</p>
        {d.expected && last && (<>
          <Stack label={`${data.season} expected`} parts={shellParts(d.expected.shell)} />
          <Stack label={<span className="dim">{last.season}</span>} parts={shellParts(last.shell)} />
          <Stack label={<span className="dim">League {base.season}</span>} parts={shellParts(base.shell)} />
          <StackKey names={SHELLS.map((s) => SHELL_LABEL[s])} />
        </>)}
        <h6>Pressure (tick = league)</h6>
        {c && <Meter label="Blitz rate" v={c.blitz_exp} tick={lg.cur.blitz} max={0.6} val={p0(c.blitz_exp)} note={`now ${p0(c.blitz)}`} thin={c.games < 6} title="Blend of this season so far and last season (5 games of prior)" />}
        {c && <Meter label="5+ rushers" v={c.pr5} tick={lg.cur.pr5} max={0.6} val={p0(c.pr5)} note={`lg ${p0(lg.cur.pr5)}`} thin={c.games < 6} />}
        {last && <Meter label={`Pressure ${last.season}`} v={last.pressure} tick={(lg[String(last.season)] as { pressure: number }).pressure} max={0.5} val={p0(last.pressure)} note={`lg ${p0((lg[String(last.season)] as { pressure: number }).pressure)}`} />}
      </Sec>
      <Sec title="This season so far" aside={c ? `${c.games} games · ${c.n} dropbacks${c.games < 6 ? " · thin" : ""}` : undefined}>
        {!c ? <p className="dim">No games yet.</p> : (
          <table className="mx-tbl"><thead><tr><th /><th>{d.abbr}</th><th>League</th><th>vs lg</th></tr></thead><tbody>
            {([
              ["Pass EPA / dropback", c.epa_db, lg.cur.epa_db, "e"], ["Success rate", c.success, lg.cur.success, "p"], ["Completion", c.comp, lg.cur.comp, "p"],
              ["Sack rate", c.sack, lg.cur.sack, "p"],
              ...(r ? [["Run EPA / rush", r.epa, lg.cur_run.epa, "e"], ["Yards / carry", r.ypc, lg.cur_run.ypc, "y"], ["Runs of 10+", r.explosive, lg.cur_run.explosive, "p"]] : []),
            ] as [string, number, number, "e" | "p" | "y"][]).map(([k, a, b, t]) => (
              <tr key={k}><td>{k}</td><td className="num">{t === "e" ? sg2(a) : t === "p" ? p1(a) : n2(a, 1)}</td><td className="num dim">{t === "e" ? sg2(b) : t === "p" ? p1(b) : n2(b, 1)}</td>
                <td className="num">{t === "e" ? sg2(a - b) : t === "p" ? pp(a - b) : sg2(a - b, 1)}</td></tr>
            ))}
          </tbody></table>
        )}
        {c && c.games < 6 && <p className="mx-note">Early-season defensive numbers are thin: lean on the scheme history above, not these.</p>}
      </Sec>
      <Sec title="What they allow" aside={`${data.season}, shrunk toward the league`}>
        <div className="mx-hd"><span /><span /><span>share</span><span>tgt</span><span>PPR/t</span></div>
        {(["WR", "TE", "RB"] as const).map((k) => {
          const a = d.allowed[k];
          return <Meter key={k} label={k} v={a.share} tick={a.league_share} max={0.7} val={p0(a.share)} note={`${a.tgt}`} aux={n2(a.ppr_t)} thin={a.tgt < 40} />;
        })}
        <p className="mx-note r">Tick = league share of targets. PPR per target is pulled toward the league: WR {n2(d.allowed.WR.league_ppr_t)}, TE {n2(d.allowed.TE.league_ppr_t)}, RB {n2(d.allowed.RB.league_ppr_t)} league-wide. Dashed = under 40 targets.</p>
        <h6>By depth of target</h6>
        <table className="mx-tbl"><thead><tr><th /><th>tgt</th><th>EPA/t</th><th>lg</th><th>comp</th><th>lg</th></tr></thead><tbody>
          {DEPTH.map(([k, n]) => { const x = d.depth[k]; return x ? <tr key={k}><td>{n}</td><td className="num dim">{x.tgt}</td><td className="num">{sg2(x.epa_t)}</td><td className="num dim">{sg2(x.league_epa_t)}</td><td className="num">{p0(x.comp)}</td><td className="num dim">{p0(x.league_comp)}</td></tr> : null; })}
        </tbody></table>
      </Sec>
      {mineFacing.length > 0 && (
        <Sec title="On your slate, facing them">
          <p className="mx-chips">{mineFacing.map((x) => <span key={x.id} className={`mx-chip ${x.side}`}>{lastName(x.name)}</span>)}</p>
        </Sec>
      )}
      <Sec title="Who has hurt them" aside={`${data.season}, by PPR on targets`}>
        {d.faced.length === 0 ? <p className="dim">Nobody yet.</p> : (
          <table className="mx-tbl"><thead><tr><th /><th>pos</th><th>tgt</th><th>rec</th><th>yds</th><th>td</th><th>PPR</th></tr></thead><tbody>
            {d.faced.map((f) => <tr key={f.gsis}><td>{f.name}</td><td className="dim">{f.pos}</td><td className="num">{f.tgt}</td><td className="num">{f.rec}</td><td className="num">{f.yds}</td><td className="num">{f.td}</td><td className="num">{n2(f.ppr, 1)}</td></tr>)}
          </tbody></table>
        )}
      </Sec>
      <Sec title="Their defenders" aside={<span className="seg sm" role="group" aria-label="Defender view"><button type="button" aria-pressed={tab === "play"} onClick={() => setTab("play")}>Playmakers</button><button type="button" aria-pressed={tab === "onoff"} onClick={() => setTab("onoff")}>On / off</button></span>}>
        <p className="mx-note t">The free data records who sacks, hits, tackles and breaks up passes, and who is on the field. It does <b>not</b> record who covered whom or whose receiver caught the ball, so there is no per-defender coverage grade here.</p>
        {tab === "play" ? <Playmakers v={v} /> : <OnOff v={v} />}
      </Sec>
    </div>
  );
}

function Playmakers({ v }: { v: MxDefenseView }) {
  const t = (rows: MxDefenseView["defenders"]["current"], title: string) => (
    <>
      <h6>{title}</h6>
      <table className="mx-tbl"><thead><tr><th /><th>pos</th><th>sk</th><th>hit</th><th>tfl</th><th>int</th><th>pbu</th><th>tkl</th></tr></thead><tbody>
        {rows.slice(0, 8).map((x) => <tr key={x.gsis} className={x.still === false ? "gone" : ""}><td>{x.name}{x.still === false && <small> left</small>}</td><td className="dim">{x.pos}</td><td className="num">{x.sacks}</td><td className="num">{x.hit}</td><td className="num">{x.tfl}</td><td className="num">{x.int}</td><td className="num">{x.pbu}</td><td className="num dim">{x.tkl}</td></tr>)}
      </tbody></table>
    </>
  );
  return <>{t(v.defenders.current, "This season")}{t(v.defenders.last, "Last season")}</>;
}

function OnOff({ v }: { v: MxDefenseView }) {
  const rows = v.defenders.onoff ?? [];
  if (!rows.length) return <p className="dim">No on-field data for this team last season.</p>;
  return (
    <>
      <p className="mx-note t">Team EPA per dropback allowed with each player on the field and off it, {v.defenders.onoff_season}. Positive = the defense did <i>worse</i> with him out there. Other players on the field with him are not controlled for, so this is context, not a rating. Grey = within two standard errors of nothing.</p>
      <table className="mx-tbl onoff"><thead><tr><th /><th>pos</th><th>snaps</th><th>on</th><th>off</th><th>on − off</th></tr></thead><tbody>
        {rows.map((x) => {
          const diff = x.off_epa == null ? null : x.on_epa - x.off_epa;
          const real = diff != null && x.se != null && Math.abs(diff) > 2 * x.se;
          return (
            <tr key={x.gsis} className={x.still === false ? "gone" : ""}>
              <td>{x.name}{x.still === false && <small> left</small>}</td><td className="dim">{x.pos}</td><td className="num">{p0(x.pct)}</td>
              <td className="num">{sg2(x.on_epa)}</td><td className="num dim">{x.off_epa == null ? "–" : sg2(x.off_epa)}</td>
              <td className={`num ${real ? "" : "noise"}`}>{diff == null ? "always on" : <>{sg2(diff)}<small> ±{n2(2 * (x.se ?? 0))}</small></>}</td>
            </tr>
          );
        })}
      </tbody></table>
    </>
  );
}

/* ---------------- all defenses ---------------- */
type SortKey = "abbr" | "man" | "blitz" | "pressure" | "pass_rank" | "run_rank";
const rankOf = (t: MxBoard["teams"][string], k: "pass_rank" | "run_rank") => { const g = t[k === "pass_rank" ? "pass_grade" : "run_grade"]; return g ? g.rank : null; };
function DefenseTable({ data, rows, onPick }: { data: MxBoard; rows: MxRow[]; onPick: (a: string) => void }) {
  const [sort, setSort] = useState<{ k: SortKey; dir: 1 | -1 }>({ k: "man", dir: -1 });
  const facing = useMemo(() => {
    const m = new Map<string, MxRow[]>();
    for (const r of rows) if (r.opp) m.set(r.opp, [...(m.get(r.opp) ?? []), r]);
    return m;
  }, [rows]);
  const list = Object.entries(data.teams).sort((a, b) => {
    const val = (e: [string, MxBoard["teams"][string]]) => sort.k === "abbr" ? e[0] : sort.k === "pass_rank" || sort.k === "run_rank" ? (rankOf(e[1], sort.k) ?? 99) : (e[1][sort.k] ?? -9);
    const x = val(a), y = val(b);
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
  });
  const base = data.league.base.man, L = data.league;
  const th = (k: SortKey, label: string, title?: string) => (
    <th><button type="button" className={sort.k === k ? "on" : ""} title={title} onClick={() => setSort({ k, dir: sort.k === k ? (sort.dir === 1 ? -1 : 1) : (k === "abbr" ? 1 : -1) })}>{label}{sort.k === k ? (sort.dir === 1 ? " ↑" : " ↓") : ""}</button></th>
  );
  return (
    <div className="mx-dt">
      <h2>All defenses <small>week {data.week} · click one for its file</small></h2>
      <table className="mx-tbl big"><thead><tr>
        {th("abbr", "Defense")}<th>Faces</th>
        {th("man", "Man (exp.)", "Expected man-coverage rate this season: league baseline plus the persistent part of last season’s distance from it")}
        {th("blitz", "Blitz", "This season so far blended with last season’s rate")}
        {th("pressure", "Pressure ’25", "Share of dropbacks with a pressure, last season")}
        {th("pass_rank", "Vs the pass", "Rating from every dropback: EPA allowed this season so far, blended with last season. Rank 1 = toughest. Tough = top 11, Soft = bottom 11.")}
        {th("run_rank", "Vs the run", "Same rating on every rush. Rank 1 = toughest.")}<th>Your players facing them</th></tr></thead>
        <tbody>
          {list.map(([a, t]) => {
            const fl = facing.get(a) ?? [];
            return (
              <tr key={a} onClick={() => onPick(a)} tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") onPick(a); }}>
                <td><b>{a}</b></td><td className="dim">{data.schedule[a]?.opp ?? "bye"}</td>
                <td><Meter label="" v={t.man} tick={base} max={0.5} val={p0(t.man)} /></td>
                <td><Meter label="" v={t.blitz} tick={L.cur.blitz} max={0.5} val={p0(t.blitz)} thin={t.games < 6} /></td>
                <td><Meter label="" v={t.pressure} tick={L["2025"] ? (L["2025"] as { pressure: number }).pressure : L.pressure} max={0.45} val={p0(t.pressure)} /></td>
                <td><Grade g={t.pass_grade} /></td><td><Grade g={t.run_grade} /></td>
                <td>{fl.map((x) => <span key={x.id} className={`mx-chip ${x.side}`}>{lastName(x.name)}</span>)}</td>
              </tr>
            );
          })}
        </tbody></table>
      <p className="mx-note">“Vs the pass” and “Vs the run” are one rating per defense (EPA allowed on every play, this season so far blended with last season), ranked 1 = toughest of 32. ESPN’s projections already use it, so a soft defense is a reason a number is high, not a reason to raise it. Ticks mark the league. Man rate is the expected {data.season} value from each team’s own history against the league baseline ({p0(base)}); the other columns are this season so far ({"≤"}{Math.max(...Object.values(data.teams).map((t) => t.games))} games), which is thin until midseason.</p>
    </div>
  );
}
