import { type CSSProperties, useMemo, useState } from "react";
import type { Pos } from "../../types";
import { f1, lastName, pct, signed } from "../../format";
import { POS_VAR } from "../../components/bits";
import { useFile } from "../../components/FileContext";
import "./live.css";

/* M — Live: the matchup while games are on. It reads like a broadcast: a scorebug on top (points so
   far, the live win chance, where each side is headed), the win-chance line under it (one dot per
   scoring play of yours or theirs), then every starter as a bullet: the bar is what he has, the
   dashed tick is what his projection says he'd have by this point of his game, the grey band is
   where he's likely to finish. Pick a player for his game as a timeline. Green/red = good/bad for
   you, so their player having a big night shows red. Weakness: the rest-of-game model isn't fitted
   yet (see the notes at the bottom), so trust the bars and the counts more than the decimals. */

export interface LiveEvent { seq: number; period: number; clock: string; elapsed: number; wall?: string; text: string; rz: boolean; kind: "catch" | "target" | "carry" | "pass" | "fumble" | "two_pt"; pts: number; yards?: number; td?: boolean }
export interface LivePlayer {
  id: number; name: string; pos: Pos; team: string; slot: string; side: "me" | "opp"; proj: number; injury?: string; headshot?: string | null;
  event: string | null; state: "pre" | "in" | "post" | "bye" | "out"; elapsed: number; opp: string | null; home?: boolean;
  pts: number; line: Record<string, number>; events: LiveEvent[]; tilt: number; expected_now?: number;
  live_mean?: number; p10?: number; p50?: number; p90?: number;
  opps?: { now: number; per_game: number; expected_now: number }; share?: { tgt: number; team_pass: number; season: number | null };
  injury_live?: { period: number; clock: string; text: string };
}
export interface LiveGame { event: string; kickoff: string; state: "pre" | "in" | "post"; detail: string; period: number; clock: string; elapsed: number;
  home: { abbr: string; score: number; name: string }; away: { abbr: string; score: number; name: string }; possession: string | null; down_dist: string | null; red_zone: boolean; last_play: string | null }
export interface LiveBoard {
  generated_at: number; week: number; me_name: string; opp_name: string; games: LiveGame[]; me: LivePlayer[]; opp: LivePlayer[];
  matchup: { p_win: number; p_win_pre: number; me: { mean: number; p10: number; p90: number }; opp: { mean: number; p10: number; p90: number }; me_pts: number; opp_pts: number; me_left: number; opp_left: number };
  series: { wall: string; elapsed: number; event: string; p_win: number; who: number | null; pts: number; text: string; period: number; clock: string }[];
  insights: { tone: "up" | "down" | "flat" | "alert"; kind: string; text: string; id?: number }[];
  notes: string[]; errors: string[]; poll_s: number;
}

const chip = (pos: Pos) => ({ background: POS_VAR[pos], color: pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" });
const kickoff = (iso?: string) => (iso ? new Date(iso).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" }) : "");
const stateText = (p: LivePlayer, g?: LiveGame) =>
  p.state === "in" ? `${g?.clock ?? ""} Q${g?.period ?? ""}` : p.state === "post" ? "Final" : p.state === "bye" ? "Bye" : p.state === "out" ? "Out" : g ? kickoff(g.kickoff) : "";

function lineText(p: LivePlayer): string {
  const L = p.line || {};
  if (p.pos === "DST") return `${L.pa ?? 0} pts allowed · ${L.ya ?? 0} yds · ${L.sacks ?? 0} sk · ${(L.ints ?? 0) + (L.fr ?? 0)} TO`;
  if (p.pos === "K") return `FG ${L.fgm ?? 0}/${L.fga ?? 0} · XP ${L.xpm ?? 0}/${L.xpa ?? 0}`;
  const bits: string[] = [];
  if (L.att) bits.push(`${L.cmp ?? 0}/${L.att} · ${L.pass_yd ?? 0} yds · ${L.pass_td ?? 0} TD${L.ints ? ` · ${L.ints} INT` : ""}`);
  if (L.tgt || L.rec) bits.push(`${L.rec ?? 0} rec on ${L.tgt ?? 0} tgt${L.deep_tgt ? ` (${L.deep_tgt} deep)` : ""} · ${L.rec_yd ?? 0} yds${L.rec_td ? ` · ${L.rec_td} TD` : ""}`);
  if (L.car) bits.push(`${L.car} car · ${L.rush_yd ?? 0} yds${L.rush_td ? ` · ${L.rush_td} TD` : ""}`);
  if (L.fl) bits.push(`${L.fl} fumble lost`);
  return bits.join("  ·  ") || "no touches yet";
}

export function LiveScreen({ data, err }: { data: LiveBoard | null; err: string | null }) {
  const [selId, setSelId] = useState<number | null>(null);
  const { open } = useFile();
  const gameOf = useMemo(() => new Map((data?.games ?? []).map((g) => [g.event, g])), [data]);
  if (!data) return <div className="live"><p className="lv-empty">{err ?? "Reading the live feeds…"}</p></div>;
  const all = [...data.me, ...data.opp];
  const sel = all.find((p) => p.id === selId) ?? all.find((p) => p.side === "me" && p.state === "in") ?? all.find((p) => p.state === "in")
    ?? [...data.me].filter((p) => p.state === "post").sort((a, b) => b.pts - a.pts)[0] ?? null;
  const m = data.matchup;
  const anyLive = all.some((p) => p.state === "in" || p.state === "post");
  const scaleMax = Math.max(30, ...all.map((p) => Math.max(p.p90 ?? p.proj * 1.6, p.pts)));
  const tracked = new Set(all.filter((p) => p.event).map((p) => p.event!));
  const games = data.games.filter((g) => tracked.has(g.event)).sort((a, b) => ({ in: 0, pre: 1, post: 2 }[a.state] - { in: 0, pre: 1, post: 2 }[b.state] || a.kickoff.localeCompare(b.kickoff)));
  const d = m.p_win - m.p_win_pre;
  return (
    <div className="live">
      <section className="lv-bug" aria-label="Matchup">
        <div className="side me">
          <span className="nm">{data.me_name}</span>
          <b className="pts num">{f1(m.me_pts)}</b>
          <span className="sub">headed for <b className="num">{f1(m.me.mean)}</b> <em className="num">({f1(m.me.p10)}–{f1(m.me.p90)})</em> · {m.me_left} still to play</span>
        </div>
        <div className="mid">
          <span className="lbl">Win chance</span>
          <b className="wp num">{pct(m.p_win)}</b>
          <span className={`dl ${d > 0.005 ? "up" : d < -0.005 ? "down" : ""}`}>{Math.abs(d) < 0.005 ? "same as kickoff" : `${signed(d * 100)} pts since kickoff (${pct(m.p_win_pre)})`}</span>
          <span className="tug" aria-hidden="true"><i style={{ width: `${m.p_win * 100}%` }} /></span>
        </div>
        <div className="side opp">
          <span className="nm">{data.opp_name}</span>
          <b className="pts num">{f1(m.opp_pts)}</b>
          <span className="sub">headed for <b className="num">{f1(m.opp.mean)}</b> <em className="num">({f1(m.opp.p10)}–{f1(m.opp.p90)})</em> · {m.opp_left} still to play</span>
        </div>
      </section>

      <section className="lv-games" aria-label="Games">
        {games.map((g) => (
          <div key={g.event} className={`gm ${g.state}`}>
            <span className="tm"><b>{g.away.abbr}</b><i className="num">{g.state === "pre" ? "" : g.away.score}</i>{g.possession === g.away.abbr && g.state === "in" ? <em className="ball" title="Has the ball" /> : null}</span>
            <span className="tm"><b>{g.home.abbr}</b><i className="num">{g.state === "pre" ? "" : g.home.score}</i>{g.possession === g.home.abbr && g.state === "in" ? <em className="ball" title="Has the ball" /> : null}</span>
            <span className="st num">{g.state === "pre" ? kickoff(g.kickoff) : g.detail}</span>
            {g.state === "in" && g.down_dist && <span className={`dd ${g.red_zone ? "rz" : ""}`}>{g.down_dist}</span>}
            {g.state === "in" && <span className="who">{all.filter((p) => p.event === g.event).map((p) => <i key={p.id} className={p.side}>{p.pos === "DST" ? p.name.replace(" D/ST", " D") : lastName(p.name)}</i>)}</span>}
          </div>
        ))}
      </section>

      {anyLive && <WinLine data={data} onPick={setSelId} />}
      {anyLive && <Swings data={data} onPick={setSelId} />}

      <div className="lv-cols">
        {(["me", "opp"] as const).map((side) => (
          <section key={side} className={`lv-roster ${side}`} aria-label={side === "me" ? "Your starters" : "Their starters"}>
            <h5>{side === "me" ? data.me_name : data.opp_name}<span>bar = now · dashed tick = projected by now · band = likely finish</span></h5>
            {(side === "me" ? data.me : data.opp).map((p) => (
              <button key={p.id} type="button" className={`lv-row ${p.state} ${sel?.id === p.id ? "on" : ""}`} onClick={() => setSelId(p.id)} onDoubleClick={() => p.id > 0 && open(p.id, [])}
                title={`${p.name}: projection ${f1(p.proj)}${p.live_mean != null ? `, now headed for ${f1(p.live_mean)} (${f1(p.p10)}–${f1(p.p90)})` : ""}. Double-click for his file.`}>
                <span className="sl">{p.slot}</span>
                <span className="nm"><i className="pchip" style={chip(p.pos)}>{p.pos}</i>{p.pos === "DST" ? p.name : lastName(p.name)}{p.injury_live ? <em className="hurt" title={p.injury_live.text}>injury?</em> : null}</span>
                <span className="clk num">{stateText(p, p.event ? gameOf.get(p.event) : undefined)}</span>
                <Bullet p={p} max={scaleMax} />
                <span className="pt num" title={p.pos === "DST" && p.state === "in" ? "As it stands: includes the points-allowed and yards-allowed bonus at the current score, the way ESPN shows it. The arrow is the model's finish, which doesn't bank that bonus yet." : undefined}>{p.state === "pre" || p.state === "bye" || p.state === "out" ? <em>{f1(p.proj)}</em> : f1(p.pts)}{p.pos === "DST" && p.state === "in" ? <sup>*</sup> : null}</span>
                <span className={`hd num ${p.state === "in" ? tone(p, (p.live_mean ?? p.proj) - p.proj) : ""}`}>{p.state === "in" ? `→ ${f1(p.live_mean)}` : ""}</span>
              </button>
            ))}
          </section>
        ))}
      </div>

      {sel && <PlayerTrack p={sel} g={sel.event ? gameOf.get(sel.event) : undefined} />}

      <section className="lv-feed" aria-label="What's happening">
        <h5>What's happening</h5>
        {data.insights.length ? (
          <ul>{data.insights.map((x, i) => <li key={i}><i className={`tone ${x.tone}`} />{x.text}</li>)}</ul>
        ) : <p className="dim">Nothing yet. Notes show up once your players or theirs are playing.</p>}
        <Recent data={data} />
      </section>

      <footer className="lv-notes">
        <p>Updated {new Date(data.generated_at * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" })} · refreshes every {data.poll_s}s while a game is on · ESPN's public game feeds (unofficial).</p>
        {data.notes.map((n, i) => <p key={i}>{n}</p>)}
        {data.errors.length > 0 && <p className="err">Feed hiccups (showing the last good read): {data.errors.join(" · ")}</p>}
      </footer>
    </div>
  );
}

const tone = (p: LivePlayer, delta: number) => {
  if (Math.abs(delta) < 1) return "";
  const good = p.side === "me" ? delta > 0 : delta < 0;
  return good ? "up" : "down";
};

function Bullet({ p, max }: { p: LivePlayer; max: number }) {
  const x = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  const live = p.state === "in" || p.state === "post";
  const lo = p.p10 ?? 0, hi = p.p90 ?? p.proj;
  return (
    <span className={`bul ${p.side}`} aria-hidden="true">
      {p.state !== "bye" && p.state !== "out" && <i className="band" style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})` }} />}
      {live && <i className="now" style={{ width: x(p.pts) }} />}
      {p.state === "in" && p.expected_now != null && <i className="exp" style={{ left: x(p.expected_now) }} />}
      {p.proj > 0 && <i className="proj" style={{ left: x(p.proj) }} />}
      {p.state === "in" && p.live_mean != null && <i className="mean" style={{ left: x(p.live_mean) }} />}
    </span>
  );
}

function WinLine({ data, onPick }: { data: LiveBoard; onPick: (id: number) => void }) {
  const [hov, setHov] = useState<number | null>(null);
  const W = 1000, H = 150, padL = 34, padR = 10, padT = 10, padB = 22;
  // x = time since the first tracked kickoff (wall clock), so a Sunday with overlapping games reads in order
  const t0s = data.games.filter((g) => g.state !== "pre" && [...data.me, ...data.opp].some((p) => p.event === g.event)).map((g) => Date.parse(g.kickoff));
  const t0 = t0s.length ? Math.min(...t0s) : Date.now();
  const tNow = data.generated_at * 1000;
  const pts = [{ t: t0, p: data.matchup.p_win_pre, s: null as null | (typeof data.series)[number] },
    ...data.series.filter((s) => s.wall).map((s) => ({ t: Date.parse(s.wall), p: s.p_win, s })),
    { t: Math.max(tNow, t0 + 60_000), p: data.matchup.p_win, s: null }];
  // x = wall clock with the dead time between games squeezed out, so Thursday, the Sunday slots and
  // Monday sit side by side; tonight's game keeps room to its likely end so the line grows across.
  const GAP = 600_000, BREAK = 40 * 60_000;
  const ts = pts.map((q) => q.t).sort((a, b) => a - b);
  const segs: { a: number; b: number }[] = [];
  for (const t of ts) { const last = segs[segs.length - 1]; if (last && t - last.b <= BREAK) last.b = Math.max(last.b, t); else segs.push({ a: t, b: t }); }
  const liveKick = data.games.filter((g) => g.state === "in").map((g) => Date.parse(g.kickoff));
  if (liveKick.length) segs[segs.length - 1].b = Math.max(segs[segs.length - 1].b, Math.max(...liveKick) + 3.4 * 3600_000);
  const span = segs.reduce((x, g) => x + (g.b - g.a), 0) + GAP * (segs.length - 1) || 1;
  const off: number[] = []; segs.reduce((x, g, i) => { off[i] = x; return x + (g.b - g.a) + GAP; }, 0);
  const X = (t: number) => {
    let i = segs.findIndex((g) => t <= g.b + 1);
    if (i < 0) i = segs.length - 1;
    const g = segs[i];
    return padL + ((off[i] + Math.max(0, Math.min(g.b, t) - g.a)) / span) * (W - padL - padR);
  };
  // Zoom the y axis to where the line actually lives: a win chance that stays between 40% and 55% all
  // night drew a flat line on a 0-100% axis. 50% is always inside the range; the axis says when it is zoomed.
  const lo = Math.min(0.5, ...pts.map((q) => q.p)), hi = Math.max(0.5, ...pts.map((q) => q.p));
  let y0 = Math.max(0, Math.floor((lo - 0.06) * 20) / 20), y1 = Math.min(1, Math.ceil((hi + 0.06) * 20) / 20);
  if (y1 - y0 < 0.2) { const c = (y0 + y1) / 2; y0 = c - 0.1; y1 = c + 0.1; if (y0 < 0) { y1 -= y0; y0 = 0; } if (y1 > 1) { y0 -= y1 - 1; y1 = 1; } }
  const zoomed = y0 > 0.001 || y1 < 0.999;
  const step = y1 - y0 > 0.6 ? 0.25 : y1 - y0 > 0.3 ? 0.1 : 0.05;
  const gridAt: number[] = []; for (let g = Math.ceil(y0 / step - 1e-9) * step; g <= y1 + 1e-9; g += step) gridAt.push(Math.round(g * 100) / 100);
  const Y = (p: number) => padT + (1 - (p - y0) / (y1 - y0)) * (H - padT - padB);
  let d = `M${X(pts[0].t)},${Y(pts[0].p)}`;
  for (let i = 1; i < pts.length; i++) d += ` H${X(pts[i].t)} V${Y(pts[i].p)}`;
  const area = `${d} V${Y(0.5)} H${X(pts[0].t)} Z`;
  const ticks: { t: number; first: boolean }[] = [];
  for (const g of segs) {
    const step = g.b - g.a > 4 * 3600_000 ? 3600_000 : 1800_000;
    let first = true;
    for (let t = Math.ceil(g.a / step) * step; t <= g.b; t += step) { ticks.push({ t, first }); first = false; }
  }
  for (let i = ticks.length - 1, lastX = Infinity; i >= 0; i--) { // thin labels that would collide (keep segment starts)
    const x = X(ticks[i].t);
    if (lastX - x < 95 && !ticks[i].first) ticks.splice(i, 1); else lastX = x;
  }
  for (let i = 1; i < ticks.length; i++) if (X(ticks[i].t) - X(ticks[i - 1].t) < 95) { ticks.splice(i - 1, 1); i--; }
  const tickLabel = (t: number, first: boolean) => new Date(t).toLocaleString(undefined, first && segs.length > 1 ? { weekday: "short", hour: "numeric", minute: "2-digit" } : { hour: "numeric", minute: "2-digit" });
  const all = new Map([...data.me, ...data.opp].map((p) => [p.id, p]));
  const h = hov != null ? pts[hov] : null;
  return (
    <section className="lv-wp" aria-label="Win chance through the night">
      <h5>Win chance through the night <span>big dots = a starter of yours (blue) or theirs (plum) scoring · small dots = a score change or quarter end in their games</span></h5>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onMouseLeave={() => setHov(null)}>
        <defs>
          <clipPath id="lv-above"><rect x="0" y="0" width={W} height={Y(0.5)} /></clipPath>
          <clipPath id="lv-below"><rect x="0" y={Y(0.5)} width={W} height={H} /></clipPath>
        </defs>
        {gridAt.map((g) => <line key={g} className={Math.abs(g - 0.5) < 1e-6 ? "mid" : "grid"} x1={padL} x2={W - padR} y1={Y(g)} y2={Y(g)} />)}
        {gridAt.map((g) => <text key={g} className="ax" x={padL - 6} y={Y(g) + 4} textAnchor="end">{`${Math.round(g * 100)}%`}</text>)}
        {ticks.map(({ t, first }) => <text key={t} className="ax" x={X(t)} y={H - 6} textAnchor="middle">{tickLabel(t, first)}</text>)}
        {segs.slice(1).map((g, i) => <line key={`b${i}`} className="brk" x1={X(g.a) - 3} x2={X(g.a) - 3} y1={padT} y2={H - padB} />)}
        <path d={area} className="fill me" clipPath="url(#lv-above)" />
        <path d={area} className="fill opp" clipPath="url(#lv-below)" />
        <path d={d} className="ln" vectorEffect="non-scaling-stroke" />
        {pts.map((q, i) => q.s ? (
          <circle key={i} cx={X(q.t)} cy={Y(q.p)} r={(hov === i ? 2 : 0) + (q.s.who != null ? 4 : 2.5)} className={`dot ${q.s.who != null ? all.get(q.s.who)?.side ?? "" : "evt"}`} vectorEffect="non-scaling-stroke"
            onMouseEnter={() => setHov(i)} onClick={() => { if (q.s?.who != null) onPick(q.s.who); }} />
        ) : null)}
        <circle cx={X(pts[pts.length - 1].t)} cy={Y(data.matchup.p_win)} r={4} className="dot now" />
      </svg>
      <p className="lv-read">{h?.s
        ? <>{h.s.who != null ? <><b>{all.get(h.s.who)?.name}</b> {signed(h.s.pts)} · </> : null}Q{h.s.period} {h.s.clock} · win chance {pct(h.s.p_win)}. <span className="dim">{h.s.text}</span></>
        : <>Now {pct(data.matchup.p_win)}; {pct(data.matchup.p_win_pre)} at kickoff. Hover a dot for the play.{zoomed ? <span className="dim"> The axis is zoomed to {Math.round(y0 * 100)}–{Math.round(y1 * 100)}% so small moves show.</span> : null}</>}</p>
    </section>
  );
}

/* What moved the win chance tonight: each scoring play's change in your win chance, the five biggest,
   and the net by player. Descriptive: it explains the line above, it predicts nothing. */
function Swings({ data, onPick }: { data: LiveBoard; onPick: (id: number) => void }) {
  const ser = [...data.series].sort((a, b) => (a.wall && b.wall ? Date.parse(a.wall) - Date.parse(b.wall) : a.elapsed - b.elapsed));
  if (ser.length < 2) return null;
  let prev = data.matchup.p_win_pre;
  const steps = ser.map((x) => { const from = prev; prev = x.p_win; return { ...x, from, d: x.p_win - from }; });
  const all = new Map([...data.me, ...data.opp].map((p) => [p.id, p]));
  const nm = (p: LivePlayer) => (p.pos === "DST" ? p.name : lastName(p.name));
  // a play with no player of yours or theirs: say which game and whose starters are in it
  const where = (ev: string) => {
    const g = data.games.find((x) => x.event === ev);
    const in_ = (ps: LivePlayer[]) => ps.filter((p) => p.event === ev).map(nm).join(", ");
    const mine = in_(data.me), theirs = in_(data.opp);
    return `${g ? `${g.away.abbr} @ ${g.home.abbr}` : "Their game"}${mine ? ` · your ${mine}` : ""}${theirs ? ` · their ${theirs}` : ""}`;
  };
  // the first point's change from kickoff covers every play before it (the series keeps only the latest
  // 120 points), so it goes to "Everything else" instead of being pinned on whoever made that play
  const early = steps.shift()!;
  const top = [...steps].filter((x) => Math.abs(x.d) >= 0.005).sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, 5);
  const by = new Map<number, number>();
  let rest = early.d;
  for (const x of steps) if (x.who != null) by.set(x.who, (by.get(x.who) ?? 0) + x.d); else rest += x.d;
  const movers: [number | null, number][] = [...by.entries()].filter(([, v]) => Math.abs(v) >= 0.005).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6);
  if (movers.length && Math.abs(rest) >= 0.005) movers.push([null, rest]);
  const mx = Math.max(0.01, ...movers.map(([, v]) => Math.abs(v)));
  const end = steps[steps.length - 1].p_win;
  if (!top.length) return null;
  return (
    <section className="lv-swing" aria-label="What moved the win chance">
      <h5>What moved it <span>the five plays that moved your win chance most · what happened, not a forecast</span></h5>
      <div className={movers.length ? "sw-two" : ""}>
        <ol className="sw-top">
          {top.map((x, i) => {
            const p = x.who != null ? all.get(x.who) : undefined;
            return (
              <li key={i} onClick={() => p && onPick(p.id)} className={p ? "pick" : ""}>
                <b className={`num d ${x.d > 0 ? "up" : "down"}`}>{signed(x.d * 100)}</b>
                <span className="t">
                  {p ? <><em className={p.side}>{p.side === "me" ? "Your" : "Their"} {nm(p)}</em> {signed(x.pts)} pts</> : <>{x.text.startsWith("End of Q") ? "Quarter ended" : "Score change"} in {where(x.event)}</>}
                  {" · "}Q{x.period} {x.clock} · <span className="num">{pct(x.from)} → {pct(x.p_win)}</span>
                  <small>{x.text}</small>
                </span>
              </li>
            );
          })}
        </ol>
        {movers.length > 0 && <div className="sw-who">
          {movers.map(([id, v]) => {
            const p = id != null ? all.get(id) : undefined;
            return (
              <button key={id ?? "rest"} type="button" className="sw-row" disabled={id == null} onClick={() => id != null && onPick(id)}
                title={id == null ? "Time running off the clock, score changes and quarter ends that weren't a starter's own play, plus everything before the first play shown" : `${p?.name ?? id}: net ${signed(v * 100)} points of win chance from his plays tonight`}>
                <span className={`nm ${p?.side ?? ""}`}>{id == null ? "Everything else" : p ? `${p.side === "me" ? "your" : "their"} ${nm(p)}` : id}</span>
                <span className="trk"><i className={v >= 0 ? "up" : "down"} style={{ width: `${(50 * Math.abs(v)) / mx}%`, [v >= 0 ? "left" : "right"]: "50%" } as CSSProperties} /></span>
                <b className="num">{signed(v * 100)}</b>
              </button>
            );
          })}
          <p className="dim">Net change in win chance from each player's plays, in percentage points; right of center helped you. Together they add up to tonight's move: {pct(data.matchup.p_win_pre)} at kickoff to {pct(end)}.</p>
        </div>}
      </div>
    </section>
  );
}

function PlayerTrack({ p, g }: { p: LivePlayer; g?: LiveGame }) {
  const W = 1000, H = 170, padL = 34, padR = 64, padT = 12, padB = 40;
  const top = Math.max(p.p90 ?? 0, p.proj * 1.3, p.pts + 2, 10);
  const X = (e: number) => padL + e * (W - padL - padR);
  const Y = (v: number) => padT + (1 - v / top) * (H - padT - padB);
  const evs = p.events ?? [];
  let cum = 0;
  let d = `M${X(0)},${Y(0)}`;
  for (const e of evs) { if (!e.pts) continue; d += ` H${X(e.elapsed)} V${Y((cum += e.pts))}`; }
  const nowE = p.state === "post" ? 1 : p.elapsed;
  d += ` H${X(nowE)}`;
  const mine = p.side === "me";
  const o = p.opps, sh = p.share;
  const live = p.state === "in" || p.state === "post";
  const yMax = Math.max(20, ...evs.map((e) => Math.abs(e.yards ?? 0)));
  return (
    <section className={`lv-track ${p.side}`} aria-label={`${p.name}'s game`}>
      <h5><i className="pchip" style={chip(p.pos)}>{p.pos}</i>{p.name} <span>{p.team}{p.opp ? ` ${p.home ? "vs" : "@"} ${p.opp}` : ""} · {mine ? "yours" : "theirs"} · projection {f1(p.proj)}</span></h5>
      {!live ? <p className="lv-read">{p.state === "pre" ? `Kicks off ${g ? kickoff(g.kickoff) : "later"}. Projected ${f1(p.proj)} (${f1(p.p10)}–${f1(p.p90)}).` : p.state === "bye" ? "On bye." : "Not playing."}</p> : <>
        <div className="facts">
          <span><b className="num">{f1(p.pts)}</b> pts{p.state === "in" && p.expected_now != null ? <> · <em className={tone(p, p.pts - p.expected_now)}>{signed(p.pts - p.expected_now)}</em> vs pace</> : null}</span>
          <span>{lineText(p)}</span>
          {o && o.per_game > 0 && <span><b className="num">{o.now}</b> opportunities · usual by now <b className="num">{f1(o.expected_now)}</b> (of {f1(o.per_game)} a game)</span>}
          {sh && <span><b className="num">{sh.tgt === 0 ? "0%" : pct(sh.tgt)}</b> of {p.team}'s {sh.team_pass} pass plays{sh.season ? <> · season <b className="num">{pct(sh.season)}</b></> : null}</span>}
          {p.state === "in" && <span>rest of game ×<b className="num">{p.tilt.toFixed(2)}</b> for usage · headed for <b className="num">{f1(p.live_mean)}</b> <em className="num">({f1(p.p10)}–{f1(p.p90)})</em></span>}
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {[0.25, 0.5, 0.75].map((q) => <line key={q} className="grid" x1={X(q)} x2={X(q)} y1={padT} y2={H - padB} />)}
          {[1, 2, 3, 4].map((q) => <text key={q} className="ax" x={X((q - 0.5) / 4)} y={H - 4} textAnchor="middle">Q{q}</text>)}
          <line className="pace" x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(p.proj)} vectorEffect="non-scaling-stroke" />
          <text className="ax" x={X(1) + 6} y={Y(p.proj) + 4}>proj {f1(p.proj)}</text>
          {p.state === "in" && p.p10 != null && p.p90 != null && <>
            <rect className="band" x={X(1) - 3} width={6} y={Y(p.p90)} height={Math.max(1, Y(p.p10) - Y(p.p90))} />
            <line className="proj-to" x1={X(nowE)} y1={Y(p.pts)} x2={X(1)} y2={Y(p.live_mean ?? p.pts)} vectorEffect="non-scaling-stroke" />
          </>}
          <path d={d} className={`cum ${p.side}`} vectorEffect="non-scaling-stroke" />
          {p.state === "in" && <line className="nowline" x1={X(nowE)} x2={X(nowE)} y1={padT} y2={H - padB} />}
          {/* every look along the bottom: stem = yards, hollow = incomplete target, red ring = inside the 20 */}
          {evs.filter((e) => e.kind !== "pass").map((e, i, arr) => {
            // looks on back-to-back plays sit seconds apart: stack them instead of drawing one blob
            let lift = 0;
            for (let j = i - 1; j >= 0 && X(e.elapsed) - X(arr[j].elapsed) < 9; j--) lift++;
            const base = H - padB + 18 - lift * 7, len = (Math.abs(e.yards ?? 0) / yMax) * 14;
            return <g key={`${e.seq}${e.kind}`} className={`ev ${e.kind} ${e.rz ? "rz" : ""} ${e.td ? "td" : ""}`}>
              <title>{`Q${e.period} ${e.clock}: ${e.text}`}</title>
              {e.kind !== "target" && <line x1={X(e.elapsed)} x2={X(e.elapsed)} y1={base} y2={base - Math.max(2, len)} vectorEffect="non-scaling-stroke" />}
              <circle cx={X(e.elapsed)} cy={base} r={e.td ? 4.5 : 3} vectorEffect="non-scaling-stroke" />
            </g>;
          })}
        </svg>
        <p className="lv-read dim">Solid line: his points through the game. Diagonal: his projection spread evenly over 60 minutes. The thin line to the right is where the model now has him finishing; the bar at the end is the likely range. Dots along the bottom are his looks (hollow = incomplete, red ring = inside the 20); hover for the play.</p>
        {evs.length > 0 && <ol className="plays">{[...evs].reverse().slice(0, 8).map((e) => <li key={`${e.seq}${e.kind}`}><span className="num">Q{e.period} {e.clock}</span>{e.pts ? <b className={`num ${mine ? (e.pts > 0 ? "up" : "down") : (e.pts > 0 ? "down" : "up")}`}>{signed(e.pts)}</b> : <b className="num dim">0</b>}<span>{e.text}</span></li>)}</ol>}
      </>}
    </section>
  );
}

function Recent({ data }: { data: LiveBoard }) {
  const ev = [...data.me, ...data.opp].flatMap((p) => (p.events ?? []).filter((e) => e.kind !== "pass" || e.pts).map((e) => ({ e, p })));
  ev.sort((a, b) => (b.e.wall ?? "").localeCompare(a.e.wall ?? "") || b.e.seq - a.e.seq);
  if (!ev.length) return null;
  return (
    <>
      <h6>Latest looks</h6>
      <ol className="plays">{ev.slice(0, 8).map(({ e, p }) => (
        <li key={`${p.id}${e.seq}${e.kind}`}><span className="num">Q{e.period} {e.clock}</span><i className={`who ${p.side}`}>{lastName(p.name)}</i>
          {e.pts ? <b className={`num ${(p.side === "me") === (e.pts > 0) ? "up" : "down"}`}>{signed(e.pts)}</b> : <b className="num dim">0</b>}<span>{e.text}</span></li>
      ))}</ol>
    </>
  );
}

export function liveMarkdown(b: LiveBoard): string {
  const m = b.matchup;
  const row = (p: LivePlayer) => `| ${p.slot} | ${p.name} | ${p.state} | ${f1(p.pts)} | ${f1(p.proj)} | ${p.live_mean != null ? `${f1(p.live_mean)} (${f1(p.p10)}–${f1(p.p90)})` : ""} | ${p.state === "in" || p.state === "post" ? lineText(p) : ""} |`;
  return [
    `# Live: ${b.me_name} vs ${b.opp_name} (week ${b.week})`,
    `Score so far ${f1(m.me_pts)}–${f1(m.opp_pts)}. Win chance ${pct(m.p_win)} (kickoff ${pct(m.p_win_pre)}). Headed for ${f1(m.me.mean)} vs ${f1(m.opp.mean)}.`,
    ``, `| Slot | ${b.me_name} | State | Pts | Proj | Headed for | Line |`, `|---|---|---|---|---|---|---|`, ...b.me.map(row),
    ``, `| Slot | ${b.opp_name} | State | Pts | Proj | Headed for | Line |`, `|---|---|---|---|---|---|---|`, ...b.opp.map(row),
    ``, `Notes:`, ...b.insights.map((x) => `- ${x.text}`),
    ...(() => { // the biggest single moves in win chance tonight (descriptive)
      const ser = [...b.series].sort((x, y) => x.elapsed - y.elapsed); let prev = m.p_win_pre;
      const st = ser.map((x) => { const d = x.p_win - prev; prev = x.p_win; return { ...x, d }; }).sort((x, y) => Math.abs(y.d) - Math.abs(x.d)).slice(0, 5).filter((x) => Math.abs(x.d) >= 0.005);
      return st.length ? ["", "Biggest moves in win chance:", ...st.map((x) => `- ${signed(x.d * 100)} pts, Q${x.period} ${x.clock}: ${x.text}`)] : [];
    })(),
    ``, ...b.notes.map((n) => `_${n}_`),
  ].join("\n");
}
