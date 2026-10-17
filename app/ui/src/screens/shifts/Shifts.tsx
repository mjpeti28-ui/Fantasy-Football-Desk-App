import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Pos, ShiftRow, Shifts as ShiftsT, ShiftWin } from "../../types";
import { f1, lastName, pct, signed, throughWeek } from "../../format";
import { Face, POS_VAR, Spark } from "../../components/bits";
import { useFile } from "../../components/FileContext";
import "./shifts.css";

/* L — Shifts: the league's usage changes on one map. Across = change in snap share, up = change in
   target (or opportunity) share. The map is cut into the same zones the trend tags use, and each
   zone carries what changes like that have meant since 2016 — so where a dot lands already says
   how much to believe it. The list below is the same data as dumbbells (from → to), filterable. */

type Win = "recent" | "season";
type Who = "all" | "fa" | "mine" | "others";
type Dir = "both" | "up" | "down";
type ZoneDef = { key: string; label: string; short?: string; tone: "up" | "down" | "noise" };
const ZONES_RECENT: ZoneDef[] = [
  { key: "snap_up", label: "Snap jump", tone: "up" },
  { key: "share_up_with_snaps", label: "More looks, more snaps", short: "Both up", tone: "up" },
  { key: "share_spike", label: "Target spike, same snaps", short: "Target spike", tone: "noise" },
  { key: "share_drop", label: "Fewer looks, same snaps", short: "Fewer looks", tone: "down" },
  { key: "snap_down", label: "Snap drop", tone: "down" },
];
const ZONES_SEASON: ZoneDef[] = [
  { key: "snap_up", label: "More snaps than last year", tone: "up" },
  { key: "share_up", label: "More looks than last year", tone: "up" },
  { key: "share_down", label: "Fewer looks than last year", tone: "down" },
  { key: "snap_down", label: "Fewer snaps than last year", tone: "down" },
];
const UPS = new Set(["snap_up", "share_up_with_snaps", "share_up", "share_spike"]);
const dead = (r: ShiftRow) => /OUT|RESERVE|SUSPEN/i.test(r.injury ?? "");

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(640);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Pooled zone stats for the current position filter (RB/WR/TE, n-weighted). */
function zoneStats(z: ShiftsT["zones"], pos: Pos | "ALL", key: string, win: Win) {
  const ps = pos === "ALL" || pos === "QB" ? ["RB", "WR", "TE"] : [pos];
  let h = 0, n = 0, e = 0, en = 0, sig = 0;
  for (const p of ps) {
    const s = win === "season" ? z[p]?.season?.[key] : (z[p]?.[key] as { hold: number | null; n: number | null; espn?: { rel: number; lo: number; hi: number } | null } | undefined);
    if (!s || s.hold == null || !s.n) continue;
    h += s.hold * s.n; n += s.n;
    const es = (s as { espn?: { rel: number; lo: number; hi: number } | null }).espn;
    if (es) { e += es.rel * s.n; en += s.n; if (es.lo > 0 || es.hi < 0) sig += s.n; }
  }
  return n ? { hold: h / n, n, espn: en ? e / en : null, espnReal: en > 0 && sig / en > 0.5 } : null;
}

function ChangeMap({ rows, win, myTeam, zoneSel, onZone, data, pos }: {
  rows: ShiftRow[]; win: Win; myTeam: number; zoneSel: string | null; onZone: (z: string | null) => void; data: ShiftsT; pos: Pos | "ALL";
}) {
  const { open } = useFile();
  const [ref, W] = useWidth<HTMLDivElement>();
  const H = 380, L = 40, R = 12, T = 12, B = 34;
  const X = 60, Y = 20;
  const sx = (v: number) => L + ((Math.max(-X, Math.min(X, v)) + X) / (2 * X)) * (W - L - R);
  const sy = (v: number) => T + (1 - (Math.max(-Y, Math.min(Y, v)) + Y) / (2 * Y)) * (H - T - B);
  const th = data.thresholds;
  // QBs have no share axis: they only go on the map when you're looking at QBs (along the zero line)
  const pts = rows.map((r) => ({ r, w: r[win]! })).filter((p) => p.w && p.w.d_snap != null && (p.w.d_share != null || pos === "QB"));
  const [hov, setHov] = useState<number | null>(null);
  const near = (e: React.PointerEvent<SVGSVGElement>) => {
    const b = e.currentTarget.getBoundingClientRect(); const mx = e.clientX - b.left, my = e.clientY - b.top;
    let best = -1, bd = 16 * 16;
    pts.forEach((p, i) => { if (zoneSel && p.w.zone !== zoneSel) return; const d = (sx(p.w.d_snap!) - mx) ** 2 + (sy(p.w.d_share ?? 0) - my) ** 2; if (d < bd) { bd = d; best = i; } });
    setHov(best >= 0 ? best : null);
  };
  // zone rectangles in data units: [x0, x1, y0, y1]
  const Z: Record<string, number[]> = win === "recent"
    ? { snap_up: [th.snap, X, -Y, Y], share_up_with_snaps: [th.snap_with, th.snap, th.share, Y], share_spike: [-th.snap_with, th.snap_with, th.share, Y],
        share_drop: [-th.snap_with, th.snap_with, -Y, -th.share], snap_down: [-X, -th.snap, -Y, Y] }
    : { snap_up: [th.snap, X, -Y, Y], share_up: [-th.snap, th.snap, th.share, Y], share_down: [-th.snap, th.snap, -Y, -th.share], snap_down: [-X, -th.snap, -Y, Y] };
  const zones = win === "recent" ? ZONES_RECENT : ZONES_SEASON;
  const label = [...pts].sort((a, b) => Math.abs(b.w.d_snap!) + Math.abs(b.w.d_share ?? 0) * 2 - (Math.abs(a.w.d_snap!) + Math.abs(a.w.d_share ?? 0) * 2)).slice(0, 7).map((p) => p.r.id);
  const vmax = 20;
  const hp = hov != null ? pts[hov] : null;
  // zone labels: wide zones get their full label in a bottom corner (the top is where the big movers
  // land); the narrow "both up" strip gets a short label set vertically.
  type Box = { x0: number; x1: number; y0: number; y1: number };
  const tw = (t: string, px = 6.1) => t.length * px + 4;
  const placed: Box[] = [];
  const zoneLabs = zones.map((z) => {
    const [x0, x1, y0, y1] = Z[z.key];
    const st = zoneStats(data.zones, pos, z.key, win);
    const xa = sx(x0), xb = sx(x1), ya = sy(y1), yb = sy(y0);
    const narrow = xb - xa < 90;
    const text = `${xb - xa < 300 ? z.short ?? z.label : z.label}${st ? ` · held ${Math.round(st.hold * 100)}%` : ""}`;
    const w = tw(text, 5.9);
    let lab: { x: number; y: number; anchor: "start" | "end" | "middle"; rot?: boolean; text: string };
    if (narrow) { lab = { x: (xa + xb) / 2 + 3.5, y: ya + 6, anchor: "end", rot: true, text }; placed.push({ x0: (xa + xb) / 2 - 7, x1: (xa + xb) / 2 + 7, y0: ya + 6, y1: ya + 6 + w }); }
    else {
      const top = z.key === "share_spike" || z.key === "share_up";
      const y = top ? ya + 14 : yb - 8;
      const anchor = z.key.startsWith("snap_down") ? "start" : z.key.startsWith("snap_up") ? "end" : "middle";
      const x = anchor === "start" ? xa + 6 : anchor === "end" ? xb - 6 : (xa + xb) / 2;
      lab = { x, y, anchor, text };
      const bx0 = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
      placed.push({ x0: bx0, x1: bx0 + w, y0: y - 10, y1: y + 3 });
    }
    return { z, st, on: zoneSel === z.key, xa, xb, ya, yb, lab };
  });
  // direct labels for the biggest movers: try right, left, above, below; skip if nothing fits
  const hit = (a: Box) => placed.some((b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0);
  const dotLabs = pts.filter((p) => label.includes(p.r.id)).flatMap((p) => {
    const x = sx(p.w.d_snap!), y = sy(p.w.d_share ?? 0), t = lastName(p.r.name), w = tw(t, 6.3);
    const cands: { x: number; y: number; anchor: "start" | "end" | "middle"; b: Box }[] = [
      { x: x + 9, y: y + 3.5, anchor: "start", b: { x0: x + 8, x1: x + 8 + w, y0: y - 7, y1: y + 5 } },
      { x: x - 9, y: y + 3.5, anchor: "end", b: { x0: x - 8 - w, x1: x - 8, y0: y - 7, y1: y + 5 } },
      { x, y: y - 10, anchor: "middle", b: { x0: x - w / 2, x1: x + w / 2, y0: y - 20, y1: y - 8 } },
      { x, y: y + 18, anchor: "middle", b: { x0: x - w / 2, x1: x + w / 2, y0: y + 7, y1: y + 20 } },
    ];
    const c = cands.find((c) => c.b.x0 >= L && c.b.x1 <= W - R && c.b.y0 >= T - 2 && c.b.y1 <= H - B && !hit(c.b));
    if (!c) return [];
    placed.push(c.b);
    return [{ id: p.r.id, t, ...c }];
  });
  return (
    <div className="sh-map" ref={ref}>
      <svg width={W} height={H} role="img" aria-label={`Change map: ${pts.length} players by change in snap share (across) and ${win === "recent" ? "target share" : "share"} (up).`}
        onPointerMove={near} onPointerLeave={() => setHov(null)} onClick={() => { if (hp) open(hp.r.id, pts.map((p) => p.r.id)); }} style={{ cursor: hp ? "pointer" : "default" }}>
        {zoneLabs.map(({ z, st, on, xa, xb, ya, yb, lab }) => (
          <g key={z.key} className={`zone ${z.tone} ${on ? "on" : ""} ${zoneSel && !on ? "dim" : ""}`} onClick={(e) => { e.stopPropagation(); onZone(on ? null : z.key); }}>
            <rect x={xa} y={ya} width={xb - xa} height={yb - ya}><title>{z.label}{st ? ` · held ${Math.round(st.hold * 100)}%` : ""}</title></rect>
            <text transform={lab.rot ? `translate(${lab.x} ${lab.y}) rotate(-90)` : undefined} x={lab.rot ? 0 : lab.x} y={lab.rot ? 0 : lab.y} textAnchor={lab.anchor} className="zl">{lab.text}</text>
          </g>
        ))}
        {[-60, -40, -20, 0, 20, 40, 60].map((t) => <g key={`x${t}`}><line x1={sx(t)} x2={sx(t)} y1={H - B} y2={H - B + 4} className="tk" /><text x={sx(t)} y={H - B + 16} textAnchor="middle">{t > 0 ? `+${t}` : t}</text></g>)}
        {[-20, -10, 0, 10, 20].map((t) => <g key={`y${t}`}><text x={L - 6} y={sy(t) + 3} textAnchor="end">{t > 0 ? `+${t}` : t}</text></g>)}
        <line x1={sx(0)} x2={sx(0)} y1={T} y2={H - B} className="axis0" /><line x1={L} x2={W - R} y1={sy(0)} y2={sy(0)} className="axis0" />
        <text x={W - R} y={H - 4} textAnchor="end" className="axl">snap share change, points →</text>
        <text transform={`translate(11 ${T + 4}) rotate(90)`} className="axl">↑ {win === "recent" ? "target / opportunity" : "share"} change, points</text>
        {pts.map((p, i) => {
          const fa = !p.r.owner, mine = p.r.owner?.team_id === myTeam;
          const r = 3 + 5 * Math.sqrt(Math.max(0, Math.min(vmax, p.r.ledger?.est_ppg ?? 4)) / vmax);
          return <circle key={p.r.id} cx={sx(p.w.d_snap!)} cy={sy(p.w.d_share ?? 0)} r={r}
            className={`dot ${fa ? "fa" : mine ? "mine" : "other"} ${dead(p.r) ? "dead" : ""} ${hov === i ? "hov" : ""} ${zoneSel && p.w.zone !== zoneSel ? "dim" : ""}`}
            style={fa ? { stroke: POS_VAR[p.r.pos] } : undefined} />;
        })}
        {dotLabs.map((d) => <text key={`l${d.id}`} x={d.x} y={d.y} textAnchor={d.anchor} className="dl">{d.t}</text>)}
        {hp && (
          <g className="tip" transform={`translate(${Math.min(Math.max(sx(hp.w.d_snap!) + 12, 0), W - 236)} ${Math.max(sy(hp.w.d_share ?? 0) - 64, 2)})`}>
            <rect width={226} height={56} rx={6} />
            <text x={9} y={16} className="t1">{hp.r.name} · {hp.r.pos} {hp.r.team ?? ""}</text>
            <text x={9} y={31}>snaps {f1(hp.w.from_snap)}% → {f1(hp.w.to_snap)}%{hp.w.to_share != null ? ` · share ${f1(hp.w.from_share)}% → ${f1(hp.w.to_share)}%` : ""}</text>
            <text x={9} y={46}>{hp.r.owner ? (hp.r.owner.team_id === myTeam ? "yours" : hp.r.owner.name) : "free agent"}{hp.r.ledger?.est_ppg != null ? ` · model ${f1(hp.r.ledger.est_ppg)} ppg` : ""} · click for file</text>
          </g>
        )}
      </svg>
      <p className="sh-key">
        <span><i className="k mine" />yours</span><span><i className="k other" />other teams</span><span><i className="k fa" />free agent</span>
        <span>dot size = model's rest-of-season points per game</span><span>click a zone to filter · click a dot to open his file</span>
      </p>
    </div>
  );
}

function Dumbbell({ a, b, max, tone }: { a: number | null; b: number | null; max: number; tone: string }) {
  if (a == null || b == null) return <span className="dumb none">–</span>;
  const x = (v: number) => `${(Math.max(0, Math.min(max, v)) / max) * 100}%`;
  const lo = Math.min(a, b), hi = Math.max(a, b);
  return (
    <span className={`dumb ${tone}`} title={`${f1(a)}% → ${f1(b)}%`}>
      <span className="trk">
        <i className="rail" />
        <i className="dseg" style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})` }} />
        <i className="from" style={{ left: x(a) }} />
        <i className="to" style={{ left: x(b) }} />
      </span>
      <em className="num">{Math.round(a)}→{Math.round(b)}</em>
    </span>
  );
}

function zoneLabel(win: Win, z: string | null) {
  const zs = win === "recent" ? ZONES_RECENT : ZONES_SEASON;
  return zs.find((x) => x.key === z) ?? null;
}

function why(w: ShiftWin, win: Win): string {
  if (w.hold == null) return "";
  let s = w.cause?.length && w.hold_if_still_out != null
    ? `Held ${Math.round(w.hold * 100)}%; ${Math.round(w.hold_if_still_out * 100)}% while ${w.cause.join(", ")} stays out.`
    : `${win === "season" ? "Early-season changes like this" : "Changes like this"} held ${Math.round(w.hold * 100)}% of the time.`;
  if (w.espn) s += w.espn.lo > 0 || w.espn.hi < 0 ? ` Next week they scored ${signed(w.espn.rel)} vs ESPN's projection.` : " ESPN's next projection already reflects it.";
  return s;
}

export function shiftsMarkdown(rows: ShiftRow[], win: Win, data: ShiftsT): string {
  const L = [`## Usage shifts, ${throughWeek(data.through_week, data.partial)} (${win === "recent" ? "last two games vs earlier" : "this season vs last"}) — from Desk`, "",
    "| Player | Pos | Team | Rostered by | Snaps | Share | Zone | Held | Model ppg |", "|---|---|---|---|---|---|---|---|---|"];
  for (const r of rows.slice(0, 40)) {
    const w = r[win]!;
    L.push(`| ${r.name} | ${r.pos} | ${r.team ?? ""} | ${r.owner?.name ?? "FA"} | ${f1(w.from_snap)}→${f1(w.to_snap)}% | ${w.to_share == null ? "–" : `${f1(w.from_share)}→${f1(w.to_share)}%`} | ${zoneLabel(win, w.zone)?.label ?? "–"}${w.cause?.length ? ` (${w.cause.join(", ")} out)` : ""} | ${w.hold == null ? "–" : pct(w.hold)} | ${f1(r.ledger?.est_ppg)} |`);
  }
  L.push("", "Held = how often changes like this kept at least half their size over the next two games (2016-25).");
  return L.join("\n");
}

export function ShiftsScreen({ data, err, myTeam, onCopy, inCompare, onCompare }: {
  data: ShiftsT | null; err: string | null; myTeam: number; onCopy: (md: string) => void; inCompare: (id: number) => boolean; onCompare: (id: number) => void;
}) {
  const { open } = useFile();
  const [win, setWin] = useState<Win>("recent");
  const [pos, setPos] = useState<Pos | "ALL">("ALL");
  const [who, setWho] = useState<Who>("all");
  const [dir, setDir] = useState<Dir>("both");
  const [holds, setHolds] = useState(false);
  const [cause, setCause] = useState(false);
  const [healthy, setHealthy] = useState(true);
  const [zoneSel, setZoneSel] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"size" | "snap" | "share" | "value">("size");
  const [more, setMore] = useState(false);

  const base = useMemo(() => (data?.rows ?? []).filter((r) => {
    const w = r[win];
    if (!w || w.d_snap == null) return false;
    if (pos !== "ALL" && r.pos !== pos) return false;
    if (who === "fa" && r.owner) return false;
    if (who === "mine" && r.owner?.team_id !== myTeam) return false;
    if (who === "others" && (!r.owner || r.owner.team_id === myTeam)) return false;
    if (healthy && dead(r)) return false;
    if (q && !r.name.toLowerCase().includes(q.toLowerCase())) return false;
    return true;
  }), [data, win, pos, who, myTeam, healthy, q]);
  const list = useMemo(() => base.filter((r) => {
    const w = r[win]!;
    if (!w.zone) return false;
    if (zoneSel && w.zone !== zoneSel) return false;
    if (dir === "up" && !UPS.has(w.zone)) return false;
    if (dir === "down" && UPS.has(w.zone)) return false;
    if (holds && (w.hold ?? 0) < 0.55) return false;
    if (cause && !w.cause?.length) return false;
    return true;
  }).sort((a, b) => {
    const A = a[win]!, B = b[win]!;
    if (sort === "snap") return Math.abs(B.d_snap ?? 0) - Math.abs(A.d_snap ?? 0);
    if (sort === "share") return Math.abs(B.d_share ?? 0) - Math.abs(A.d_share ?? 0);
    if (sort === "value") return (b.ledger?.est_ppg ?? 0) - (a.ledger?.est_ppg ?? 0);
    return (Math.abs(B.d_snap ?? 0) + 2 * Math.abs(B.d_share ?? 0)) - (Math.abs(A.d_snap ?? 0) + 2 * Math.abs(A.d_share ?? 0));
  }), [base, win, zoneSel, dir, holds, cause, sort]);

  if (err) return <div className="shifts"><p className="sh-err">{err}</p></div>;
  if (!data) return <div className="shifts"><header className="sh-hd"><h2>Shifts</h2><p className="sub">Reading every team's usage…</p></header></div>;
  const ups = base.filter((r) => r[win]!.zone === "snap_up");
  const faUps = ups.filter((r) => !r.owner && !dead(r));
  const headline = win === "recent"
    ? `${ups.length} player${ups.length === 1 ? "" : "s"} jumped 15+ points in snap share over the last two games${faUps.length ? `; ${faUps.length} ${faUps.length === 1 ? "is" : "are"} on waivers` : ""}.`
    : `${ups.length} player${ups.length === 1 ? "" : "s"} are playing 15+ points more snaps than last season${faUps.length ? `; ${faUps.length} on waivers` : ""}.`;
  const shown = more ? list : list.slice(0, 40);
  // the window label differs by player (byes, missed games); show the one most players share
  const winLabel = (() => {
    const n = new Map<string, number>();
    for (const r of data.rows) { const l = r[win]?.label; if (l) n.set(l, (n.get(l) ?? 0) + 1); }
    let best = "", top = 0;
    n.forEach((c, l) => { if (c > top) { best = l; top = c; } });
    return best;
  })();
  return (
    <div className="shifts">
      <header className="sh-hd">
        <span className="eyebrow">Shifts · {throughWeek(data.through_week, data.partial)} · {winLabel}</span>
        <h2>{headline}</h2>
        <p className="sub">{win === "recent" ? "Snap jumps have held about 6 times in 10; target spikes without more snaps mostly fade, and ESPN's next projection already prices most of these. Where a dot lands tells you how much to believe it."
          : "Early-season gains over last year have held about 6 in 10 times; early drops mostly come back."}</p>
      </header>
      <div className="sh-filters" role="toolbar" aria-label="Filters">
        <div className="seg" role="group" aria-label="Window">
          <button type="button" aria-pressed={win === "recent"} onClick={() => { setWin("recent"); setZoneSel(null); }}>Last 2 games</button>
          <button type="button" aria-pressed={win === "season"} onClick={() => { setWin("season"); setZoneSel(null); }}>vs last season</button>
        </div>
        <div className="seg" role="group" aria-label="Position">
          {(["ALL", "QB", "RB", "WR", "TE"] as const).map((p) => <button key={p} type="button" aria-pressed={pos === p} onClick={() => setPos(p)}>{p === "ALL" ? "All" : p}</button>)}
        </div>
        <div className="seg" role="group" aria-label="Who has them">
          {([["all", "Everyone"], ["fa", "Free agents"], ["mine", "Yours"], ["others", "Other teams"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={who === k} onClick={() => setWho(k)}>{l}</button>)}
        </div>
        <div className="seg" role="group" aria-label="Direction">
          {([["both", "Up & down"], ["up", "Up"], ["down", "Down"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={dir === k} onClick={() => setDir(k)}>{l}</button>)}
        </div>
        <label className="tg"><input type="checkbox" checked={holds} onChange={(e) => setHolds(e.target.checked)} /> Usually holds</label>
        {win === "recent" && <label className="tg"><input type="checkbox" checked={cause} onChange={(e) => setCause(e.target.checked)} /> Teammate out</label>}
        <label className="tg"><input type="checkbox" checked={healthy} onChange={(e) => setHealthy(e.target.checked)} /> Hide out / IR</label>
        <input className="sh-q" type="search" placeholder="Find a player" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Find a player" />
        <button className="tbtn" type="button" onClick={() => onCopy(shiftsMarkdown(list, win, data))}>Copy for Claude</button>
      </div>
      <div className="sh-top">
        <ChangeMap rows={base} win={win} myTeam={myTeam} zoneSel={zoneSel} onZone={setZoneSel} data={data} pos={pos} />
        <aside className="sh-guide" aria-label="What each zone has meant">
          <h6>What changes like these have meant, 2016–25{pos !== "ALL" && pos !== "QB" ? ` (${pos}s)` : ""}</h6>
          {(win === "recent" ? ZONES_RECENT : ZONES_SEASON).map((z) => {
            const st = zoneStats(data.zones, pos, z.key, win);
            const n = base.filter((r) => r[win]!.zone === z.key).length;
            return (
              <button key={z.key} type="button" className={`zg ${z.tone} ${zoneSel === z.key ? "on" : ""}`} onClick={() => setZoneSel(zoneSel === z.key ? null : z.key)}>
                <span className="zt"><i />{z.label} <small>{n} now</small></span>
                <span className="hb"><i style={{ width: `${(st?.hold ?? 0) * 100}%` }} /></span>
                <span className="zs">{st ? `kept half its size ${Math.round(st.hold * 100)}% of the time` : "–"}{st?.espn != null && win === "recent" ? (st.espnReal ? `; ESPN misses it by ${signed(st.espn)} next week` : "; ESPN already prices it") : ""}</span>
              </button>
            );
          })}
          <p className="note">Zones use the same thresholds as the trend tags: ±15 snap points, ±5 share points. QBs have no share axis and no history here.</p>
        </aside>
      </div>
      <section className="sh-list">
        <div className="sh-lhd">
          <b>{list.length} change{list.length === 1 ? "" : "s"}</b>{zoneSel && <button type="button" className="lnk" onClick={() => setZoneSel(null)}>clear zone</button>}
          <span className="sp" />
          <span className="dim">Sort</span>
          <div className="seg" role="group" aria-label="Sort">
            {([["size", "Biggest"], ["snap", "Snaps"], ["share", "Share"], ["value", "Model value"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={sort === k} onClick={() => setSort(k)}>{l}</button>)}
          </div>
        </div>
        <table className="sh-table">
          <thead><tr><th aria-label="Compare" /><th>Player</th><th>Snap share</th><th>{pos === "RB" ? "Opportunity share" : "Target share"}</th><th>Zone</th><th className="r">Held</th><th className="r">Model ppg</th><th className="r">ESPN wk {data.week}</th><th>Weekly snaps</th></tr></thead>
          <tbody>
            {shown.map((r) => {
              const w = r[win]!, z = zoneLabel(win, w.zone);
              const tone = z?.tone ?? "noise";
              return (
                <tr key={r.id} onDoubleClick={() => open(r.id, list.map((x) => x.id))} className={dead(r) ? "dead" : ""}>
                  <td><input type="checkbox" checked={inCompare(r.id)} onChange={() => onCompare(r.id)} aria-label={`Compare ${r.name}`} title="Add to compare" /></td>
                  <td><button type="button" className="who" onClick={() => open(r.id, list.map((x) => x.id))} title="Open player file">
                    <Face p={{ headshot: r.headshot, pos: r.pos, name: r.name }} className="round" />
                    <span><b>{r.name}</b><small><i className="pchip" style={{ background: POS_VAR[r.pos], color: r.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{r.pos}</i>
                      {r.team ?? ""} · <span className={`own ${r.owner ? (r.owner.team_id === myTeam ? "mine" : "") : "fa"}`}>{r.owner ? (r.owner.team_id === myTeam ? "yours" : r.owner.name) : "free agent"}</span>
                      {r.injury && !/^(ACTIVE|NORMAL)$/i.test(r.injury) ? <em className="inj"> · {r.injury.replace(/_/g, " ").toLowerCase()}</em> : null}</small></span>
                  </button></td>
                  <td><Dumbbell a={w.from_snap} b={w.to_snap} max={100} tone={(w.d_snap ?? 0) >= 0 ? "up" : "down"} /></td>
                  <td><Dumbbell a={w.from_share} b={w.to_share} max={40} tone={(w.d_share ?? 0) >= 0 ? "up" : "down"} /></td>
                  <td><span className={`zchip ${tone}`} title={why(w, win)}>{z?.label ?? "–"}</span>{w.cause?.length ? <small className="cause"> {w.cause.join(", ")} out</small> : null}</td>
                  <td className="r num"><span className={`held ${(w.hold ?? 0) >= 0.55 ? "real" : "weak"}`} title={why(w, win)}>{w.hold == null ? "–" : pct(w.hold)}</span></td>
                  <td className="r num">{f1(r.ledger?.est_ppg)}</td>
                  <td className="r num">{f1(r.espn_week)}</td>
                  <td><Spark w={r.series.snap} bw={5} h={16} color={POS_VAR[r.pos]} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {list.length > 40 && !more && <button type="button" className="tbtn" onClick={() => setMore(true)}>Show all {list.length}</button>}
        {!list.length && <p className="dim">No changes match these filters.</p>}
        <p className="sh-foot">Snap and share come from nflverse play-by-play and snap counts. “Held” = how often changes like this kept at least half their size over the next two games, 2016–25, by position. Spark = snap share by week, scaled to 25+. Double-click a row or click a name for the full file; tick a box to compare.</p>
      </section>
    </div>
  );
}
