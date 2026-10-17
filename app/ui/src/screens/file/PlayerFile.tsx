import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Comps, FpMetric, LogRow, PlayerFile as PF, Pos, RoleTrend } from "../../types";
import { f1, pct, signed } from "../../format";
import { Face, POS_VAR, Spark } from "../../components/bits";
import { type TeamColor, tilePalette } from "../../teamColor";
import { holdText } from "../../components/Role";
import { useFile } from "../../components/FileContext";
import "./file.css";
import type { APlayer, Analytics } from "../../types";
import { FileAnalytics, analyticsLines } from "./FileAnalytics";

/* K — the player file. A card back for one player: who he is (role map), which of his numbers are
   real yet (fingerprint), what happened week by week, and what happened to players like him. */

const fmtv = (fmt: string | undefined, v: number | null | undefined) =>
  v == null ? "–" : fmt === "pct100" ? `${Math.round(v)}%` : fmt === "pct" ? `${Math.round(v * 100)}%` : fmt === "num2" ? v.toFixed(2) : v.toFixed(1);
const ord = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th"}`;

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function ticks(lo: number, hi: number, n = 5) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

/* ---------------- role map ---------------- */
function RoleMap({ f, myTeam, onOpen }: { f: PF; myTeam: number; onOpen: (id: number) => void }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const m = f.map, pos = f.pos;
  const H = 340, L = 48, R = 14, T = 14, B = 40;
  const me = m.trail[m.trail.length - 1];
  const peers = m.peers.filter((p) => p.id !== f.id);
  const xs = [...peers.map((p) => p.x), ...m.trail.map((t) => t.x), ...m.groups.map((g) => g.x), ...(m.last_season ? [m.last_season.x] : [])];
  const ys = [...peers.map((p) => p.y), ...m.trail.map((t) => t.y), ...m.groups.map((g) => g.y), ...(m.last_season ? [m.last_season.y] : [])];
  const q = (a: number[], p: number) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]; };
  let [x0, x1] = [q(xs, 0.01), q(xs, 0.99)], [y0, y1] = [q(ys, 0.01), q(ys, 0.99)];
  const px = (x1 - x0) * 0.08 || 0.05, py = (y1 - y0) * 0.08 || 0.05;
  x0 = Math.max(m.fmt[m.x]?.startsWith("pct") ? 0 : -Infinity, x0 - px); x1 += px; y0 = Math.max(m.fmt[m.y]?.startsWith("pct") ? 0 : -Infinity, y0 - py); y1 += py;
  const sx = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const kx = (W - L - R) / (x1 - x0), ky = (H - T - B) / (y1 - y0);
  const vmax = Math.max(0.01, ...peers.map((p) => p.vol ?? 0));
  const [hover, setHover] = useState<number | null>(null);
  const near = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect(); const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 18 * 18;
    peers.forEach((p, i) => { const d = (sx(p.x) - mx) ** 2 + (sy(p.y) - my) ** 2; if (d < bd) { bd = d; best = i; } });
    setHover(best >= 0 ? best : null);
  };
  const hp = hover != null ? peers[hover] : null;
  const mine = f.role.style?.name;
  return (
    <div className="pf-map" ref={ref}>
      <svg width={W} height={H} role="img" aria-label={`Role map: ${m.xlabel} across, ${m.ylabel} up. ${f.name} at ${fmtv(m.fmt[m.x], me?.x)}, ${fmtv(m.fmt[m.y], me?.y)}.`}
        onPointerMove={near} onPointerLeave={() => setHover(null)} onClick={() => { if (hp?.id) onOpen(hp.id); }} style={{ cursor: hp?.id ? "pointer" : "default" }}>
        {ticks(x0, x1).map((t) => <g key={`x${t}`}><line x1={sx(t)} x2={sx(t)} y1={T} y2={H - B} className="gl" /><text x={sx(t)} y={H - B + 14} textAnchor="middle">{fmtv(m.fmt[m.x], t)}</text></g>)}
        {ticks(y0, y1).map((t) => <g key={`y${t}`}><line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className="gl" /><text x={L - 6} y={sy(t) + 3} textAnchor="end">{fmtv(m.fmt[m.y], t)}</text></g>)}
        <text x={L + (W - L - R) / 2} y={H - 6} textAnchor="middle" className="axl">{m.xlabel} →</text>
        <text transform={`translate(12 ${T + (H - T - B) / 2}) rotate(-90)`} textAnchor="middle" className="axl">{m.ylabel} →</text>
        <defs><clipPath id={`clip-${f.id}`}><rect x={L} y={T} width={W - L - R} height={H - T - B} /></clipPath></defs>
        <g clipPath={`url(#clip-${f.id})`}>
        {m.groups.map((g) => {
          const a = g.cov[0][0] * kx * kx, b = -g.cov[0][1] * kx * ky, c = g.cov[1][1] * ky * ky;
          const tr = (a + c) / 2, dd = Math.sqrt(((a - c) / 2) ** 2 + b * b);
          const l1 = tr + dd, l2 = Math.max(tr - dd, 1e-6), ang = (0.5 * Math.atan2(2 * b, a - c) * 180) / Math.PI;
          const cx = sx(g.x), cy = sy(g.y);
          return (
            <g key={g.name} className={`grp ${g.name === mine ? "on" : ""}`} style={g.name === mine ? { ["--pc" as string]: POS_VAR[pos] } : undefined}>
              <ellipse cx={cx} cy={cy} rx={Math.sqrt(l1)} ry={Math.sqrt(l2)} transform={`rotate(${ang} ${cx} ${cy})`} />
            </g>
          );
        })}
        {peers.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.5 + 5 * Math.sqrt((p.vol ?? 0) / vmax)}
            className={`peer ${p.owner?.team_id === myTeam ? "mine" : ""} ${hover === i ? "hov" : ""}`} />
        ))}
        </g>
        {m.groups.map((g) => <text key={`l${g.name}`} x={Math.min(Math.max(sx(g.x), L + 60), W - R - 60)} y={Math.min(Math.max(sy(g.y), T + 10), H - B - 6)} textAnchor="middle" dy="0.35em" className={`glab ${g.name === mine ? "on" : ""}`}>{g.name}</text>)}
        {m.last_season && (
          <g className="last">
            {m.trail.length > 0 && <line x1={sx(m.last_season.x)} y1={sy(m.last_season.y)} x2={sx(m.trail[0].x)} y2={sy(m.trail[0].y)} />}
            <circle cx={sx(m.last_season.x)} cy={sy(m.last_season.y)} r={5} />
            <text x={sx(m.last_season.x) - 8} y={sy(m.last_season.y) + 14} textAnchor="end">{f.season - 1}</text>
          </g>
        )}
        {m.trail.length > 1 && <polyline className="trail" points={m.trail.map((t) => `${sx(t.x)},${sy(t.y)}`).join(" ")} style={{ stroke: POS_VAR[pos] }} />}
        {m.trail.map((t, i) => i < m.trail.length - 1 && <circle key={t.week} cx={sx(t.x)} cy={sy(t.y)} r={2.5} className="tdot" style={{ fill: POS_VAR[pos] }} />)}
        {me && <g><circle cx={sx(me.x)} cy={sy(me.y)} r={7} className="me" style={{ fill: POS_VAR[pos] }} />
          <text x={sx(me.x) > W - 130 ? sx(me.x) - 11 : sx(me.x) + 11} y={sy(me.y) - 8} textAnchor={sx(me.x) > W - 130 ? "end" : "start"} className="mel">{f.name.split(" ").slice(-1)[0]} · wk {me.week}</text></g>}
        {hp && (
          <g className="tip" transform={`translate(${Math.min(sx(hp.x) + 10, W - 190)} ${Math.max(sy(hp.y) - 46, 4)})`}>
            <rect width={180} height={40} rx={6} />
            <text x={8} y={16} className="t1">{hp.name ?? "Player"}{hp.owner ? ` · ${hp.owner.team_id === myTeam ? "yours" : hp.owner.name}` : " · free agent"}</text>
            <text x={8} y={31}>{fmtv(m.fmt[m.x], hp.x)} · {fmtv(m.fmt[m.y], hp.y)}{hp.id ? " · click to open" : ""}</text>
          </g>
        )}
      </svg>
      <p className="pf-cap">
        Every {pos} with 2+ games this season, placed by the two usage traits that separate the {f.style_info.names.length} styles (dot size = {pos === "RB" ? "opportunity" : pos === "QB" ? "dropback" : "target"} share,
        <span className="k-mine" /> on your roster). Ovals cover the middle of each style group, {(m.seasons ?? [2016, 2025])[0]}–{String((m.seasons ?? [2016, 2025])[1]).slice(2)}.{f.style_info.pff ? " Alignment (slot/inline) is PFF season to date, so his point has no weekly path." : ""} The colored path is {f.name.split(" ").slice(-1)[0]} week by week, regressed for sample size{m.last_season ? `; the ring is ${f.season - 1}` : ""}.
      </p>
    </div>
  );
}

/* ---------------- fingerprint ---------------- */
function Fingerprint({ f }: { f: PF }) {
  const groups = ["Role", "Style", "Scoring", "Efficiency"] as const;
  const G = f.season_line?.games ?? 0;
  const row = (m: FpMetric) => {
    const rawPct = m.raw_pct ?? null;
    const p = m.pct ?? 50;
    const sig = Math.round((m.rel ?? 0) * 100);
    return (
      <div key={m.key} className={`fp-row ${sig < 35 ? "noisy" : ""}`}
        title={`${m.label}. Settles after about ${m.stab_games < 1 ? "1 game" : `${Math.round(m.stab_games)} games`}: at ${G} game${G === 1 ? "" : "s"}, about ${sig}% of the gap from average is real. Raw ${fmtv(m.fmt, m.raw)}, best estimate ${fmtv(m.fmt, m.est)}${m.prev != null ? `, last season ${fmtv(m.fmt, m.prev)}` : ""}.`}>
        <span className="lab">{m.label}</span>
        <span className="track" aria-hidden="true">
          <i className="mid" />
          {rawPct != null && Math.abs(rawPct - p) > 1.5 && <i className="conn" style={{ left: `${Math.min(rawPct, p)}%`, width: `${Math.abs(rawPct - p)}%` }} />}
          {rawPct != null && <i className="raw" style={{ left: `${rawPct}%` }} />}
          <i className="est" style={{ left: `${p}%`, background: POS_VAR[f.pos] }} />
        </span>
        <span className="val num"><b>{fmtv(m.fmt, m.est)}</b>{m.raw != null && fmtv(m.fmt, m.raw) !== fmtv(m.fmt, m.est) && <small> raw {fmtv(m.fmt, m.raw)}</small>}</span>
        <span className="sig num"><span className="m"><i style={{ width: `${sig}%` }} /></span>{sig}%</span>
        <span className="rk num" title={m.group === "Style" ? "Where he sits among this season's players (style traits have no better or worse)" : "Rank this season, raw numbers"}>{m.rank && m.of ? (m.group === "Style" ? `${ord(Math.max(1, Math.round(100 * (1 - (m.rank - 1) / m.of))))} pct` : `${ord(m.rank)} / ${m.of}`) : "–"}</span>
      </div>
    );
  };
  return (
    <div className="pf-fp">
      <div className="fp-head"><span /><span className="scale"><em>0</em><em>50th</em><em>100th percentile</em></span><span>Best guess</span><span>Signal</span><span>This year</span></div>
      {groups.map((g) => {
        const ms = f.fingerprint.filter((m) => m.group === g);
        return ms.length ? <div key={g} className="fp-grp"><h6>{g}</h6>{ms.map(row)}</div> : null;
      })}
      <p className="pf-cap"><span className="k-raw" /> raw so far · <span className="k-est" style={{ background: POS_VAR[f.pos] }} /> best guess: the raw number pulled toward {f.last_season ? "last season" : "the position average"} by how much of it is real yet.
        Percentiles are against {f.pos}s with a real role, 2016–25. <b>Signal</b> = share of the gap from average that is real at this sample, from split-half reliability; faded rows are mostly noise.</p>
    </div>
  );
}
/* ---------------- by week ---------------- */
function Weeks({ f, ar }: { f: PF; ar?: APlayer }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const log = f.gamelog.filter((r) => r.week >= 1), s = f.role.series;
  const weeks = log.map((r) => r.week);
  const [hov, setHov] = useState<number | null>(null);
  if (!weeks.length) return null;
  const L = 34, R = 10, H1 = 120, H2 = 132;
  const bw = Math.min(24, ((W - L - R) / weeks.length) * 0.5);
  const x = (i: number) => L + ((W - L - R) / weeks.length) * (i + 0.5);
  const y1 = (v: number) => 8 + (1 - v / 100) * (H1 - 26);
  const snapByWeek = new Map((s?.weeks ?? []).map((w, i) => [w, s!.snap[i]]));
  const shareByWeek = new Map((s?.weeks ?? []).map((w, i) => [w, s!.share[i]]));
  const xfpOf = (w: number) => (ar && w >= 1 && w <= ar.weekly.length ? ar.weekly[w - 1]?.[2] ?? null : null);
  const pmax = Math.max(20, ...log.map((r) => Math.max(r.espn_pts ?? r.ppr ?? 0, r.espn_proj ?? 0, xfpOf(r.week) ?? 0))) * 1.1;
  const y2 = (v: number) => 8 + (1 - v / pmax) * (H2 - 30);
  const snapPts = weeks.map((w, i) => (snapByWeek.has(w) ? `${x(i)},${y1(snapByWeek.get(w)!)}` : null));
  const segs: string[][] = []; let cur: string[] = [];
  snapPts.forEach((p) => { if (p) cur.push(p); else if (cur.length) { segs.push(cur); cur = []; } }); if (cur.length) segs.push(cur);
  const shareLabel = s?.share_label ?? "Share";
  const hr = hov != null ? log[hov] : null;
  return (
    <div className="pf-weeks" ref={ref} onPointerLeave={() => setHov(null)}>
      <div className="legend">{f.pos !== "QB" && <span><i className="k-bar" style={{ background: POS_VAR[f.pos] }} />{shareLabel.toLowerCase()}</span>}<span><i className="k-line" />snap share</span></div>
      <svg width={W} height={H1} role="img" aria-label="Usage by week">
        {[0, 50, 100].map((v) => <g key={v}><line x1={L} x2={W - R} y1={y1(v)} y2={y1(v)} className="gl" /><text x={L - 6} y={y1(v) + 3} textAnchor="end">{v}%</text></g>)}
        {weeks.map((w, i) => {
          const sh = shareByWeek.get(w);
          return <g key={w}>
            <rect x={x(i) - ((W - L - R) / weeks.length) / 2} y={0} width={(W - L - R) / weeks.length} height={H1} className="hit" onPointerEnter={() => setHov(i)} />
            {sh != null && f.pos !== "QB" && <path className="bar" style={{ fill: POS_VAR[f.pos] }} d={barPath(x(i) - bw / 2, y1(sh), bw, y1(0) - y1(sh))} />}
            {!log[i].played && !log[i].upcoming && <rect x={x(i) - bw / 2} y={y1(100)} width={bw} height={y1(0) - y1(100)} className="out" />}
          </g>;
        })}
        {segs.map((sg, i) => <polyline key={i} points={sg.join(" ")} className="snapline" />)}
        {weeks.map((w, i) => snapByWeek.has(w) && <circle key={w} cx={x(i)} cy={y1(snapByWeek.get(w)!)} r={4} className="snapdot" />)}
        {hov != null && <line x1={x(hov)} x2={x(hov)} y1={4} y2={H1 - 18} className="cross" />}
      </svg>
      <div className="legend"><span><i className="k-bar up" />beat ESPN's projection</span><span><i className="k-bar down" />missed it</span><span><i className="k-tick" />ESPN projection</span>{ar && <span><i className="k-ring" />what his usage usually produces</span>}</div>
      <svg width={W} height={H2} role="img" aria-label="Fantasy points by week against ESPN's projection">
        {ticks(0, pmax, 4).map((v) => <g key={v}><line x1={L} x2={W - R} y1={y2(v)} y2={y2(v)} className="gl" /><text x={L - 6} y={y2(v) + 3} textAnchor="end">{v}</text></g>)}
        {log.map((r, i) => {
          const pts = r.espn_pts ?? (r.played ? r.ppr : null);
          const cls = r.espn_proj == null || pts == null ? "flat" : pts >= r.espn_proj ? "up" : "down";
          return <g key={r.week}>
            <rect x={x(i) - ((W - L - R) / weeks.length) / 2} y={0} width={(W - L - R) / weeks.length} height={H2} className="hit" onPointerEnter={() => setHov(i)} />
            {pts != null && <path className={`pbar ${cls}`} d={barPath(x(i) - bw / 2, y2(Math.max(pts, 0)), bw, y2(0) - y2(Math.max(pts, 0)))} />}
            {r.espn_proj != null && <line x1={x(i) - bw / 2 - 4} x2={x(i) + bw / 2 + 4} y1={y2(r.espn_proj)} y2={y2(r.espn_proj)} className="ptick" />}
            {xfpOf(r.week) != null && r.played && <circle cx={x(i)} cy={y2(xfpOf(r.week)!)} r={5} className="xring" />}
            {r.upcoming && r.espn_proj != null && <text x={x(i)} y={y2(r.espn_proj) - 6} textAnchor="middle" className="up-l">{f1(r.espn_proj)}</text>}
            {!r.played && !r.upcoming && <text x={x(i)} y={y2(0) - 6} textAnchor="middle" className="out-l">out</text>}
            <text x={x(i)} y={H2 - 6} textAnchor="middle">{r.upcoming ? `${r.week}·next` : r.week}</text>
          </g>;
        })}
        {hov != null && <line x1={x(hov)} x2={x(hov)} y1={4} y2={H2 - 20} className="cross" />}
      </svg>
      {hr && (
        <div className="wk-tip" style={{ left: Math.min(Math.max(x(hov!) - 120, 0), W - 250) }}>
          <b>Week {hr.week}{hr.opp ? ` vs ${hr.opp}` : ""}</b>
          {hr.upcoming ? <span>ESPN projects {f1(hr.espn_proj)}</span> : !hr.played ? <span>Didn't play</span> : <>
            <span>{boxLine(f.pos, hr)}</span>
            <span>{f1(hr.espn_pts ?? hr.ppr)} pts{hr.espn_proj != null ? ` vs ${f1(hr.espn_proj)} projected (${signed((hr.espn_pts ?? hr.ppr ?? 0) - hr.espn_proj)})` : ""}</span>
            {xfpOf(hr.week) != null && <span>His usage usually produces {f1(xfpOf(hr.week))}</span>}
            <span>{hr.snap != null ? `${hr.snap}% of snaps` : ""}{shareByWeek.get(hr.week) != null ? ` · ${shareByWeek.get(hr.week)}% ${shareLabel.toLowerCase()}` : ""}</span>
          </>}
        </div>
      )}
    </div>
  );
}
const barPath = (x: number, y: number, w: number, h: number) => {
  if (h <= 0) return "";
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
};
function boxLine(pos: Pos, r: LogRow) {
  const p: string[] = [];
  if (pos === "QB") p.push(`${r.cmp}/${r.att}, ${r.pass_yd} yds, ${r.pass_td} TD, ${r.ints} INT`);
  if (pos === "QB" || pos === "RB" || (r.car ?? 0) > 0) p.push(`${r.car} car, ${r.rush_yd} yds${r.rush_td ? `, ${r.rush_td} TD` : ""}`);
  if (pos !== "QB") p.push(`${r.rec}/${r.tgt} for ${r.rec_yd}${r.rec_td ? `, ${r.rec_td} TD` : ""}`);
  if (r.fl) p.push(`${r.fl} fumble lost`);
  return p.join(" · ");
}

/* ---------------- comps ---------------- */
function CompsView({ f }: { f: PF }) {
  const c = f.comps as Comps;
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hov, setHov] = useState<number | null>(null);
  const L = 16, R = 16, H = 150, base = 104;
  const est = f.ledger?.est_ppg ?? null;
  const xmax = Math.max(24, c.q.p90 + 4, c.me.ppg + 3, est ?? 0);
  const x = (v: number) => L + (Math.min(Math.max(v, 0), xmax) / xmax) * (W - L - R);
  // stack dots in 0.75-ppg bins (a simple beeswarm)
  const named = new Set(c.list.map((r) => `${r.name}${r.season}`));
  const bins = new Map<number, number>();
  const dots = c.all.map((d) => { const b = Math.round(d.ros_ppg / 0.75); const n = bins.get(b) ?? 0; bins.set(b, n + 1); return { ...d, cx: x(b * 0.75), cy: base - 6 - n * 9 }; });
  const hd = hov != null ? dots[hov] : null;
  const wk = c.weeks, hmax = Math.max(...wk.hist, 1);
  return (
    <div className="pf-comps">
      <div className="cp-top" ref={ref}>
        <svg width={W} height={H} role="img" aria-label={`Rest-of-season points per game for the ${c.k} most similar player-seasons: middle 80% ${c.q.p10} to ${c.q.p90}, median ${c.q.p50}.`}
          onPointerLeave={() => setHov(null)}>
          <rect x={x(c.q.p10)} y={base - 2} width={x(c.q.p90) - x(c.q.p10)} height={10} rx={3} className="band80" />
          <rect x={x(c.q.p25)} y={base - 2} width={x(c.q.p75) - x(c.q.p25)} height={10} rx={3} className="band50" />
          <line x1={x(c.q.p50)} x2={x(c.q.p50)} y1={base - 6} y2={base + 12} className="med" style={{ stroke: POS_VAR[f.pos] }} />
          {ticks(0, xmax, 6).map((t) => <g key={t}><line x1={x(t)} x2={x(t)} y1={base + 12} y2={base + 16} className="tk" /><text x={x(t)} y={base + 28} textAnchor="middle">{t}</text></g>)}
          {c.starter_ppg != null && <g className="mark starter"><line x1={x(c.starter_ppg)} x2={x(c.starter_ppg)} y1={14} y2={base + 12} /><text x={x(c.starter_ppg) + 4} y={22}>starter level {c.starter_ppg}</text></g>}
          <g className="mark now"><line x1={x(c.me.ppg)} x2={x(c.me.ppg)} y1={30} y2={base + 12} /><text x={x(c.me.ppg) + 4} y={38}>his {f.season} so far {c.me.ppg}</text></g>
          {est != null && <g className="mark model"><path d={`M${x(est)},${base + 13} l5,8 h-10 Z`} /><text x={x(est)} y={base + 42} textAnchor="middle">our model {f1(est)}</text></g>}
          {dots.map((d, i) => (
            <circle key={i} cx={d.cx} cy={d.cy} r={hov === i ? 5 : 4} className={`cdot ${named.has(`${d.name}${d.season}`) ? "named" : ""}`}
              style={named.has(`${d.name}${d.season}`) ? { fill: POS_VAR[f.pos] } : undefined} onPointerEnter={() => setHov(i)} />
          ))}
          {hd && <g className="tip" transform={`translate(${Math.min(Math.max(hd.cx - 90, 0), W - 180)} ${Math.max(hd.cy - 44, 0)})`}><rect width={180} height={34} rx={6} />
            <text x={8} y={14} className="t1">{hd.name} {hd.season}</text><text x={8} y={28}>{hd.ppg_then} then → {hd.ros_ppg} rest of season</text></g>}
        </svg>
        <p className="pf-cap">Each dot is one of the {c.k} player-seasons (2016–25) that looked most like him after week {c.W}, placed at the PPR points per game they scored the rest of that season.
          Bar: middle 50% (dark) and 80% (light). Highlighted dots are listed below.</p>
      </div>
      <div className="cp-tiles">
        <div><b>{c.q.p50}</b><span>median rest-of-season PPR/game</span><small>{c.q.p10}–{c.q.p90} middle 80%</small></div>
        <div><b>{pct(c.kept_role)}</b><span>kept their snap share</span><small>within 10 points</small></div>
        <div><b>{pct(c.avail)}</b><span>of remaining games played</span><small>injuries included</small></div>
        <div className="hist">
          <svg width={150} height={46} role="img" aria-label={`Weekly scores of comps after week ${c.W}: 15 or more ${pct(wk.ge15)}, under 5 ${pct(wk.lt5)}`}>
            {wk.hist.map((h, i) => <rect key={i} x={i * 7.5} y={40 - (36 * h) / hmax} width={6} height={(36 * h) / hmax} rx={1.5} className={i * 2 >= 15 ? "hi" : i * 2 < 5 ? "lo" : ""}
              style={i * 2 >= 15 ? { fill: POS_VAR[f.pos] } : undefined}><title>{`${i * 2}–${i * 2 + 2} pts: ${h} weeks`}</title></rect>)}
            <line x1={0} x2={150} y1={40.5} y2={40.5} className="gl" />
          </svg>
          <span>their weeks: <b>{pct(wk.ge15)}</b> were 15+, <b>{pct(wk.lt5)}</b> under 5</span>
        </div>
      </div>
      <table className="cp-table">
        <thead><tr><th>Similar player-season</th><th className="r">Then: PPR/g</th><th className="r">snaps</th><th className="r">After: PPR/g</th><th className="r">games</th><th>Week by week after</th></tr></thead>
        <tbody>
          {c.list.map((r) => {
            const d = r.ros_ppg - r.ppg_then;
            return (
              <tr key={`${r.name}${r.season}`}>
                <td><span className="simbar" title={`Similarity ${r.sim}`}><i style={{ width: `${Math.max(8, r.sim)}%` }} /></span><b>{r.name}</b> <span className="dim">{r.season} · {r.team}{r.age ? ` · ${Math.floor(r.age)}` : ""}</span></td>
                <td className="r num">{f1(r.ppg_then)}</td>
                <td className="r num">{Math.round(r.snap_then)}%</td>
                <td className="r num"><b>{f1(r.ros_ppg)}</b> <em className={Math.abs(d) < 1 ? "flat" : d > 0 ? "up" : "down"}>{Math.abs(d) < 1 ? "≈" : signed(d)}</em></td>
                <td className="r num">{r.ros_games}/{r.ros_team_games}</td>
                <td><Spark w={r.traj} bw={4} h={18} color={POS_VAR[f.pos]} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {c.bt && <p className="pf-cap">How much to trust this: tested on 2016–25 with each season held out, ranges like these held the real outcome {pct(c.bt.cov80)} of the time (aiming for 80%).
        As a single number the comp median misses by {c.bt.mae.toFixed(1)} PPR/game on average, the same as a simple regression on points so far and last season ({c.bt.mae_lin.toFixed(1)}) and better than points so far alone ({c.bt.mae_sofar.toFixed(1)}).
        Treat the range as the honest part; the median is not an edge over the model.</p>}
    </div>
  );
}

/* ---------------- box score ---------------- */
function BoxScore({ f }: { f: PF }) {
  const rows = f.gamelog.filter((r) => !r.upcoming && r.week >= 1);
  const pos = f.pos;
  const cols: { k: keyof LogRow | "ca" | "rt"; h: string }[] = pos === "QB"
    ? [{ k: "ca", h: "Cmp/Att" }, { k: "pass_yd", h: "Yds" }, { k: "pass_td", h: "TD" }, { k: "ints", h: "Int" }, { k: "car", h: "Car" }, { k: "rush_yd", h: "Rush yds" }, { k: "rush_td", h: "Rush TD" }]
    : pos === "RB"
      ? [{ k: "snap", h: "Snap %" }, { k: "car", h: "Car" }, { k: "rush_yd", h: "Yds" }, { k: "rush_td", h: "TD" }, { k: "rt", h: "Rec/Tgt" }, { k: "rec_yd", h: "Rec yds" }, { k: "rec_td", h: "Rec TD" }]
      : [{ k: "snap", h: "Snap %" }, { k: "rt", h: "Rec/Tgt" }, { k: "rec_yd", h: "Yds" }, { k: "rec_td", h: "TD" }, { k: "car", h: "Car" }, { k: "rush_yd", h: "Rush yds" }];
  const cell = (r: LogRow, k: string) => (!r.played ? "" : k === "ca" ? `${r.cmp}/${r.att}` : k === "rt" ? `${r.rec}/${r.tgt}` : (r as unknown as Record<string, number | null>)[k] ?? "–");
  const tot = (k: string) => rows.filter((r) => r.played).reduce((s, r) => s + (Number((r as unknown as Record<string, number | null>)[k]) || 0), 0);
  return (
    <table className="pf-box">
      <thead><tr><th>Wk</th><th>Opp</th>{cols.map((c) => <th key={c.h} className="r">{c.h}</th>)}<th className="r">Pts</th><th className="r">ESPN proj</th><th className="r">±</th></tr></thead>
      <tbody>
        {rows.map((r) => {
          const pts = r.espn_pts ?? r.ppr; const d = pts != null && r.espn_proj != null ? pts - r.espn_proj : null;
          return <tr key={r.week} className={r.played ? "" : "dnp"}><td className="num">{r.week}</td><td>{r.played ? r.opp ?? "" : r.upcoming ? "Upcoming" : "Did not play"}</td>
            {cols.map((c) => <td key={c.h} className="r num">{cell(r, c.k as string)}</td>)}
            <td className="r num"><b>{r.played ? f1(pts) : "–"}</b></td><td className="r num dim">{f1(r.espn_proj)}</td>
            <td className={`r num ${d == null ? "" : d >= 0 ? "up" : "down"}`}>{d == null || !r.played ? "" : signed(d)}</td></tr>;
        })}
        <tr className="tot"><td colSpan={2}>Season</td>
          {cols.map((c) => <td key={c.h} className="r num">{c.k === "snap" ? (f.season_line ? `${f.season_line.snap}` : "–") : c.k === "ca" ? `${tot("cmp")}/${tot("att")}` : c.k === "rt" ? `${tot("rec")}/${tot("tgt")}` : tot(c.k as string)}</td>)}
          <td className="r num"><b>{f1(rows.filter((r) => r.played).reduce((s, r) => s + (r.espn_pts ?? r.ppr ?? 0), 0))}</b></td><td className="r num dim">{f1(rows.filter((r) => r.played).reduce((s, r) => s + (r.espn_proj ?? 0), 0))}</td><td /></tr>
      </tbody>
    </table>
  );
}

/* ---------------- the file ---------------- */
const KIND: Record<string, { label: string; icon: string }> = {
  role: { label: "Who he is", icon: "◎" }, signal: { label: "Real", icon: "●" }, caution: { label: "Noise", icon: "○" },
  change: { label: "Changed", icon: "◆" }, outlook: { label: "Outlook", icon: "▸" },
};

export function fileMarkdown(f: PF, ar?: APlayer, an?: Analytics | null): string {
  const L = [`## ${f.name} (${f.pos}, ${f.pro_team ?? f.team ?? "FA"}) — player file from Desk, through week ${f.through_week}`, ""];
  f.read.forEach((r) => L.push(`- **${KIND[r.kind]?.label ?? r.kind}:** ${r.text}`));
  L.push("", "| Metric | Best guess | Raw | Percentile | Signal |", "|---|---|---|---|---|");
  f.fingerprint.forEach((m) => L.push(`| ${m.label} | ${fmtv(m.fmt, m.est)} | ${fmtv(m.fmt, m.raw)} | ${m.pct == null ? "–" : Math.round(m.pct)} | ${Math.round((m.rel ?? 0) * 100)}% |`));
  if (f.comps) {
    const c = f.comps;
    L.push("", `Comps after week ${c.W} (${c.k} most similar 2016–25): rest-of-season PPR/g median ${c.q.p50}, 80% range ${c.q.p10}–${c.q.p90}; kept snap share ${pct(c.kept_role)}; played ${pct(c.avail)} of games.`,
      ...c.list.slice(0, 8).map((r) => `- ${r.name} ${r.season} (${r.team}): ${r.ppg_then} → ${r.ros_ppg}`));
  }
  L.push(...analyticsLines(ar, an));
  L.push("", "| Wk | Opp | Pts | ESPN proj | Snap % |", "|---|---|---|---|---|", ...f.gamelog.filter((r) => !r.upcoming).map((r) => `| ${r.week} | ${r.opp ?? "–"} | ${r.played ? f1(r.espn_pts ?? r.ppr) : "DNP"} | ${f1(r.espn_proj)} | ${r.snap ?? "–"} |`));
  return L.join("\n");
}

export function PlayerFileView({ file, err, loading, nameHint, myTeam, colors, dark, hasPrev, hasNext, onPrev, onNext, onClose, onOpen, onCopy, onLog, onTrade, onWaivers, an }: {
  an?: Analytics | null;
  file: PF | null; err: string | null; loading: boolean; nameHint?: string; myTeam: number; colors?: Record<string, TeamColor>; dark: boolean;
  hasPrev: boolean; hasNext: boolean; onPrev: () => void; onNext: () => void; onClose: () => void; onOpen: (id: number) => void;
  onCopy: (f: PF) => void; onLog: (f: PF) => void; onTrade: (f: PF) => void; onWaivers: (f: PF) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const { openCompare } = useFile();
  useEffect(() => { scroller.current?.scrollTo(0, 0); }, [file?.id]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.("input, textarea")) return;
      if (e.key === "Escape" || e.key === " ") { e.preventDefault(); onClose(); }
      if (e.key === "ArrowLeft" && hasPrev) onPrev();
      if (e.key === "ArrowRight" && hasNext) onNext();
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose, onPrev, onNext, hasPrev, hasNext]);
  const pal = useMemo(() => tilePalette(file?.pro_team ?? file?.team ?? null, colors, dark), [file, colors, dark]);
  const go = (id: string) => scroller.current?.querySelector(`#pf-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const st = pal ? { ["--tbg" as string]: pal.bg, ["--tac" as string]: pal.accent, ["--thi" as string]: pal.hi, ["--tlo" as string]: pal.lo } : {};
  const f = file;
  const owner = f?.owner;
  const ar = f ? an?.players.find((p) => p.id === f.id) : undefined;
  return (
    <div className="pf-wrap" role="dialog" aria-modal="true" aria-label={f ? `${f.name} player file` : "Player file"}>
      <div className="pf-scroll" ref={scroller}>
        <header className={`pf-hd ${pal ? "team" : ""}`} style={st}>
          <div className="pf-nav">
            <button type="button" onClick={onPrev} disabled={!hasPrev} aria-label="Previous player">←</button>
            <button type="button" onClick={onNext} disabled={!hasNext} aria-label="Next player">→</button>
            <button type="button" onClick={onClose} aria-label="Close player file" className="x">Close <kbd>esc</kbd></button>
          </div>
          {f && <Face p={{ headshot: f.headshot ?? null, pos: f.pos, name: f.name }} className="pf-face" />}
          <div className="pf-id">
            <h1>{f?.name ?? nameHint ?? "Loading…"}</h1>
            {f && <div className="meta">
              <i className={`pchip ${f.pos === "WR" ? "wr" : ""}`} style={{ background: POS_VAR[f.pos] }}>{f.pos}</i>
              <span>{f.pro_team ?? f.team ?? "FA"}</span>
              {f.age && <span>age {Math.floor(f.age)}</span>}
              {f.draft && <span>{f.draft.year}{f.draft.round ? ` · round ${f.draft.round}${f.draft.pick ? `, pick ${f.draft.pick}` : ""}` : " · undrafted"}</span>}
              {f.ledger?.bye && <span>bye week {f.ledger.bye}</span>}
              <span className={`own ${owner ? (owner.team_id === myTeam ? "mine" : "other") : "fa"}`}>{owner ? (owner.team_id === myTeam ? "Your roster" : owner.name) : "Free agent"}</span>
              {owner?.injury && owner.injury !== "ACTIVE" && <span className="inj">{owner.injury.replace(/_/g, " ").toLowerCase()}</span>}
            </div>}
          </div>
          {f && <div className="pf-tiles">
            <div><span>This week, ESPN</span><b>{f1(f.espn_week)}</b><small>{f.game ? (f.game.bye ? "bye" : `${f.game.home ? "vs" : "@"} ${f.game.opp} ${f.game.day ?? ""}`) : "–"}</small></div>
            <div><span>Rest of season, our model</span><b>{f1(f.ledger?.est_ppg)}</b><small>per game · plays {pct(f.ledger?.avail_ros)}</small></div>
            <div><span>{f.season} so far</span><b>{f1(f.season_line?.ppr_g)}</b><small>{f.season_line?.rank ? `${f.pos}${f.season_line.rank} of ${f.season_line.of} · ${f.season_line.games} games` : "no games"}</small></div>
            <div><span>Players like him</span><b>{f.comps ? f.comps.q.p50 : "–"}</b><small>{f.comps ? `${f.comps.q.p10}–${f.comps.q.p90} rest of season` : "needs 2+ games"}</small></div>
          </div>}
        </header>
        {f && <nav className="pf-tabs" aria-label="Sections">
          {[["read", "The read"], ["an", "Season analytics"], ["map", "Role map"], ["fingerprint", "Fingerprint"], ["usage", "By week"], ["comps", "Similar players"], ["box", "Box score"]].map(([id, l]) =>
            <button key={id} type="button" onClick={() => go(id)}>{l}</button>)}
          <span className="sp" />
          <button type="button" className="act" onClick={() => openCompare(f.id)}>Compare with…</button>
          <button type="button" className="act" onClick={() => onCopy(f)}>Copy for Claude</button>
          <button type="button" className="act" onClick={() => onLog(f)}>Log a note…</button>
          {owner && owner.team_id !== myTeam && <button type="button" className="act" onClick={() => onTrade(f)}>Build a trade</button>}
          {!owner && <button type="button" className="act" onClick={() => onWaivers(f)}>See on Waivers</button>}
        </nav>}
        {err && <p className="pf-err">{err}</p>}
        {loading && !f && <p className="pf-load">Reading the file… (a few seconds the first time)</p>}
        {f && <div className={`pf-body ${loading ? "stale" : ""}`}>
          <section id="pf-read" className="pf-read">
            <h2>The read</h2>
            <ul>{f.read.map((r, i) => (
              <li key={i} className={r.kind}>
                <button type="button" onClick={() => go(r.anchor)} title="Show the evidence">
                  <span className="kd"><i aria-hidden="true">{KIND[r.kind]?.icon}</i>{KIND[r.kind]?.label}</span><span className="tx">{r.text}</span>
                </button>
              </li>))}
            </ul>
            {f.role.traits.length > 0 && <div className="pf-traits">{f.role.traits.map((t) => <span key={t.label} className="chip" title={t.text}>{t.label}</span>)}</div>}
          </section>
          <section id="pf-an">
            <h2>Season analytics <small>{an ? `through week ${an.through_week} · each stat with how much 2021–25 trusts it` : ""}</small></h2>
            <FileAnalytics ar={ar} a={an} />
          </section>
          <div className="pf-two">
            <section id="pf-map">
              <h2>Role map <small>{f.role.style ? `${f.role.style.name} · ${Math.round(f.role.style.p * 100)}%` : ""}{f.role.route_sh ? ` · routes on ${f.role.route_sh.v}% of dropbacks (PFF)` : ""}</small></h2>
              <StyleMix f={f} />
              <RoleMap f={f} myTeam={myTeam} onOpen={onOpen} />
            </section>
            <section id="pf-fingerprint">
              <h2>Fingerprint <small>what's real yet</small></h2>
              <Fingerprint f={f} />
            </section>
          </div>
          <section id="pf-usage">
            <h2>By week <small>usage on top, points against ESPN's projection below</small></h2>
            <Trends ts={f.role.trends} />
            <Weeks f={f} ar={ar} />
          </section>
          <section id="pf-comps">
            <h2>Similar players <small>{f.comps ? `${f.comps.k} closest of ${f.comps.pool} ${f.pos} seasons at week ${f.comps.W}, 2016–25` : ""}</small></h2>
            {f.comps ? <CompsView f={f} /> : <p className="pf-cap">Comps need at least two games this season.</p>}
          </section>
          <section id="pf-box">
            <h2>Box score</h2>
            <BoxScore f={f} />
          </section>
          <footer className="pf-foot">
            Usage from nflverse play-by-play, FTN charting and snap counts{f.style_info.pff ? "; WR/TE alignment from PFF" : ""}; points and projections from ESPN. Styles, stabilization and comps are fit on 2016–25 and tested out of sample (app/DESIGN.md).
            Nothing here feeds the valuation model; it explains, it doesn't forecast better than the model.
          </footer>
        </div>}
      </div>
    </div>
  );
}

function StyleMix({ f }: { f: PF }) {
  const st = f.role.style;
  if (!st) return null;
  const probs = Object.entries(st.probs).sort((a, b) => b[1] - a[1]);
  const G = f.season_line?.games ?? 0;
  const acc = f.style_info.early_acc?.[G < 6 ? "G3" : "G6"];
  const tr = f.style_info.transitions?.[st.name];
  return (
    <div className="pf-mix">
      <div className="bar" role="img" aria-label={probs.map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(", ")}>
        {probs.filter(([, v]) => v >= 0.02).map(([k, v], i) => <i key={k} className={i === 0 ? "top" : ""} style={{ width: `${v * 100}%`, ...(i === 0 ? { background: POS_VAR[f.pos] } : {}) }}><span>{v >= 0.18 ? `${k} ${Math.round(v * 100)}%` : ""}</span></i>)}
      </div>
      <p className="pf-cap">
        {st.from_last_season ? "Based on last season until he has 2 games. " : ""}
        {acc && G < 10 ? `At ${G < 6 ? "3" : "6"} games, labels like this matched the full-season call ${pct(acc.acc)} of the time (they claimed ${pct(acc.conf)}), so the bar is honest about its doubt. ` : ""}
        {tr ? `${st.name}s stay one the next year ${pct(tr[st.name])} of the time.` : ""}
      </p>
    </div>
  );
}

function Trends({ ts }: { ts: RoleTrend[] }) {
  if (!ts.length) return <p className="pf-cap">No snap or share changes big enough to flag.</p>;
  return (
    <ul className="pf-trends">
      {ts.map((t) => <li key={t.kind} className={["snap_up", "share_up", "bigger_role"].includes(t.kind) ? "up" : ["snap_down", "share_down", "smaller_role"].includes(t.kind) ? "down" : "flat"}>
        <b>{t.label}</b> {t.text} <span className="h">{holdText(t)}</span></li>)}
    </ul>
  );
}
