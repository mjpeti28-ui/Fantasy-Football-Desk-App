import { useMemo, useState } from "react";
import type { APlayer, Analytics, RoleProfile, RolesBoard, TeamWeek } from "../../types";
import { f1, lastName, signed } from "../../format";
import { POS_VAR } from "../../components/bits";
import { useFile } from "../../components/FileContext";
import { Ev, evTip, useWidth } from "../../components/Analytics";
import { catalog, CLUSTER_SETS, fmt, GROUP_ORDER, PRESETS, type Stat } from "./stats";
import { clColor, ClusterControls, clusterMd, ClusterView, runClusters, type Row } from "./Clusters";
import { CorrView } from "./Corr";
import "./lab.css";

/* P — Lab: a bench for poking at the season, in three views.
   Scatter: any stat against any stat, the fitted line and its r for this season, the players furthest from the line
   named, each axis wearing the chip its 2021–25 backtest earned, colour by owner, position, a third stat or cluster.
   Clusters: k-means on any set of stats, with how well the groups separate and how stable they are.
   Correlations: every stat against every other. Fingerprint stats can be raw or noise-adjusted (shrunk toward the
   position average by how many games they take to settle). Overpass Mono readouts on a hairline grid;
   you = cobalt, this week's opponent = plum, other rosters grey, free agents hollow. */

const POS = ["All", "QB", "RB", "WR", "TE"] as const;
type Who = "all" | "rostered" | "fa" | "mine";
type View = "scatter" | "clusters" | "corr";
type ColorBy = "owner" | "pos" | "stat" | "cluster";
const ls = { get: (k: string, d: string) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } }, set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* not persisted */ } } };

function StatSelect({ value, onChange, stats, avail }: { value: string; onChange: (k: string) => void; stats: Stat[]; avail: Set<string> }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {GROUP_ORDER.map((g) => { const ss = stats.filter((s) => s.group === g && avail.has(s.k)); return ss.length ? <optgroup key={g} label={g}>{ss.map((s) => <option key={s.k} value={s.k}>{s.label}</option>)}</optgroup> : null; })}
    </select>
  );
}

export function LabScreen({ a, err, tw, teams, roles, onCopy }: { a: Analytics | null; err: string | null; tw: TeamWeek; teams: { id: number; name: string }[]; roles?: RolesBoard | null; onCopy: (md: string) => void }) {
  const { open } = useFile();
  const STATS = useMemo(() => catalog(roles), [roles]);
  const SBY = useMemo(() => new Map(STATS.map((s) => [s.k, s])), [STATS]);
  const [view, setViewS] = useState<View>(() => ls.get("desk.lab.view", "scatter") as View);
  const [xk, setXk] = useState(() => ls.get("desk.lab.x", "xfp"));
  const [yk, setYk] = useState(() => ls.get("desk.lab.y", "ppg"));
  const [pos, setPos] = useState<(typeof POS)[number]>(() => ls.get("desk.lab.pos", "WR") as (typeof POS)[number]);
  const [who, setWho] = useState<Who>("all");
  const [rel, setRel] = useState(() => ls.get("desk.lab.rel", "1") === "1");
  const [adj, setAdj] = useState(() => ls.get("desk.lab.adj", "1") === "1");
  const [color, setColor] = useState<ColorBy>("owner");
  const [ck, setCk] = useState("ppg");
  const [clKeys, setClKeysS] = useState<Record<string, string[]>>(() => { try { return JSON.parse(ls.get("desk.lab.clkeys", "{}")); } catch { return {}; } });
  const [kSel, setKSel] = useState<number | "auto">("auto");
  const [hov, setHov] = useState<number | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [q, setQ] = useState("");
  const [ref, W] = useWidth<HTMLDivElement>(820);
  const [wref, WW] = useWidth<HTMLDivElement>(1100);
  const setView = (v: View) => { setViewS(v); ls.set("desk.lab.view", v); };
  const setX = (k: string) => { setXk(k); ls.set("desk.lab.x", k); };
  const setY = (k: string) => { setYk(k); ls.set("desk.lab.y", k); };
  const setP = (p: (typeof POS)[number]) => { setPos(p); ls.set("desk.lab.pos", p); };
  const me = tw.team_id, opp = tw.matchup?.opp_id ?? null;

  const rows: Row[] = useMemo(() => {
    if (!a) return [];
    return a.players.filter((p) => (pos === "All" || p.pos === pos) && p.games >= 2 && (!rel || p.pool) && (who === "all" || (who === "fa" ? p.owner == null : who === "mine" ? p.owner === me : p.owner != null)))
      .map((p) => ({ p, r: roles?.players[String(p.id)] }));
  }, [a, pos, who, rel, me, roles]);
  // a stat is offered when at least half the players shown have it
  const avail = useMemo(() => new Set(STATS.filter((s) => { const n = rows.filter((r) => s.get(r.p, r.r, adj) != null).length; return n >= 8 && n >= rows.length * 0.5; }).map((s) => s.k)), [STATS, rows, adj]);
  const X = SBY.get(avail.has(xk) ? xk : "xfp") ?? STATS[1], Y = SBY.get(avail.has(yk) ? yk : "ppg") ?? STATS[0];

  const sets = CLUSTER_SETS[pos] ?? CLUSTER_SETS.All;
  const keys = (clKeys[pos] ?? sets.map((s) => s.keys.filter((k) => avail.has(k))).find((k) => k.length >= 2) ?? []).filter((k) => avail.has(k));
  const setKeys = (ks: string[]) => { const n = { ...clKeys, [pos]: ks }; setClKeysS(n); ls.set("desk.lab.clkeys", JSON.stringify(n)); };
  const needCl = view === "clusters" || color === "cluster";
  const keyStr = keys.join(",");
  const cl = useMemo(() => (needCl ? runClusters(rows, keys.map((k) => SBY.get(k)!).filter(Boolean), kSel, adj) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [needCl, rows, keyStr, kSel, adj, SBY]);
  const clOf = useMemo(() => new Map((cl?.rows ?? []).map((r, i) => [r.p.id, cl!.labels[i]])), [cl]);

  const pts = useMemo(() => rows.map(({ p, r }) => ({ p, x: X.get(p, r, adj), y: Y.get(p, r, adj) }))
    .filter((d): d is { p: APlayer; x: number; y: number } => d.x != null && d.y != null && Number.isFinite(d.x) && Number.isFinite(d.y)), [rows, X, Y, adj]);

  const fit = useMemo(() => {
    const n = pts.length;
    if (n < 5) return null;
    const mx = pts.reduce((s, d) => s + d.x, 0) / n, my = pts.reduce((s, d) => s + d.y, 0) / n;
    let sxx = 0, syy = 0, sxy = 0;
    for (const d of pts) { sxx += (d.x - mx) ** 2; syy += (d.y - my) ** 2; sxy += (d.x - mx) * (d.y - my); }
    if (sxx === 0 || syy === 0) return null;
    const b = sxy / sxx, c = my - b * mx, r = sxy / Math.sqrt(sxx * syy);
    const res = pts.map((d) => ({ ...d, res: d.y - (c + b * d.x) })).sort((u, v) => Math.abs(v.res) - Math.abs(u.res));
    const sd = Math.sqrt(res.reduce((s, d) => s + d.res ** 2, 0) / Math.max(1, n - 2));
    return { b, c, r, n, res, sd, mx, my };
  }, [pts]);

  if (!a) return <div className="lab-screen"><p className="lab-empty">{err ?? "Building the season analytics…"}</p></div>;
  const M = pos === "All" ? a.metrics : a.metrics_pos[pos] ?? a.metrics;
  const evFor = (s: Stat) => (s.fixed ? <Ev l={s.fixed} tip={s.note} /> : s.ev && M[s.ev] ? <Ev l={M[s.ev].label} tip={evTip(M[s.ev])} /> : null);
  const teamName = (t: number | null) => (t == null ? "free agent" : t === me ? "you" : teams.find((x) => x.id === t)?.name ?? `team ${t}`);
  const hasFp = STATS.some((s) => s.fp);

  const H = 470, L = 58, R = 18, T = 14, B = 46;
  const xs = pts.map((d) => d.x), ys = pts.map((d) => d.y);
  const pad = (lo: number, hi: number) => { const s = hi - lo || 1; return [lo - 0.06 * s, hi + 0.06 * s]; };
  const [x0, x1] = xs.length ? pad(Math.min(...xs), Math.max(...xs)) : [0, 1];
  const [y0, y1] = ys.length ? pad(Math.min(...ys), Math.max(...ys)) : [0, 1];
  const sx = (v: number) => L + ((W - L - R) * (v - x0)) / (x1 - x0);
  const sy = (v: number) => T + ((H - T - B) * (y1 - v)) / (y1 - y0);
  const ticks = (lo: number, hi: number) => { const span = hi - lo, st = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50].find((s) => span / s <= 7) ?? 100; const out: number[] = []; for (let t = Math.ceil(lo / st) * st; t <= hi + 1e-9; t += st) out.push(Math.round(t * 1000) / 1000); return out; };
  const med = (v: number[]) => { const s = [...v].sort((p, q2) => p - q2); return s.length ? s[Math.floor(s.length / 2)] : 0; };
  const mxx = med(xs), myy = med(ys);
  const named = new Set((fit?.res ?? []).slice(0, 6).map((d) => d.p.id));
  // also the extremes on each axis: the top dot is often the one people ask about, even when it sits on the line
  for (const k of ["x", "y"] as const) if (pts.length) named.add(pts.reduce((m, d) => (d[k] > m[k] ? d : m)).p.id);
  const hit = q.trim().length >= 2 ? pts.filter((d) => d.p.name.toLowerCase().includes(q.trim().toLowerCase())).map((d) => d.p.id) : [];
  for (const id of hit.slice(0, 3)) named.add(id);
  if (sel != null) named.add(sel);

  const C = SBY.get(avail.has(ck) ? ck : "ppg") ?? STATS[0];
  const cvals = color === "stat" ? rows.map((r) => C.get(r.p, r.r, adj)).filter((v): v is number => v != null).sort((u, v) => u - v) : [];
  const [c0, c1] = cvals.length ? [cvals[Math.floor(cvals.length * 0.05)], cvals[Math.floor(cvals.length * 0.95)]] : [0, 1];
  const cOf = new Map(color === "stat" ? rows.map((r) => [r.p.id, C.get(r.p, r.r, adj)] as const) : []);
  const fill = (p: APlayer) => {
    if (color === "pos") return POS_VAR[p.pos];
    if (color === "cluster") { const c = clOf.get(p.id); return c == null ? "var(--lab-other)" : clColor(c); }
    if (color === "stat") { const v = cOf.get(p.id); if (v == null) return "var(--lab-other)"; const t = Math.max(0, Math.min(1, (v - c0) / ((c1 - c0) || 1))); return `color-mix(in srgb, var(--grad-hi) ${Math.round(t * 100)}%, var(--grad-lo))`; }
    return p.owner === me ? "var(--you)" : p.owner === opp && opp != null ? "var(--them)" : p.owner != null ? "var(--lab-other)" : "none";
  };
  const hp = hov != null ? pts.find((d) => d.p.id === hov) : null;
  const sp = sel != null ? pts.find((d) => d.p.id === sel)?.p ?? a.players.find((p) => p.id === sel) : null;
  const preset = PRESETS.find((p) => p.x === X.k && p.y === Y.k);
  const popTxt = `${pos === "All" ? "all positions" : pos}, ${who}${rel ? ", fantasy-relevant" : ""}`;

  const md = () => {
    if (view === "clusters" && cl) return clusterMd(cl, popTxt, adj, teamName);
    const L2 = [`## Lab: ${Y.label} vs ${X.label} (${popTxt}${X.fp || Y.fp ? (adj ? ", noise-adjusted" : ", raw") : ""}), through week ${a.through_week}`, "",
      fit ? `This season: r = ${fit.r.toFixed(2)} across ${fit.n} players; line ${Y.short} = ${fit.c.toFixed(2)} + ${fit.b.toFixed(2)} × ${X.short}; typical distance from the line ±${fit.sd.toFixed(1)}.` : "Too few players for a line.",
      `Backtest labels (2021–25): ${X.short}: ${X.fixed ?? M[X.ev ?? ""]?.label_text ?? "–"}; ${Y.short}: ${Y.fixed ?? M[Y.ev ?? ""]?.label_text ?? "–"}.`, "",
      "Furthest from the line:", "", `| Player | Pos | Team | Rostered by | ${X.short} | ${Y.short} | Off the line |`, "|---|---|---|---|---|---|---|",
      ...(fit?.res ?? []).slice(0, 12).map((d) => `| ${d.p.name} | ${d.p.pos} | ${d.p.team ?? ""} | ${teamName(d.p.owner)} | ${fmt(X, d.x)} | ${fmt(Y, d.y)} | ${signed(d.res)} |`)];
    return L2.join("\n");
  };

  return (
    <div className="lab-screen" ref={wref}>
      <header className="lab-hd">
        <h2>Lab</h2>
        <div className="seg lab-views" role="tablist" aria-label="View">
          {([["scatter", "Scatter"], ["clusters", "Clusters"], ["corr", "Correlations"]] as const).map(([k, l]) => <button key={k} type="button" role="tab" aria-selected={view === k} aria-pressed={view === k} onClick={() => setView(k)}>{l}</button>)}
        </div>
        <p>{view === "scatter" ? "Any stat against any stat." : view === "clusters" ? "Group players by any set of stats." : "Every stat against every other."} This season through week {a.through_week}. Chips carry what 2021–25 says about each stat; everything drawn here describes this season only.</p>
        <button className="tbtn" type="button" onClick={() => onCopy(md())}>Copy for Claude</button>
      </header>
      <div className="lab-ctl">
        <div className="seg" role="group" aria-label="Position">{POS.map((p) => <button key={p} type="button" aria-pressed={pos === p} onClick={() => setP(p)}>{p}</button>)}</div>
        <div className="seg" role="group" aria-label="Who">{([["all", "Everyone"], ["rostered", "Rostered"], ["fa", "Free agents"], ["mine", "Yours"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={who === k} onClick={() => setWho(k)}>{l}</button>)}</div>
        <label className="chk" title="The top players at each position by points per game (what the app's percentiles use). Off: everyone with 2+ games."><input type="checkbox" checked={rel} onChange={(e) => { setRel(e.target.checked); ls.set("desk.lab.rel", e.target.checked ? "1" : "0"); }} /> Fantasy-relevant only</label>
        {hasFp && <div className="seg" role="group" aria-label="Fingerprint values" title="Fingerprint stats: this season as-is, or pulled toward the position average until they have enough games to mean something.">
          {([[false, "Raw"], [true, "Noise-adjusted"]] as const).map(([v, l]) => <button key={l} type="button" aria-pressed={adj === v} onClick={() => { setAdj(v); ls.set("desk.lab.adj", v ? "1" : "0"); }}>{l}</button>)}
        </div>}
        <span className="dim">{rows.length} players</span>
      </div>

      {view === "scatter" && <>
        <div className="lab-presets">{PRESETS.filter((p) => !p.pos || pos === "All" || p.pos === pos || (p.pos === "WR" && pos === "TE")).map((p) => <button key={p.label} type="button" aria-pressed={p.x === X.k && p.y === Y.k} onClick={() => { if (p.pos && pos !== p.pos && !(p.pos === "WR" && pos === "TE")) setP(p.pos); setX(p.x); setY(p.y); }}>{p.label}</button>)}</div>
        <div className="lab-ctl">
          <label>Y <StatSelect value={Y.k} onChange={setY} stats={STATS} avail={avail} /></label>
          <button type="button" className="swap" onClick={() => { const t = X.k; setX(Y.k); setY(t); }} title="Swap axes">⇅</button>
          <label>X <StatSelect value={X.k} onChange={setX} stats={STATS} avail={avail} /></label>
          <div className="seg" role="group" aria-label="Color">{([["owner", "By owner"], ["pos", "By position"], ["stat", "By stat"], ["cluster", "By cluster"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={color === k} onClick={() => setColor(k)}>{l}</button>)}</div>
          {color === "stat" && <label>Color <StatSelect value={C.k} onChange={setCk} stats={STATS} avail={avail} /></label>}
          <input type="search" placeholder="Find…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Highlight a player" />
        </div>
        <div className="lab-main">
          <div className="lab-plot" ref={ref} onMouseLeave={() => setHov(null)}>
            <div className="lab-read num">
              {fit ? <>r <b>{fit.r.toFixed(2)}</b> · n {fit.n} · slope {fit.b.toFixed(2)} · ±{fit.sd.toFixed(1)} off the line</> : `n ${pts.length}`}
              {preset && <span className="why">{preset.why}</span>}
            </div>
            <svg width={W} height={H} role="img" aria-label={`${Y.label} against ${X.label}, ${pts.length} players`}>
              {ticks(x0, x1).map((t) => <g key={`x${t}`}><line className="gl" x1={sx(t)} x2={sx(t)} y1={T} y2={H - B} /><text className="tk" x={sx(t)} y={H - B + 16} textAnchor="middle">{fmt(X, t)}</text></g>)}
              {ticks(y0, y1).map((t) => <g key={`y${t}`}><line className="gl" x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} /><text className="tk" x={L - 8} y={sy(t) + 3.5} textAnchor="end">{fmt(Y, t)}</text></g>)}
              <line className="medl" x1={sx(mxx)} x2={sx(mxx)} y1={T} y2={H - B} /><line className="medl" x1={L} x2={W - R} y1={sy(myy)} y2={sy(myy)} />
              <defs><clipPath id="lab-clip"><rect x={L} y={T} width={Math.max(0, W - L - R)} height={H - T - B} /></clipPath></defs>
              {fit && <line className="fit" clipPath="url(#lab-clip)" x1={sx(x0)} x2={sx(x1)} y1={sy(fit.c + fit.b * x0)} y2={sy(fit.c + fit.b * x1)} />}
              <text className="axl" x={(L + W - R) / 2} y={H - 8} textAnchor="middle">{X.label}{X.fp ? (adj ? " (noise-adjusted)" : " (raw)") : ""}</text>
              <text className="axl" transform={`translate(14 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">{Y.label}{Y.fp ? (adj ? " (noise-adjusted)" : " (raw)") : ""}</text>
              {pts.map((d) => {
                const f = fill(d.p);
                const ring = color !== "owner" && (d.p.owner === me ? "you" : d.p.owner === opp && opp != null ? "them" : "");
                return <circle key={d.p.id} className={`pt ${f === "none" ? "fa" : ""} ${ring || ""} ${sel === d.p.id ? "sel" : ""} ${hit.includes(d.p.id) ? "hit" : ""}`} cx={sx(d.x)} cy={sy(d.y)} r={sel === d.p.id || hov === d.p.id ? 6.5 : 5}
                  style={f === "none" ? undefined : { fill: f }} onMouseEnter={() => setHov(d.p.id)} onClick={() => setSel(d.p.id)} onDoubleClick={() => open(d.p.id, pts.map((x) => x.p.id))} />;
              })}
              {pts.filter((d) => named.has(d.p.id)).map((d) => {
                const right = sx(d.x) < W - 150;
                return <text key={`n${d.p.id}`} className="nm" x={sx(d.x) + (right ? 9 : -9)} y={sy(d.y) - 7} textAnchor={right ? "start" : "end"}>{lastName(d.p.name)}</text>;
              })}
            </svg>
            {hp && <div className="lab-tip" style={{ left: Math.min(sx(hp.x) + 14, W - 230), top: Math.max(sy(hp.y) - 10, 30) }}>
              <b>{hp.p.name}</b> <span className="dim">{hp.p.pos} · {hp.p.team ?? "FA"} · {teamName(hp.p.owner)}</span>
              <span>{Y.short} {fmt(Y, hp.y)} · {X.short} {fmt(X, hp.x)}{fit ? ` · ${signed(hp.y - (fit.c + fit.b * hp.x))} off the line` : ""}</span>
              {color === "stat" && <span>{C.short} {fmt(C, cOf.get(hp.p.id) ?? null)}</span>}
              {color === "cluster" && clOf.has(hp.p.id) && <span>{cl!.names[clOf.get(hp.p.id)!]}</span>}
              <span className="dim">{hp.p.games} games · double-click for his file</span>
            </div>}
            <div className="lab-axes">
              <span>Y · {Y.label} {evFor(Y)}</span><span>X · {X.label} {evFor(X)}</span>
              <span className="leg">{color === "owner" ? <><i className="k you" />yours <i className="k them" />this week's opponent <i className="k oth" />other rosters <i className="k fa" />free agents </>
                : color === "pos" ? <>{(["QB", "RB", "WR", "TE"] as const).map((p2) => <span key={p2}><i className="k" style={{ background: POS_VAR[p2] }} />{p2} </span>)}</>
                : color === "stat" ? <><span className="grad" /> {C.label}: {fmt(C, c0)} → {fmt(C, c1)} {evFor(C)} </>
                : cl ? <>{cl.names.map((n, i) => <span key={i}><i className="k" style={{ background: clColor(i) }} />{n} </span>)} <span>(from the Clusters view's stats)</span></> : <>Pick at least two stats in Clusters. </>}
                <span> · hairlines = medians · line = this season's fit</span></span>
            </div>
          </div>
          <aside className="lab-side">
            {sp ? <SelCard p={sp} r={roles?.players[String(sp.id)]} X={X} Y={Y} adj={adj} fit={fit} a={a} onOpen={() => open(sp.id, [sp.id])} teamName={teamName} />
              : <p className="dim">Click a dot to pin a player here; double-click to open his file.</p>}
            <h5>Furthest from the line</h5>
            <ol className="lab-out">
              {(fit?.res ?? []).slice(0, 10).map((d) => (
                <li key={d.p.id} className={`${d.p.owner === me ? "me" : d.p.owner === opp && opp != null ? "opp" : ""} ${sel === d.p.id ? "on" : ""}`} onClick={() => setSel(d.p.id)} onDoubleClick={() => open(d.p.id, fit!.res.slice(0, 10).map((x) => x.p.id))}>
                  <span className="nm">{d.p.name}<small>{d.p.pos} · {d.p.owner == null ? "FA" : d.p.owner === me ? "yours" : d.p.team}</small></span>
                  <span className={`num ${d.res > 0 ? "up" : "down"}`}>{signed(d.res)}</span>
                </li>
              ))}
            </ol>
            <p className="dim note">Distance from this season's line, in {Y.short} units. A big gap is a question to ask, not an answer: check each axis's chip for whether it has meant anything before.</p>
          </aside>
        </div>
      </>}

      {view === "clusters" && <>
        <ClusterControls stats={STATS} avail={avail} keys={keys} setKeys={setKeys} k={kSel} setK={setKSel} sets={sets} adj={adj} />
        {cl ? <ClusterView res={cl} W={Math.max(320, WW - 48 - 22)} me={me} opp={opp} sel={sel} setSel={setSel} open={open} teamName={teamName} />
          : <p className="lab-empty">{keys.length < 2 ? "Pick at least two stats to group players by." : "Too few players have all of these stats. Drop one, or widen who's shown."}</p>}
      </>}

      {view === "corr" && <CorrView rows={rows} stats={STATS.filter((s) => avail.has(s.k))} adj={adj} W={Math.max(320, WW - 48 - 300)} onPick={(x, y) => { setX(x); setY(y); setView("scatter"); }} />}
    </div>
  );
}

function SelCard({ p, r, X, Y, adj, fit, a, onOpen, teamName }: { p: APlayer; r?: RoleProfile; X: Stat; Y: Stat; adj: boolean; fit: { b: number; c: number } | null; a: Analytics; onOpen: () => void; teamName: (t: number | null) => string }) {
  const x = X.get(p, r, adj), y = Y.get(p, r, adj);
  return (
    <div className="lab-sel">
      <b>{p.name}</b><span className="dim">{p.pos} · {p.team ?? "FA"} · {teamName(p.owner)} · {p.games} games</span>
      <dl>
        <dt>{Y.short}</dt><dd>{fmt(Y, y)}</dd>
        <dt>{X.short}</dt><dd>{fmt(X, x)}</dd>
        {fit && x != null && y != null && <><dt>Off the line</dt><dd>{signed(y - (fit.c + fit.b * x))}</dd></>}
        <dt>Model ROS</dt><dd>{f1(p.est_ppg)}</dd>
        {p.pct.ppg != null && <><dt>Pts/g percentile</dt><dd>{Math.round(p.pct.ppg)}th</dd></>}
      </dl>
      {p.flags.length > 0 && <p className="fl">{p.flags.map((f) => f.short).join(" · ")}</p>}
      <button type="button" className="tbtn" onClick={onOpen}>Open player file</button>
      <span className="dim">through week {a.through_week}</span>
    </div>
  );
}
