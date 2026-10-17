import { useMemo, useState } from "react";
import type { ALabel, APlayer, Analytics, TeamWeek } from "../../types";
import { f1, signed } from "../../format";
import { POS_VAR } from "../../components/bits";
import { useFile } from "../../components/FileContext";
import { Ev, evTip, useWidth } from "../../components/Analytics";
import "./lab.css";

/* P — Lab: a bench for poking at the season. Any stat against any stat for every relevant QB/RB/WR/TE,
   the fitted line and its r for this season, the players furthest from the line named on the plot, and
   each axis wearing the chip its 2021–25 backtest earned. Overpass Mono readouts on a hairline grid;
   you = cobalt, this week's opponent = plum, other rosters grey, free agents hollow. */

type Fmt = "num" | "pct" | "pp" | "sgn" | "rank";
interface Stat { k: string; label: string; short: string; fmt: Fmt; get: (p: APlayer) => number | null; ev?: string; fixed?: ALabel; note?: string }
const pctOf = (x: number | null) => (x == null ? null : 100 * x);
const STATS: Stat[] = [
  { k: "ppg", label: "Points per game", short: "Pts/g", fmt: "num", get: (p) => p.ppg, ev: "ppg" },
  { k: "xfp", label: "Expected points per game (usage)", short: "Usage xFP", fmt: "num", get: (p) => p.xfp, ev: "xfp" },
  { k: "fpoe", label: "Scored over usage, per game", short: "Over usage", fmt: "sgn", get: (p) => p.fpoe, ev: "fpoe" },
  { k: "est_ppg", label: "Model, rest of season (ppg)", short: "Model ROS", fmt: "num", get: (p) => p.est_ppg, fixed: "descriptive", note: "The model's own number: what everything else is measured against." },
  { k: "opp", label: "Targets + carries per game", short: "Opps/g", fmt: "num", get: (p) => p.opp, ev: "opp" },
  { k: "snap", label: "Snap share", short: "Snap %", fmt: "pct", get: (p) => pctOf(p.snap), ev: "snap" },
  { k: "tsh", label: "Target share", short: "Tgt share", fmt: "pct", get: (p) => pctOf(p.tsh), ev: "tsh" },
  { k: "ays", label: "Air-yards share", short: "Air share", fmt: "pct", get: (p) => pctOf(p.ays), ev: "ays" },
  { k: "wopr", label: "WOPR", short: "WOPR", fmt: "num", get: (p) => p.wopr, ev: "wopr" },
  { k: "csh", label: "Team carry share", short: "Carry share", fmt: "pct", get: (p) => pctOf(p.csh), ev: "csh" },
  { k: "start", label: "Startable-week rate", short: "Startable", fmt: "pct", get: (p) => pctOf(p.start), ev: "start" },
  { k: "smash", label: "Smash-week rate", short: "Smash", fmt: "pct", get: (p) => pctOf(p.smash), ev: "smash" },
  { k: "dud", label: "Dud-week rate", short: "Dud", fmt: "pct", get: (p) => pctOf(p.dud), ev: "dud" },
  { k: "cv", label: "Swing (weekly sd ÷ average)", short: "Swing", fmt: "num", get: (p) => p.cv, ev: "cv" },
  { k: "resid", label: "Points vs ESPN's projection, per game", short: "vs ESPN", fmt: "sgn", get: (p) => p.resid, ev: "resid" },
  { k: "snap_tr", label: "Snap share, last 3 minus season (pts)", short: "Snap trend", fmt: "pp", get: (p) => pctOf(p.snap_tr), ev: "snap_tr" },
  { k: "xfp_tr", label: "Usage xFP, last 3 minus season", short: "Usage trend", fmt: "sgn", get: (p) => p.xfp_tr, ev: "xfp_tr" },
  { k: "est_tr", label: "Model value, 3-week change", short: "Value move", fmt: "sgn", get: (p) => p.est_tr, ev: "est_tr" },
  { k: "ecr", label: "Experts' rank at position", short: "ECR", fmt: "rank", get: (p) => p.ecr_rank, fixed: "descriptive", note: "FantasyPros consensus; the model already blends it in." },
];
const SBY = new Map(STATS.map((s) => [s.k, s]));
const fmt = (s: Stat, v: number | null) => (v == null ? "–" : s.fmt === "pct" ? `${v.toFixed(0)}%` : s.fmt === "pp" ? `${signed(v).replace(".0", "")} pts` : s.fmt === "sgn" ? signed(v) : s.fmt === "rank" ? `#${Math.round(v)}` : s.k === "cv" || s.k === "wopr" ? v.toFixed(2) : f1(v));

const PRESETS: { label: string; x: string; y: string; why: string }[] = [
  { label: "Luck vs role", x: "xfp", y: "ppg", why: "Above the line = scoring more than his usage usually produces. About 9% of that has carried forward." },
  { label: "Season vs model", x: "ppg", y: "est_ppg", why: "Below the line = the model expects less than he has scored. Mostly efficiency it expects to fade." },
  { label: "Snaps moving", x: "snap_tr", y: "est_tr", why: "The two small edges: snap-share trend and model value moves, each worth a fraction of a point a game." },
  { label: "Volume vs swing", x: "opp", y: "cv", why: "Swing falls as volume rises. Swing repeats a little but doesn't predict points." },
  { label: "Startable vs swing", x: "start", y: "cv", why: "Same average, different shapes: steady starters vs boom-bust ones." },
];
const POS = ["All", "QB", "RB", "WR", "TE"] as const;
type Who = "all" | "rostered" | "fa" | "mine";

export function LabScreen({ a, err, tw, teams, onCopy }: { a: Analytics | null; err: string | null; tw: TeamWeek; teams: { id: number; name: string }[]; onCopy: (md: string) => void }) {
  const { open } = useFile();
  const [xk, setXk] = useState(() => { try { return localStorage.getItem("desk.lab.x") ?? "xfp"; } catch { return "xfp"; } });
  const [yk, setYk] = useState(() => { try { return localStorage.getItem("desk.lab.y") ?? "ppg"; } catch { return "ppg"; } });
  const [pos, setPos] = useState<(typeof POS)[number]>("WR");
  const [who, setWho] = useState<Who>("all");
  const [color, setColor] = useState<"owner" | "pos">("owner");
  const [hov, setHov] = useState<number | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [q, setQ] = useState("");
  const [ref, W] = useWidth<HTMLDivElement>(820);
  const setX = (k: string) => { setXk(k); try { localStorage.setItem("desk.lab.x", k); } catch { /* not persisted */ } };
  const setY = (k: string) => { setYk(k); try { localStorage.setItem("desk.lab.y", k); } catch { /* not persisted */ } };
  const X = SBY.get(xk) ?? STATS[1], Y = SBY.get(yk) ?? STATS[0];
  const me = tw.team_id, opp = tw.matchup?.opp_id ?? null;

  const pts = useMemo(() => {
    if (!a) return [];
    return a.players.filter((p) => (pos === "All" || p.pos === pos) && p.games >= 2 && (who === "all" || (who === "fa" ? p.owner == null : who === "mine" ? p.owner === me : p.owner != null)))
      .map((p) => ({ p, x: X.get(p), y: Y.get(p) })).filter((d): d is { p: APlayer; x: number; y: number } => d.x != null && d.y != null && Number.isFinite(d.x) && Number.isFinite(d.y));
  }, [a, pos, who, xk, yk, me, X, Y]);

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

  const H = 470, L = 58, R = 18, T = 14, B = 46;
  const xs = pts.map((d) => d.x), ys = pts.map((d) => d.y);
  const pad = (lo: number, hi: number) => { const s = hi - lo || 1; return [lo - 0.06 * s, hi + 0.06 * s]; };
  const [x0, x1] = xs.length ? pad(Math.min(...xs), Math.max(...xs)) : [0, 1];
  const [y0, y1] = ys.length ? pad(Math.min(...ys), Math.max(...ys)) : [0, 1];
  const sx = (v: number) => L + ((W - L - R) * (v - x0)) / (x1 - x0);
  const sy = (v: number) => T + ((H - T - B) * (y1 - v)) / (y1 - y0);
  const ticks = (lo: number, hi: number) => { const span = hi - lo, st = [0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50].find((s) => span / s <= 7) ?? 100; const out: number[] = []; for (let t = Math.ceil(lo / st) * st; t <= hi + 1e-9; t += st) out.push(Math.round(t * 1000) / 1000); return out; };
  const med = (v: number[]) => { const s = [...v].sort((p, q2) => p - q2); return s.length ? s[Math.floor(s.length / 2)] : 0; };
  const mxx = med(xs), myy = med(ys);
  const named = new Set((fit?.res ?? []).slice(0, 6).map((d) => d.p.id));
  const hit = q.trim().length >= 2 ? pts.filter((d) => d.p.name.toLowerCase().includes(q.trim().toLowerCase())).map((d) => d.p.id) : [];
  for (const id of hit.slice(0, 3)) named.add(id);
  if (sel != null) named.add(sel);
  const fill = (p: APlayer) => (color === "pos" ? POS_VAR[p.pos] : p.owner === me ? "var(--you)" : p.owner === opp && opp != null ? "var(--them)" : p.owner != null ? "var(--lab-other)" : "none");
  const hp = hov != null ? pts.find((d) => d.p.id === hov) : null;
  const sp = sel != null ? pts.find((d) => d.p.id === sel)?.p ?? a.players.find((p) => p.id === sel) : null;
  const preset = PRESETS.find((p) => p.x === xk && p.y === yk);
  const teamName = (t: number | null) => (t == null ? "free agent" : t === me ? "you" : teams.find((x) => x.id === t)?.name ?? `team ${t}`);

  const md = () => {
    const L2 = [`## Lab: ${Y.label} vs ${X.label} (${pos === "All" ? "all positions" : pos}, ${who}), through week ${a.through_week}`, "",
      fit ? `This season: r = ${fit.r.toFixed(2)} across ${fit.n} players; line ${Y.short} = ${fit.c.toFixed(2)} + ${fit.b.toFixed(2)} × ${X.short}; typical distance from the line ±${fit.sd.toFixed(1)}.` : "Too few players for a line.",
      `Backtest labels (2021–25): ${X.short}: ${X.fixed ?? M[X.ev ?? ""]?.label_text ?? "–"}; ${Y.short}: ${Y.fixed ?? M[Y.ev ?? ""]?.label_text ?? "–"}.`, "",
      "Furthest from the line:", "", `| Player | Pos | Team | Rostered by | ${X.short} | ${Y.short} | Off the line |`, "|---|---|---|---|---|---|---|",
      ...(fit?.res ?? []).slice(0, 12).map((d) => `| ${d.p.name} | ${d.p.pos} | ${d.p.team ?? ""} | ${teamName(d.p.owner)} | ${fmt(X, d.x)} | ${fmt(Y, d.y)} | ${signed(d.res)} |`)];
    return L2.join("\n");
  };

  return (
    <div className="lab-screen">
      <header className="lab-hd">
        <h2>Lab</h2>
        <p>Any stat against any stat, this season through week {a.through_week}. Each axis carries what 2021–25 says about it; the line and r describe this season only.</p>
        <button className="tbtn" type="button" onClick={() => onCopy(md())}>Copy for Claude</button>
      </header>
      <div className="lab-presets">{PRESETS.map((p) => <button key={p.label} type="button" aria-pressed={p.x === xk && p.y === yk} onClick={() => { setX(p.x); setY(p.y); }}>{p.label}</button>)}</div>
      <div className="lab-ctl">
        <label>Y <select value={yk} onChange={(e) => setY(e.target.value)}>{STATS.map((s) => <option key={s.k} value={s.k}>{s.label}</option>)}</select></label>
        <button type="button" className="swap" onClick={() => { const t = xk; setX(yk); setY(t); }} title="Swap axes">⇅</button>
        <label>X <select value={xk} onChange={(e) => setX(e.target.value)}>{STATS.map((s) => <option key={s.k} value={s.k}>{s.label}</option>)}</select></label>
        <div className="seg" role="group" aria-label="Position">{POS.map((p) => <button key={p} type="button" aria-pressed={pos === p} onClick={() => setPos(p)}>{p}</button>)}</div>
        <div className="seg" role="group" aria-label="Who">{([["all", "Everyone"], ["rostered", "Rostered"], ["fa", "Free agents"], ["mine", "Yours"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={who === k} onClick={() => setWho(k)}>{l}</button>)}</div>
        <div className="seg" role="group" aria-label="Color">{([["owner", "By owner"], ["pos", "By position"]] as const).map(([k, l]) => <button key={k} type="button" aria-pressed={color === k} onClick={() => setColor(k)}>{l}</button>)}</div>
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
            <text className="axl" x={(L + W - R) / 2} y={H - 8} textAnchor="middle">{X.label}</text>
            <text className="axl" transform={`translate(14 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">{Y.label}</text>
            {pts.map((d) => {
              const f = fill(d.p);
              return <circle key={d.p.id} className={`pt ${f === "none" ? "fa" : ""} ${sel === d.p.id ? "sel" : ""} ${hit.includes(d.p.id) ? "hit" : ""}`} cx={sx(d.x)} cy={sy(d.y)} r={sel === d.p.id || hov === d.p.id ? 6.5 : 5}
                style={f === "none" ? undefined : { fill: f }} onMouseEnter={() => setHov(d.p.id)} onClick={() => setSel(d.p.id)} onDoubleClick={() => open(d.p.id, pts.map((x) => x.p.id))} />;
            })}
            {pts.filter((d) => named.has(d.p.id)).map((d) => {
              const right = sx(d.x) < W - 150;
              return <text key={`n${d.p.id}`} className="nm" x={sx(d.x) + (right ? 9 : -9)} y={sy(d.y) - 7} textAnchor={right ? "start" : "end"}>{d.p.name.split(" ").slice(-1)[0]}</text>;
            })}
          </svg>
          {hp && <div className="lab-tip" style={{ left: Math.min(sx(hp.x) + 14, W - 230), top: Math.max(sy(hp.y) - 10, 30) }}>
            <b>{hp.p.name}</b> <span className="dim">{hp.p.pos} · {hp.p.team ?? "FA"} · {teamName(hp.p.owner)}</span>
            <span>{Y.short} {fmt(Y, hp.y)} · {X.short} {fmt(X, hp.x)}{fit ? ` · ${signed(hp.y - (fit.c + fit.b * hp.x))} off the line` : ""}</span>
            <span className="dim">{hp.p.games} games · double-click for his file</span>
          </div>}
          <div className="lab-axes">
            <span>Y · {Y.label} {evFor(Y)}</span><span>X · {X.label} {evFor(X)}</span>
            <span className="leg">{color === "owner" ? <><i className="k you" />yours <i className="k them" />this week's opponent <i className="k oth" />other rosters <i className="k fa" />free agents </>
              : <>{(["QB", "RB", "WR", "TE"] as const).map((p2) => <span key={p2}><i className="k" style={{ background: POS_VAR[p2] }} />{p2} </span>)}</>}
              <span> · hairlines = medians · line = this season's fit</span></span>
          </div>
        </div>
        <aside className="lab-side">
          {sp ? <SelCard p={sp} X={X} Y={Y} fit={fit} a={a} onOpen={() => open(sp.id, [sp.id])} teamName={teamName} />
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
    </div>
  );
}

function SelCard({ p, X, Y, fit, a, onOpen, teamName }: { p: APlayer; X: Stat; Y: Stat; fit: { b: number; c: number } | null; a: Analytics; onOpen: () => void; teamName: (t: number | null) => string }) {
  const x = X.get(p), y = Y.get(p);
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
