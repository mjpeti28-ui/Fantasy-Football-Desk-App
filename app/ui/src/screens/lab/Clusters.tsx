import { useMemo, useState } from "react";
import type { APlayer, RoleProfile } from "../../types";
import { f1, lastName } from "../../format";
import { kmeans, neighbors, pca2, silhouette, stability, zscore, type Mat } from "./cluster";
import { fmt, GROUP_ORDER, type Stat } from "./stats";

/* Lab · Clusters: k-means on any set of stats (z-scored, so each counts equally), auto-k by separation × stability,
   a PCA map of the groups, group cards named by what sets them apart, and nearest neighbours for a pinned player.
   Purely descriptive: nothing here has been backtested, and the copy says so. */

export interface Row { p: APlayer; r?: RoleProfile }
export interface ClusterResult {
  rows: Row[]; stats: Stat[]; X: Mat; Z: Mat; k: number; auto: boolean; labels: number[]; centers: Mat;
  sil: number; stab: number; byK: { k: number; sil: number; stab: number }[]; dropped: number;
  names: string[]; typical: number[]; xy: [number, number][]; load: [number[], number[]]; ve: [number, number];
}
export const CL_N = 8;
export const clColor = (c: number) => `var(--cl${c % CL_N})`;

export function runClusters(rows: Row[], stats: Stat[], kSel: number | "auto", adj: boolean): ClusterResult | null {
  if (stats.length < 2) return null;
  const keep: Row[] = [], X: Mat = [];
  for (const row of rows) {
    const v = stats.map((s) => s.get(row.p, row.r, adj));
    if (v.every((x) => x != null && Number.isFinite(x))) { keep.push(row); X.push(v as number[]); }
  }
  if (keep.length < 12) return null;
  const { Z } = zscore(X);
  const ks = [2, 3, 4, 5, 6, 7].filter((k) => k * 4 <= keep.length);
  const byK = ks.map((k) => { const m = kmeans(Z, k); return { k, m, sil: silhouette(Z, m.labels, k), stab: stability(Z, k, m.labels) }; });
  // auto: best separation × stability; a smaller k wins near-ties (within 3%), since extra groups are cheap to invent
  const score = (x: { sil: number; stab: number }) => Math.max(0, x.sil) * Math.max(0, x.stab);
  const top = Math.max(...byK.map(score));
  const pick = kSel === "auto" ? byK.find((x) => score(x) >= top * 0.97) ?? byK[0] : byK.find((x) => x.k === kSel) ?? byK[byK.length - 1];
  const { m } = pick, k = pick.k;
  const pc = pca2(Z);
  const typical = m.centers.map((c, ci) => {
    let bi = -1, bd = Infinity;
    Z.forEach((z, i) => { if (m.labels[i] !== ci) return; const d = z.reduce((s, x, j) => s + (x - c[j]) ** 2, 0); if (d < bd) { bd = d; bi = i; } });
    return bi;
  });
  const names = m.centers.map((c) => {
    const parts = c.map((z, j) => ({ z, j })).filter((x) => Math.abs(x.z) >= 0.5).sort((a, b) => Math.abs(b.z) - Math.abs(a.z)).slice(0, 3)
      .map(({ z, j }) => `${z > 0 ? (z >= 1.2 ? "very high" : "high") : (z <= -1.2 ? "very low" : "low")} ${stats[j].short}`);
    return parts.length ? parts.join(" · ") : "middle of the pack";
  });
  const seen = new Map<string, number>();
  const uniq = names.map((n) => { const c = (seen.get(n) ?? 0) + 1; seen.set(n, c); return c > 1 ? `${n} (${c})` : n; });
  return { rows: keep, stats, X, Z, k, auto: kSel === "auto", labels: m.labels, centers: m.centers, sil: pick.sil, stab: pick.stab,
    byK: byK.map(({ k: kk, sil, stab }) => ({ k: kk, sil, stab })), dropped: rows.length - keep.length, names: uniq, typical, xy: pc.xy, load: pc.load, ve: pc.ve };
}

const silWord = (s: number) => (s >= 0.5 ? "clear groups" : s >= 0.3 ? "real but overlapping groups" : s >= 0.18 ? "weak structure" : "little real structure");
const stabWord = (s: number) => (s >= 0.8 ? "stable" : s >= 0.6 ? "mostly stable" : "unstable");

export function ClusterControls({ stats, avail, keys, setKeys, k, setK, sets, adj }: {
  stats: Stat[]; avail: Set<string>; keys: string[]; setKeys: (k: string[]) => void; k: number | "auto"; setK: (k: number | "auto") => void;
  sets: { label: string; keys: string[]; why: string }[]; adj: boolean;
}) {
  const groups = GROUP_ORDER.map((g) => ({ g, ss: stats.filter((s) => s.group === g && avail.has(s.k)) })).filter((x) => x.ss.length);
  const on = new Set(keys);
  const flip = (sk: string) => setKeys(on.has(sk) ? keys.filter((x) => x !== sk) : [...keys, sk]);
  const usable = (ks: string[]) => ks.filter((x) => avail.has(x));
  return (
    <div className="cl-ctl">
      <div className="lab-presets">
        {sets.map((s) => { const ks = usable(s.keys); return ks.length >= 2 && <button key={s.label} type="button" title={s.why} aria-pressed={ks.length === keys.length && ks.every((x) => on.has(x))} onClick={() => setKeys(ks)}>{s.label}</button>; })}
        <button type="button" className="clear" onClick={() => setKeys([])}>Clear</button>
      </div>
      <div className="cl-pick">
        {groups.map(({ g, ss }) => (
          <div key={g} className="cl-grp"><span className="gl">{g}</span>
            {ss.map((s) => <button key={s.k} type="button" aria-pressed={on.has(s.k)} title={`${s.label}${s.note ? `. ${s.note}` : ""}`} className={s.fixed === "noise" ? "nz" : ""} onClick={() => flip(s.k)}>{s.short}</button>)}
          </div>
        ))}
      </div>
      <div className="lab-ctl">
        <span className="lbl">Groups</span>
        <div className="seg" role="group" aria-label="Number of groups">
          {(["auto", 2, 3, 4, 5, 6, 7] as const).map((x) => <button key={x} type="button" aria-pressed={k === x} onClick={() => setK(x)}>{x === "auto" ? "Auto" : x}</button>)}
        </div>
        <span className="dim">{keys.length} stat{keys.length === 1 ? "" : "s"} · k-means on z-scores{adj ? " · fingerprint stats noise-adjusted" : " · raw values"} · dashed chips are mostly noise</span>
      </div>
    </div>
  );
}

export function ClusterView({ res, W, me, opp, sel, setSel, open, teamName }: {
  res: ClusterResult; W: number; me: number; opp: number | null; sel: number | null; setSel: (id: number | null) => void;
  open: (id: number, list: number[]) => void; teamName: (t: number | null) => string;
}) {
  const [hov, setHov] = useState<number | null>(null);
  const [more, setMore] = useState<Set<number>>(new Set());
  const { rows, stats, X, Z, k, labels, xy } = res;
  const si = sel != null ? rows.findIndex((r) => r.p.id === sel) : -1;
  const nb = useMemo(() => (si >= 0 ? neighbors(Z, si, 6) : []), [Z, si]);
  const nbSet = new Set(nb.map((x) => rows[x.j].p.id));

  const H = 420, L = 18, R = 18, T = 16, B = 40;
  const xs = xy.map((d) => d[0]), ys = xy.map((d) => d[1]);
  const pad = (lo: number, hi: number) => { const s = hi - lo || 1; return [lo - 0.06 * s, hi + 0.06 * s]; };
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const sx = (v: number) => L + ((W - L - R) * (v - x0)) / (x1 - x0), sy = (v: number) => T + ((H - T - B) * (y1 - v)) / (y1 - y0);
  const axisWords = (ld: number[]) => {
    const o = ld.map((v, j) => ({ v, j })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).filter((x) => Math.abs(x.v) >= 0.3).slice(0, 3);
    const up = o.filter((x) => x.v > 0).map((x) => stats[x.j].short), dn = o.filter((x) => x.v < 0).map((x) => stats[x.j].short);
    return [up.length ? `more ${up.join(", ")}` : "", dn.length ? `less ${dn.join(", ")}` : ""].filter(Boolean).join(" · ") || "a mix";
  };
  const named = new Set<number>(res.typical.filter((i) => i >= 0).map((i) => rows[i].p.id));
  if (sel != null) { named.add(sel); nb.slice(0, 3).forEach((x) => named.add(rows[x.j].p.id)); }
  const hp = hov != null ? rows.findIndex((r) => r.p.id === hov) : -1;
  const mean = (c: number, f: (r: Row) => number | null) => { const v = rows.filter((_, i) => labels[i] === c).map(f).filter((x): x is number => x != null); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; };

  const colMean = (c: number, j: number) => { const v = X.filter((_, i) => labels[i] === c).map((x) => x[j]); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; };

  return (
    <div className="cl-view">
      <div className="lab-plot">
        <div className="lab-read num">
          k <b>{k}</b>{res.auto ? " (auto)" : ""} · separation <b>{res.sil.toFixed(2)}</b> {silWord(res.sil)} · resamples agree <b>{Math.round(res.stab * 100)}%</b> {stabWord(res.stab)} · n {rows.length}{res.dropped ? ` (${res.dropped} missing a stat left out)` : ""}
        </div>
        <svg width={W} height={H} role="img" aria-label={`${k} groups of ${rows.length} players`}>
          <line className="gl" x1={sx(0)} x2={sx(0)} y1={T} y2={H - B} /><line className="gl" x1={L} x2={W - R} y1={sy(0)} y2={sy(0)} />
          <text className="axl" x={W - R} y={H - 22} textAnchor="end">→ {axisWords(res.load[0])} ({Math.round(res.ve[0] * 100)}% of the spread)</text>
          <text className="axl" x={L} y={H - 6}>↑ {axisWords(res.load[1])} ({Math.round(res.ve[1] * 100)}%)</text>
          {nb.length > 0 && si >= 0 && nb.map((x) => <line key={`nb${x.j}`} className="nbl" x1={sx(xy[si][0])} y1={sy(xy[si][1])} x2={sx(xy[x.j][0])} y2={sy(xy[x.j][1])} />)}
          {rows.map((r, i) => {
            const own = r.p.owner === me ? "you" : r.p.owner === opp && opp != null ? "them" : r.p.owner == null ? "fa" : "";
            return <circle key={r.p.id} className={`pt cl ${own} ${sel === r.p.id ? "sel" : ""} ${nbSet.has(r.p.id) ? "nb" : ""}`} cx={sx(xy[i][0])} cy={sy(xy[i][1])} r={sel === r.p.id || hov === r.p.id ? 6.5 : 5}
              style={{ fill: clColor(labels[i]) }} onMouseEnter={() => setHov(r.p.id)} onMouseLeave={() => setHov(null)} onClick={() => setSel(r.p.id)} onDoubleClick={() => open(r.p.id, rows.map((x) => x.p.id))} />;
          })}
          {rows.map((r, i) => named.has(r.p.id) && <text key={`n${r.p.id}`} className="nm" x={sx(xy[i][0]) + (sx(xy[i][0]) < W - 120 ? 9 : -9)} y={sy(xy[i][1]) - 7} textAnchor={sx(xy[i][0]) < W - 120 ? "start" : "end"}>{lastName(r.p.name)}</text>)}
        </svg>
        {hp >= 0 && <div className="lab-tip" style={{ left: Math.min(sx(xy[hp][0]) + 14, W - 230), top: Math.max(sy(xy[hp][1]) - 10, 30) }}>
          <b>{rows[hp].p.name}</b> <span className="dim">{rows[hp].p.pos} · {rows[hp].p.team ?? "FA"} · {teamName(rows[hp].p.owner)}</span>
          <span><i className="sw" style={{ background: clColor(labels[hp]) }} />{res.names[labels[hp]]}</span>
          <span className="dim">{stats.map((s, j) => `${s.short} ${fmt(s, X[hp][j])}`).join(" · ")}</span>
        </div>}
        <div className="lab-axes"><span className="leg">Map: the two directions that spread these players out most (PCA). Rings: <i className="k you" />yours <i className="k them" />opponent; faded dots are free agents. Labelled dots are each group's most typical player.</span></div>
        <div className="cl-k">
          <span className="dim">Separation × stability by number of groups:</span>
          {res.byK.map((x) => <span key={x.k} className={`kb ${x.k === k ? "on" : ""}`} title={`k=${x.k}: separation ${x.sil.toFixed(2)}, resamples agree ${Math.round(x.stab * 100)}%`}>
            <i style={{ height: `${Math.max(2, 26 * Math.max(0, x.sil) * Math.max(0, x.stab) / 0.6)}px` }} />{x.k}</span>)}
        </div>
      </div>
      <div className="cl-cards">
        {res.centers.map((c, ci) => {
          const mem = rows.map((r, i) => ({ r, i })).filter((x) => labels[x.i] === ci).sort((a, b) => (b.r.p.est_ppg ?? 0) - (a.r.p.est_ppg ?? 0));
          const mine = mem.filter((x) => x.r.p.owner === me).length, fa = mem.filter((x) => x.r.p.owner == null).length;
          const showAll = more.has(ci);
          return (
            <div key={ci} className="cl-card" style={{ borderTopColor: clColor(ci) }}>
              <b className="cn"><i className="sw" style={{ background: clColor(ci) }} />{res.names[ci]}</b>
              <span className="dim">{mem.length} players · {f1(mean(ci, (r) => r.p.ppg))} pts/g · model {f1(mean(ci, (r) => r.p.est_ppg))} ROS{mine ? ` · ${mine} yours` : ""}{fa ? ` · ${fa} free` : ""}</span>
              <table className="cl-cen"><tbody>
                {stats.map((s, j) => { const z = c[j]; const v = colMean(ci, j); return (
                  <tr key={s.k}><td>{s.short}</td><td className="num">{fmt(s, v)}</td>
                    <td className="zb"><span className="mid" /><span className={`bar ${z >= 0 ? "pos" : "neg"}`} style={{ [z >= 0 ? "left" : "right"]: "50%", width: `${Math.min(50, Math.abs(z) * 22)}%` } as React.CSSProperties} /></td></tr>); })}
              </tbody></table>
              <div className="cl-mem">
                {(showAll ? mem : mem.slice(0, 10)).map(({ r }) => <button key={r.p.id} type="button" className={`${r.p.owner === me ? "me" : r.p.owner === opp && opp != null ? "opp" : r.p.owner == null ? "fa" : ""} ${sel === r.p.id ? "on" : ""}`}
                  onClick={() => setSel(r.p.id)} onDoubleClick={() => open(r.p.id, mem.map((x) => x.r.p.id))} title={`${r.p.name} · ${teamName(r.p.owner)} · model ${f1(r.p.est_ppg)}`}>{lastName(r.p.name)}</button>)}
                {mem.length > 10 && <button type="button" className="more" onClick={() => setMore((s) => { const n = new Set(s); if (n.has(ci)) n.delete(ci); else n.add(ci); return n; })}>{showAll ? "fewer" : `+${mem.length - 10}`}</button>}
              </div>
            </div>
          );
        })}
      </div>
      {si >= 0 && <div className="cl-nb">
        <h5>Closest to {rows[si].p.name} on these {stats.length} stats</h5>
        <ol className="lab-out">
          {nb.map((x) => { const r = rows[x.j]; return (
            <li key={r.p.id} className={r.p.owner === me ? "me" : r.p.owner === opp && opp != null ? "opp" : ""} onClick={() => setSel(r.p.id)} onDoubleClick={() => open(r.p.id, nb.map((y) => rows[y.j].p.id))}>
              <span className="nm"><i className="sw" style={{ background: clColor(labels[x.j]) }} />{r.p.name}<small>{r.p.pos} · {r.p.owner == null ? "FA" : r.p.owner === me ? "yours" : r.p.team} · model {f1(r.p.est_ppg)}</small></span>
              <span className="num">{x.d.toFixed(1)}</span>
            </li>); })}
        </ol>
        <p className="dim note">Distance in standard deviations across the chosen stats; under ~1 is a near twin. Similar usage, not a promise of similar points.</p>
      </div>}
      <p className="dim note cl-note">Groups describe this season; none of this has been backtested. Raw stats mostly sort players by how big their job is, so if every group reads "high … / low …" on volume stats, try a style-only set. A stat that takes many games to settle (dashed chips) can split players on noise.</p>
    </div>
  );
}

export function clusterMd(res: ClusterResult, pos: string, adj: boolean, teamName: (t: number | null) => string): string {
  const out = [`## Lab clusters: ${pos}, ${res.stats.map((s) => s.short).join(", ")}${adj ? " (fingerprint stats noise-adjusted)" : ""}`, "",
    `k-means on z-scores, k = ${res.k}${res.auto ? " (auto)" : ""}. Separation (silhouette) ${res.sil.toFixed(2)}: ${silWord(res.sil)}. Resample agreement (ARI) ${res.stab.toFixed(2)}. ${res.rows.length} players. Descriptive only; not backtested.`, ""];
  res.centers.forEach((_, ci) => {
    const mem = res.rows.filter((_, i) => res.labels[i] === ci).sort((a, b) => (b.p.est_ppg ?? 0) - (a.p.est_ppg ?? 0));
    out.push(`### ${ci + 1}. ${res.names[ci]} (${mem.length})`, "", mem.map((r) => `${r.p.name} (${teamName(r.p.owner)})`).join(", "), "");
  });
  return out.join("\n");
}
