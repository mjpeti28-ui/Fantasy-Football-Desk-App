import { useMemo, useState } from "react";
import { spearman } from "./cluster";
import { GROUP_ORDER, type Stat } from "./stats";
import type { Row } from "./Clusters";

/* Lab · Correlations: every stat against every other for the current population (Spearman rank r, pairwise
   complete). Click a cell to open that pair on the scatter. Same-season correlation only: it says what travels
   together this year, not what predicts what. */

export function CorrView({ rows, stats, adj, W, onPick }: { rows: Row[]; stats: Stat[]; adj: boolean; W: number; onPick: (x: string, y: string) => void }) {
  const [hov, setHov] = useState<[number, number] | null>(null);
  const [tk, setTk] = useState("ppg");
  const ss = useMemo(() => [...stats].sort((a, b) => GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group)), [stats]);
  const cols = useMemo(() => ss.map((s) => rows.map((r) => s.get(r.p, r.r, adj))), [ss, rows, adj]);
  const M = useMemo(() => ss.map((_, i) => ss.map((__, j) => (i === j ? { r: 1, n: rows.length } : j < i ? null : spearman(cols[i], cols[j])))), [ss, cols, rows.length]);
  const at = (i: number, j: number) => (i <= j ? M[i][j] : M[j][i]);
  const pairs = useMemo(() => {
    const out: { i: number; j: number; r: number; n: number }[] = [];
    for (let i = 0; i < ss.length; i++) for (let j = i + 1; j < ss.length; j++) { const c = M[i][j]; if (c) out.push({ i, j, r: c.r, n: c.n }); }
    return out.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  }, [M, ss.length]);

  const n = ss.length, LBL = 118, cell = Math.max(12, Math.min(26, Math.floor((Math.min(W, 900) - LBL - 10) / Math.max(1, n))));
  const size = LBL + cell * n;
  const col = (r: number) => `color-mix(in srgb, ${r >= 0 ? "var(--cor-pos)" : "var(--cor-neg)"} ${Math.round(Math.abs(r) * 100)}%, var(--win))`;
  const h = hov ? at(hov[0], hov[1]) : null;
  // links between how players are used, leaving out points restated (scoring, weekly shape, market) and the same stat measured twice
  const OUTCOME = new Set(["Scoring", "Weekly shape", "Market"]);
  // WOPR is a formula of target and air-yards share; above r 0.9 two stats are usually one quantity measured twice
  const twin = (a: string, b: string) => a === "wopr" || b === "wopr" || Math.abs(M[ss.findIndex((x) => x.k === a)]?.[ss.findIndex((x) => x.k === b)]?.r ?? 0) > 0.9;
  const notable = pairs.filter((p) => !OUTCOME.has(ss[p.i].group) && !OUTCOME.has(ss[p.j].group) && !twin(ss[p.i].k, ss[p.j].k)).slice(0, 10);
  const ti = Math.max(0, ss.findIndex((s) => s.k === tk));
  const withT = ss.map((_, j) => ({ j, c: at(ti, j) })).filter((x) => x.j !== ti && x.c).sort((a, b) => Math.abs(b.c!.r) - Math.abs(a.c!.r)).slice(0, 10);

  return (
    <div className="lab-main">
      <div className="lab-plot">
        <div className="lab-read num">
          {h && hov ? <>{ss[hov[0]].label} × {ss[hov[1]].label}: r <b>{h.r.toFixed(2)}</b> · n {h.n}</> : <>{n} stats · {rows.length} players · Spearman rank r, pairs where both exist (10+ players)</>}
        </div>
        <svg width={size + 4} height={size + 4} role="img" aria-label={`Correlation grid of ${n} stats`} onMouseLeave={() => setHov(null)}>
          {ss.map((s, i) => <text key={`r${s.k}`} className={`cr-l ${hov?.[0] === i ? "on" : ""}`} x={LBL - 6} y={LBL + i * cell + cell / 2 + 4} textAnchor="end">{s.short}</text>)}
          {ss.map((s, j) => <text key={`c${s.k}`} className={`cr-l ${hov?.[1] === j ? "on" : ""}`} transform={`translate(${LBL + j * cell + cell / 2 + 4} ${LBL - 6}) rotate(-60)`}>{s.short}</text>)}
          {ss.map((si, i) => ss.map((sj, j) => {
            const c = at(i, j);
            return <rect key={`${si.k}-${sj.k}`} className={`cr-c ${i === j ? "diag" : ""}`} x={LBL + j * cell} y={LBL + i * cell} width={cell - 1} height={cell - 1}
              style={{ fill: c && i !== j ? col(c.r) : undefined }} onMouseEnter={() => setHov([i, j])} onClick={() => i !== j && c && onPick(sj.k, si.k)} />;
          }))}
        </svg>
        <div className="lab-axes"><span className="leg"><i className="k" style={{ background: "var(--cor-pos)" }} />move together <i className="k" style={{ background: "var(--cor-neg)" }} />move opposite · blank = too few players with both · click a cell to plot the pair</span></div>
      </div>
      <aside className="lab-side">
        <h5>What goes with <select className="cr-t" value={ss[ti]?.k} onChange={(e) => setTk(e.target.value)}>{ss.map((s) => <option key={s.k} value={s.k}>{s.short}</option>)}</select></h5>
        <ol className="lab-out">
          {withT.map(({ j, c }) => <li key={ss[j].k} onClick={() => onPick(ss[j].k, ss[ti].k)} title={`${ss[j].label}, n ${c!.n}`}>
            <span className="nm">{ss[j].short}<small>{ss[j].group}</small></span><span className={`num ${c!.r > 0 ? "up" : "down"}`}>{c!.r.toFixed(2)}</span></li>)}
        </ol>
        <h5>Strongest links between usage and style stats</h5>
        <ol className="lab-out">
          {notable.map((p) => <li key={`${p.i}-${p.j}`} onClick={() => onPick(ss[p.j].k, ss[p.i].k)} title={`${ss[p.i].label} × ${ss[p.j].label}, n ${p.n}`}>
            <span className="nm">{ss[p.i].short} × {ss[p.j].short}</span><span className={`num ${p.r > 0 ? "up" : "down"}`}>{p.r.toFixed(2)}</span></li>)}
        </ol>
        <p className="dim note">The second list leaves out points restated (scoring, weekly shape, experts) WOPR (a blend of two shares) and pairs above 0.9, which are usually one thing measured twice. One season of the players shown: correlation here is what moves together, not what predicts what. The scatter's chips say which stats have predicted anything.</p>
      </aside>
    </div>
  );
}
