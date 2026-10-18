import type { PlayerFile as PF } from "../../types";
import type { MxCompare } from "../matchups/types";
import { f1, lastName, pct } from "../../format";
import { POS_VAR } from "../../components/bits";

/* This-week visuals for Compare's start/sit view. Only things that held up in backtests, or are
   labeled as context: the outcome range (ESPN projection × the position's actual/forecast spread,
   the same grid as the head-to-head odds), recent usage (settles within a few games), points vs
   ESPN's line (context: how he's done, not a forecast) and defense rank (already in ESPN's number). */

const LET = ["A", "B", "C", "D"];
const dead = (f: PF) => !!f.game?.bye || /^(OUT|INJURY_RESERVE|IR|SUSPENSION)$/i.test((f.owner?.injury ?? f.ledger?.status ?? "") || "");

function qs(f: PF, q?: Record<string, number[]>): number[] | null {
  const r = q?.[f.pos];
  if (!r || !f.espn_week || dead(f)) return null;
  return r.map((x) => x * f.espn_week!).sort((a, b) => a - b);
}
const at = (s: number[], p: number) => s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))];
const share = (s: number[], x: number) => s.filter((v) => v >= x).length / s.length;

export function OutcomeRange({ files, ratioQ }: { files: PF[]; ratioQ?: Record<string, number[]> }) {
  const S = files.map((f) => qs(f, ratioQ));
  if (!S.some(Boolean)) return null;
  const xmax = Math.max(20, ...S.map((s) => (s ? at(s, 0.9) + 2 : 0)));
  const x = (v: number) => `${(Math.max(0, Math.min(xmax, v)) / xmax) * 100}%`;
  const ticks = [0, 5, 10, 15, 20, 25, 30, 35].filter((t) => t <= xmax);
  return (
    <div className="cmp-ranges wk">
      {files.map((f, i) => {
        const s = S[i];
        return (
          <div key={f.id} className="rr">
            <span className="nm"><i className="let">{LET[i]}</i>{lastName(f.name)}</span>
            <span className="trk">
              {s && <><i className="b80" style={{ left: x(at(s, 0.1)), width: `calc(${x(at(s, 0.9))} - ${x(at(s, 0.1))})` }} />
                <i className="b50" style={{ left: x(at(s, 0.25)), width: `calc(${x(at(s, 0.75))} - ${x(at(s, 0.25))})` }} />
                <i className="med" style={{ left: x(at(s, 0.5)), background: POS_VAR[f.pos] }} title={`median ${f1(at(s, 0.5))}`} />
                <i className="mdl" style={{ left: x(f.espn_week!) }} title={`ESPN ${f1(f.espn_week)}`} /></>}
            </span>
            <span className="vals num">{s ? <>{f1(at(s, 0.1))}–{f1(at(s, 0.9))} <small className="dim">15+ {pct(share(s, 15))} · 20+ {pct(share(s, 20))}</small></> : <small className="dim">not playing / no projection</small>}</span>
          </div>
        );
      })}
      <div className="rr axis"><span /><span className="trk">{ticks.map((t) => <em key={t} style={{ left: x(t) }}>{t}</em>)}</span><span /></div>
      <p className="cap"><i className="k80" /> 80% and <i className="k50" /> 50% of where this week should land: ESPN's projection (▲) times how far actual scores at his position have landed from forecasts, 2021–25.
        Colored tick = median. The 15+/20+ chances come from the same spread; boom chances showed skill in the scorecard backtest, while a "bust" chance showed none, so it is left out.</p>
    </div>
  );
}

/** Opponent's pass and run defense rank on one 1–32 line per kind; letters are the players. */
export function DefenseDots({ files, mx }: { files: PF[]; mx: Record<number, MxCompare | null | undefined> }) {
  const kinds = (["pass", "run"] as const).filter((k) => files.some((f) => mx[f.id]?.defense?.[k]));
  if (!kinds.length) return null;
  return (
    <div className="cmp-dd">
      {kinds.map((k) => (
        <div key={k} className="ddr">
          <span className="nm">{k === "pass" ? "Pass D" : "Run D"}</span>
          <span className="trk"><i className="zone tough" /><i className="zone soft" />
            {files.map((f, i) => { const d = mx[f.id]?.defense?.[k];
              // players facing equally ranked defenses (teammates, or the same rank) sit side by side instead of on top of each other
              const n = d ? files.slice(0, i).filter((g) => mx[g.id]?.defense?.[k]?.rank === d.rank).length : 0;
              return d ? <b key={f.id} style={{ left: `${((d.rank - 1) / Math.max(1, d.of - 1)) * 100}%`, marginLeft: -10 + n * 18, ["--c" as string]: POS_VAR[f.pos] }}
              title={`${f.name}: ${mx[f.id]!.opp} ${d.label.toLowerCase()}, ${d.rank} of ${d.of}`}>{LET[i]}</b> : null; })}
          </span>
        </div>
      ))}
      <div className="ddr axis"><span /><span className="trk"><em style={{ left: 0 }}>1 toughest</em><em style={{ left: "100%" }}>softest 32</em></span></div>
      <p className="cap">Defense rated on every play this season, blended with last. Already inside ESPN's projection, so this explains the number rather than adding to it.</p>
    </div>
  );
}

function Bars({ v, max, h = 22, bw = 7, color }: { v: (number | null)[]; max: number; h?: number; bw?: number; color: string }) {
  const W = v.length * (bw + 2);
  return (
    <svg width={W} height={h} viewBox={`0 0 ${W} ${h}`}>
      {v.map((x, i) => x == null ? <rect key={i} x={i * (bw + 2)} y={h - 1} width={bw} height={1} fill={color} opacity={0.35} />
        : <rect key={i} x={i * (bw + 2)} y={h - Math.max(1.5, (h * x) / max)} width={bw} height={Math.max(1.5, (h * x) / max)} rx={1} fill={color} />)}
    </svg>
  );
}

const done = (f: PF, n: number) => f.gamelog.filter((r) => !r.upcoming).slice(-n);

/** Last 5 games of one usage stat, on a scale shared by every player in the comparison. */
export function usageMax(files: PF[], k: "snap" | "tgt" | "car") {
  return Math.max(k === "snap" ? 100 : 6, ...files.flatMap((f) => done(f, 5).map((r) => (r[k] ?? 0) * (k === "snap" && (r.snap ?? 0) <= 1 ? 100 : 1))));
}
export function UsageCell({ f, k, max }: { f: PF; k: "snap" | "tgt" | "car"; max: number }) {
  const g = done(f, 5);
  if (!g.length) return <span className="dim">–</span>;
  const v = g.map((r) => (r.played && r[k] != null ? r[k]! * (k === "snap" && r[k]! <= 1 ? 100 : 1) : null));
  const last = v.filter((x): x is number => x != null);
  return <span className="usg"><Bars v={v} max={max} color={POS_VAR[f.pos]} /><small className="num dim">{last.length ? `${Math.round(last.reduce((a, b) => a + b, 0) / last.length)}${k === "snap" ? "%" : ""} avg` : "–"}</small></span>;
}

/** Last 6 games: bar = points scored, tick = ESPN's projection that week. Context only. */
export function VsEspnCell({ f }: { f: PF }) {
  const g = done(f, 6);
  if (!g.length) return <span className="dim">–</span>;
  const pts = g.map((r) => (r.played ? r.espn_pts ?? r.ppr : null));
  const max = Math.max(20, ...pts.map((x) => x ?? 0), ...g.map((r) => r.espn_proj ?? 0));
  const h = 26, bw = 9, W = g.length * (bw + 3);
  const y = (v: number) => h - Math.max(1.5, (h * v) / max);
  const beat = g.filter((r, i) => pts[i] != null && r.espn_proj != null && pts[i]! >= r.espn_proj).length, n = g.filter((r, i) => pts[i] != null && r.espn_proj != null).length;
  return (
    <span className="usg">
      <svg width={W} height={h} viewBox={`0 0 ${W} ${h}`}>
        {g.map((r, i) => <g key={r.week}><title>{`Week ${r.week}: ${pts[i] == null ? "did not play" : f1(pts[i])}${r.espn_proj != null ? ` (ESPN ${f1(r.espn_proj)})` : ""}`}</title>
          {pts[i] == null ? <rect x={i * (bw + 3)} y={h - 1} width={bw} height={1} fill="var(--ink-3)" />
            : <rect x={i * (bw + 3)} y={y(pts[i]!)} width={bw} height={h - y(pts[i]!)} rx={1} fill={r.espn_proj != null && pts[i]! >= r.espn_proj ? POS_VAR[f.pos] : "var(--ink-3)"} opacity={0.85} />}
          {r.espn_proj != null && <rect x={i * (bw + 3) - 1} y={y(r.espn_proj) - 1} width={bw + 2} height={2} fill="var(--ink)" />}</g>)}
      </svg>
      <small className="num dim">{n ? `beat ESPN ${beat} of ${n}` : "–"}</small>
    </span>
  );
}
