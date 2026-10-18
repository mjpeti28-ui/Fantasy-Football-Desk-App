import type { Accuracy, Player, PlayerFile as PF, TeamWeek } from "../../types";
import { pBeats, phi } from "../../odds";
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
/** ESPN over-projects players coming back from missed games (2021-25, vs the same position's other games; pff/creative.md E10). */
export function backPts(f: PF): number | null {
  const b = f.back;
  if (!b || f.game?.bye) return null;
  if (b.game === 2) return f.pos === "WR" ? -0.8 : null;
  return f.pos === "WR" ? -1.3 : f.pos === "QB" ? -1.5 : f.pos === "RB" ? (b.missed >= 2 ? -1.5 : -0.7) : null;
}
const asPt = (f: PF, add = 0) => ({ pos: f.pos, espn_week: f.espn_week != null ? Math.max(0.1, f.espn_week + add) : null, game: f.game ?? null,
  injury: (f.owner?.injury ?? f.ledger?.status ?? "ACTIVE") || "ACTIVE" } as unknown as Player);

const at = (s: number[], p: number) => s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))];
const share = (s: number[], x: number) => s.filter((v) => v >= x).length / s.length;

export function OutcomeRange({ files, ratioQ, acc }: { files: PF[]; ratioQ?: Record<string, number[]>; acc?: Accuracy | null }) {
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
                <i className="mdl" style={{ left: x(f.espn_week!) }} title={`ESPN ${f1(f.espn_week)}`} />
                {backPts(f) != null && <i className="adj" style={{ left: x(f.espn_week! + backPts(f)!) }} title={`ESPN ${f1(f.espn_week)} adjusted for his first game back: ${f1(f.espn_week! + backPts(f)!)}`} />}</>}
            </span>
            <span className="vals num">{s ? <>{f1(at(s, 0.1))}–{f1(at(s, 0.9))} <small className="dim">15+ {pct(share(s, 15))} · 20+ {pct(share(s, 20))}</small></> : <small className="dim">not playing / no projection</small>}</span>
          </div>
        );
      })}
      <div className="rr axis"><span /><span className="trk">{ticks.map((t) => <em key={t} style={{ left: x(t) }}>{t}</em>)}</span><span /></div>
      <p className="cap"><i className="k80" /> 80% and <i className="k50" /> 50% of where his score should land if he plays: ESPN's projection (▲) times how far actual scores at his position landed from ESPN's, 2021–25. Colored tick = median.
        {acc?.range ? ` This season ${Math.round(acc.range.rate * 100)}% of ${acc.range.n.toLocaleString()} rostered player-games landed inside the 80% range.` : ""}
        {acc?.boom ? ` 15+ chances said ${Math.round(acc.boom["15"].pred * 100)}% on average and ${Math.round(acc.boom["15"].actual * 100)}% did; 20+ said ${Math.round(acc.boom["20"].pred * 100)}%, ${Math.round(acc.boom["20"].actual * 100)}% did.` : ""}
        {" "}No bust chance: it showed no skill in backtests.</p>
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

/** What would change the call: the projection gap that flips a 2-player start/sit to 50/50, the known
 *  adjustments on this page drawn against it, and (for your own players) the win-chance difference. */
export function DecisionBox({ files, ratioQ, tw }: { files: PF[]; ratioQ?: Record<string, number[]>; tw?: TeamWeek | null }) {
  if (files.length !== 2 || !ratioQ || files.some((f) => !f.espn_week || dead(f))) return null;
  const p = pBeats(asPt(files[0]), asPt(files[1]), ratioQ);
  if (p == null) return null;
  const [fav, dog] = p >= 0.5 ? [files[0], files[1]] : [files[1], files[0]];
  // bisection: points added to the underdog's projection until it's a coin flip
  let lo = 0, hi = 40;
  for (let k = 0; k < 30; k++) { const m = (lo + hi) / 2; (pBeats(asPt(fav), asPt(dog, m), ratioQ) ?? 0.5) > 0.5 ? (lo = m) : (hi = m); }
  const need = (lo + hi) / 2;
  const s = qs(fav, ratioQ)!, noise = (at(s, 0.9) - at(s, 0.1)) / 2;
  // known adjustments, signed toward the underdog (positive = closes the gap)
  const adj = [fav, dog].map((f) => ({ f, v: backPts(f) })).filter((x) => x.v != null).map((x) => ({ who: lastName(x.f.name), v: x.f === dog ? x.v! : -x.v!, raw: x.v! }));
  const net = adj.reduce((t, x) => t + x.v, 0);
  const weight = (v: number) => (v < 0 ? "backs the favourite" : v >= need ? "could flip it" : v >= need / 3 ? "a lean" : "a tiebreaker");
  const pAdj = adj.length ? pBeats(asPt(fav, adj.filter((x) => x.who === lastName(fav.name)).reduce((t, x) => t + x.raw, 0)), asPt(dog, adj.filter((x) => x.who === lastName(dog.name)).reduce((t, x) => t + x.raw, 0)), ratioQ) : null;
  const scale = Math.max(need, noise, ...adj.map((x) => Math.abs(x.v))) * 1.1;
  const w = (v: number) => `${(Math.abs(v) / scale) * 100}%`;
  // win chance: only when both are yours and one of them is in the current lineup
  const mine = tw ? files.map((f) => tw.players.find((x) => x.id === f.id)) : [];
  const starter = mine.find((x) => x && !["BE", "IR"].includes(x.slot));
  const other = starter ? mine.find((x) => x && x !== starter) : undefined;
  const m = tw?.matchup;
  let win: { s: string; o: string; ps: number; po: number } | null = null;
  if (m && starter && other && starter.espn_week != null && other.espn_week != null) {
    const base = m.my_proj;
    win = { s: lastName(starter.name), o: lastName(other.name), ps: phi((base - m.opp_proj) / m.margin_sd), po: phi((base - starter.espn_week + other.espn_week - m.opp_proj) / m.margin_sd) };
  }
  return (
    <div className="cmp-dec">
      <div className="row"><span className="nm">To make it 50/50</span><span className="bar"><i className="need" style={{ width: w(need) }} /></span>
        <span className="v"><b>{lastName(dog.name)} +{f1(need)}</b> <small className="dim">in his projection</small></span></div>
      <div className="row"><span className="nm">A normal week's miss</span><span className="bar"><i className="noise" style={{ width: w(noise) }} /></span>
        <span className="v"><b>±{f1(noise)}</b> <small className="dim">either way, for {lastName(fav.name)}</small></span></div>
      {adj.map((x) => <div key={x.who} className="row"><span className="nm">{x.who}: first game back</span><span className="bar"><i className={x.v > 0 ? "adj up" : "adj dn"} style={{ width: w(x.v) }} /></span>
        <span className="v"><b>{x.raw > 0 ? "+" : "−"}{f1(Math.abs(x.raw))}</b> <small className="dim">{x.v > 0 ? "closes" : "widens"} the gap · {weight(x.v)}</small></span></div>)}
      <p className="cap">{adj.length
        ? `Known ESPN misses on this page move it ${net >= 0 ? "toward" : "away from"} ${lastName(dog.name)} by ${f1(Math.abs(net))}; with them the call is ${pAdj != null ? `${Math.round(Math.max(pAdj, 1 - pAdj) * 100)}/${Math.round(Math.min(pAdj, 1 - pAdj) * 100)}` : "unchanged"}.`
        : "Nothing on this page measurably moves ESPN's number for either player: the matchup is already in it and the PFF/context rows tested null."}
        {" "}The defense, Vegas and PFF rows describe the game; none of them is a measured adjustment.
        {win && ` Win chance this week: ${Math.round(win.ps * 100)}% starting ${win.s}, ${Math.round(win.po * 100)}% with ${win.o} instead (${signedPct(win.po - win.ps)}).`}</p>
    </div>
  );
}
const signedPct = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(Math.round(d * 1000) / 10)} percentage points`;
