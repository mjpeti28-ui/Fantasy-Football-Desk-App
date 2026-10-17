import { useMemo, useState } from "react";
import "./waivers.css";
import type { DepthRow, Player, Pos, RolesBoard, Shifts, Usage, WaiverBoard, WaiverRow } from "../../types";
import { RoleSection, RoleTag } from "../../components/Role";
import { CompareBtn, useFile } from "../../components/FileContext";
import { f1, lastName, pct, signed } from "../../format";
import { Face, POS_VAR } from "../../components/bits";

/** ladder labels: last name, or the team for a defense ("49ers", not "49ers D/...") */
const ladderName = (n: string) => (n.endsWith("D/ST") ? n.replace(/\s*D\/ST$/, "") : lastName(n));

const POSITIONS: Pos[] = ["QB", "RB", "WR", "TE", "K", "DST"];
const LADDER_H = 230;
const LABEL_GAP = 15;

/** Which usage number tells the story for a position. */
export function usageLine(pos: Pos, u: Usage | null): { label: string; vals: (number | null)[]; unit: string } | null {
  if (!u || pos === "K" || pos === "DST") return null;
  if (pos === "QB") return { label: "Snaps", vals: u.snap_pct, unit: "%" };
  if (pos === "RB") return { label: "Touches", vals: u.carries.map((c, i) => (c == null && u.targets[i] == null ? null : (c ?? 0) + (u.targets[i] ?? 0))), unit: "" };
  return { label: "Targets", vals: u.targets, unit: "" };
}

function UsageSpark({ vals, color }: { vals: (number | null)[]; color: string }) {
  const mx = Math.max(1, ...vals.filter((x): x is number => x != null));
  return (
    <span className="wv-spark" aria-hidden="true">
      {vals.map((v, i) => <i key={i} className={v == null ? "none" : ""} style={{ height: v == null ? 2 : `${Math.max(2, (16 * v) / mx)}px`, background: v == null ? undefined : color }} />)}
    </span>
  );
}

/** Spread labels so they don't collide; keeps order, stays inside the column. */
function spread(ys: number[], h: number): number[] {
  const out = [...ys];
  for (let i = 1; i < out.length; i++) out[i] = Math.max(out[i], out[i - 1] + LABEL_GAP);
  const over = out.length ? out[out.length - 1] - (h - 6) : 0;
  if (over > 0) for (let i = out.length - 1; i >= 0; i--) out[i] = Math.min(out[i], (i === out.length - 1 ? h - 6 : out[i + 1] - LABEL_GAP));
  return out;
}

function Ladder({ pos, mine, fas, repl, top, sel, onSel, byes, first }: {
  first: boolean; pos: Pos; mine: DepthRow[]; fas: WaiverRow[]; repl: number; top: number; sel: number | null; onSel: (id: number) => void; byes: Map<number, number | null>;
}) {
  const y = (v: number | null) => LADDER_H - (LADDER_H * Math.min(Math.max(v ?? 0, 0), top)) / top;
  const m = [...mine].sort((a, b) => (b.est_ppg ?? 0) - (a.est_ppg ?? 0));
  const f = [...fas].sort((a, b) => (b.est_ppg ?? 0) - (a.est_ppg ?? 0)).slice(0, 3);
  const my = spread(m.map((p) => y(p.est_ppg)), LADDER_H), fy = spread(f.map((p) => y(p.est_ppg)), LADDER_H);
  const healthy = m.filter((p) => !p.ir && (p.avail ?? 1) >= 0.6).length;
  const byeList = m.filter((p) => byes.get(p.id)).map((p) => `${lastName(p.name)} ${byes.get(p.id)}`);
  return (
    <figure className="ladder" style={{ ["--pc" as string]: POS_VAR[pos] }}>
      <figcaption><b>{pos}</b><span>{healthy} healthy</span></figcaption>
      <div className="lane" style={{ height: LADDER_H }}>
        {[5, 10, 15, 20].filter((t) => t < top).map((t) => <span key={t} className="tick" style={{ top: y(t) }}>{first ? <em>{t}</em> : null}</span>)}
        <span className="spine" />
        <span className="repl" style={{ top: y(repl) }} title={`Replacement level: the best free agent here scores about ${f1(repl)} a game`} />
        {f.map((p, i) => (
          <button key={p.id} type="button" className={`fa ${sel === p.id ? "on" : ""}`} onClick={() => onSel(p.id)} aria-label={`${p.name}, free agent, ${f1(p.est_ppg)} per game`} title={`${p.name}: ${f1(p.est_ppg)} a game, plays ${pct(p.avail)}, season gain ${signed(p.gain)}`}>
            <i className="dot" style={{ top: y(p.est_ppg) }} />
            <span className="lab" style={{ top: fy[i] }}>{ladderName(p.name)}</span>
          </button>
        ))}
        {m.map((p, i) => {
          const shaky = p.ir || (p.avail ?? 1) < 0.6;
          return (
            <span key={p.id} className={`mine ${shaky ? "shaky" : ""}`} title={`${p.name}: ${f1(p.est_ppg)} a game when he plays, plays ${pct(p.avail)} of the rest${p.ir ? " (IR)" : ""}${p.drop_cost != null ? `; your lineups lose about ${f1(p.drop_cost)} season points without him` : ""}`}>
              <i className="dot" style={{ top: y(p.est_ppg) }} />
              <span className="lab" style={{ top: my[i] }}>{ladderName(p.name)}{p.ir ? " IR" : shaky ? ` ${pct(p.avail)}` : ""}</span>
            </span>
          );
        })}
      </div>
      <div className="foot">{byeList.length ? `Byes: ${byeList.join(", ")}` : " "}</div>
    </figure>
  );
}

function GainBar({ g, noise, max }: { g: number | null; noise: number; max: number }) {
  if (g == null) return <span className="dim">–</span>;
  const w = (x: number) => `${(50 * Math.min(Math.abs(x), max)) / max}%`;
  const real = Math.abs(g) >= noise;
  return (
    <span className={`gbar ${real ? (g > 0 ? "up" : "down") : "nz"}`}>
      <span className="track"><span className="band" style={{ left: `calc(50% - ${w(noise)})`, width: `calc(2 * ${w(noise)})` }} />
        <span className="fill" style={g >= 0 ? { left: "50%", width: w(g) } : { right: "50%", width: w(g) }} /></span>
      <b className="num">{signed(g)}</b>
    </span>
  );
}

export function headline(b: WaiverBoard): { main: string; sub: string } {
  const real = b.candidates.filter((c) => (c.gain ?? 0) >= b.noise).sort((a, c) => (c.gain ?? 0) - (a.gain ?? 0));
  const best = [...b.candidates].sort((a, c) => (c.gain ?? 0) - (a.gain ?? 0))[0];
  if (real.length) {
    const r = real[0];
    const sub = real.length > 1 ? `Biggest of ${real.length} moves that clear the noise (±${b.noise} season points).` : `The one move this week that clears the noise (±${b.noise} season points).`;
    const mine = (b.depth[r.pos] ?? []).filter((d) => !d.ir && (d.avail ?? 1) >= 0.6);
    const why = mine.length === 1 ? ` You carry one healthy ${r.pos}, so most of his value is cover for ${mine[0].name}'s bye and injury risk.` : "";
    return { main: `Add ${r.name}, drop ${r.drop_name}: ${signed(r.gain)} season points.`, sub: sub + why };
  }
  return { main: "Nothing on the wire beats what you have.", sub: best ? `Best is ${best.name} (${signed(best.gain)}), inside the ±${b.noise} noise.` : "" };
}

export function waiversMarkdown(b: WaiverBoard): string {
  const h = headline(b);
  const rows = [...b.candidates].sort((a, c) => (c.gain ?? 0) - (a.gain ?? 0)).slice(0, 12);
  return [`## Waivers, week ${b.week} (from Desk)`, "", `${h.main} ${h.sub}`, "",
    "| Player | Pos | ESPN wk | ROS ppg | Plays | Season gain | Drop |", "|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.name}${r.tag ? ` (${r.tag})` : ""} | ${r.pos} | ${f1(r.espn_wk)} | ${f1(r.est_ppg)} | ${pct(r.avail)} | ${signed(r.gain)} | ${r.drop_name ?? "–"} |`),
    "", `Season gain = simulated rest-of-season starting-lineup points with the best drop; under ±${b.noise} is noise.`,
    ...(b.keep?.length ? ["", `Won't drop (never offered as the drop): ${Object.values(b.depth).flat().filter((d) => b.keep!.includes(d.id)).map((d) => d.name).join(", ")}.`] : []),
    "", "My depth:", ...POSITIONS.map((p) => `- ${p}: ${(b.depth[p] ?? []).map((d) => `${d.name} ${f1(d.est_ppg)}${d.ir ? " IR" : ""}`).join(", ")}`)].join("\n");
}

const RISE: Record<string, string> = { snap_up: "Snap jump", share_up_with_snaps: "More looks, more snaps", share_spike: "Target spike" };

/** Free agents whose usage just jumped (the Shifts board, filtered to the wire). */
function Risers({ shifts, board, onSel }: { shifts: Shifts | null; board: WaiverBoard; onSel: (id: number) => void }) {
  const { open } = useFile();
  if (!shifts) return null;
  const inList = new Set(board.candidates.map((c) => c.id));
  const rows = shifts.rows.filter((r) => !r.owner && r.recent?.zone && RISE[r.recent.zone] && !/OUT|RESERVE|SUSPEN/i.test(r.injury ?? ""))
    .sort((a, b) => (b.recent!.hold ?? 0.4) * (b.recent!.d_snap ?? 0) + (b.recent!.d_share ?? 0) - ((a.recent!.hold ?? 0.4) * (a.recent!.d_snap ?? 0) + (a.recent!.d_share ?? 0)))
    .slice(0, 6);
  if (!rows.length) return null;
  return (
    <section className="risers" aria-label="Usage jumps on the wire">
      <h3>Usage jumps on the wire <span>{rows[0].recent?.label ?? "last 2 games vs the 2 before"} · from Shifts</span></h3>
      <div className="rs-grid">
        {rows.map((r) => {
          const w = r.recent!;
          return (
            <button key={r.id} type="button" className="rs" onClick={() => (inList.has(r.id) ? onSel(r.id) : open(r.id, rows.map((x) => x.id)))}>
              <span className="rs-top"><i className="pchip" style={{ background: POS_VAR[r.pos], color: r.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{r.pos}</i><b>{r.name}</b><small>{r.team}</small></span>
              <span className="rs-mid"><span className="num">snaps {Math.round(w.from_snap ?? 0)}→{Math.round(w.to_snap ?? 0)}%</span>{w.to_share != null ? <span className="num"> · share {Math.round(w.from_share ?? 0)}→{Math.round(w.to_share)}%</span> : null}</span>
              <span className="rs-bot"><em>{RISE[w.zone!]}{w.cause?.length ? `, ${w.cause.join(" & ")} out` : ""}</em><span>{w.hold != null ? `held ${pct(w.hold)}` : ""}</span>
                <span className="dim">{r.ledger?.est_ppg != null ? `model ${f1(r.ledger.est_ppg)}` : ""}{inList.has(r.id) ? " · on the shortlist" : ""}</span></span>
            </button>
          );
        })}
      </div>
      <p className="dim">"Held" = how often changes like this kept at least half their size over the next two games (2016–25). In backtests ESPN's next projection already reflected most of these, so the edge is getting there before your league, not before ESPN.</p>
    </section>
  );
}

function NextWeek({ board, onSel }: { board: WaiverBoard; onSel: (id: number) => void }) {
  const { open } = useFile();
  const n = board.next_week;
  if (!n || !n.byes.length) return null;
  const inList = new Set(board.candidates.map((c) => c.id));
  return (
    <section className="risers" aria-label={`Week ${n.week} byes`}>
      <h3>Week {n.week}: {n.byes.map((b) => `${lastName(b.name)} (${b.pos})`).join(", ")} on bye <span>your lineup that week projects {f1(n.lineup)} · one-week adds, ignoring the drop</span></h3>
      {n.adds.length ? (
        <div className="rs-grid">
          {n.adds.map((a) => (
            <button key={a.id} type="button" className="rs" onClick={() => (inList.has(a.id) ? onSel(a.id) : open(a.id, n.adds.map((x) => x.id)))}>
              <span className="rs-top"><i className="pchip" style={{ background: POS_VAR[a.pos], color: a.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{a.pos}</i><b>{a.name}</b><small>{a.team ?? "FA"}</small></span>
              <span className="rs-bot"><em>+{f1(a.gain)} to your week {n.week} lineup</em><span className="dim">{f1(a.pts)} expected{a.tag && a.tag !== "ACTIVE" ? ` · ${a.tag.toLowerCase()}` : ""}</span></span>
            </button>
          ))}
        </div>
      ) : <p className="dim">Your bench covers every bye that week.</p>}
      <p className="dim">Expected = our rest-of-season points per game times his chance of playing. For one week only: the season gain below is what decides whether he's worth a roster spot.</p>
    </section>
  );
}

export function Waivers({ board, err, sel, onSel, players, roles, shifts, onKeep, busy }: { board: WaiverBoard | null; err: string | null; sel: number | null; onSel: (id: number) => void; players: Player[]; roles: RolesBoard | null; shifts?: Shifts | null; onKeep?: (ids: number[]) => void; busy?: boolean }) {
  const [filter, setFilter] = useState<Pos | "ALL">("ALL");
  const [pick, setPick] = useState(false);
  const { open, compare } = useFile();
  const byes = useMemo(() => new Map(players.map((p) => [p.id, p.bye ?? null])), [players]);
  if (err) return <div className="waivers"><p className="wv-off">{err}</p></div>;
  if (!board) return <div className="waivers"><header className="wv-hd"><h2>Waivers</h2><p className="wv-sub">Simulating every top free agent with each possible drop… about 15 seconds the first time.</p></header></div>;
  const top = Math.max(20, ...Object.values(board.depth).flat().map((d) => d.est_ppg ?? 0), ...board.candidates.map((c) => c.est_ppg ?? 0)) + 1;
  const rows = board.candidates.filter((c) => filter === "ALL" || c.pos === filter).sort((a, b) => (b.gain ?? 0) - (a.gain ?? 0));
  const gmax = Math.max(board.noise * 2, ...board.candidates.map((c) => Math.abs(c.gain ?? 0)));
  const h = headline(board);
  const kept = new Set(board.keep ?? []);
  const toggle = (id: number) => onKeep?.(kept.has(id) ? [...kept].filter((x) => x !== id) : [...kept, id]);
  const droppable = Object.values(board.depth).flat().filter((d) => !d.ir).sort((a, b) => (board.drop_cost?.[String(a.id)] ?? 0) - (board.drop_cost?.[String(b.id)] ?? 0));
  const keptRows = droppable.filter((d) => kept.has(d.id));
  const usual = (() => {
    const n = new Map<number, number>(); for (const c of board.candidates) if (c.drop_id != null) n.set(c.drop_id, (n.get(c.drop_id) ?? 0) + 1);
    const [id, k] = [...n].sort((x, y) => y[1] - x[1])[0] ?? [null, 0];
    const c = id != null ? board.drop_cost?.[String(id)] : undefined, nm = board.candidates.find((x) => x.drop_id === id)?.drop_name;
    return id != null && k >= board.candidates.length * 0.6 && c != null ? { id, name: nm ?? "", cost: c } : null;
  })();
  return (
    <div className="waivers">
      <header className="wv-hd">
        <span className="eyebrow">Waivers · week {board.week}</span>
        <h2>{h.main}</h2>
        <p className="wv-sub">{h.sub}</p>
      </header>
      <section className="ladders" aria-label="Depth by position">
        {POSITIONS.map((p) => <Ladder key={p} pos={p} mine={board.depth[p] ?? []} fas={board.candidates.filter((c) => c.pos === p)} repl={board.replacement[p]} top={top} sel={sel} onSel={onSel} byes={byes} first={p === "QB"} />)}
      </section>
      <p className="wv-key"><span className="k-mine" /> yours <span className="k-fa" /> best free agents <span className="k-repl" /> replacement level · height is rest-of-season points per game when playing (hover for numbers), same scale everywhere</p>
      <NextWeek board={board} onSel={onSel} />
      <Risers shifts={shifts ?? null} board={board} onSel={onSel} />
      <section className="wv-list">
        <div className="wv-tools">
          <div className="seg" role="group" aria-label="Position">
            {(["ALL", ...POSITIONS] as const).map((p) => <button key={p} type="button" aria-pressed={filter === p} onClick={() => setFilter(p)}>{p === "ALL" ? "All" : p}</button>)}
          </div>
          <span className="dim">Sorted by season gain with the best drop{usual ? ` · usual drop: ${usual.name}, who carries ${f1(usual.cost)} season points for you` : ""}</span>
          {usual && onKeep && <button type="button" className="tbtn" onClick={() => toggle(usual.id)} title="Stop suggesting him as the drop">Keep {lastName(usual.name)}</button>}
        </div>
        {onKeep && (
          <div className="wv-keep">
            <span className="lbl">Won't drop</span>
            {keptRows.length ? keptRows.map((d) => <button key={d.id} type="button" className="chip" onClick={() => toggle(d.id)} title="Click to make him droppable again">{d.name} <i aria-hidden="true">×</i></button>) : <span className="dim">nobody: every non-IR player is fair game</span>}
            <button type="button" className="tbtn" aria-expanded={pick} onClick={() => setPick(!pick)}>{pick ? "Done" : "Choose who you'd drop…"}</button>
            {busy && <span className="dim">Rebuilding without them…</span>}
          </div>
        )}
        {pick && onKeep && (
          <div className="wv-pick" role="group" aria-label="Players you'd be willing to drop">
            <p className="dim">Checked = you'd drop him. The board picks the best drop among the checked players. Cheapest to lose first.</p>
            <div className="pk-grid">
              {droppable.map((d) => (
                <label key={d.id} className={kept.has(d.id) ? "off" : ""}>
                  <input type="checkbox" checked={!kept.has(d.id)} onChange={() => toggle(d.id)} />
                  <b>{d.name}</b>
                  <small className="num">{f1(d.est_ppg)} ppg · plays {pct(d.avail)} · carries {f1(d.drop_cost ?? null)}</small>
                </label>
              ))}
            </div>
          </div>
        )}
        <table className="wv-table">
          <thead><tr><th>Free agent</th><th>Role</th><th className="r">ESPN wk</th><th className="r">ROS ppg</th><th className="r" title="Chance his rest-of-season ppg reaches a weekly starter at his position, from his ROS ppg alone (2021-25 backtest)">Starter odds</th><th className="r">Plays</th><th>Last 4 weeks</th><th className="gcol">Season gain</th><th>Drop</th></tr></thead>
          <tbody>
            {rows.map((r) => {
              const u = usageLine(r.pos, r.usage);
              return (
                <tr key={r.id} className={sel === r.id ? "on" : ""} onClick={(e) => { if (e.metaKey && r.pos !== "K" && r.pos !== "DST") compare(r.id); else onSel(r.id); }} onDoubleClick={() => open(r.id, rows.map((x) => x.id))} tabIndex={0}
                  onKeyDown={(e) => { if (e.key === "Enter") onSel(r.id); }} title="Double-click (or select and press space) for the full player file">
                  <td><span className="who"><i className="pchip" style={{ background: POS_VAR[r.pos], color: r.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{r.pos}</i><Face p={r} /><span><b>{r.name}</b><small>{r.team ?? "FA"}{r.tag ? <em className="tag"> · {r.tag}</em> : ""}</small></span></span></td>
                  <td>{r.pos === "K" || r.pos === "DST" ? <span className="dim">–</span> : <RoleTag r={roles?.players[String(r.id)]} />}</td>
                  <td className="r num">{f1(r.espn_wk)}</td>
                  <td className="r num">{f1(r.est_ppg)}</td>
                  <td className="r num" title={r.p_boom != null ? `Top-half starter: ${pct(r.p_boom)}` : undefined}>{pct(r.p_start)}</td>
                  <td className="r num">{pct(r.avail)}</td>
                  <td>{u ? <span className="use"><UsageSpark vals={u.vals} color={POS_VAR[r.pos]} /><small>{u.label}<span className="nw">{u.vals.map((v) => (v == null ? "–" : Math.round(v))).join(" ")}{u.unit}</span></small></span> : <span className="dim">–</span>}</td>
                  <td className="gcol"><GainBar g={r.gain} noise={board.noise} max={gmax} /></td>
                  <td className="drop" title={r.drop_name ?? undefined}>{r.drop_name ? lastName(r.drop_name) : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {board.left_off.length > 0 && <p className="dim">Left off (out or on IR): {board.left_off.map((l) => `${l.name} ${l.pos}`).join(", ")}.</p>}
        <p className="dim">Season gain is rest-of-season starting-lineup points (regular season and playoffs) after adding him and dropping the player shown, simulated against your current roster. The grey band is ±{board.noise}: moves inside it are coin flips.</p>
      </section>
    </div>
  );
}

export function WaiverInspector({ r, board, onLog, onCopy, roles, onKeep }: { r: WaiverRow; board: WaiverBoard; onLog: () => void; onCopy: () => void; roles: RolesBoard | null; onKeep?: (ids: number[]) => void }) {
  const drop = Object.values(board.depth).flat().find((d) => d.id === r.drop_id);
  const real = Math.abs(r.gain ?? 0) >= board.noise;
  const u = r.usage;
  const showUse = u && r.pos !== "K" && r.pos !== "DST";
  return (
    <div className="ins wv-ins">
      <div className="ins-hd"><Face p={r} /><div><div className="nm">{r.name}</div><div className="dim">{r.pos} · {r.team ?? "FA"}{r.tag ? ` · ${r.tag}` : ""}</div></div></div>
      <section className={`wv-verdict ${real ? (r.gain! > 0 ? "up" : "down") : "nz"}`}>
        <b className="num">{signed(r.gain)}</b>
        <span>season points if you add him and drop {r.drop_name ?? "nobody"}. {real ? "Clears the noise." : `Inside the ±${board.noise} noise: a coin flip.`}</span>
      </section>
      <section>
        <h6>Him vs the drop</h6>
        <dl className="kv cmp">
          <dt /><dd>{lastName(r.name)}</dd><dd>{drop ? lastName(drop.name) : "–"}</dd>
          <dt>ROS ppg when playing</dt><dd>{f1(r.est_ppg)}</dd><dd>{f1(drop?.est_ppg ?? null)}</dd>
          <dt title="Chance his rest-of-season ppg reaches a weekly starter / a top-half starter at his position, from ROS ppg alone">Starter · top-half odds</dt><dd>{pct(r.p_start)} · {pct(r.p_boom)}</dd><dd>–</dd>
          <dt>Plays rest of season</dt><dd>{pct(r.avail)}</dd><dd>{pct(drop?.avail ?? null)}{drop?.ir ? " IR" : ""}</dd>
          <dt>ESPN this week</dt><dd>{f1(r.espn_wk)}</dd><dd>–</dd>
          <dt>Experts' rank</dt><dd>{r.ecr_rank ? `${r.pos}${Math.round(r.ecr_rank)}` : "–"}</dd><dd>–</dd>
          <dt title="Season lineup points your roster loses without him">Carries for you</dt><dd>–</dd><dd>{drop?.drop_cost != null ? f1(drop.drop_cost) : "–"}</dd>
        </dl>
      </section>
      <RoleSection r={roles?.players[String(r.id)]} board={roles} pos={r.pos} id={r.pos === "K" || r.pos === "DST" ? undefined : r.id} list={board.candidates.map((c) => c.id)} />
      {showUse && (
        <section>
          <h6>Usage, weeks {u!.weeks[0]}–{u!.weeks[u!.weeks.length - 1]}</h6>
          <table className="wv-use">
            <thead><tr><th />{u!.weeks.map((w) => <th key={w}>{w}</th>)}</tr></thead>
            <tbody>
              <tr><th>Snaps %</th>{u!.snap_pct.map((v, i) => <td key={i}>{v == null ? "–" : Math.round(v)}</td>)}</tr>
              {r.pos !== "QB" && <tr><th>Targets</th>{u!.targets.map((v, i) => <td key={i}>{v == null ? "–" : Math.round(v)}</td>)}</tr>}
              {r.pos !== "QB" && <tr><th>Tgt share %</th>{u!.target_share.map((v, i) => <td key={i}>{v == null ? "–" : Math.round(v)}</td>)}</tr>}
              {(r.pos === "RB" || r.pos === "QB") && <tr><th>Carries</th>{u!.carries.map((v, i) => <td key={i}>{v == null ? "–" : Math.round(v)}</td>)}</tr>}
            </tbody>
          </table>
        </section>
      )}
      <section><h6>Why this number</h6><p className="wv-why">{r.why}</p></section>
      <div className="ins-acts"><button className="tbtn pri" type="button" onClick={onLog}>Log to journal…</button>{r.pos !== "K" && r.pos !== "DST" && <CompareBtn id={r.id} />}{drop && onKeep && <button className="tbtn" type="button" onClick={() => onKeep([...(board.keep ?? []), drop.id])} title="Stop suggesting him as the drop">Don't drop {lastName(drop.name)}</button>}<button className="tbtn" type="button" onClick={onCopy}>Copy for Claude</button></div>
      <p className="ins-note">Desk is read-only with ESPN for now: make the claim in the ESPN app.</p>
    </div>
  );
}
