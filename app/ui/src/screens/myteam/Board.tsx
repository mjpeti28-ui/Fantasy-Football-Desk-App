import { TrendGlyph } from "../../components/Role";
import { useFile } from "../../components/FileContext";
import { useEffect, useMemo, useRef, useState } from "react";
import type { RolesBoard, Player, TeamWeek } from "../../types";
import { f1, gameLabel, lastName, pct, phi, signed } from "../../format";
import { Face, H2H, POS_VAR, Proj, isDead, statusOf } from "../../components/bits";
import { pBeats } from "../../odds";
import { type ColorMode, DEAD_PALETTE, type TeamColor, tilePalette } from "../../teamColor";
import { STARTERS, type Plan, type Slot, canPlay, deadStarters, fixDead, move, planFrom, planPoints, rival, slotOf } from "./lineup";
import { Carries, LineupCheck, SlotDuel } from "./Insights";
import "./board.css";

interface Props {
  roles?: RolesBoard | null;
  tw: TeamWeek;
  plan: Plan;
  setPlan: (p: Plan) => void;
  selected: number | null;
  onSelect: (id: number) => void;
  ratioQ?: Record<string, number[]>;
  colorMode: ColorMode;
  teamColors?: Record<string, TeamColor>;
  dropCost?: Record<string, number> | null;
  onCopy?: (text: string) => void;
}

function useDark() {
  const q = typeof window !== "undefined" ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  const [dark, setDark] = useState(!!q?.matches);
  useEffect(() => { if (!q) return; const f = () => setDark(q.matches); q.addEventListener("change", f); return () => q.removeEventListener("change", f); }, [q]);
  return dark;
}

type Target = Slot | "BENCH";
const LAYOUT: Slot[] = ["QB", "RB", "WR", "TE", "RB/WR", "WR/TE", "FLEX", "K", "DST"];
const DRAG_PX = 5;

/** Pointer-based drag (HTML5 drag-and-drop is unreliable inside the macOS webview). */
function usePointerDrag(onDrop: (id: number, to: Target) => void, onClick: (id: number) => void) {
  const [drag, setDrag] = useState<{ id: number; x: number; y: number; over: Target | null } | null>(null);
  const pending = useRef<{ id: number; x0: number; y0: number; started: boolean } | null>(null);
  const targetAt = (x: number, y: number): Target | null => {
    const el = document.elementFromPoint(x, y)?.closest("[data-drop]") as HTMLElement | null;
    return (el?.dataset.drop as Target) ?? null;
  };
  useEffect(() => {
    const move = (e: PointerEvent) => {
      const p = pending.current;
      if (!p) return;
      if (!p.started && Math.hypot(e.clientX - p.x0, e.clientY - p.y0) < DRAG_PX) return;
      p.started = true;
      setDrag({ id: p.id, x: e.clientX, y: e.clientY, over: targetAt(e.clientX, e.clientY) });
    };
    const up = (e: PointerEvent) => {
      const p = pending.current;
      pending.current = null;
      if (!p) return;
      if (p.started) { const t = targetAt(e.clientX, e.clientY); if (t) onDrop(p.id, t); }
      else onClick(p.id);
      setDrag(null);
    };
    const cancel = (e: KeyboardEvent) => { if (e.key === "Escape") { pending.current = null; setDrag(null); } };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("keydown", cancel);
    return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("keydown", cancel); };
  }, [onDrop, onClick]);
  const start = (id: number) => (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    pending.current = { id, x0: e.clientX, y0: e.clientY, started: false };
  };
  return { drag, start };
}

export function Board({ tw, plan, setPlan, selected, onSelect, ratioQ, colorMode, teamColors, roles, dropCost, onCopy }: Props) {
  const { open: openFile, compare } = useFile();
  const dark = useDark();
  const by = useMemo(() => new Map(tw.players.map((p) => [p.id, p])), [tw]);
  const espnPlan = useMemo(() => planFrom(tw.lineup.current), [tw]);
  const planRef = useRef(plan);
  planRef.current = plan;
  const { drag, start } = usePointerDrag(
    useMemo(() => (id: number, to: Target) => { const next = move(planRef.current, by, id, to); if (next) setPlan(next); }, [by, setPlan]),
    onSelect,
  );
  const pts = planPoints(plan, by);
  const m = tw.matchup;
  const pWin = m ? phi((pts - m.opp_proj) / m.margin_sd) : null;
  const dead = deadStarters(plan, by);
  const changed = STARTERS.filter((s) => plan[s] !== espnPlan[s]);
  const dragP = drag ? by.get(drag.id) ?? null : null;
  const legal = (to: Target) => !!dragP && (to === "BENCH" ? !!slotOf(plan, dragP.id) : canPlay(dragP, to) && slotOf(plan, dragP.id) !== to);

  const tip = (p: Player) => {
    const r = rival(p, plan, by);
    if (!r) return <span className="h2h"><b>No healthy alternative on the roster</b><span className="sub">ESPN {f1(p.espn_week)} this week · {pct(p.avail_now)} to play</span></span>;
    const prob = r.role === "bench" ? pBeats(p, r.other, ratioQ) : pBeats(p, r.other, ratioQ);
    return <H2H p={p} other={r.other} slot={r.slot} role={r.role} prob={prob} />;
  };

  // 3D: a slight tilt toward the pointer (Tint mode only).
  const tiltMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget, r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    el.style.setProperty("--rx", `${(0.5 - y) * 3}deg`); el.style.setProperty("--ry", `${(x - 0.5) * 4}deg`);
    el.style.setProperty("--mx", `${x * 100}%`); el.style.setProperty("--my", `${y * 100}%`);
  };
  const tiltReset = (e: React.PointerEvent<HTMLDivElement>) => {
    for (const k of ["--rx", "--ry", "--mx", "--my"]) e.currentTarget.style.removeProperty(k);
  };

  const tile = (p: Player, label: string, extra = "") => {
    const st = statusOf(p);
    const deadNow = isDead(p);
    const movable = p.slot !== "IR";
    // Players who can't score lose their team color so the bye/out cue always reads first.
    const pal = colorMode === "position" ? null : deadNow ? DEAD_PALETTE(dark) : tilePalette(p.team, teamColors, dark);
    const vars: Record<string, string> = { "--c": POS_VAR[p.pos] };
    if (pal) Object.assign(vars, { "--tc": pal.sig, "--tc2": pal.second, "--tbg": pal.bg, "--tac": pal.accent, "--thi": pal.hi, "--tlo": pal.lo });
    const tilt = colorMode === "tint" && !drag;
    return (
      <div
        key={p.id}
        className={`tile ${pal ? "team" : ""} ${deadNow ? "dead" : ""} ${selected === p.id ? "sel" : ""} ${drag?.id === p.id ? "lifted" : ""} ${movable ? "" : "fixed"} ${extra}`}
        style={vars}
        title={pal && p.team ? `${p.team}` : undefined}
        onPointerDown={movable ? start(p.id) : undefined}
        onPointerMove={tilt ? tiltMove : undefined}
        onPointerLeave={tilt ? tiltReset : undefined}
        onClick={movable ? undefined : () => onSelect(p.id)}
        onDoubleClick={() => openFile(p.id, tw.players.map((x) => x.id))}
        onClickCapture={(e) => { if (e.metaKey && p.pos !== "K" && p.pos !== "DST") { e.stopPropagation(); compare(p.id); } }}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(p.id); } }}
        tabIndex={0}
        role="button"
        aria-pressed={selected === p.id}
        aria-label={`${p.name}, ${label}, ${f1(p.espn_week)} projected, ${st.label}`}
      >
        {deadNow && <span className="tag">{st.cls === "bye" ? "BYE" : st.label.toUpperCase()}</span>}
        {pal && colorMode === "tint" && <span className="sleeve-clip" aria-hidden="true"><span className="sleeve"><i /><i /></span></span>}
        <span className="fw"><Face p={p} className="round" /></span>
        <span className="n">{lastName(p.name)}<TrendGlyph r={roles?.players[String(p.id)]} /></span>
        <span className="p"><Proj p={p} tip={drag ? undefined : tip(p)} /></span>
        <span className="s">{pal && <i className={`pchip ${p.pos === "WR" ? "wr" : ""}`}>{p.pos === "DST" ? "D/ST" : p.pos}</i>}{label} · {gameLabel(p.game)}{p.game && !p.game.bye ? ` ${p.game.day}` : ""}</span>
      </div>
    );
  };

  const slotBox = (s: Slot) => {
    const id = plan[s];
    const p = id != null ? by.get(id) : undefined;
    const ok = legal(s);
    return (
      <div key={s} data-drop={s} className={`slot ${dragP ? (ok ? "can" : "cannot") : ""} ${drag?.over === s && ok ? "over" : ""} ${changed.includes(s) ? "changed" : ""}`}>
        <span className="slab">{changed.includes(s) && <i className="chg-dot" aria-hidden="true" />}{s}{changed.includes(s) && <em> · changed</em>}</span>
        {p ? tile(p, s) : <div className="empty">Empty: drag a {s === "FLEX" ? "RB, WR or TE" : s.replace("/", " or ")} here</div>}
      </div>
    );
  };

  const bench = tw.players.filter((p) => !slotOf(plan, p.id));
  const weeks = Array.from({ length: 17 - tw.week + 1 }, (_, i) => tw.week + i);
  const startersNow = STARTERS.filter((s) => plan[s] != null).map((s) => by.get(plan[s]!)!);
  const fix = fixDead(plan, by);
  const fixGain = planPoints(fix.plan, by) - pts;

  return (
    <div className={`board ${drag ? "dragging" : ""}`} data-cmode={colorMode}>
      <header className="score">
        <div className="me"><span className="t">{tw.team_name}</span><span className="v num">{f1(pts)}</span>
          <span className="sub">{changed.length ? <>Plan differs from ESPN in {changed.length} slot{changed.length > 1 ? "s" : ""} · ESPN lineup {f1(tw.lineup.current_pts)} · <button type="button" className="linkbtn" onClick={() => setPlan(espnPlan)}>Reset</button></> : <>Matches your ESPN lineup</>}</span></div>
        <div className="mid">Week {tw.week}<br />vs</div>
        {m && <div className="them"><span className="t">{m.opp_name}</span><span className="v num">{f1(m.opp_proj)}</span><span className="sub">their lineup as set</span></div>}
        {pWin != null && <div className="wp"><span className="v num">{pct(pWin)}</span><span className="sub">to win</span></div>}
      </header>

      {dead.length > 0 && (
        <div className="alert" role="status">
          <span className="alert-txt">{dead.map((d) => `${lastName(d.p.name)} ${d.p.game?.bye ? "is on bye" : "is " + statusOf(d.p).label.toLowerCase()} at ${d.slot}`).join("; ")}.</span>
          {fix.moves.length > 0 && <button className="tbtn pri" type="button" onClick={() => setPlan(fix.plan)}>
            {fix.moves.map((mv) => `Start ${lastName(mv.in.name)} at ${mv.slot}`).join(", ")} <span className="num">{signed(fixGain)}</span></button>}
        </div>
      )}

      <LineupCheck tw={tw} plan={plan} by={by} ratioQ={ratioQ} setPlan={setPlan} onCopy={onCopy} />

      <section aria-label="Starting lineup"><h5>Starting lineup <span>drag a tile onto a slot or the bench, or select a player and use Move in the side panel</span></h5><div className="slots">{LAYOUT.map(slotBox)}</div></section>

      <section className={`tray ${dragP && legal("BENCH") ? "can" : ""} ${drag?.over === "BENCH" && legal("BENCH") ? "over" : ""}`} aria-label="Bench" data-drop="BENCH">
        <h5>Bench &amp; IR <span>{bench.filter((p) => !isDead(p)).length} usable this week</span></h5>
        <div className="tray-row">{bench.map((p) => tile(p, p.slot === "IR" ? "IR" : "BN", "small"))}</div>
      </section>

      <div className="bd-two">
        <SlotDuel tw={tw} plan={plan} by={by} ratioQ={ratioQ} />
        <Carries tw={tw} cost={dropCost} plan={plan} />
      </div>

      <section className="byes" aria-label="Bye weeks ahead">
        <h5>Byes ahead for this lineup <span>weeks 15–17 are the playoffs</span></h5>
        <table>
          <thead><tr><th />{weeks.map((w) => <th key={w} className={w > 14 ? "po" : ""}>{w}</th>)}</tr></thead>
          <tbody>
            {startersNow.map((p) => (
              <tr key={p.id}><td className="n">{lastName(p.name)}</td>{weeks.map((w) => <td key={w} className={`${w > 14 ? "po" : ""} ${p.bye === w ? "bye" : ""}`} title={p.bye === w ? `${p.name}: bye in week ${w}` : undefined} />)}</tr>
            ))}
            <tr className="cnt"><td />{weeks.map((w) => { const c = startersNow.filter((p) => p.bye === w).length; return <td key={w}>{c ? `${c} out` : ""}</td>; })}</tr>
          </tbody>
        </table>
      </section>
      <p className="foot">Points are ESPN's weekly projections, the best single-week source in backtests. Hover a number for how often that player outscores his real alternative. Weekly scores swing so much that two players a few points apart split often. Changes here are a plan; set them in ESPN until writes are built.</p>

      {drag && dragP && (
        <div className="ghost" style={{ left: drag.x + 12, top: drag.y + 12, ["--c" as string]: POS_VAR[dragP.pos] }} aria-hidden="true">
          <span className="n">{lastName(dragP.name)}</span><span className="num">{f1(dragP.espn_week)}</span>
          <span className="s">{drag.over ? (legal(drag.over) ? `→ ${drag.over === "BENCH" ? "bench" : drag.over}` : `can't play ${drag.over}`) : dragP.pos}</span>
        </div>
      )}
    </div>
  );
}
